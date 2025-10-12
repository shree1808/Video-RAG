import torch
import numpy as np
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
import clip
import json
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv
load_dotenv()

class VideoRAGChromaDB:
    def __init__(self, db_path="chroma_db"):
        # Initialize models
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create collections with proper distance metrics
        try:
            self.client.delete_collection("all_text_approach")
        except:
            pass
        
        # define the index. 
        self.transcript_collection = self.client.create_collection(
            name="all_text_approach",
            metadata={"hnsw:space": "cosine"}
        )
     
    def add_to_index(self, clip_data: dict):
        try:
            clip_id = clip_data['clip_id']
            transcript_id = f"{clip_id}_transcript"
            raw_transcript = clip_data['transcript']
            
            # Get the transcript embeddings
            transcript_embeddings = self.text_model.encode(raw_transcript)
            doc_text = f'For clip with clip id: {clip_id} following is the transcription: {raw_transcript}'

            # Flatten metadata - ChromaDB only supports primitive types (str, int, float, bool)
            all_metadata = {
                "transcript_id": transcript_id, 
                "video_name": clip_data['video_name'], 
                "start_time": float(clip_data['start_time']), 
                "end_time": float(clip_data['end_time']), 
                "audio_path": clip_data['audio_path'],
                "content_type": "audio_transcribe",
                "num_frames": len(clip_data['frames'])
            }
            
            # Store frame details as separate metadata fields
            frame_ids = []
            frame_times = []
            frame_paths = []
            frame_emotions = []
            frame_actions = []
            
            frame_text_embeddings = []
            for i, frame in enumerate(clip_data['frames']):
                frame_id = f"{clip_id}_frame_{i}"
                frame_text = (
                    f"Inside the frame, entities captured: "
                    f"emotion is {frame['emotion']}, "
                    f"action is {frame['action']}, "
                    f"caption says {frame['caption']}, "
                    f"description is {frame['description']}."
                )
                doc_text += f' And following are the details captured from the frame: {frame_text}'

                # Get Frame text embeddings
                frame_text_embeddings.append(self.text_model.encode(frame_text))
                
                # Collect frame data for serialization
                frame_ids.append(frame_id)
                frame_times.append(float(frame['frame_time']))
                frame_paths.append(frame['image_path'])
                frame_emotions.append(frame['emotion'])
                frame_actions.append(frame['action'])
            
            # Store frames as JSON string (workaround for ChromaDB limitation)
            if frame_ids:
                all_metadata["frames_json"] = json.dumps({
                    "frame_ids": frame_ids,
                    "frame_times": frame_times,
                    "image_paths": frame_paths,
                    "emotions": frame_emotions,
                    "actions": frame_actions
                })
            
            # Combine embeddings by averaging
            if frame_text_embeddings:
                # Stack frame embeddings vertically
                stacked_frame_embeddings = np.vstack(frame_text_embeddings)
                # Average the frame embeddings
                avg_frame_embeddings = np.mean(stacked_frame_embeddings, axis=0)
                # Average transcript and frame embeddings to maintain dimension
                multimodal_emb = (transcript_embeddings + avg_frame_embeddings) / 2
            else:
                multimodal_emb = transcript_embeddings
            
            # Convert to list for ChromaDB
            multimodal_emb = multimodal_emb.tolist()
            
            self.transcript_collection.add(
                ids=[clip_id],
                embeddings=[multimodal_emb],
                documents=[doc_text],
                metadatas=[all_metadata]
            )
            
            print(f"Successfully added clip: {clip_id}")

        except Exception as e:
            print(f"Error while adding multimodal clip to index: {clip_id} with error: {e}")
            import traceback
            traceback.print_exc()

    def get_frames_from_metadata(self, metadata: dict) -> List[Dict]:
        """Helper method to deserialize frames from metadata"""
        if "frames_json" in metadata:
            frames_data = json.loads(metadata["frames_json"])
            # Reconstruct frames list
            frames = []
            for i in range(len(frames_data["frame_ids"])):
                frames.append({
                    "frame_id": frames_data["frame_ids"][i],
                    "frame_time": frames_data["frame_times"][i],
                    "image_path": frames_data["image_paths"][i],
                    "emotion": frames_data["emotions"][i],
                    "action": frames_data["actions"][i],
                    "content_type": "frames"
                })
            return frames
        return []

    def search(self, query: str, n_results: int = 5, 
               filter_metadata: Optional[Dict[str, Any]] = None,
               return_for_llm: bool = False):
        """Search in transcript collection."""
        query_emb = self.text_model.encode(query).tolist()
        
        results = self.transcript_collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            where=filter_metadata
        )
        
        # Sort results by distance (ascending - lower is better)
        sorted_indices = np.argsort(results['distances'][0])
        
        # If return_for_llm flag is set, return formatted context
        if return_for_llm:
            return self._format_results_for_llm(results, sorted_indices, query)
        
        # # Pretty print results
        # # print(f"\n{'='*80}")
        # # print(f"Query: {query}")
        # # print(f"{'='*80}\n")
        
        # for rank, idx in enumerate(sorted_indices, 1):
        #     doc_id = results['ids'][0][idx]
        #     doc = results['documents'][0][idx]
        #     metadata = results['metadatas'][0][idx]
        #     distance = results['distances'][0][idx]
            
        #     i = rank - 1
        #     # print(f"Result {rank} (Distance: {distance:.4f}, Similarity Score: {1-distance:.4f})")
        #     # print(f"Clip ID: {doc_id}")
        #     # print(f"Video: {metadata.get('video_name', 'N/A')}")
        #     # print(f"Time Range: {metadata.get('start_time', 0):.1f}s - {metadata.get('end_time', 0):.1f}s")
        #     # print(f"Audio Path: {metadata.get('audio_path', 'N/A')}")
        #     # print(f"Number of Frames: {metadata.get('num_frames', 0)}")
            
        #     # Get frame details
        #     frames = self.get_frames_from_metadata(metadata)
        #     if frames:
        #         print(f"\nFrame Details:")
        #         for frame in frames:
        #             print(f"  - Frame at {frame['frame_time']:.1f}s: {frame['emotion']} / {frame['action']}")
        #             print(f"    Image: {frame['image_path']}")
            
        #     # print(f"\nDocument Preview: {doc[:200]}...")
        #     # print(f"{'-'*80}\n")
        
        return results
    
    def _format_results_for_llm(self, results: Dict, sorted_indices: np.ndarray, query: str) -> str:
        """Format search results as context for LLM."""
        context_parts = []
        
        for rank, idx in enumerate(sorted_indices, 1):
            doc_id = results['ids'][0][idx]
            doc = results['documents'][0][idx]
            metadata = results['metadatas'][0][idx]
            distance = results['distances'][0][idx]
            similarity = 1 - distance
            
            # Get frame details
            frames = self.get_frames_from_metadata(metadata)
            
            context_part = f"""
--- Clip {rank} (Relevance Score: {similarity:.2f}) ---
Clip ID: {doc_id}
Video: {metadata.get('video_name', 'N/A')}
Time Range: {metadata.get('start_time', 0):.1f}s to {metadata.get('end_time', 0):.1f}s
Audio Path: {metadata.get('audio_path', 'N/A')}

Transcript:
{doc}

Visual Information:
"""
            if frames:
                for frame in frames:
                    context_part += f"- At {frame['frame_time']:.1f}s: Emotion={frame['emotion']}, Action={frame['action']}\n"
            else:
                context_part += "No frame information available.\n"
            
            context_parts.append(context_part)
        
        full_context = "\n".join(context_parts)
        return full_context
    
    def query_with_llm(self, query: str, n_results: int = 5, 
                       groq_api_key: str = os.getenv('GROQ_API_KEY'), model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        """Search and get LLM response using Groq."""
        try:
            from groq import Groq
        except ImportError:
            print("Please install groq: pip install groq")
            return None
        
        # Get formatted context
        context = self.search(query, n_results=n_results, return_for_llm=True)
        
        # Initialize Groq client
        client = Groq(api_key=groq_api_key)
        
        # Create prompt
        prompt = f"""You are a helpful video analysis assistant. Based on the video clips and their transcripts provided below, answer the user's question accurately.

User Question: {query}

Retrieved Video Clips Context:
{context}

Instructions:
- Answer the question based ONLY on the information provided in the clips above
- Include specific timestamps when referencing information
- If the answer cannot be found in the provided clips, say so clearly
- Be concise but thorough

Answer:"""
        
        # Get LLM response
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )
        
        response = chat_completion.choices[0].message.content
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")
        print(f"LLM Response:\n{response}")
        print(f"\n{'='*80}\n")
        
        return {
            "query": query,
            "context": context,
            "response": response
        }
    
def process_video_dataset(clips_data: List[Dict[str, Any]]):
        """Process entire video dataset."""
        
        # Initialize the RAG system
        rag = VideoRAGChromaDB()
        
        # Process all clips
        for clip_data in clips_data:
            print(f"\nProcessing clip: {clip_data['clip_id']}")
            rag.add_to_index(clip_data)
        
        print(f"\n{'='*80}")
        print(f"Successfully indexed {len(clips_data)} clips")
        print(f"{'='*80}\n")
        return rag



if __name__ == "__main__":

    import json 
    with open('C:/Users/Anil.Bhallavi/Desktop/work/Data Science/VideoRAG/video_index.json') as f:
        dataset  = json.load(f)
    
    def process_video_dataset(clips_data: List[Dict[str, Any]]):

        """Process entire video dataset."""
        
        # Initialize the RAG system
        rag = VideoRAGChromaDB()
        
        # Process all clips
        for clip_data in clips_data:
            print(f"\nProcessing clip: {clip_data['clip_id']}")
            rag.add_to_index(clip_data)
        
        print(f"\n{'='*80}")
        print(f"Successfully indexed {len(clips_data)} clips")
        print(f"{'='*80}\n")
        
        return rag
    
    # Example usage:
    rag = process_video_dataset(clips_data=dataset)
    response = rag.search("Neel Nanda mentored how many mentees", 5)
    # response = rag.query_with_llm("Neel Nanda mentored how many mentees", 5)
    print('------------====================----------', response)
