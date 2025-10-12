from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
import time 
import os
import base64
from langchain_community.chat_models import ChatOllama
import whisper
import re
from dotenv import load_dotenv

load_dotenv()
###
from models import MetadataExtraction_Frame
###

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

parser = PydanticOutputParser(pydantic_object= MetadataExtraction_Frame)

class AudioTranscriber:
    def __init__(self, audio_folder: str, model_size="medium"):
        self.audio_folder = audio_folder
        self.model = whisper.load_model(model_size)

    def transcribe_audios(self):
        transcripts = {}
        for file in os.listdir(self.audio_folder):
            if file.endswith(".wav"):
                file_path = os.path.join(self.audio_folder, file)
                result = self.model.transcribe(file_path)
                transcripts[file] = result["text"]
                print(f"Transcribed: {file}")
        return transcripts


class FrameCaptioner:
    def __init__(self, frames_folder: str, model="gemma3:12b"):  # Using your available vision model
        self.frames_folder = frames_folder
        # self.llm = ChatOllama(model=model)
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_caption(self, image_path: str) -> str:
        """Generate caption for a given image using Ollama VLM."""
        try:
            # Method 1: Using base64 encoding (recommended for Ollama)
            base64_image = self.encode_image_to_base64(image_path)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant that extracts structured metadata from images."),
                ("human", "Analyze this image and extract emotion, action, and caption.\n{format_instructions}")
            ])

            # Inject pydantic class instructions
            final_prompt = prompt.partial(format_instructions = parser.get_format_instructions())

            message = HumanMessage(
                content=[
                    {"type": "text", "text": final_prompt.format()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )

            # Try async invoke if available are else fallback to normal invoke.
            try:
                response = self.llm.ainvoke([message])
                result = parser.parse(response.content)
                # caption = f"In the frame,Emotion:{result.emotion},Action:{result.action},Caption:{result.caption}, Description: {result.description}" 
            except:
                response = self.llm.invoke([message])
                result = parser.parse(response.content)
                # caption = f"In the frame,Emotion:{result.emotion},Action:{result.action},Caption:{result.caption}, Description: {result.description}" 
            return result
            # Returns a Pydantic Object. (result.var) 
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return "Failed to generate caption"

    def caption_frames(self):
        """Caption all frames in the folder."""
        captions = {}
        rpm_counter = 0
        if not os.path.exists(self.frames_folder):
            print(f"Frames folder '{self.frames_folder}' does not exist!")
            return captions
        
        image_files = [f for f in os.listdir(self.frames_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"No image files found in '{self.frames_folder}'!")
            return captions
        
        print(f"Found {len(image_files)} images to caption...")
        
        for file in image_files:
            frame_path = os.path.join(self.frames_folder, file)
            caption = self.generate_caption(frame_path)
            captions[file] = caption
            print(f"🖼️ Captioned {file}: {caption}")
            rpm_counter +=1

            if rpm_counter >= 10:
                print("⏳ Rate limit reached. Sleeping for 60s...")
                time.sleep(60)
                rpm_counter = 0  # reset after sleep
        return captions
    
 #########
class DataBuilder:
    def __init__(self, transcripts, captions):
        self.transcripts = transcripts
        self.captions = captions

    def build(self):
        data = []

        for audio_file, transcript in self.transcripts.items():
            # Parse audio filename → clip metadata
            # Example: clip_video1_0_30.wav
            # match = re.match(r"clip_(.+)_(\d+)_(\d+)\.wav", audio_file)
            match = re.match(r"clip_(.+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.wav", audio_file)

            if not match:
                continue
            filename, start, end = match.groups()
            clip_id = f"{filename}_clip_{start}_{end}"

            # Clip-level dict
            clip_dict = {
                "clip_id": clip_id,
                "video_name": filename + ".mp4",
                "start_time": float(start),
                "end_time": float(end),
                "transcript": transcript,
                "audio_path" : os.path.join("audio_from_clips", audio_file),
                "frames": []
            }

            # Add frames belonging to this clip
            for frame_file, caption in self.captions.items():
                # Example: frame_video1_0_5.jpg
                # frame_match = re.match(r"frame_(.+)_(\d+)_(\d+)\.jpg", frame_file)
                frame_match = re.match(r"frame_(.+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.jpg", frame_file)
                if not frame_match:
                    continue
                frame_name, f_start, f_end = frame_match.groups()
                if frame_name == filename:
                    clip_dict["frames"].append({
                        "frame_time": float(f_start),
                        "emotion": caption.emotion,
                        "action": caption.action,  
                        "caption": caption.caption,
                        "description" : caption.description, 
                        "image_path": os.path.join("frames_from_clips", frame_file)
                    })

            data.append(clip_dict)
        return data


# if __name__ == "__main__":
#     # Step 1: Transcribe audio
#     transcriber = AudioTranscriber("audio_from_clips")
#     transcripts = transcriber.transcribe_audios()

#     # # Step 2: Caption frames
#     captioner = FrameCaptioner("frames_from_clips")
#     captions = captioner.caption_frames()

#     # # Step 3: Build data structure
#     builder = DataBuilder(transcripts, captions)
#     dataset = builder.build()

#     # # # Save as JSON
#     import json
#     with open("video_index.json", "w") as f:
#         json.dump(dataset, f, indent=2)

#     print("✅ Data structure built and saved to video_index.json")
