import os
import streamlit as st
from typing import List

from Indexing import process_video_dataset
from chunker import ClipProcessor
from data_builder import DataBuilder, AudioTranscriber, FrameCaptioner
from utils import GetResponseClip

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(layout="wide", page_title="🎥 VideoRAG Simple UI")
st.title("🎥 VideoRAG Simple UI")

# -------------------
# Sidebar (left side)
# -------------------
with st.sidebar:
    st.header("⚙️ Setup")

    # Step 1: Get folder path
    folder_path = st.text_input("Enter folder path containing videos:")

    if folder_path and os.path.isdir(folder_path):
        st.success(f"✅ Found folder: {folder_path}")

        # Initialize session state
        if "rag" not in st.session_state:
            st.session_state.rag = None

        if st.button("Start Video Analysis"):
            st.info("🔹 Starting video processing... This may take a few minutes depending on the dataset.")

            # Init clip processor
            chunker = ClipProcessor(folder_path)

            # Extract audio and frames
            with st.spinner("🎬 Splitting videos into clips and extracting frames & audio..."):
                chunker.main()
            st.success("✅ Video splitting done.")

            # Audio transcription
            with st.spinner("🎤 Transcribing audio using Whisper..."):
                transcriber = AudioTranscriber("audio_from_clips")
                transcripts = transcriber.transcribe_audios()
            st.success("✅ Audio transcription completed.")

            # Caption frames
            with st.spinner("🖼️ Generating captions for frames using Gemini..."):
                captioner = FrameCaptioner("frames_from_clips")
                captions = captioner.caption_frames()
            st.success("✅ Frame captioning completed.")

            # Build dataset
            with st.spinner("🛠️ Building dataset for RAG..."):
                builder = DataBuilder(transcripts, captions)
                dataset = builder.build()
            st.success("✅ Dataset prepared.")

            # Process dataset for RAG
            with st.spinner("📚 Indexing dataset for retrieval..."):
                st.session_state.rag = process_video_dataset(clips_data=dataset)
            st.success("✅ Indexing completed.")
    else:
        st.warning("No video files found in the folder or folder path is invalid.")

# -------------------
# Main Page (right side)
# -------------------
st.header("💬 Ask Questions")

query = st.text_input("Ask a query about the video:")

if query and st.session_state.get("rag") is not None:
    with st.spinner("🔎 Retrieving relevant clips and generating answer..."):
        response = st.session_state.rag.query_with_llm(query)

    st.subheader("📜 Response")
    st.write(response['response'])
    with st.spinner("🎥 Retrieving relevant video clips..."):
        # Get top 2 clips
        # clipper = GetResponseClip(response['context'], 2)
        retrieved_clips = st.session_state.rag.search(query)
        # Get top 2 clips
        clipper = GetResponseClip(retrieved_clips, 2)
        retrieved_video_paths = clipper.extract_video_from_metadata()
        
        if retrieved_video_paths:
            num_clips = len(retrieved_video_paths)
            cols = st.columns(num_clips)
            for idx, col in enumerate(cols):
                col.video(retrieved_video_paths[idx])
        else:
            st.warning("No relevant clips found to display.")


