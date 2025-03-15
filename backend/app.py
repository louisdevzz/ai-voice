import streamlit as st
import sounddevice as sd
import numpy as np
import threading
import time
import queue
from dotenv import load_dotenv
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_chatbot import text_to_speech, transcribe_audio, get_ai_response

# Load environment variables
load_dotenv()

# Global variables for thread-safe operation
audio_queue = queue.Queue()
recording_state = threading.Event()
stream = None

# Streamlit page config
st.set_page_config(
    page_title="Omni Voice Assistant",
    page_icon="🎤",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
    border: none;
}
.stButton>button:hover {
    background-color: #45a049;
}
.main {
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Title
st.title("🎤 Omni Voice Assistant")
st.markdown("---")

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Chat history display
    st.subheader("Chat History")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Omni:** {message['content']}")

with col2:
    # Recording controls
    st.subheader("Voice Controls")
    
    def audio_callback(indata, frames, time_info, status):
        """Callback function to handle audio input"""
        if status:
            print('Error:', status)
        if recording_state.is_set():
            audio_queue.put(indata.copy())
    
    def process_audio():
        """Process recorded audio data"""
        try:
            audio_chunks = []
            while not audio_queue.empty():
                audio_chunks.append(audio_queue.get())
                
            if audio_chunks:
                # Convert audio data to the correct format
                audio_data = np.concatenate(audio_chunks)
                
                # Process the audio
                with st.spinner('Processing your message...'):
                    # Transcribe audio
                    transcription = transcribe_audio(audio_data.tobytes(), 
                                                  samplerate=16000,
                                                  use_whisper=True)
                    
                    if transcription and transcription != "Không thể nhận dạng giọng nói":
                        # Add user message to chat
                        st.session_state.messages.append({
                            "role": "user",
                            "content": transcription
                        })
                        
                        # Get AI response
                        response = get_ai_response(transcription)
                        
                        # Add AI response to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        return True
                    else:
                        st.error("Could not transcribe audio. Please try again.")
                        return False
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return False
        return False

    def start_recording():
        """Start the recording stream"""
        global stream
        try:
            # Clear any existing audio data
            while not audio_queue.empty():
                audio_queue.get()
            
            recording_state.set()
            stream = sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=16000,
                blocksize=int(16000 * 0.05)  # 50ms blocks
            )
            stream.start()
            return True
        except Exception as e:
            st.error(f"Error starting recording: {str(e)}")
            recording_state.clear()
            return False

    def stop_recording():
        """Stop the recording stream"""
        global stream
        try:
            recording_state.clear()
            if stream:
                stream.stop()
                stream.close()
                stream = None
            return process_audio()
        except Exception as e:
            st.error(f"Error stopping recording: {str(e)}")
            return False

    # Recording controls
    if not recording_state.is_set():
        if st.button("Start Recording 🎙️"):
            if start_recording():
                st.info("Recording started... Click 'Stop Recording' when finished.")
                st.experimental_rerun()
    else:
        if st.button("Stop Recording ⏹️"):
            if stop_recording():
                st.experimental_rerun()

    # Clear chat button
    if st.button("Clear Chat 🗑️"):
        st.session_state.messages = []
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Team") 