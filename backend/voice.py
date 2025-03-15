import os
import json
import queue
import sounddevice as sd
import numpy as np
import threading
import time
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from io import BytesIO
import wave
import tempfile
import webrtcvad
import collections
import audioop

# Load environment variables
load_dotenv()

# Initialize clients
eleven_labs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# VAD configuration
FRAME_DURATION = 30  # ms
vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (most aggressive)
ring_buffer = collections.deque(maxlen=8)  # Store 8 frames for VAD
voiced_frames = []
silence_duration = 0
SILENCE_THRESHOLD = 1.0  # seconds

def is_speech(audio_segment, sample_rate):
    """Check if audio segment contains speech"""
    try:
        return vad.is_speech(audio_segment, sample_rate)
    except:
        # Fallback to simple energy-based detection
        rms = audioop.rms(audio_segment, 2)  # 2 bytes per sample for int16
        return rms > 500  # Adjust threshold as needed

def save_audio_to_wav(audio_data, samplerate):
    """Convert audio data to WAV format"""
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(samplerate)
        wf.writeframes(audio_data)
    return BytesIO(buffer.getvalue())

def save_temp_wav_file(audio_data, samplerate):
    """Save audio data to a temporary WAV file"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data)
    return temp_file.name

def transcribe_audio(audio_data, samplerate, use_whisper=True):
    """Transcribe audio using both Whisper and ElevenLabs"""
    try:
        results = []
        
        # Convert audio data to WAV format for ElevenLabs
        wav_buffer = save_audio_to_wav(audio_data, samplerate)
        
        # Try OpenAI Whisper first
        if use_whisper:
            try:
                # Save to temporary WAV file for Whisper
                temp_wav_path = save_temp_wav_file(audio_data, samplerate)
                
                with open(temp_wav_path, 'rb') as audio_file:
                    whisper_response = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="vi",
                        response_format="text"
                    )
                results.append(("Whisper", whisper_response))
                
                # Clean up temporary file
                os.unlink(temp_wav_path)
                
            except Exception as e:
                print(f"Lỗi Whisper: {str(e)}")
        
        # Try ElevenLabs
        # try:
        #     eleven_response = eleven_labs.speech_to_text.convert(
        #         file=wav_buffer,
        #         model_id="scribe_v1",
        #         language_code="vi",
        #         tag_audio_events=True
        #     )
        #     results.append(("ElevenLabs", eleven_response.text))
        # except Exception as e:
        #     print(f"Lỗi ElevenLabs: {str(e)}")
        
        # Return results
        if results:
            print("\nKết quả nhận dạng:")
            for source, text in results:
                print(f"{source}: {text}")
            
            # Return the Whisper result if available, otherwise ElevenLabs
            return next((text for source, text in results if source == "Whisper"), results[0][1])
        else:
            return "Không thể nhận dạng giọng nói"
            
    except Exception as e:
        print(f"Lỗi nhận dạng: {str(e)}")
        return None

def listen_for_speech():
    """Lắng nghe và nhận dạng giọng nói từ microphone với streaming mode"""
    # Thiết lập các tham số âm thanh
    device_info = sd.query_devices(kind='input')
    samplerate = int(device_info['default_samplerate'])
    
    # Tạo queue để lưu trữ audio data
    audio_buffer = []
    is_recording = False
    last_voice_time = time.time()
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal is_recording, last_voice_time
        
        if status:
            print(f"Status: {status}")
            
        # Convert to mono if needed and ensure correct format
        audio_chunk = indata.copy()
        if audio_chunk.shape[1] > 1:  # If stereo
            audio_chunk = np.mean(audio_chunk, axis=1)
        
        # Check for voice activity
        chunk_bytes = audio_chunk.tobytes()
        is_voice = is_speech(chunk_bytes, samplerate)
        
        if is_voice:
            last_voice_time = time.time()
            if not is_recording:
                print("\nPhát hiện giọng nói, bắt đầu ghi âm...")
                is_recording = True
            audio_buffer.append(audio_chunk)
        elif is_recording:
            audio_buffer.append(audio_chunk)
            # Check if silence duration exceeds threshold
            if time.time() - last_voice_time > SILENCE_THRESHOLD:
                if len(audio_buffer) > 0:
                    print("\nPhát hiện im lặng, xử lý audio...")
                    # Process the recorded audio
                    audio_data = np.concatenate(audio_buffer)
                    audio_bytes = audio_data.tobytes()
                    
                    # Transcribe the audio
                    transcription = transcribe_audio(audio_bytes, samplerate)
                    if transcription:
                        print(f"\nKết quả cuối cùng: {transcription}")
                    
                    # Reset for next recording
                    audio_buffer.clear()
                is_recording = False
                print("\nĐang lắng nghe...")
    
    try:
        print("\nThông tin thiết bị âm thanh đầu vào:")
        print(device_info)
        print("\nĐang lắng nghe... (Ctrl+C để thoát)")
        
        # Bắt đầu stream âm thanh
        with sd.InputStream(callback=audio_callback,
                          samplerate=samplerate,
                          channels=1,
                          dtype=np.int16,
                          blocksize=int(samplerate * FRAME_DURATION / 1000)):
            while True:
                time.sleep(0.1)  # Reduce CPU usage
                
    except KeyboardInterrupt:
        print("\nĐã dừng lắng nghe.")
    except Exception as e:
        print(f"\nLỗi: {str(e)}")

if __name__ == "__main__":
    print("Bắt đầu nhận dạng giọng nói...")
    listen_for_speech()


def text_to_speech(text, is_system_message=False):
    """Convert text to speech using Zalo AI TTS"""
    try:
        # Process text if it's not a system message
        if not is_system_message:
            text = preprocess_text(text)
        
        print(f"Đang chuyển đổi văn bản thành giọng nói: {text}")
        
        # Prepare the API request
        url = f"https://api.zalo.ai/{ZALO_API_VERSION}/tts/synthesize"
        
        headers = {
            "apikey": ZALO_API_KEY,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "input": text,
            "speaker_id": SPEAKER_ID,
            "speed": VOICE_SPEED,
            "encode_type": 1  # 1: WAV format
        }
        
        # Make the API request
        response = requests.post(url, headers=headers, data=data)
        
        if response.status_code == 200:
            # Parse the JSON response
            json_response = response.json()
            print(f"API Response: {response.text}")  # Debug log
            
            if json_response["error_code"] == 0:
                # Get the audio URL
                audio_url = json_response["data"]["url"]
                print(f"Audio URL: {audio_url}")
                
                # Download the audio file
                audio_response = requests.get(audio_url)
                if audio_response.status_code == 200:
                    # Create a unique filename in the cache directory
                    temp_file = os.path.join(CACHE_DIR, f"audio_{int(time.time())}.wav")
                    
                    # Save the audio data
                    with open(temp_file, 'wb') as f:
                        f.write(audio_response.content)
                    
                    print(f"Đã lưu file âm thanh: {temp_file}")
                    print(f"Kích thước file: {os.path.getsize(temp_file)} bytes")
                    
                    try:
                        # Play the audio file
                        data, samplerate = sf.read(temp_file)
                        sd.play(data, samplerate)
                        sd.wait()  # Wait until audio finishes playing
                        print("Phát âm thanh thành công")
                    except Exception as e:
                        print(f"Lỗi phát âm thanh: {str(e)}")
                    finally:
                        # Clean up
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                else:
                    print(f"Lỗi tải file âm thanh: {audio_response.status_code}")
            else:
                print(f"Lỗi API Zalo: {json_response['error_message']}")
        else:
            error_msg = response.json().get('error_message', 'Unknown error')
            print(f"Lỗi API Zalo: {response.status_code} - {error_msg}")
            
    except Exception as e:
        print(f"Lỗi chuyển đổi văn bản thành giọng nói: {str(e)}")
        import traceback
        print("Chi tiết lỗi:")
        print(traceback.format_exc())
