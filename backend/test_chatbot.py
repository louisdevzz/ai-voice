import os
import sounddevice as sd
import numpy as np
import threading
import time
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
import wave
import tempfile
import webrtcvad
import collections
import audioop
import re
from urllib.parse import quote
import subprocess
from elevenlabs.client import ElevenLabs
from elevenlabs import stream, play, VoiceSettings

# Load environment variables
load_dotenv()

# Create cache directory for audio files
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# VAD configuration
FRAME_DURATION = 30  # ms
vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (most aggressive)
ring_buffer = collections.deque(maxlen=8)  # Store 8 frames for VAD
voiced_frames = []
silence_duration = 0
SILENCE_THRESHOLD = 1.0  # seconds

# Voice configuration
VOICE_ID = "foH7s9fX31wFFH2yqrFa"  # ElevenLabs voice ID
MODEL_ID = "eleven_turbo_v2_5"  # ElevenLabs model ID

# Audio messages
WELCOME_MESSAGE = "Xin chào! Tôi là Omni, tôi đang nghe bạn. Hãy đặt câu hỏi."
PROCESSING_MESSAGE = "Vui lòng chờ trong giây lát, tôi đang xử lý câu trả lời."
LISTENING_MESSAGE = "Tôi đang lắng nghe bạn, hãy đặt câu hỏi."

# Audio playback configuration
def play_audio_file(file_path):
    """Play audio file using mpg123"""
    try:
        if not os.path.exists(file_path):
            print(f"Lỗi: Không tìm thấy file âm thanh tại {file_path}")
            return
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"Lỗi: File âm thanh rỗng {file_path}")
            return
            
        print(f"Phát file âm thanh: {file_path} (kích thước: {file_size} bytes)")
        
        # Check if mpg123 is installed
        try:
            subprocess.run(['which', 'mpg123'], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("Lỗi: Chưa cài đặt mpg123. Vui lòng cài đặt bằng lệnh: sudo apt-get install mpg123")
            return
        
        # Use mpg123 to play the audio file directly
        result = subprocess.run(['mpg123', '-q', file_path], 
                             check=True, 
                             capture_output=True,
                             text=True)
        
        if result.returncode == 0:
            print("Phát âm thanh thành công")
        else:
            print(f"Lỗi khi phát âm thanh: {result.stderr}")
            
    except subprocess.CalledProcessError as e:
        print(f"Lỗi phát âm thanh: {str(e)}")
        if e.stderr:
            print(f"Chi tiết lỗi: {e.stderr}")
    except Exception as e:
        print(f"Lỗi không xác định: {str(e)}")
        import traceback
        print("Chi tiết lỗi:")
        print(traceback.format_exc())

def is_speech(audio_segment, sample_rate):
    """Check if audio segment contains speech"""
    try:
        # Try VAD first
        vad_result = vad.is_speech(audio_segment, sample_rate)
        
        # If VAD detects speech, check the volume level
        if vad_result:
            # Calculate RMS (volume level)
            rms = audioop.rms(audio_segment, 2)  # 2 bytes per sample for int16
            
            # Define RMS thresholds for distance control
            MIN_RMS = 1000  # Minimum volume (too far, > 150cm)
            MAX_RMS = 3000  # Maximum volume (too close, < 100cm)
            
            # Check if the volume is within the desired range
            if MIN_RMS <= rms <= MAX_RMS:
                return True, None
            else:
                if rms > MAX_RMS:
                    return False, "too_close"
                elif rms < MIN_RMS:
                    return False, "too_far"
        return False, None
    except:
        # Fallback to simple energy-based detection with adjusted threshold
        rms = audioop.rms(audio_segment, 2)
        MIN_RMS = 1000
        MAX_RMS = 3000
        
        if MIN_RMS <= rms <= MAX_RMS:
            return True, None
        else:
            if rms > MAX_RMS:
                return False, "too_close"
            elif rms < MIN_RMS:
                return False, "too_far"
            return False, None

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

def preprocess_text(text):
    """Process text to improve speech quality"""
    # Replace markdown symbols
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold markers
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic markers
    
    # Replace special characters with their spoken equivalents
    replacements = {
        '#': 'mục',
        '&': 'và',
        '@': 'tại',
        '%': 'phần trăm',
        '+': 'cộng',
        '=': 'bằng',
        '/': 'trên',
        '\\': 'phần',
        '|': 'hoặc',
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, f' {replacement} ')
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add natural pauses with punctuation
    text = text.replace('. ', '. ... ')
    text = text.replace('? ', '? ... ')
    text = text.replace('! ', '! ... ')
    text = text.replace(': ', ': ... ')
    
    return text

def text_to_speech(text, is_system_message=False):
    """Convert text to speech using ElevenLabs with streaming"""
    try:
        # Process text if it's not a system message
        if not is_system_message:
            text = preprocess_text(text)
        
        print(f"Đang chuyển đổi văn bản thành giọng nói: {text}")
        
        # Convert text to speech using ElevenLabs with streaming
        def text_stream():
            # Add slight pause at the beginning for better audio initialization
            yield "... "
            
            # Split text into smaller chunks at punctuation marks
            chunks = re.split('([.!?]+)', text)
            for i in range(0, len(chunks)-1, 2):
                # Combine chunk with its punctuation
                chunk = chunks[i] + chunks[i+1] if i+1 < len(chunks) else chunks[i]
                chunk = chunk.strip()
                if chunk:
                    # Add small pauses between chunks for better clarity
                    yield chunk + " ... "
            # Handle the last chunk if it exists and doesn't end with punctuation
            if len(chunks) % 2 == 1 and chunks[-1].strip():
                yield chunks[-1].strip() + " ... "
        
        # Voice settings optimized for Vietnamese clarity
        voice_settings = VoiceSettings(
            stability=0.75,  # Higher stability for clearer pronunciation
            similarity_boost=0.75,  # Higher similarity for better voice quality
            style=0.35,  # Lower style for more natural speech
            use_speaker_boost=True  # Enable speaker boost for clearer audio
        )
        
        # Generate audio stream with optimized settings
        audio_stream = eleven_client.generate(
            text=text_stream(),
            voice=VOICE_ID,
            model=MODEL_ID,
            stream=True,
            voice_settings=voice_settings,  # Apply voice settings
            optimize_streaming_latency=2  # Moderate latency optimization for better quality
        )
        
        # Stream and play the audio in real-time
        stream(audio_stream)
        print("Phát âm thanh thành công")
            
    except Exception as e:
        print(f"Lỗi chuyển đổi văn bản thành giọng nói: {str(e)}")
        import traceback
        print("Chi tiết lỗi:")
        print(traceback.format_exc())

def get_ai_response(user_input):
    """Get AI response using GPT-4 with web search capability"""
    try:
        # Play processing message
        text_to_speech(PROCESSING_MESSAGE, is_system_message=True)
        
        response = openai_client.responses.create(
            model="gpt-4o",
            tools=[{ "type": "web_search_preview" }],
            input=user_input,
        )
        
        # Process the response
        if response.output:
            for output in response.output:
                if output.type == "message":
                    # Get the message content
                    for content in output.content:
                        if content.type == "output_text":
                            response_text = content.text
                            
                            # Check if response is long (more than 100 words)
                            word_count = len(response_text.split())
                            if word_count > 100:
                                # Create a shorter summary using OpenAI
                                summary_prompt = f"""Hãy tóm tắt nội dung sau đây, đảm bảo:
1. Giữ nguyên ý chính và thông tin quan trọng
2. Sử dụng câu văn rõ ràng, dễ hiểu
3. Độ dài khoảng 3-4 câu
4. Tập trung vào các điểm chính yếu

Nội dung cần tóm tắt:
{response_text}

Tóm tắt ngắn gọn:"""
                                
                                summary_response = openai_client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": "Bạn là một chuyên gia tóm tắt văn bản, luôn đảm bảo giữ được ý nghĩa chính và tạo ra các câu dễ hiểu."},
                                        {"role": "user", "content": summary_prompt}
                                    ],
                                    temperature=0.3,  # Lower temperature for more focused summaries
                                    max_tokens=150,   # Limit length but ensure complete sentences
                                    presence_penalty=0.0,
                                    frequency_penalty=0.0
                                )
                                
                                # Get the summary
                                summary = summary_response.choices[0].message.content
                                
                                # Convert summary to speech
                                text_to_speech(summary)
                                return summary
                            else:
                                # Convert original response to speech if it's short enough
                                text_to_speech(response_text)
                                return response_text
                elif output.type == "web_search_call":
                    # Web search was performed
                    continue
        
        default_response = "Tôi không thể tìm thấy câu trả lời cho câu hỏi của bạn."
        text_to_speech(default_response, is_system_message=True)
        return default_response
        
    except Exception as e:
        print(f"Error getting AI response: {str(e)}")
        error_message = "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn."
        text_to_speech(error_message, is_system_message=True)
        return error_message

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
        
        # Return results and get AI response
        if results:
            print("\nKết quả nhận dạng:")
            for source, text in results:
                print(f"{source}: {text}")
            
            # Get the transcription
            transcription = next((text for source, text in results if source == "Whisper"), results[0][1])
            
            # Get AI response
            print("\nOmni đang xử lý câu hỏi của bạn...")
            ai_response = get_ai_response(transcription)
            print(f"\nOmni: {ai_response}")
            
            return transcription
        else:
            return "Không thể nhận dạng giọng nói"
            
    except Exception as e:
        print(f"Lỗi nhận dạng: {str(e)}")
        return None

def listen_for_speech():
    """Listen and recognize speech from microphone with streaming mode"""
    # Play welcome message
    text_to_speech(WELCOME_MESSAGE, is_system_message=True)
    
    # Setup audio parameters
    device_info = sd.query_devices(kind='input')
    samplerate = int(device_info['default_samplerate'])
    
    # Create queue for audio data
    audio_buffer = []
    is_recording = False
    last_voice_time = time.time()
    is_processing = False  # Flag to track if we're currently processing audio
    processing_lock = threading.Lock()  # Lock for synchronizing processing
    last_distance_warning = None  # Track last distance warning
    last_warning_time = 0  # Track when the last warning was shown
    WARNING_COOLDOWN = 2  # Seconds between warnings
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal is_recording, last_voice_time, is_processing, last_distance_warning, last_warning_time
        
        if status:
            print(f"Status: {status}")
            
        # If we're currently processing, ignore new audio input
        if is_processing:
            return
            
        # Convert to mono if needed and ensure correct format
        audio_chunk = indata.copy()
        if audio_chunk.shape[1] > 1:  # If stereo
            audio_chunk = np.mean(audio_chunk, axis=1)
        
        # Check for voice activity
        chunk_bytes = audio_chunk.tobytes()
        is_voice, distance_status = is_speech(chunk_bytes, samplerate)
        
        # Handle distance warnings
        current_time = time.time()
        if distance_status != last_distance_warning and current_time - last_warning_time > WARNING_COOLDOWN:
            if distance_status == "too_close":
                print("\nGiọng nói quá gần, vui lòng nói xa hơn (>100cm)")
                last_warning_time = current_time
            elif distance_status == "too_far":
                print("\nGiọng nói quá xa, vui lòng nói gần hơn (<150cm)")
                last_warning_time = current_time
            last_distance_warning = distance_status
        
        if is_voice:
            last_voice_time = current_time
            if not is_recording:
                print("\nPhát hiện giọng nói, bắt đầu ghi âm...")
                is_recording = True
            audio_buffer.append(audio_chunk)
        elif is_recording:
            audio_buffer.append(audio_chunk)
            # Check if silence duration exceeds threshold
            if current_time - last_voice_time > SILENCE_THRESHOLD:
                if len(audio_buffer) > 0 and not is_processing:
                    # Try to acquire the lock
                    if processing_lock.acquire(blocking=False):
                        try:
                            is_processing = True
                            print("\nPhát hiện im lặng, xử lý audio...")
                            # Process the recorded audio
                            audio_data = np.concatenate(audio_buffer)
                            audio_bytes = audio_data.tobytes()
                            
                            def process_audio():
                                nonlocal is_processing
                                try:
                                    # Transcribe the audio
                                    transcription = transcribe_audio(audio_bytes, samplerate)
                                    if transcription:
                                        print(f"\nKết quả cuối cùng: {transcription}")
                                finally:
                                    is_processing = False
                                    processing_lock.release()
                            
                            # Start processing in a separate thread
                            threading.Thread(target=process_audio).start()
                            
                            # Reset for next recording
                            audio_buffer.clear()
                        except:
                            is_processing = False
                            processing_lock.release()
                            raise
                
                is_recording = False
                print("\nĐang lắng nghe...")
    
    try:
        print("\nThông tin thiết bị âm thanh đầu vào:")
        print(device_info)
        print("\nĐang lắng nghe... (Ctrl+C để thoát)")
        
        # Start audio stream
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
