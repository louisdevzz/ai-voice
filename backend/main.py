from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import tempfile
from typing import Optional, List, Dict
import wave
import shutil
import base64
import asyncio
from datetime import datetime
import json
from firecrawl import FirecrawlApp

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://ttu-ai.vercel.app"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Change PDF file path to TXT file path
TXT_FILE_PATH = "Ver2. Đại học Tân Tạo Tuyển sinh 2025.txt"

# Keywords for CS/IT related queries
CS_KEYWORDS = [
    'khoa học máy tính', 'công nghệ thông tin', 'cntt', 'it', 'computer science',
    'lập trình', 'phần mềm', 'trí tuệ nhân tạo', 'ai', 'artificial intelligence',
    'khoa kỹ thuật', 'sit', 'school of information technology'
]

# Add TTU-related keywords constant
TTU_KEYWORDS = [
    'tân tạo', 'ttu', 'đại học tân tạo', 'trường tân tạo', 
    'university tân tạo', 'ttu.edu.vn', 'sit.ttu.edu.vn'
]

# Add validation keywords for TTU content
TTU_VALIDATION_KEYWORDS = [
    'long an', 'đức hòa', 'y khoa', 'y học', 'điều dưỡng',
    'khoa học sự sống', 'khoa học ứng dụng', 'công nghệ thông tin',
    'ngôn ngữ anh', 'quản trị kinh doanh'
]

# Add common query keywords that imply TTU-related questions
IMPLICIT_KEYWORDS = [
    'học phí', 'giảm học phí', 'học bổng', 'tuyển sinh', 'đăng ký', 'xét tuyển',
    'ngành', 'khoa', 'chương trình', 'điểm chuẩn', 'đào tạo', 'thi', 'nhập học',
    'ký túc xá', 'phòng', 'cơ sở', 'cơ sở vật chất', 'giảng viên', 'giáo sư'
]

# Cache for scraped content
scraped_content_cache = {}

# Cache for PDF content
pdf_content_cache = {}

def load_content():
    """Load and cache TXT content"""
    if not pdf_content_cache.get('content'):
        try:
            with open(TXT_FILE_PATH, 'r', encoding='utf-8') as file:
                content = file.read()
                
                pdf_content_cache['content'] = content
                pdf_content_cache['timestamp'] = datetime.now()
                
                # Split content into sections based on keywords
                sections = {
                    'tuyển sinh': [],
                    'học phí': [],
                    'học bổng': [],
                    'ngành học': [],
                    'thông tin chung': []
                }
                
                # Split content into lines
                lines = content.split('\n')
                current_section = 'thông tin chung'
                
                for line in lines:
                    lower_line = line.lower()
                    if 'tuyển sinh' in lower_line:
                        current_section = 'tuyển sinh'
                    elif 'học phí' in lower_line:
                        current_section = 'học phí'
                    elif 'học bổng' in lower_line:
                        current_section = 'học bổng'
                    elif 'ngành' in lower_line:
                        current_section = 'ngành học'
                    
                    sections[current_section].append(line)
                
                pdf_content_cache['sections'] = sections
                
        except Exception as e:
            print(f"Error loading TXT file: {str(e)}")
            return None
    
    return pdf_content_cache.get('content'), pdf_content_cache.get('sections')

# Voice configuration
VOICE_ID = "foH7s9fX31wFFH2yqrFa"  # ElevenLabs voice ID
MODEL_ID = "eleven_turbo_v2_5"  # ElevenLabs model ID

# Status messages for different processing stages
STATUS_MESSAGES = {
    "start": "Xin chào, tôi có thể giúp gì cho bạn.",
    "processing": "Tôi đang xử lý yêu cầu của bạn.",
    "thinking": "Tôi đang suy nghĩ.",
    "error": "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại."
}

class ChatRequest(BaseModel):
    message: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatSearchRequest(BaseModel):
    message: str
    search: bool = False

class StatusRequest(BaseModel):
    status: str
    custom_message: Optional[str] = None

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

def validate_audio_file(file_path: str) -> bool:
    """Validate if the file is a proper audio file"""
    try:
        with wave.open(file_path, 'rb') as wave_file:
            # Check basic audio file properties
            if wave_file.getnchannels() == 0 or wave_file.getsampwidth() == 0:
                return False
        return True
    except Exception:
        return False

@app.post("/ai/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file provided")

    if not audio.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid audio file format. Supported formats: WAV, MP3, M4A, WEBM")

    try:
        # Create a temporary directory to store the uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio_path = os.path.join(temp_dir, "audio_file.wav")
            
            # Save the uploaded file
            with open(temp_audio_path, "wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
            
            # Validate the audio file
            if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                raise HTTPException(status_code=400, detail="Empty or invalid audio file")

            # Use OpenAI Whisper for transcription
            try:
                with open(temp_audio_path, 'rb') as audio_file:
                    transcription = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="vi",
                        response_format="text"
                    )
                return {"text": transcription}
            except Exception as e:
                print(f"Transcription error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcription failed: {str(e)}"
                )
                
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )
    finally:
        # Make sure the uploaded file is closed
        await audio.close()

def text_to_speech(text: str) -> bytes:
    """Convert text to speech using ElevenLabs"""
    try:
        # Generate audio with optimized settings for Vietnamese
        audio_generator = eleven_client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.75,
                similarity_boost=0.75,
                style=0.35,
                use_speaker_boost=True
            )
        )
        
        # Convert generator to bytes
        audio_bytes = b"".join(chunk for chunk in audio_generator)
        
        # Convert audio bytes to base64 string
        return base64.b64encode(audio_bytes)
        
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        raise

@app.post("/ai/api/status-message")
async def get_status_message(request: StatusRequest):
    """Get audio message for a specific status"""
    try:
        message = request.custom_message or STATUS_MESSAGES.get(request.status, STATUS_MESSAGES["error"])
        audio_data = text_to_speech(message)
        return {
            "message": message,
            "audio": audio_data.decode()
        }
    except Exception as e:
        print(f"Status message error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Status message error: {str(e)}"
        )

@app.post("/ai/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Clean input message - remove special characters
        cleaned_message = request.message
        for char in ['*', '#', '_', '`', '~', '>', '<']:
            cleaned_message = cleaned_message.replace(char, '')

        # Get AI response using GPT-4 with web search
        response = openai_client.responses.create(
            model="gpt-4o",
            tools=[{ "type": "web_search_preview" }],
            input=cleaned_message
        )
        
        # Extract response text from the response format
        response_text = ""
        if response.output:
            for output in response.output:
                if output.type == "message":
                    for content in output.content:
                        if content.type == "output_text":
                            response_text = content.text
                            break
                    if response_text:  # If we found the text, break the outer loop
                        break
                elif output.type == "web_search_call":
                    # Web search was performed
                    continue

        if not response_text:
            response_text = "Xin lỗi, tôi không thể tìm thấy thông tin phù hợp cho câu hỏi của bạn."

        # Clean response text - remove special characters
        for char in ['*', '#', '_', '`', '~', '>', '<']:
            response_text = response_text.replace(char, '')

        # If response is too long or contains potential inaccuracies, improve it
        word_count = len(response_text.split())
        if word_count > 100 or "không chính xác" in response_text.lower() or "sai" in response_text.lower():
            verification_prompt = f"""Hãy phân tích và cải thiện nội dung sau đây:

1. Kiểm tra tính chính xác của thông tin
2. Xác định và loại bỏ thông tin không chính xác hoặc không đáng tin cậy
3. Tập trung vào các sự kiện đã được xác minh
4. Trình bày ngắn gọn trong 3-4 câu
5. Sử dụng ngôn ngữ rõ ràng, dễ hiểu
6. Nếu không chắc chắn về điều gì, hãy nói rõ

Nội dung cần phân tích:
{response_text}

Phiên bản cải thiện:"""
            
            improved_response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """Bạn là một chuyên gia phân tích và kiểm chứng thông tin.
Nhiệm vụ của bạn là:
- Kiểm tra tính chính xác của thông tin
- Chỉ giữ lại những thông tin đã được xác minh
- Nêu rõ nếu có thông tin không chắc chắn
- Tạo phiên bản ngắn gọn, dễ hiểu"""},
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.3,
                max_tokens=200,
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            response_text = improved_response.choices[0].message.content

            # Clean improved response - remove any remaining special characters
            for char in ['*', '#', '_', '`', '~', '>', '<']:
                response_text = response_text.replace(char, '')

        # Generate speech from the response
        audio_data = text_to_speech(response_text)
            
        return {
            "response": response_text,
            "audio": audio_data.decode()
        }
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {str(e)}"
        )

@app.post("/ai/api/chat-search")
async def chat_search_endpoint(request: ChatSearchRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Clean input message
        cleaned_message = request.message.lower()
        for char in ['*', '#', '_', '`', '~', '>', '<']:
            cleaned_message = cleaned_message.replace(char, '')

        # Check if query is explicitly or implicitly related to TTU
        is_explicit_ttu = any(keyword in cleaned_message for keyword in TTU_KEYWORDS)
        is_implicit_ttu = any(keyword in cleaned_message for keyword in IMPLICIT_KEYWORDS)
        
        # If not related to TTU at all, return a polite message
        if not is_explicit_ttu and not is_implicit_ttu:
            return {
                "role": "assistant",
                "content": """Xin lỗi, tôi chỉ có thể trả lời các câu hỏi liên quan đến Đại học Tân Tạo (TTU).

Vui lòng đặt câu hỏi về:
* Tuyển sinh TTU
* Ngành học tại TTU
* Học phí & học bổng TTU
* Thông tin về TTU""",
                "sections": [],
                "citations": None
            }

        # Add TTU context to implicit queries for search
        search_message = cleaned_message
        if not is_explicit_ttu and is_implicit_ttu:
            search_message = f"đại học tân tạo {cleaned_message}"

        # Check if query is CS/IT related
        is_cs_query = any(keyword in cleaned_message for keyword in CS_KEYWORDS)

        # Enhanced context with specific TTU information
        ttu_context = """Trợ lý tư vấn tuyển sinh Đại học Tân Tạo (TTU).
Thông tin cơ bản về TTU:
- Địa chỉ: Đại học Tân Tạo, Tân Đức E-City, Đức Hòa, Long An
- Các ngành đào tạo chính: Y khoa, Điều dưỡng, Công nghệ thông tin, Khoa học sự sống, Ngôn ngữ Anh, Quản trị kinh doanh
- Website: ttu.edu.vn

Yêu cầu:
1. Chỉ sử dụng thông tin từ tài liệu tuyển sinh 2025 và website chính thức của TTU
2. Không sử dụng thông tin chung chung hoặc không xác thực
3. Nếu không có thông tin chính xác, đề xuất liên hệ TTU
4. Mọi câu trả lời đều ngầm hiểu là về Đại học Tân Tạo"""

        # 1. Get web search results with enhanced validation
        def get_web_search():
            try:
                # Use OpenAI web search with modified query
                search_query = f"site:ttu.edu.vn {search_message}"
                openai_response = openai_client.responses.create(
                    model="gpt-4o",
                    tools=[{"type": "web_search_preview"}],
                    input=f"""Tìm thông tin về Đại học Tân Tạo (TTU) theo yêu cầu sau:
{search_message}

Yêu cầu:
1. Chỉ tìm thông tin từ website chính thức ttu.edu.vn
2. Thông tin phải chính xác và có nguồn gốc rõ ràng
3. Bỏ qua thông tin chung chung hoặc không xác thực
4. Nếu không tìm thấy thông tin chính xác, trả về "Không tìm thấy thông tin"
5. Mọi thông tin đều phải là về Đại học Tân Tạo."""
                )

                # Extract web search results
                web_content = ""
                citations = []
                
                if openai_response.output:
                    for output in openai_response.output:
                        if output.type == "message":
                            for content in output.content:
                                if content.type == "output_text":
                                    web_content = content.text
                                    # Extract citations if available
                                    if hasattr(content, 'annotations'):
                                        citations = [
                                            {
                                                'url': ann.url,
                                                'title': ann.title
                                            }
                                            for ann in content.annotations
                                            if ann.type == "url_citation"
                                        ]

                # Validate web content
                if not any(keyword in web_content.lower() for keyword in TTU_VALIDATION_KEYWORDS):
                    web_content = ""

                # If CS/IT related, also get Firecrawl content
                if is_cs_query:
                    cache_key = 'sit_content'
                    if cache_key not in scraped_content_cache:
                        scrape_result = firecrawl_app.scrape_url(
                            'https://sit.ttu.edu.vn',
                            params={
                                'formats': ['markdown'],
                                'onlyMainContent': True,
                                'waitFor': 1000
                            }
                        )
                        if scrape_result.get('success'):
                            content = scrape_result['data']['markdown']
                            if any(keyword in content.lower() for keyword in TTU_VALIDATION_KEYWORDS):
                                if len(content) > 2000:
                                    content = content[:2000] + "..."
                                scraped_content_cache[cache_key] = content
                                scraped_content_cache['timestamp'] = datetime.now()
                                # Combine with web search results
                                web_content = f"{web_content}\n\nThông tin từ Khoa CNTT:\n{content}"

                return web_content, citations
            except Exception as e:
                print(f"Web search error: {str(e)}")
                return "", []

        # 2. Get TXT content with optimized section selection
        def get_txt_content():
            txt_content, txt_sections = load_content()
            if not txt_content:
                return "", []

            # Keywords mapping
            keywords = {
                'tuyển sinh': ['tuyển sinh', 'đăng ký', 'xét tuyển', 'hồ sơ', 'điểm chuẩn'],
                'học phí': ['học phí', 'tiền học', 'phí'],
                'học bổng': ['học bổng', 'hỗ trợ', 'miễn giảm'],
                'ngành học': ['ngành', 'chuyên ngành', 'khoa', 'chương trình']
            }

            # Find most relevant section only
            most_relevant_section = 'thông tin chung'
            max_matches = 0
            
            for section, words in keywords.items():
                matches = sum(1 for word in words if word in cleaned_message)
                if matches > max_matches:
                    max_matches = matches
                    most_relevant_section = section

            # Get content from most relevant section only
            content = '\n'.join(txt_sections[most_relevant_section])
            if len(content) > 2000:  # Limit to ~2000 characters
                content = content[:2000] + "..."

            return content, [most_relevant_section]

        # Execute searches
        web_text, citations = get_web_search()
        txt_text, matched_sections = get_txt_content()

        # Enhance the synthesis prompt to enforce accuracy
        synthesis_prompt = f"""Câu hỏi: {cleaned_message}

Thông tin chính thức từ tài liệu tuyển sinh TTU:
{txt_text}

Thông tin bổ sung từ website TTU:
{web_text}

Yêu cầu: 
1. Trả lời với ngầm hiểu đây là thông tin về Đại học Tân Tạo
2. Chỉ tổng hợp thông tin chính thức và đã xác thực
3. Ưu tiên thông tin từ tài liệu tuyển sinh 2025
4. Bỏ qua thông tin chung chung hoặc không rõ nguồn gốc
5. Nếu không đủ thông tin chính xác, đề xuất liên hệ TTU
6. Trả lời ngắn gọn, súc tích
7. Nếu có trích dẫn từ website, đính kèm link nguồn"""

        # Get final response with reduced token count
        final_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": ttu_context},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        response_text = final_response.choices[0].message.content

        # If no useful information found
        if not response_text or "không tìm thấy thông tin" in response_text.lower():
            response_text = """### Thông tin liên hệ TTU

Xin lỗi, tôi không tìm thấy thông tin chi tiết về vấn đề này. Vui lòng liên hệ TTU để được tư vấn cụ thể:
* **Hotline:** 0981 152 153
* **Email:** info@ttu.edu.vn
* **Website:** [ttu.edu.vn](https://ttu.edu.vn)"""

        if is_cs_query:
            response_text += "\n\n### Khoa CNTT TTU\n* **Email:** sit@ttu.edu.vn\n* **Website:** [sit.ttu.edu.vn](https://sit.ttu.edu.vn)"

        # Clean up formatting
        response_text = response_text.replace("\n\n\n", "\n\n").strip()

        # Add TTU context to response if it was an implicit query
        if not is_explicit_ttu and is_implicit_ttu and "TTU" not in response_text[:50]:
            response_text = f"### Thông tin Đại học Tân Tạo (TTU)\n\n{response_text}"

        return {
            "role": "assistant",
            "content": response_text,
            "sections": matched_sections,
            "citations": citations if citations else None
        }

    except Exception as e:
        print(f"Chat search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat search error: {str(e)}"
        )

@app.get("/ai/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://localhost:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info") 