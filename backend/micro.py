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
from typing import Optional, List
import wave
import shutil
import base64
import httpx
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

# Initialize Firecrawl client
firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

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
        cleaned_message = request.message
        for char in ['*', '#', '_', '`', '~', '>', '<']:
            cleaned_message = cleaned_message.replace(char, '')

        # Define TTU page categories and their URLs for context
        ttu_context = """
        Bạn đang trả lời câu hỏi về Đại học Tân Tạo (TTU). Các thông tin chính:
        - Website chính: https://ttu.edu.vn
        - Tuyển sinh: /tuyen-sinh, /thong-tin-tuyen-sinh
        - Đào tạo: /dao-tao, /chuong-trinh-dao-tao
        - Nghiên cứu: /nghien-cuu, /nghien-cuu-khoa-hoc
        - Các khoa: Y, CNTT, Sinh học, Ngôn ngữ, Kinh tế, Điều dưỡng
        - Học phí & học bổng: /hoc-phi, /hoc-bong
        - Tin tức & sự kiện: /tin-tuc, /su-kien
        """

        # 1. Get OpenAI web search results
        search_query = f"site:ttu.edu.vn {cleaned_message}"
        openai_response = openai_client.responses.create(
            model="gpt-4o",
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "medium"
            }],
            input=f"{ttu_context}\n\nCâu hỏi: {search_query}"
        )

        # Extract OpenAI response text and citations
        openai_text = ""
        citations = []

        if openai_response.output:
            for output in openai_response.output:
                if output.type == "message" and output.content:
                    for content in output.content:
                        if content.type == "output_text":
                            openai_text = content.text
                            # Extract citations if available
                            if hasattr(content, 'annotations'):
                                for annotation in content.annotations:
                                    if annotation.type == "url_citation":
                                        citations.append({
                                            'url': annotation.url,
                                            'title': annotation.title,
                                            'text': openai_text[annotation.start_index:annotation.end_index]
                                        })

        # 2. Get Firecrawl results from relevant pages
        ttu_pages = {
            "tuyển sinh": [
                "https://tuyensinh.ttu.edu.vn/",
                "https://tuyensinh.ttu.edu.vn/thong-tin-tuyen-sinh",
                "https://tuyensinh.ttu.edu.vn/chuong-trinh-dao-tao",
                "https://tuyensinh.ttu.edu.vn/hoc-phi-hoc-bong"
            ],
            "đào tạo": [
                "https://ttu.edu.vn/dao-tao",
                "https://ttu.edu.vn/chuong-trinh-dao-tao",
                "https://tuyensinh.ttu.edu.vn/chuong-trinh-dao-tao"
            ],
            "nghiên cứu": [
                "https://ttu.edu.vn/nghien-cuu",
                "https://ttu.edu.vn/nghien-cuu-khoa-hoc"
            ],
            "khoa y": [
                "https://medicine.ttu.edu.vn/?lang=vi",
                "https://tuyensinh.ttu.edu.vn/khoa-y"
            ],
            "khoa cntt": [
                "https://sit.ttu.edu.vn/",
                "https://sit.ttu.edu.vn/dao-tao",
                "https://sit.ttu.edu.vn/sinh-vien/tuyen-sinh",
                "https://tuyensinh.ttu.edu.vn/khoa-cong-nghe-thong-tin"
            ],
            "khoa sinh": [
                "https://ttu.edu.vn/khoa-cong-nghe-sinh-hoc",
                "https://tuyensinh.ttu.edu.vn/khoa-cong-nghe-sinh-hoc"
            ],
            "khoa ngôn ngữ": [
                "https://shl.ttu.edu.vn/vi/",
                "https://shl.ttu.edu.vn/vi/dao-tao",
                "https://tuyensinh.ttu.edu.vn/khoa-ngon-ngu"
            ],
            "khoa kinh tế": [
                "https://sbe.ttu.edu.vn/",
                "https://sbe.ttu.edu.vn/dao-tao",
                "https://tuyensinh.ttu.edu.vn/khoa-kinh-te"
            ],
            "học phí": [
                "https://tuyensinh.ttu.edu.vn/hoc-phi-hoc-bong",
                "https://ttu.edu.vn/hoc-phi"
            ],
            "học bổng": [
                "https://tuyensinh.ttu.edu.vn/hoc-phi-hoc-bong",
                "https://ttu.edu.vn/hoc-bong"
            ]
        }

        # Determine relevant pages based on question
        question = cleaned_message.lower()
        relevant_urls = []
        
        # Add more specific keywords for each faculty
        faculty_keywords = {
            "khoa y": ["y", "y khoa", "bác sĩ", "y học", "medicine"],
            "khoa cntt": ["cntt", "công nghệ thông tin", "it", "máy tính", "computer", "phần mềm", "lập trình"],
            "khoa sinh": ["sinh học", "công nghệ sinh học", "biology"],
            "khoa ngôn ngữ": ["ngôn ngữ", "anh văn", "tiếng anh", "language"],
            "khoa kinh tế": ["kinh tế", "quản trị", "business", "marketing", "tài chính"]
        }
        
        for category, urls in ttu_pages.items():
            # Check main category keywords
            if category in question:
                relevant_urls.extend(urls)
                continue
                
            # Check specific faculty keywords
            if category in faculty_keywords:
                if any(keyword in question for keyword in faculty_keywords[category]):
                    relevant_urls.extend(urls)

        if not relevant_urls:
            if any(word in question for word in ["học", "đăng ký", "xét tuyển", "thi"]):
                relevant_urls.extend(ttu_pages["tuyển sinh"])
            elif any(word in question for word in ["chương trình", "môn học"]):
                relevant_urls.extend(ttu_pages["đào tạo"])

        # Remove duplicates while preserving order
        relevant_urls = list(dict.fromkeys(relevant_urls))

        # Crawl selected pages
        firecrawl_content = []
        for url in relevant_urls[:2]:  # Limit to top 2 most relevant pages
            try:
                crawl_response = firecrawl_app.scrape_url(
                    url,
                    params={
                        "formats": ["markdown"],
                        "onlyMainContent": True,
                        "waitFor": 2000,
                        "location": {"languages": ["vi"]}
                    }
                )
                
                if crawl_response.get("success"):
                    content = crawl_response.get("data", {}).get("markdown", "")
                    if content:
                        firecrawl_content.append(content)
            except Exception as e:
                print(f"Firecrawl error for {url}: {str(e)}")
                continue

        # 3. Combine and synthesize information using GPT-4
        synthesis_prompt = f"""
        Hãy tổng hợp và trả lời câu hỏi dựa trên hai nguồn thông tin sau:

        Câu hỏi: {cleaned_message}

        Nguồn 1 (OpenAI Web Search):
        {openai_text}

        Nguồn 2 (Nội dung trang web):
        {'\n---\n'.join(firecrawl_content)}

        Yêu cầu:
        1. Tổng hợp thông tin từ cả hai nguồn
        2. Ưu tiên thông tin mới nhất và chính xác nhất
        3. Trả lời ngắn gọn, rõ ràng và dễ hiểu
        4. Nếu thông tin không đầy đủ, đề xuất liên hệ:
           - Điện thoại: (+84) 272 376 9216 | Hotline: 0981 152 153
           - Email: info@ttu.edu.vn
           - Website: https://ttu.edu.vn
        """

        final_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """Bạn là trợ lý thông tin chính thức của Đại học Tân Tạo (TTU).
Nhiệm vụ:
- Tổng hợp thông tin từ nhiều nguồn
- Đảm bảo tính chính xác và cập nhật
- Trả lời ngắn gọn, rõ ràng và chuyên nghiệp
- Đề xuất liên hệ TTU nếu cần thêm thông tin"""},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        response_text = final_response.choices[0].message.content

        # If no useful information found
        if not response_text or "không tìm thấy thông tin" in response_text.lower():
            response_text = """### Thông báo

Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu hiện có.

### Thông tin liên hệ

Vui lòng liên hệ trực tiếp với Đại học Tân Tạo để được tư vấn chi tiết:

* **Điện thoại:** (+84) 272 376 9216
* **Hotline:** 0981 152 153
* **Email:** info@ttu.edu.vn
* **Website:** [https://ttu.edu.vn](https://ttu.edu.vn)
* **Địa chỉ:** Đại lộ Đại học Tân Tạo, Tân Đức E.City, Đức Hòa, Tỉnh Long An"""

        # Add citations if available
        if citations:
            response_text += "\n\n### Nguồn tham khảo\n"
            for citation in citations:
                response_text += f"\n* [{citation['title']}]({citation['url']})"

        # Clean up any double newlines and ensure consistent formatting
        response_text = response_text.replace("\n\n\n", "\n\n").strip()

        return {
            "role": "assistant",
            "content": response_text,
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