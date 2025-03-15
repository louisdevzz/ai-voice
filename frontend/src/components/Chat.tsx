import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import MessageStore from '../store/MessageStore';
import LoadingOverlay from './LoadingOverlay';

interface ChatProps {
  onSendMessage?: (message: string) => void;
}

const Chat: React.FC<ChatProps> = () => {
  const navigate = useNavigate();
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  }, [input]);


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const chatId = uuidv4();
    const userMessage = input.trim();
    setIsLoading(true);
    
    // Add initial message and navigate immediately
    const messageStore = MessageStore.getInstance();
    messageStore.addMessage(chatId, { role: 'user', content: userMessage });
    
    // Initialize search steps in MessageStore
    messageStore.setSearchSteps(chatId, [
      { message: "Bắt đầu tìm kiếm", status: 'current' },
      { message: "Đang tìm kiếm thông tin", status: 'pending' },
      { message: "Đang trả về kết quả", status: 'pending' },
      { message: "Trả về kết quả", status: 'pending' }
    ]);
    
    navigate(`/chat/${chatId}`);
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/ai/api/chat-search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          chatId: chatId,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      messageStore.addMessage(chatId, { role: 'assistant', content: data.content });
      
      // Clear search steps after getting response
      messageStore.setSearchSteps(chatId, []);
    } catch (error) {
      console.error('Error:', error);
      messageStore.addMessage(chatId, { 
        role: 'assistant', 
        content: 'Sorry, there was an error processing your request. Please try again.' 
      });
      // Clear search steps on error
      messageStore.setSearchSteps(chatId, []);
    } finally {
      setIsLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
        stream.getTracks().forEach(track => track.stop());
        
        setIsLoading(true);
        try {
          const formData = new FormData();
          formData.append('audio', audioBlob, 'recording.wav');

          const transcriptionResponse = await fetch(`${import.meta.env.VITE_API_URL}/ai/api/transcribe`, {
            method: 'POST',
            body: formData,
          });

          if (!transcriptionResponse.ok) {
            throw new Error('Transcription failed');
          }

          const transcriptionData = await transcriptionResponse.json();
          setInput(transcriptionData.text);
        } catch (error) {
          console.error('Error processing audio:', error);
        } finally {
          setIsLoading(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-2 sm:p-4 bg-gray-50">
      {isLoading && <LoadingOverlay />}
      {
        !isLoading && (
          <div className="w-full max-w-[900px] mx-auto">
        <div className="flex items-center flex-col justify-center mb-4 sm:mb-8">
          <div>
            <h1 className="text-2xl sm:text-[40px] text-gray-800 text-center font-semibold px-2">
              Omni - AI Tư vấn Tuyển sinh
            </h1>
            <p className="text-gray-600 text-center text-sm">
              Tôi có thể giúp gì cho bạn?
            </p>
          </div>
        </div>

        <div className="relative w-full px-2">
          <div className="bg-white rounded-xl sm:rounded-2xl p-3 sm:p-4 shadow-lg border border-gray-200">
            <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:gap-4">
              {/* Textarea field */}
              <div className="relative">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  placeholder="Hãy đặt câu hỏi về tuyển sinh Đại học Tân Tạo..."
                  className="w-full bg-transparent text-gray-800 text-[14px] sm:text-[16px] focus:outline-none resize-none min-h-[30px] max-h-[200px] overflow-y-auto placeholder-gray-500"
                />
              </div>

              {/* Buttons container */}
              <div className="flex justify-end items-center space-x-2">
                {/* Voice button */}
                <button
                  type="button"
                  onClick={isRecording ? stopRecording : startRecording}
                  className={`p-2 sm:p-3 hover:bg-gray-100 rounded-full transition-colors ${
                    isRecording ? 'bg-red-500 hover:bg-red-600 text-white' : ''
                  }`}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="sm:w-6 sm:h-6">
                    <path d="M12 15C13.66 15 15 13.66 15 12V5C15 3.34 13.66 2 12 2C10.34 2 9 3.34 9 5V12C9 13.66 10.34 15 12 15Z" stroke={isRecording ? 'white' : '#374151'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M5.5 12C5.5 15.87 8.41 19 12 19C15.59 19 18.5 15.87 18.5 12" stroke={isRecording ? 'white' : '#374151'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M12 19V22" stroke={isRecording ? 'white' : '#374151'} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>

                {/* Send button with up arrow */}
                <button
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  className={`p-2 rounded-full transition-colors ${
                    input.trim() && !isLoading
                      ? 'bg-blue-600 hover:bg-blue-700 text-white'
                      : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="sm:w-6 sm:h-6">
                    <path
                      d="M12 19V5M12 5L5 12M12 5L19 12"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </button>
              </div>
            </form>
          </div>
        </div>
        </div>
        )
      }
    </div>
  );
};

export default Chat; 