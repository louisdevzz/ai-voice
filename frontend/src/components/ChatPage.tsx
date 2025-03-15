import { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import MessageStore from '../store/MessageStore';
import LoadingOverlay from './LoadingOverlay';
import { ProcessLogs } from './ProcessingSteps';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const ChatPage = () => {
  const { chatId } = useParams();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [processingStep, setProcessingStep] = useState<'thinking' | 'processing' | 'responding' | 'done'>('done');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [prevMessagesLength, setPrevMessagesLength] = useState(0);
  const [isInitializing, setIsInitializing] = useState(true);
  const [searchSteps, setSearchSteps] = useState<Array<{message: string, status: 'pending' | 'completed' | 'current'}>>([
    { message: "Bắt đầu tìm kiếm", status: 'pending' },
    { message: "Đang tìm kiếm thông tin", status: 'pending' },
    { message: "Đang trả về kết quả", status: 'pending' },
    { message: "Trả về kết quả", status: 'pending' }
]);

  // Replace the fetch initial messages effect with MessageStore
  useEffect(() => {
    if (!chatId) return;
    const messageStore = MessageStore.getInstance();
    const messages = messageStore.getMessages(chatId);
    setMessages(messages);
    setIsInitializing(false);
  }, [chatId]);

  // Auto resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  }, [input]);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  };

  // Update scroll only for user messages
  useEffect(() => {
    if (messages.length > prevMessagesLength) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage && lastMessage.role === 'user') {
        scrollToBottom();
      }
    }
    setPrevMessagesLength(messages.length);
  }, [messages, prevMessagesLength]);

  const handleSubmit = async (e: React.FormEvent) => {
    setSearchSteps(steps => steps.map(step => ({ ...step, status: 'pending' })));

    // Step 1: Processing chat message
    setSearchSteps(steps => steps.map((step, index) => 
        index === 0 ? { ...step, status: 'current' } : step
    ));

    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    
    const messageStore = MessageStore.getInstance();
    messageStore.addMessage(chatId!, { role: 'user', content: userMessage });
    setMessages(messageStore.getMessages(chatId!));
    
    // Force scroll after user message
    setTimeout(scrollToBottom, 100);
    
    setIsLoading(true);
    setProcessingStep('thinking');

    try {
      // Step 1: Bắt đầu tìm kiếm
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Step 2: Đang tìm kiếm thông tin
      setSearchSteps(steps => steps.map((step, index) => 
        index === 0 ? { ...step, status: 'completed' } :
        index === 1 ? { ...step, status: 'current' } : step
      ));
      
      // Make the API call
      const response = await fetch('https://api.slothai.xyz/ai/api/chat-search', {
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

      // Step 3: Đang trả về kết quả
      setSearchSteps(steps => steps.map((step, index) => 
        index === 1 ? { ...step, status: 'completed' } :
        index === 2 ? { ...step, status: 'current' } : step
      ));
      await new Promise(resolve => setTimeout(resolve, 1000));

      const data = await response.json();
      
      // Step 4: Trả về kết quả
      setSearchSteps(steps => steps.map((step, index) => 
        index === 2 ? { ...step, status: 'completed' } :
        index === 3 ? { ...step, status: 'current' } : step
      ));
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      messageStore.addMessage(chatId!, { role: 'assistant', content: data.content });
      setMessages(messageStore.getMessages(chatId!));
      
      // Complete all steps
      setSearchSteps(steps => steps.map(step => ({ ...step, status: 'completed' })));
    } catch (error) {
      console.error('Error:', error);
      messageStore.addMessage(chatId!, {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.'
      });
      setMessages(messageStore.getMessages(chatId!));
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

          const transcriptionResponse = await fetch('https://api.slothai.xyz/ai/api/transcribe', {
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
    <div className="flex flex-col sm:p-4 bg-gray-50">
      {isInitializing && <LoadingOverlay />}
      <div className="w-full max-w-[900px] mx-auto flex flex-col pt-16 md:pt-10 h-[96vh]">
        {/* Messages container */}
        <div className="flex-1 mt-4 sm:mt-8 space-y-3 sm:space-y-4 px-2 overflow-y-auto">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[85%] sm:max-w-[80%] p-3 sm:p-4 rounded-lg  ${
                  message.role === 'user'
                    ? 'bg-white text-gray-800 shadow-sm'
                    : ' text-gray-800'
                }`}
              >
                {message.role === 'assistant' ? (
                  <div className="whitespace-pre-wrap text-[14px] sm:text-[16px]">
                    <ReactMarkdown>
                      {message.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap text-[14px] sm:text-[16px]">{message.content}</p>
                )}
              </div>
            </div>
          ))}
          {isLoading && <ProcessLogs steps={searchSteps} />}
          <div ref={messagesEndRef} />
        </div>

        {/* Input container - now sticky to bottom */}
        <div className="flex flex-col w-full px-2 mt-4 sticky bottom-0 pb-2">
          <div className="bg-white rounded-xl sm:rounded-2xl p-3 sm:p-4 shadow-lg border border-gray-200">
            <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:gap-4">
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

              <div className="flex justify-end items-center space-x-2">
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
    </div>
  );
};

export default ChatPage; 