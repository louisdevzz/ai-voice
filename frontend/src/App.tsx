import { useState, useRef, useEffect } from 'react'
import VoiceRecorder from './components/VoiceRecorder'
import TranscriptionDisplay from './components/TranscriptionDisplay'

function App() {
  const [transcription, setTranscription] = useState('')
  const [aiResponse, setAiResponse] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [currentStatus, setCurrentStatus] = useState<string>('')

  const playAudioResponse = async (audioData: string) => {
    try {
      // Convert the base64 string back to binary
      const binaryString = atob(audioData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Create blob and URL
      const blob = new Blob([bytes], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(blob);

      // Create and play audio
      if (audioRef.current) {
        audioRef.current.src = url;
        await audioRef.current.play();
        setIsSpeaking(true);
      }
    } catch (error) {
      console.error('Error playing audio:', error);
    }
  };

  const updateStatus = async (status: string, customMessage?: string) => {
    try {
      const response = await fetch('http://localhost:3000/api/status-message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          status,
          custom_message: customMessage
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get status message');
      }

      const data = await response.json();
      setCurrentStatus(data.message);
      await playAudioResponse(data.audio);
      
      // Wait for audio to finish
      return new Promise<void>((resolve) => {
        if (audioRef.current) {
          audioRef.current.onended = () => {
            setIsSpeaking(false);
            resolve();
          };
        } else {
          resolve();
        }
      });
    } catch (error) {
      console.error('Error updating status:', error);
    }
  };

  const handleRecordingComplete = async (audioBlob: Blob) => {
    setIsLoading(true);
    try {
      // Update status to processing
      await updateStatus('processing');

      // Create form data for the audio file
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');

      // First, send the audio for transcription
      const transcriptionResponse = await fetch('http://localhost:3000/api/transcribe', {
        method: 'POST',
        body: formData,
      });

      if (!transcriptionResponse.ok) {
        throw new Error('Transcription failed');
      }

      const transcriptionData = await transcriptionResponse.json();
      setTranscription(transcriptionData.text);

      // Update status to thinking
      await updateStatus('thinking');

      // Then, get AI response based on the transcription
      const aiResponseRes = await fetch('http://localhost:3000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: transcriptionData.text }),
      });

      if (!aiResponseRes.ok) {
        throw new Error('Failed to get AI response');
      }

      const aiResponseData = await aiResponseRes.json();
      
      // Set response and play final audio
      setAiResponse(aiResponseData.response);
      if (aiResponseData.audio) {
        await playAudioResponse(aiResponseData.audio);
      }
    } catch (error) {
      console.error('Error processing audio:', error);
      setAiResponse('Có lỗi xảy ra khi xử lý âm thanh. Vui lòng thử lại.');
      await updateStatus('error');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle recording state change
  const handleRecordingStateChange = async (isRecording: boolean) => {
    setIsRecording(isRecording);
    if (isRecording) {
      await updateStatus('start');
    }
  };

  // Handle audio events
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.onended = () => setIsSpeaking(false);
      audioRef.current.onerror = () => setIsSpeaking(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white py-8 px-4">
      <audio ref={audioRef} />
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-blue-800 mb-4 drop-shadow-lg">
            AI Voice Assistant
          </h1>
          <p className="text-blue-600">
            Ghi âm giọng nói tiếng Việt và nhận phản hồi trực tiếp
          </p>
        </header>

        <div className="relative flex justify-center items-center mb-8">
          <div className="w-48 h-48 relative">
            <div className={`absolute inset-0 rounded-full bg-gradient-to-br from-blue-400 to-blue-600 shadow-2xl
              transform transition-all duration-500 ease-in-out
              ${isRecording ? 'scale-110 animate-pulse' : ''}
              ${isLoading ? 'animate-spin-slow' : ''}
              ${isSpeaking ? 'scale-105' : ''}`}
            >
              {/* Brain waves effect */}
              <div className={`absolute -inset-4 bg-blue-400/20 rounded-full blur-md transition-opacity
                ${isLoading || isRecording ? 'opacity-100 animate-pulse' : 'opacity-0'}`}
              />
              
              {/* Face elements */}
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                {/* Eyes */}
                <div className="flex space-x-6 mb-4">
                  <div className={`w-4 h-4 rounded-full bg-white shadow-inner transform transition-all duration-300
                    ${isSpeaking ? 'scale-75' : ''}`}
                  />
                  <div className={`w-4 h-4 rounded-full bg-white shadow-inner transform transition-all duration-300
                    ${isSpeaking ? 'scale-75' : ''}`}
                  />
                </div>
                
                {/* Mouth */}
                <div className={`w-12 h-2 bg-white rounded-full shadow-inner transform transition-all duration-300
                  ${isSpeaking ? 'h-3 rounded-lg animate-bounce' : ''}
                  ${isRecording ? 'w-8' : ''}`}
                />
              </div>
            </div>
          </div>
          
          {/* Status message with enhanced styling */}
          {currentStatus && (
            <div className="absolute -bottom-12 left-0 right-0 text-center">
              <div className="inline-block px-4 py-2 bg-white/80 backdrop-blur-sm rounded-lg shadow-lg">
                <p className="text-gray-700 font-medium">{currentStatus}</p>
              </div>
            </div>
          )}
        </div>

        <main className="space-y-8">
          <VoiceRecorder 
            onRecordingComplete={handleRecordingComplete}
            onRecordingStateChange={handleRecordingStateChange}
          />
          <TranscriptionDisplay
            transcription={transcription}
            aiResponse={aiResponse}
            isLoading={isLoading}
          />
        </main>
      </div>
    </div>
  );
}

export default App
