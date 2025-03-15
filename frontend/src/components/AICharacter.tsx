import { useState, useEffect } from 'react';

interface AICharacterProps {
  isListening: boolean;
  isProcessing: boolean;
  isSpeaking: boolean;
}

const AICharacter: React.FC<AICharacterProps> = ({ 
  isListening, 
  isProcessing,
  isSpeaking 
}) => {
  const [isBlinking, setIsBlinking] = useState(false);

  useEffect(() => {
    const blinkInterval = setInterval(() => {
      setIsBlinking(true);
      setTimeout(() => setIsBlinking(false), 200);
    }, 3000);

    return () => clearInterval(blinkInterval);
  }, []);

  return (
    <div className="relative w-64 h-64 mx-auto mb-8">
      {/* AI Head */}
      <div className="absolute inset-0 bg-blue-500 rounded-full shadow-lg">
        <div className="absolute inset-2 bg-white rounded-full"></div>
      </div>

      {/* Eyes */}
      <div className="absolute w-full top-1/3 flex justify-center space-x-8">
        <div className={`w-8 h-8 bg-blue-600 rounded-full ${isBlinking ? 'h-1' : ''} transition-all duration-200`}>
          <div className="w-3 h-3 bg-white rounded-full absolute top-1 left-1"></div>
        </div>
        <div className={`w-8 h-8 bg-blue-600 rounded-full ${isBlinking ? 'h-1' : ''} transition-all duration-200`}>
          <div className="w-3 h-3 bg-white rounded-full absolute top-1 left-1"></div>
        </div>
      </div>

      {/* Mouth */}
      <div className="absolute w-full bottom-1/4 flex justify-center">
        <div 
          className={`w-16 h-8 ${
            isProcessing ? 'animate-pulse' : ''
          } ${
            isListening ? 'h-12 animate-bounce' : ''
          } ${
            isSpeaking ? 'h-10 animate-bounce-slow' : ''
          } bg-blue-600 rounded-full transition-all duration-300`}
        >
          {(isListening || isSpeaking) && (
            <div className="w-full h-full flex justify-center items-center">
              <div className={`w-8 h-2 bg-white rounded-full ${
                isSpeaking ? 'animate-pulse' : ''
              }`}></div>
            </div>
          )}
        </div>
      </div>

      {/* AI Brain Waves */}
      <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
        <div className={`w-16 h-16 ${isProcessing || isSpeaking ? 'animate-spin-slow' : ''}`}>
          <div className="absolute w-full h-full border-4 border-blue-400 rounded-full opacity-20"></div>
          <div className="absolute w-full h-full border-4 border-blue-400 rounded-full opacity-20" 
               style={{ transform: 'rotate(45deg)' }}></div>
        </div>
      </div>
    </div>
  );
};

export default AICharacter; 