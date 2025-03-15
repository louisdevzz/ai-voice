interface TranscriptionDisplayProps {
  transcription: string;
  aiResponse: string;
  isLoading: boolean;
}

const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({
  transcription,
  aiResponse,
  isLoading,
}) => {
  return (
    <div className="w-full max-w-2xl mx-auto space-y-6 p-6 bg-white rounded-lg shadow-lg">
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-gray-700">Nội dung ghi âm:</h3>
        <div className="p-4 bg-gray-50 rounded-md min-h-[60px]">
          {transcription || 'Chưa có nội dung ghi âm...'}
        </div>
      </div>

      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-gray-700">Phản hồi AI:</h3>
        <div className="p-4 bg-blue-50 rounded-md min-h-[60px]">
          {isLoading ? (
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
            </div>
          ) : (
            aiResponse || 'Vui lòng chờ trong giây lát...'
          )}
        </div>
      </div>
    </div>
  );
};

export default TranscriptionDisplay; 