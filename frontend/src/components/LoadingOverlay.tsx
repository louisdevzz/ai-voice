import { Loader } from 'lucide-react';

const LoadingOverlay = () => {
  return (
    <div className="fixed inset-0 bg-white z-10 flex items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <Loader className="w-8 h-8 text-gray-800 animate-spin" />
        <p className="text-gray-800 text-sm font-medium tracking-wide">Loading chat...</p>
      </div>
    </div>
  );
};

export default LoadingOverlay; 