import { Loader } from 'lucide-react';

interface ProcessLogsProps {
    steps: {
        message: string;
        status: 'pending' | 'completed' | 'current';
    }[];
    className?: string;
}

export function ProcessLogs({ steps }: ProcessLogsProps) {
    // Only show the current step
    const currentStep = steps.find(step => step.status === 'current');

    if (!currentStep) return null;

    return (
        <div className="space-y-3">
            <div className="flex items-center gap-3 text-sm transition-all duration-200 text-black animate-fadeIn">
                <Loader className="w-4 h-4 animate-spin" />
                <span>{currentStep.message}</span>
            </div>
        </div>
    );
} 