import { useMutation, useQueryClient } from '@tanstack/react-query';
import { PlayCircle, Loader2 } from 'lucide-react';
import { startRun } from '../lib/api';
import type { Run } from '../lib/schemas';

interface StartJobButtonProps {
  type: Run['type'];
  disabled?: boolean;
  className?: string;
}

const typeLabels = {
  docs: 'Build Docs',
  train: 'Start Training',
  export: 'Export Report'
};

const typeIcons = {
  docs: '📚',
  train: '🧠', 
  export: '📊'
};

const typeStyles = {
  docs: 'tb-button-primary',
  train: 'tb-button-secondary',
  export: 'bg-black hover:bg-gray-800 text-white font-bold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-lg'
};

export function StartJobButton({ type, disabled, className }: StartJobButtonProps) {
  const queryClient = useQueryClient();
  
  const mutation = useMutation({
    mutationFn: () => startRun(type),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] });
    },
  });

  return (
    <button
      onClick={() => mutation.mutate()}
      disabled={disabled || mutation.isPending}
      className={`
        inline-flex items-center
        ${typeStyles[type]}
        disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
        ${className || ''}
      `}
    >
      {mutation.isPending ? (
        <Loader2 className="w-5 h-5 mr-3 animate-spin" />
      ) : (
        <span className="mr-3 text-lg">{typeIcons[type]}</span>
      )}
      <span className="text-lg">
        {mutation.isPending ? 'Starting...' : typeLabels[type]}
      </span>
    </button>
  );
}
