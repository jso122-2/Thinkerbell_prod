import { CheckCircle, AlertCircle, XCircle, Loader2 } from 'lucide-react';

interface Health {
  status: 'ok' | 'degraded' | 'down';
  dawn?: boolean;
  cursor?: boolean;
}

interface HealthPillProps {
  health?: Health;
  isLoading?: boolean;
  error?: Error | null;
}

export function HealthPill({ health, isLoading, error }: HealthPillProps) {

  if (isLoading) {
    return (
      <div className="inline-flex items-center px-3 py-1 rounded-full bg-gray-100">
        <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
        <span className="ml-2 text-sm text-gray-600">Checking...</span>
      </div>
    );
  }

  if (error || !health) {
    return (
      <div className="inline-flex items-center px-3 py-1 rounded-full bg-error-50">
        <XCircle className="w-4 h-4 text-error-600" />
        <span className="ml-2 text-sm text-error-700">Down</span>
      </div>
    );
  }

  const getStatusIcon = () => {
    switch (health.status) {
      case 'ok':
        return <CheckCircle className="w-4 h-4 text-success-600" />;
      case 'degraded':
        return <AlertCircle className="w-4 h-4 text-warning-600" />;
      case 'down':
        return <XCircle className="w-4 h-4 text-error-600" />;
    }
  };

  const getStatusClasses = () => {
    switch (health.status) {
      case 'ok':
        return 'bg-success-50 text-success-700';
      case 'degraded':
        return 'bg-warning-50 text-warning-700';
      case 'down':
        return 'bg-error-50 text-error-700';
    }
  };

  return (
    <div className={`inline-flex items-center px-3 py-1 rounded-full ${getStatusClasses()}`}>
      {getStatusIcon()}
      <span className="ml-2 text-sm capitalize">{health.status}</span>
      <div className="ml-2 flex space-x-1">
        <span className={`w-2 h-2 rounded-full ${health.dawn ? 'bg-green-400' : 'bg-red-400'}`} title="DAWN" />
        <span className={`w-2 h-2 rounded-full ${health.cursor ? 'bg-green-400' : 'bg-red-400'}`} title="Cursor" />
      </div>
    </div>
  );
}
