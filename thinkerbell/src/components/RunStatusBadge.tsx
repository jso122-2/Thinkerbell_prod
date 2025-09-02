import { clsx } from 'clsx';
import type { Run } from '../lib/schemas';

interface RunStatusBadgeProps {
  status: Run['status'];
  className?: string;
}

export function RunStatusBadge({ status, className }: RunStatusBadgeProps) {
  const baseClasses = "inline-flex items-center px-4 py-2 rounded-xl text-sm font-black uppercase tracking-wider border-2";
  
  const statusClasses = {
    queued: "bg-gray-100 text-gray-800 border-gray-300",
    running: "bg-tb-green-50 text-tb-green-700 border-tb-green-300 animate-pulse-tb",
    success: "bg-tb-green-50 text-tb-green-700 border-tb-green-300", 
    failed: "bg-tb-pink-50 text-tb-pink-700 border-tb-pink-300",
    cancelled: "bg-gray-100 text-gray-600 border-gray-300"
  };

  return (
    <span className={clsx(baseClasses, statusClasses[status], className)}>
      {status}
    </span>
  );
}
