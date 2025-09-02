import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: ReactNode;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'stable';
  description?: string;
  className?: string;
}

export function MetricCard({ title, value, icon: Icon, trend, description, className }: MetricCardProps) {
  const getTrendColor = () => {
    switch (trend) {
      case 'up': return 'text-green-500';
      case 'down': return 'text-red-500';
      case 'stable': return 'text-gray-500';
      default: return 'text-gray-500';
    }
  };

  return (
    <div className={`tb-card ${className || ''}`}>
      <div className="flex items-center">
        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-bold text-gray-600 uppercase tracking-wider">{title}</p>
            {Icon && <Icon className={`w-5 h-5 ${getTrendColor()}`} />}
          </div>
          <div className="mt-2 mb-3">
            <div className="tb-zigzag"></div>
          </div>
          <p className="text-3xl font-black text-black">{value}</p>
          {description && (
            <p className="text-sm text-gray-600 mt-2 font-medium">{description}</p>
          )}
        </div>
      </div>
    </div>
  );
}
