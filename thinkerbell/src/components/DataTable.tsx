import type { ReactNode } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';
import { clsx } from 'clsx';

export interface Column<T> {
  key: keyof T;
  header: string;
  render?: (value: any, item: T) => ReactNode;
  className?: string;
  sortable?: boolean;
  width?: string;
}

export interface RowAction<T> {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  onClick: (item: T) => void;
  variant?: 'default' | 'primary' | 'danger';
}

// Explicit re-exports to help with TypeScript/Vite
export type { Column as DataTableColumn, RowAction as DataTableRowAction };

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  onRowClick?: (item: T) => void;
  rowActions?: RowAction<T>[];
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
  onSort?: (key: string, direction: 'asc' | 'desc') => void;
  loading?: boolean;
  emptyMessage?: string;
  className?: string;
}

export function DataTable<T extends { id: string }>({ 
  data, 
  columns,
  onRowClick,
  rowActions,
  sortBy,
  sortDirection = 'desc',
  onSort,
  loading = false,
  emptyMessage = "No data available",
  className 
}: DataTableProps<T>) {
  const handleSort = (key: string) => {
    if (!onSort) return;
    
    const newDirection = sortBy === key && sortDirection === 'desc' ? 'asc' : 'desc';
    onSort(key, newDirection);
  };

  const getSortIcon = (key: string) => {
    if (sortBy !== key) return null;
    
    return sortDirection === 'asc' ? (
      <ChevronUp className="w-4 h-4 text-tb-magenta" />
    ) : (
      <ChevronDown className="w-4 h-4 text-tb-magenta" />
    );
  };

  if (loading) {
    return (
      <div className="tb-card">
        <div className="animate-pulse space-y-4">
          <div className="h-12 bg-gray-200 rounded"></div>
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-16 bg-gray-100 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={`tb-card overflow-hidden ${className || ''}`}>
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr className="border-b-2 border-gray-200">
              {columns.map((column) => (
                <th
                  key={String(column.key)}
                  className={clsx(
                    'px-6 py-4 text-left text-sm font-black text-gray-900 uppercase tracking-wider',
                    column.sortable && onSort && 'cursor-pointer hover:bg-gray-50 select-none',
                    column.className
                  )}
                  style={{ width: column.width }}
                  onClick={() => column.sortable && handleSort(String(column.key))}
                >
                  <div className="flex items-center space-x-2">
                    <span>{column.header}</span>
                    {column.sortable && getSortIcon(String(column.key))}
                  </div>
                </th>
              ))}
              {rowActions && rowActions.length > 0 && (
                <th className="px-6 py-4 text-right text-sm font-black text-gray-900 uppercase tracking-wider">
                  ACTIONS
                </th>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {data.map((item, index) => (
              <tr
                key={item.id}
                onClick={() => onRowClick?.(item)}
                className={clsx(
                  'transition-all duration-200',
                  onRowClick ? 'cursor-pointer hover:bg-tb-pink-50 hover:border-l-4 hover:border-tb-magenta' : '',
                  index % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                )}
              >
                {columns.map((column) => (
                  <td
                    key={String(column.key)}
                    className="px-6 py-4 text-sm font-medium text-gray-900"
                  >
                    {column.render 
                      ? column.render(item[column.key], item)
                      : String(item[column.key] || '-')
                    }
                  </td>
                ))}
                {rowActions && rowActions.length > 0 && (
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end space-x-2">
                      {rowActions.map((action, actionIndex) => {
                        const buttonClass = clsx(
                          'inline-flex items-center p-2 rounded-lg font-bold transition-all duration-200 hover:scale-105',
                          action.variant === 'primary' && 'text-tb-magenta hover:bg-tb-pink-50',
                          action.variant === 'danger' && 'text-red-600 hover:bg-red-50',
                          !action.variant && 'text-gray-600 hover:bg-gray-100'
                        );
                        
                        return (
                          <button
                            key={actionIndex}
                            onClick={(e) => {
                              e.stopPropagation();
                              action.onClick(item);
                            }}
                            className={buttonClass}
                            title={action.label}
                          >
                            <action.icon className="w-4 h-4" />
                          </button>
                        );
                      })}
                    </div>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.length === 0 && (
        <div className="py-16 text-center">
          <p className="text-xl font-bold text-gray-500">{emptyMessage}</p>
        </div>
      )}
    </div>
  );
}
