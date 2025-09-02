import { useState } from 'react';
import { Search, X, Calendar, Filter } from 'lucide-react';
import { clsx } from 'clsx';

interface FilterChip {
  id: string;
  label: string;
  value: any;
}

interface FilterBarProps {
  searchValue?: string;
  onSearchChange: (value: string) => void;
  typeFilter?: string;
  onTypeFilterChange: (type: string | undefined) => void;
  statusFilter?: string[];
  onStatusFilterChange: (status: string[]) => void;
  dateRange?: 'last24h' | 'last7d' | 'last30d' | 'custom';
  onDateRangeChange: (range: 'last24h' | 'last7d' | 'last30d' | 'custom') => void;
  customDateRange?: { from: string; to: string };
  onCustomDateRangeChange: (range: { from: string; to: string } | undefined) => void;
  activeFilters: FilterChip[];
  onRemoveFilter: (filterId: string) => void;
  onClearAllFilters: () => void;
}

export function FilterBar({
  searchValue = '',
  onSearchChange,
  typeFilter,
  onTypeFilterChange,
  statusFilter = [],
  onStatusFilterChange,
  dateRange,
  onDateRangeChange,
  customDateRange,
  onCustomDateRangeChange,
  activeFilters,
  onRemoveFilter,
  onClearAllFilters
}: FilterBarProps) {
  const [showFilters, setShowFilters] = useState(false);

  const typeOptions = [
    { value: 'docs', label: 'Docs' },
    { value: 'train', label: 'Train' },
    { value: 'export', label: 'Export' }
  ];

  const statusOptions = [
    { value: 'queued', label: 'Queued' },
    { value: 'running', label: 'Running' },
    { value: 'success', label: 'Success' },
    { value: 'failed', label: 'Failed' },
    { value: 'cancelled', label: 'Cancelled' }
  ];

  const dateRangeOptions = [
    { value: 'last24h', label: 'Last 24 hours' },
    { value: 'last7d', label: 'Last 7 days' },
    { value: 'last30d', label: 'Last 30 days' },
    { value: 'custom', label: 'Custom range' }
  ];

  return (
    <div className="space-y-4">
      {/* Search and Filter Toggle */}
      <div className="flex items-center space-x-4">
        {/* Search Input */}
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <input
            type="text"
            placeholder="Search runs..."
            value={searchValue}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-tb-magenta focus:border-transparent"
          />
        </div>

        {/* Filter Toggle */}
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={clsx(
            'inline-flex items-center px-4 py-2 border border-gray-300 rounded-lg font-medium transition-all duration-200',
            showFilters ? 'bg-tb-magenta text-white border-tb-magenta' : 'bg-white text-gray-700 hover:bg-gray-50'
          )}
        >
          <Filter className="w-4 h-4 mr-2" />
          Filters
          {activeFilters.length > 0 && (
            <span className="ml-2 px-2 py-1 text-xs bg-tb-pink-100 text-tb-magenta rounded-full">
              {activeFilters.length}
            </span>
          )}
        </button>
      </div>

      {/* Active Filter Chips */}
      {activeFilters.length > 0 && (
        <div className="flex items-center flex-wrap gap-2">
          <span className="text-sm font-medium text-gray-700">Active filters:</span>
          {activeFilters.map((filter) => (
            <span
              key={filter.id}
              className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-tb-pink-100 text-tb-magenta"
            >
              {filter.label}
              <button
                onClick={() => onRemoveFilter(filter.id)}
                className="ml-2 text-tb-magenta hover:text-tb-pink-700"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          ))}
          <button
            onClick={onClearAllFilters}
            className="text-sm text-gray-500 hover:text-gray-700 underline"
          >
            Clear all
          </button>
        </div>
      )}

      {/* Filter Panel */}
      {showFilters && (
        <div className="bg-gray-50 rounded-lg p-4 space-y-4 border border-gray-200">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Type Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Type</label>
              <select
                value={typeFilter || ''}
                onChange={(e) => onTypeFilterChange(e.target.value || undefined)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-tb-magenta focus:border-transparent"
              >
                <option value="">All types</option>
                {typeOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Status Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {statusOptions.map((option) => (
                  <label key={option.value} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={statusFilter.includes(option.value)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          onStatusFilterChange([...statusFilter, option.value]);
                        } else {
                          onStatusFilterChange(statusFilter.filter(s => s !== option.value));
                        }
                      }}
                      className="rounded border-gray-300 text-tb-magenta focus:ring-tb-magenta"
                    />
                    <span className="ml-2 text-sm text-gray-700">{option.label}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Date Range Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
              <select
                value={dateRange || ''}
                onChange={(e) => onDateRangeChange(e.target.value as any)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-tb-magenta focus:border-transparent"
              >
                <option value="">All time</option>
                {dateRangeOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Custom Date Range */}
          {dateRange === 'custom' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-gray-200">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">From</label>
                <input
                  type="datetime-local"
                  value={customDateRange?.from ? new Date(customDateRange.from).toISOString().slice(0, 16) : ''}
                  onChange={(e) => {
                    const from = e.target.value ? new Date(e.target.value).toISOString() : '';
                    onCustomDateRangeChange(from && customDateRange?.to ? { from, to: customDateRange.to } : undefined);
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-tb-magenta focus:border-transparent"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">To</label>
                <input
                  type="datetime-local"
                  value={customDateRange?.to ? new Date(customDateRange.to).toISOString().slice(0, 16) : ''}
                  onChange={(e) => {
                    const to = e.target.value ? new Date(e.target.value).toISOString() : '';
                    onCustomDateRangeChange(to && customDateRange?.from ? { from: customDateRange.from, to } : undefined);
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-tb-magenta focus:border-transparent"
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}