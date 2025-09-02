import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { listRunsFiltered } from '../lib/api';
import { DataTable } from '../components/DataTable';
import type { Column, RowAction } from '../components/DataTable';
import { RunStatusBadge } from '../components/RunStatusBadge';
import { formatDistanceToNow } from '../lib/utils';
import { getDateRangeFilter, parseRunQueryFromURL, runQueryToURLParams } from '../lib/query-utils';
import { ExternalLink, Eye, RotateCcw } from 'lucide-react';
import type { Run, RunQuery } from '../lib/schemas';

export default function RunsList() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState<RunQuery>(() => parseRunQueryFromURL(searchParams));

  // Update URL when query changes
  useEffect(() => {
    const params = runQueryToURLParams(query);
    setSearchParams(params, { replace: true });
  }, [query, setSearchParams]);

  const { data: runs = [], isLoading, error } = useQuery({
    queryKey: ['runs-filtered', query],
    queryFn: () => listRunsFiltered(query),
    refetchInterval: 10_000, // Auto-refresh every 10 seconds
  });

  const columns: Column<Run>[] = [
    {
      key: 'type',
      header: 'Type',
      sortable: true,
      render: (value) => <span className="capitalize font-black text-black uppercase">{value}</span>
    },
    {
      key: 'status',
      header: 'Status',
      sortable: true,
      render: (value) => <RunStatusBadge status={value} />
    },
    {
      key: 'startedAt',
      header: 'Started',
      sortable: true,
      render: (value) => (
        <div>
          <div className="text-sm font-bold text-gray-900">{formatDistanceToNow(value)}</div>
          <div className="text-xs text-gray-500">
            {new Date(value).toLocaleString()}
          </div>
        </div>
      )
    },
    {
      key: 'finishedAt',
      header: 'Duration',
      sortable: true,
      render: (value, item) => {
        if (!value) {
          return item.status === 'running' ? (
            <span className="text-sm font-bold text-tb-green-600">Running...</span>
          ) : (
            <span className="text-sm text-gray-400">-</span>
          );
        }
        const durationMs = new Date(value).getTime() - new Date(item.startedAt).getTime();
        const minutes = Math.floor(durationMs / 60000);
        const seconds = Math.floor((durationMs % 60000) / 1000);
        return <span className="text-sm font-bold">{minutes}m {seconds}s</span>;
      }
    },
    {
      key: 'metrics',
      header: 'Metrics',
      render: (value) => {
        if (!value || Object.keys(value).length === 0) {
          return <span className="text-gray-400">-</span>;
        }
        return (
          <div className="text-xs space-y-1 font-bold">
            {value.ticks && <div className="text-blue-600">TICKS: {value.ticks}</div>}
            {value.entropy && <div className="text-purple-600">ENTROPY: {value.entropy.toFixed(2)}</div>}
            {value.scup && <div className="text-orange-600">SCUP: {value.scup.toFixed(2)}</div>}
          </div>
        );
      }
    }
  ];

  const rowActions: RowAction<Run>[] = [
    {
      icon: Eye,
      label: 'View Details',
      onClick: (run) => navigate(`/runs/${run.id}`),
      variant: 'primary'
    },
    {
      icon: ExternalLink,
      label: 'View Logs',
      onClick: (run) => {
        if (run.logsUrl) {
          window.open(run.logsUrl, '_blank');
        }
      },
      variant: 'default'
    }
  ];

  // Filter management
  const getActiveFilters = () => {
    const filters = [];
    
    if (query.type) {
      filters.push({
        id: 'type',
        label: `Type: ${query.type}`,
        value: query.type
      });
    }
    
    if (query.status && query.status.length > 0) {
      filters.push({
        id: 'status',
        label: `Status: ${query.status.join(', ')}`,
        value: query.status
      });
    }
    
    if (query.q) {
      filters.push({
        id: 'search',
        label: `Search: "${query.q}"`,
        value: query.q
      });
    }
    
    return filters;
  };

  const handleRemoveFilter = (filterId: string) => {
    setQuery(prev => {
      const newQuery = { ...prev };
      if (filterId === 'type') delete newQuery.type;
      if (filterId === 'status') delete newQuery.status;
      if (filterId === 'search') delete newQuery.q;
      return newQuery;
    });
  };

  const handleClearAllFilters = () => {
    setQuery({});
  };

  const handleSort = (key: string, direction: 'asc' | 'desc') => {
    setQuery(prev => ({
      ...prev,
      sort: `${key}:${direction}`
    }));
  };

  const [sortField, sortDirection] = query.sort ? query.sort.split(':') : ['startedAt', 'desc'];

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-error-50 border border-error-200 rounded-md p-4">
          <h3 className="text-sm font-medium text-error-800">Failed to load runs</h3>
          <p className="text-sm text-error-700 mt-1">
            {error instanceof Error ? error.message : 'An error occurred'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Pipeline Runs</h1>
        <p className="text-gray-600">Monitor and manage pipeline executions</p>
      </div>

      {/* Simplified Filters */}
      <div className="bg-gray-50 rounded-lg p-4 mb-6">
        <div className="flex items-center space-x-4 mb-4">
          <input
            type="text"
            placeholder="Search runs..."
            value={query.q || ''}
            onChange={(e) => setQuery(prev => ({ ...prev, q: e.target.value || undefined }))}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-transparent"
          />
          <select
            value={query.type || ''}
            onChange={(e) => setQuery(prev => ({ ...prev, type: e.target.value as any || undefined }))}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500"
          >
            <option value="">All types</option>
            <option value="docs">📚 Docs</option>
            <option value="train">🧠 Train</option>
            <option value="export">🚀 Export</option>
          </select>
        </div>
        
        {/* Active Filters */}
        {getActiveFilters().length > 0 && (
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Active filters:</span>
            {getActiveFilters().map((filter) => (
              <span
                key={filter.id}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-pink-100 text-pink-700"
              >
                {filter.label}
                <button
                  onClick={() => handleRemoveFilter(filter.id)}
                  className="ml-2 text-pink-500 hover:text-pink-700"
                >
                  ×
                </button>
              </span>
            ))}
            <button
              onClick={handleClearAllFilters}
              className="text-sm text-gray-500 hover:text-gray-700 underline"
            >
              Clear all
            </button>
          </div>
        )}
      </div>

      {/* Runs Table */}
      <div className="bg-white rounded-lg border border-gray-200">
        {isLoading ? (
          <div className="p-8 text-center">
            <div className="animate-spin w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-gray-500">Loading runs...</p>
          </div>
        ) : (
          <DataTable
            data={runs}
            columns={columns}
            onRowClick={(run) => navigate(`/runs/${run.id}`)}
            rowActions={rowActions}
            sortBy={sortField}
            sortDirection={sortDirection as 'asc' | 'desc'}
            onSort={handleSort}
            loading={isLoading}
            emptyMessage="No runs found"
          />
        )}
      </div>

      {runs.length === 0 && !isLoading && (
        <div className="text-center py-12">
          <p className="text-gray-500 text-lg mb-2">No pipeline runs yet</p>
          <p className="text-gray-400">Start your first run from the dashboard</p>
        </div>
      )}
    </div>
  );
}
