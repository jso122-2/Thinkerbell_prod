import { useQuery } from '@tanstack/react-query';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, ExternalLink, Clock, Calendar, Zap } from 'lucide-react';
import { getRun } from '../lib/api';
import { RunStatusBadge } from '../components/RunStatusBadge';
import { LogViewer } from '../components/LogViewer';
import { formatDistanceToNow } from '../lib/utils';
import { STYLE_PROFILES } from '../mocks/data-transformer';

export default function RunDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data: run, isLoading, error } = useQuery({
    queryKey: ['run', id],
    queryFn: () => getRun(id!),
    enabled: !!id,
    refetchInterval: 10_000
  });

  if (!id) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <h3 className="text-sm font-medium text-red-800">Invalid Run ID</h3>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <h3 className="text-sm font-medium text-red-800">Failed to load run</h3>
          <p className="text-sm text-red-700 mt-1">
            {error instanceof Error ? error.message : 'An error occurred'}
          </p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="h-32 bg-gray-200 rounded"></div>
          <div className="h-96 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (!run) {
    return (
      <div className="p-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
          <h3 className="text-sm font-medium text-yellow-800">Run not found</h3>
        </div>
      </div>
    );
  }

  const typeProfile = STYLE_PROFILES.PIPELINE_TYPES[run.type];
  const statusProfile = STYLE_PROFILES.RUN_STATUS[run.status];

  const duration = run.finishedAt ? 
    new Date(run.finishedAt).getTime() - new Date(run.startedAt).getTime() :
    Date.now() - new Date(run.startedAt).getTime();

  const durationMinutes = Math.floor(duration / 60000);
  const durationSeconds = Math.floor((duration % 60000) / 1000);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b border-gray-200 bg-white">
        <div className="flex items-center space-x-4 mb-4">
          <button
            onClick={() => navigate('/runs')}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Runs</span>
          </button>
        </div>

        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-4">
            <div className="text-4xl">{typeProfile?.emoji}</div>
            <div>
              <h1 className="text-3xl font-black text-black">
                {typeProfile?.label} Pipeline
              </h1>
              <p className="text-gray-600 font-medium">
                {typeProfile?.description}
              </p>
              <div className="flex items-center space-x-4 mt-2">
                <span className="text-sm font-bold text-gray-500">ID: {run.id}</span>
                <RunStatusBadge status={run.status} />
              </div>
            </div>
          </div>

          {run.logsUrl && (
            <a
              href={run.logsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              <span>External Logs</span>
            </a>
          )}
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="p-6 bg-gray-50 border-b border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Started */}
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <div className="flex items-center space-x-3">
              <Calendar className="w-5 h-5 text-blue-500" />
              <div>
                <div className="text-sm font-bold text-gray-500 uppercase">Started</div>
                <div className="text-lg font-black text-black">
                  {formatDistanceToNow(run.startedAt)}
                </div>
                <div className="text-xs text-gray-500">
                  {new Date(run.startedAt).toLocaleString()}
                </div>
              </div>
            </div>
          </div>

          {/* Duration */}
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <div className="flex items-center space-x-3">
              <Clock className="w-5 h-5 text-green-500" />
              <div>
                <div className="text-sm font-bold text-gray-500 uppercase">Duration</div>
                <div className="text-lg font-black text-black">
                  {durationMinutes}m {durationSeconds}s
                </div>
                <div className="text-xs text-gray-500">
                  {run.status === 'running' ? 'In progress' : 'Total time'}
                </div>
              </div>
            </div>
          </div>

          {/* Metrics */}
          {run.metrics && Object.keys(run.metrics).length > 0 && (
            <>
              {run.metrics.entropy !== undefined && (
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                  <div className="flex items-center space-x-3">
                    <Zap className="w-5 h-5 text-purple-500" />
                    <div>
                      <div className="text-sm font-bold text-gray-500 uppercase">Entropy</div>
                      <div className="text-lg font-black text-black">
                        {run.metrics.entropy.toFixed(3)}
                      </div>
                      <div className="text-xs text-gray-500">
                        Model complexity
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {run.metrics.ticks !== undefined && (
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                  <div className="flex items-center space-x-3">
                    <Zap className="w-5 h-5 text-orange-500" />
                    <div>
                      <div className="text-sm font-bold text-gray-500 uppercase">Ticks</div>
                      <div className="text-lg font-black text-black">
                        {run.metrics.ticks.toLocaleString()}
                      </div>
                      <div className="text-xs text-gray-500">
                        Processing steps
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Logs */}
      <div className="flex-1">
        <LogViewer runId={run.id} />
      </div>
    </div>
  );
}