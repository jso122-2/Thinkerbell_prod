import { useQuery } from '@tanstack/react-query';
import { Activity, Users, BarChart3, TrendingUp, RefreshCw, Zap, Sparkles, Rocket } from 'lucide-react';
import { listRuns, listArtifacts, getHealth } from '../lib/api';
import { MetricCard } from '../components/MetricCard';
import { RunStatusBadge } from '../components/RunStatusBadge';
import { StartJobButton } from '../components/StartJobButton';
import { HealthPill } from '../components/HealthPill';
import { formatDistanceToNow } from '../lib/utils';
import { STYLE_PROFILES } from '../mocks/data-transformer';

export default function Dashboard() {
  console.log('📊 Production Dashboard rendering...');

  const { data: health, isLoading: healthLoading, error: healthError } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: 1, // Less aggressive retry for health checks
    retryDelay: 1000,
  });

  const { data: runs = [], isLoading: runsLoading, error: runsError } = useQuery({
    queryKey: ['runs'],
    queryFn: listRuns,
    refetchInterval: 10_000,
    retry: 2,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 5000),
  });

  const { data: artifacts = [], isLoading: artifactsLoading, error: artifactsError } = useQuery({
    queryKey: ['artifacts'],
    queryFn: listArtifacts,
    staleTime: 30_000,
    retry: 2,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 5000),
  });

  // Error state handling
  const hasErrors = healthError || runsError || artifactsError;
  const allLoading = healthLoading && runsLoading && artifactsLoading;

  // Enhanced metrics with style-conscious calculations
  const runningRuns = runs.filter(run => run.status === 'running').length;
  const successfulRuns = runs.filter(run => run.status === 'success').length;
  const totalRuns = runs.length;
  const successRate = totalRuns > 0 ? Math.round((successfulRuns / totalRuns) * 100) : 0;
  
  const recentRuns = runs.slice(0, 5);
  const recentArtifacts = artifacts.slice(0, 4);

  // Get creative artifact type distribution
  const artifactsByType = artifacts.reduce((acc, artifact) => {
    acc[artifact.type] = (acc[artifact.type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="space-y-8 p-8">
      {/* Connection Status Banner */}
      {hasErrors && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-lg">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <RefreshCw className="h-5 w-5 text-yellow-400 animate-spin" />
            </div>
            <div className="ml-3">
              <p className="text-sm font-bold text-yellow-800">
                🔄 Connecting to services...
              </p>
              <p className="text-xs text-yellow-700 mt-1">
                {healthError && "Health service unavailable. "}
                {runsError && "Runs service unavailable. "}
                {artifactsError && "Artifacts service unavailable. "}
                Using cached data where available.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Loading State for Initial Load */}
      {allLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="text-6xl mb-4">⚡</div>
            <h2 className="text-2xl font-black text-black mb-2">Initializing Thinkerbell</h2>
            <p className="text-gray-600">Loading pipeline data...</p>
            <div className="mt-6 flex justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-pink-600"></div>
            </div>
          </div>
        </div>
      )}

      {/* Hero Header */}
      {!allLoading && (
        <div className="relative">
          <h1 className="text-5xl md:text-7xl font-black text-black mb-4">
            Thinkerbell <span className="tb-accent-pink">⚡</span>
          </h1>
          <p className="text-2xl text-gray-600 mb-6 font-bold">
            AI Pipeline Command Center
          </p>
          <div className="tb-zigzag mb-4"></div>
          <div className="flex items-center space-x-4 text-sm font-bold">
            <span className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-tb-magenta" />
              <span>NEURAL NETWORKS ACTIVE</span>
            </span>
            <span className="flex items-center space-x-2">
              <Rocket className="w-4 h-4 text-tb-green-500" />
              <span>PIPELINES OPTIMIZED</span>
            </span>
            <span className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-blue-500" />
              <span>{runs.length} RUNS LOADED</span>
            </span>
          </div>
        </div>
      )}

      {/* Main Dashboard Content */}
      {!allLoading && (
        <>
          {/* Health Status */}
          <div className="mb-8">
            {healthLoading ? (
              <div className="tb-card">
                <div className="animate-pulse h-20 bg-gray-200 rounded"></div>
              </div>
            ) : health ? (
              <HealthPill health={health} isLoading={healthLoading} error={healthError} />
            ) : healthError ? (
              <div className="tb-card">
                <div className="flex items-center space-x-3">
                  <div className="text-2xl">⚠️</div>
                  <div>
                    <h3 className="font-black text-black">Health Service Unavailable</h3>
                    <p className="text-sm text-gray-600">Using default status indicators</p>
                  </div>
                </div>
              </div>
            ) : null}
          </div>

      {/* Enhanced Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="ACTIVE RUNS"
          value={runningRuns.toString()}
          icon={Activity}
          trend={runningRuns > 0 ? 'up' : 'stable'}
          description={`⚡ ${runningRuns} pipeline${runningRuns !== 1 ? 's' : ''} running`}
        />
        <MetricCard
          title="SUCCESS RATE"
          value={`${successRate}%`}
          icon={TrendingUp}
          trend={successRate >= 80 ? 'up' : successRate >= 60 ? 'stable' : 'down'}
          description={`🎯 ${successfulRuns}/${totalRuns} successful`}
        />
        <MetricCard
          title="TOTAL RUNS"
          value={totalRuns.toString()}
          icon={BarChart3}
          trend="up"
          description="🚀 All time executions"
        />
        <MetricCard
          title="ARTIFACTS"
          value={artifacts.length.toString()}
          icon={Users}
          trend="up"
          description="🎨 Generated outputs"
        />
      </div>

      <div className="tb-zigzag-green mb-8"></div>

      {/* Enhanced Quick Actions */}
      <div className="mb-8">
        <h2 className="text-3xl md:text-5xl font-black text-black mb-6">
          Quick <span className="tb-accent-green">Actions</span>
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <StartJobButton type="docs" />
          <StartJobButton type="train" />
          <StartJobButton type="export" />
        </div>
      </div>

      <div className="tb-zigzag mb-8"></div>

      {/* Enhanced Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Runs with Style Profiles */}
        <div className="tb-card">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-2xl font-black text-black">RECENT RUNS</h3>
            <RefreshCw className="w-6 h-6 text-tb-magenta animate-pulse-tb" />
          </div>
          {runsLoading ? (
            <div className="space-y-3">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="animate-pulse h-20 bg-gray-200 rounded-lg"></div>
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {recentRuns.map((run) => {
                const typeProfile = STYLE_PROFILES.PIPELINE_TYPES[run.type];
                return (
                  <div key={run.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-tb-pink-50 transition-all duration-200 hover:shadow-md border-l-4 hover:border-tb-magenta">
                    <div className="flex items-center space-x-4">
                      <div className="text-2xl">{typeProfile?.emoji}</div>
                      <div>
                        <div className="font-black text-black text-sm uppercase tracking-wider">{run.type}</div>
                        <div className="text-xs text-gray-500 font-medium">{formatDistanceToNow(run.startedAt)}</div>
                        {run.metrics && (
                          <div className="text-xs text-tb-green-600 font-bold">
                            {run.metrics.entropy && `ENTROPY: ${run.metrics.entropy.toFixed(2)}`}
                          </div>
                        )}
                      </div>
                    </div>
                    <RunStatusBadge status={run.status} />
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Recent Artifacts with Type Icons */}
        <div className="tb-card">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-2xl font-black text-black">RECENT ARTIFACTS</h3>
            <Zap className="w-6 h-6 text-tb-green-500 animate-bounce-tb" />
          </div>
          {artifactsLoading ? (
            <div className="space-y-3">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="animate-pulse h-20 bg-gray-200 rounded-lg"></div>
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {recentArtifacts.map((artifact) => {
                const profile = STYLE_PROFILES.ARTIFACT_TYPES[artifact.type];
                return (
                  <div key={artifact.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-tb-green-50 transition-all duration-200 hover:shadow-md border-l-4 hover:border-tb-green-500">
                    <div className="flex items-center space-x-4">
                      <div className="text-2xl">{profile?.emoji}</div>
                      <div>
                        <div className="font-black text-black text-sm uppercase tracking-wider">{artifact.type}</div>
                        <div className="text-xs text-gray-500 font-medium truncate max-w-xs">{artifact.name}</div>
                        {artifact.size && (
                          <div className="text-xs text-tb-magenta font-bold">
                            {(artifact.size / (1024 * 1024)).toFixed(1)}MB
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="text-xs text-gray-400 font-medium">{formatDistanceToNow(artifact.createdAt)}</div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Artifact Type Distribution */}
      {Object.keys(artifactsByType).length > 0 && (
        <div className="tb-card mt-8">
          <h3 className="text-2xl font-black text-black mb-6">ARTIFACT DISTRIBUTION</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {Object.entries(artifactsByType).map(([type, count]) => {
              const profile = STYLE_PROFILES.ARTIFACT_TYPES[type as keyof typeof STYLE_PROFILES.ARTIFACT_TYPES];
              return (
                <div key={type} className="text-center p-4 bg-gray-50 rounded-xl">
                  <div className="text-3xl mb-2">{profile?.emoji}</div>
                  <div className="font-black text-lg text-black">{count}</div>
                  <div className="text-xs text-gray-500 uppercase font-bold">{type}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}
        </>
      )}
    </div>
  );
}