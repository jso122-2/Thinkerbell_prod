import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Download, Eye, FileText, Plus } from 'lucide-react';
import { listArtifactsFiltered } from '../lib/api';
import { DataTable } from '../components/DataTable';
import type { Column, RowAction } from '../components/DataTable';
import { Drawer } from '../components/Drawer';
import { MarkdownPreview } from '../components/MarkdownPreview';
import { WhitepaperBuilder } from '../components/WhitepaperBuilder';
import { formatDistanceToNow, formatBytes } from '../lib/utils';
import { STYLE_PROFILES } from '../mocks/data-transformer';
import type { Artifact, ArtifactQuery } from '../lib/schemas';

export default function ArtifactsList() {
  const navigate = useNavigate();
  const [query, setQuery] = useState<ArtifactQuery>({});
  const [selectedArtifacts, setSelectedArtifacts] = useState<string[]>([]);
  const [previewArtifact, setPreviewArtifact] = useState<Artifact | null>(null);
  const [showWhitepaperBuilder, setShowWhitepaperBuilder] = useState(false);

  const { data: artifacts = [], isLoading, error } = useQuery({
    queryKey: ['artifacts-filtered', query],
    queryFn: () => listArtifactsFiltered(query),
    staleTime: 30_000,
  });

  const getStatusBadge = (status: Artifact['status']) => {
    const classes = {
      ready: "bg-green-100 text-green-700 border border-green-200",
      processing: "bg-yellow-100 text-yellow-700 border border-yellow-200",
      failed: "bg-red-100 text-red-700 border border-red-200"
    };
    
    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold ${classes[status]}`}>
        {status.toUpperCase()}
      </span>
    );
  };

  const columns: Column<Artifact>[] = [
    {
      key: 'type',
      header: 'Type',
      sortable: true,
      render: (value) => {
        const profile = STYLE_PROFILES.ARTIFACT_TYPES[value as keyof typeof STYLE_PROFILES.ARTIFACT_TYPES];
        return (
          <div className="flex items-center space-x-3">
            <span className="text-2xl">{profile?.emoji}</span>
            <span className="capitalize font-black text-sm uppercase tracking-wider">{value}</span>
          </div>
        );
      }
    },
    {
      key: 'name',
      header: 'Name',
      sortable: true,
      render: (value) => (
        <div className="max-w-xs">
          <div className="font-bold text-gray-900 truncate" title={value}>{value}</div>
        </div>
      )
    },
    {
      key: 'status',
      header: 'Status',
      sortable: true,
      render: (value) => getStatusBadge(value)
    },
    {
      key: 'size',
      header: 'Size',
      sortable: true,
      render: (value) => value ? (
        <span className="font-bold text-sm text-pink-600">{formatBytes(value)}</span>
      ) : '-'
    },
    {
      key: 'createdAt',
      header: 'Created',
      sortable: true,
      render: (value) => (
        <div>
          <div className="text-sm font-bold text-gray-900">{formatDistanceToNow(value)}</div>
          <div className="text-xs text-gray-500">
            {new Date(value).toLocaleDateString()}
          </div>
        </div>
      )
    }
  ];

  const rowActions: RowAction<Artifact>[] = [
    {
      icon: Eye,
      label: 'Preview',
      onClick: (artifact) => setPreviewArtifact(artifact),
      variant: 'primary'
    },
    {
      icon: FileText,
      label: 'Details',
      onClick: (artifact) => navigate(`/artifacts/${artifact.id}`),
      variant: 'default'
    },
    {
      icon: Download,
      label: 'Download',
      onClick: (artifact) => {
        if (artifact.downloadUrl) {
          window.open(artifact.downloadUrl, '_blank');
        }
      },
      variant: 'default'
    }
  ];

  const handleSort = (key: string, direction: 'asc' | 'desc') => {
    setQuery(prev => ({
      ...prev,
      sort: `${key}:${direction}`
    }));
  };

  const [sortField, sortDirection] = query.sort ? query.sort.split(':') : ['createdAt', 'desc'];

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <h3 className="text-sm font-medium text-red-800">Failed to load artifacts</h3>
          <p className="text-sm text-red-700 mt-1">
            {error instanceof Error ? error.message : 'An error occurred'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-black text-black">
            Artifacts <span className="tb-accent-pink">🎨</span>
          </h1>
          <p className="text-xl text-gray-600 font-bold">
            Browse and manage generated outputs
          </p>
        </div>
        <div className="flex items-center space-x-3">
          {selectedArtifacts.length > 0 && (
            <span className="text-sm font-medium text-gray-700">
              {selectedArtifacts.length} selected
            </span>
          )}
          <button
            onClick={() => setShowWhitepaperBuilder(true)}
            className="tb-button-primary flex items-center space-x-2"
          >
            <Plus className="w-5 h-5" />
            <span>Create Whitepaper</span>
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="flex items-center space-x-4">
          <select
            value={query.type || ''}
            onChange={(e) => setQuery(prev => ({ ...prev, type: e.target.value as any || undefined }))}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500"
          >
            <option value="">All types</option>
            <option value="model">🧬 Model</option>
            <option value="report">📊 Report</option>
            <option value="zine">📰 Zine</option>
            <option value="image">🎨 Image</option>
            <option value="log">📝 Log</option>
          </select>
          
          <select
            value={query.status?.join(',') || ''}
            onChange={(e) => {
              const statuses = e.target.value ? e.target.value.split(',') : [];
              setQuery(prev => ({ ...prev, status: statuses.length > 0 ? statuses as any : undefined }));
            }}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500"
          >
            <option value="">All statuses</option>
            <option value="ready">✅ Ready</option>
            <option value="processing">🔄 Processing</option>
            <option value="failed">❌ Failed</option>
          </select>
        </div>
      </div>

      {/* Artifacts Table */}
      <div className="tb-card">
        <DataTable
          data={artifacts}
          columns={columns}
          onRowClick={(artifact) => navigate(`/artifacts/${artifact.id}`)}
          rowActions={rowActions}
          sortBy={sortField}
          sortDirection={sortDirection as 'asc' | 'desc'}
          onSort={handleSort}
          loading={isLoading}
          emptyMessage="No artifacts found. Start some pipeline runs to generate artifacts!"
        />
      </div>

      {artifacts.length === 0 && !isLoading && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🎨</div>
          <p className="text-2xl font-bold text-gray-500 mb-2">No artifacts available</p>
          <p className="text-gray-400">Run some pipelines to generate artifacts</p>
        </div>
      )}

      {/* Preview Drawer */}
      {previewArtifact && (
        <Drawer
          isOpen={!!previewArtifact}
          onClose={() => setPreviewArtifact(null)}
          title={`Preview: ${previewArtifact.name}`}
        >
          <div className="p-6">
            <div className="mb-4">
              <div className="flex items-center space-x-3 mb-3">
                <span className="text-3xl">
                  {STYLE_PROFILES.ARTIFACT_TYPES[previewArtifact.type]?.emoji}
                </span>
                <div>
                  <h3 className="font-black text-lg">{previewArtifact.name}</h3>
                  <p className="text-sm text-gray-500">
                    {STYLE_PROFILES.ARTIFACT_TYPES[previewArtifact.type]?.description}
                  </p>
                </div>
              </div>
              {getStatusBadge(previewArtifact.status)}
            </div>

            {previewArtifact.type === 'report' || previewArtifact.type === 'zine' ? (
              <MarkdownPreview 
                content={`# ${previewArtifact.name}\n\nThis is a preview of the ${previewArtifact.type}. In a real implementation, this would show the actual content from the artifact URL.\n\n## Metadata\n- **Size**: ${previewArtifact.size ? formatBytes(previewArtifact.size) : 'Unknown'}\n- **Created**: ${formatDistanceToNow(previewArtifact.createdAt)}\n- **Status**: ${previewArtifact.status}`}
              />
            ) : previewArtifact.type === 'image' ? (
              <div className="text-center p-8 bg-gray-100 rounded-lg">
                <div className="text-4xl mb-4">🖼️</div>
                <p className="text-gray-600">Image preview would appear here</p>
                <p className="text-sm text-gray-500 mt-2">
                  {previewArtifact.size ? formatBytes(previewArtifact.size) : 'Unknown size'}
                </p>
              </div>
            ) : previewArtifact.type === 'model' ? (
              <div className="space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-bold text-blue-800 mb-2">Model Information</h4>
                  <div className="text-sm space-y-1 text-blue-700">
                    <p><strong>Format:</strong> {previewArtifact.name.split('.').pop()?.toUpperCase()}</p>
                    <p><strong>Size:</strong> {previewArtifact.size ? formatBytes(previewArtifact.size) : 'Unknown'}</p>
                    <p><strong>Created:</strong> {formatDistanceToNow(previewArtifact.createdAt)}</p>
                  </div>
                </div>
                <div className="text-center p-8 bg-gray-100 rounded-lg">
                  <div className="text-4xl mb-4">🧬</div>
                  <p className="text-gray-600">Model architecture visualization would appear here</p>
                </div>
              </div>
            ) : (
              <div className="text-center p-8 bg-gray-100 rounded-lg">
                <div className="text-4xl mb-4">📝</div>
                <p className="text-gray-600">Log content preview would appear here</p>
                <p className="text-sm text-gray-500 mt-2">
                  {previewArtifact.size ? formatBytes(previewArtifact.size) : 'Unknown size'}
                </p>
              </div>
            )}

            {previewArtifact.downloadUrl && (
              <div className="mt-6 pt-4 border-t border-gray-200">
                <a
                  href={previewArtifact.downloadUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="tb-button-primary w-full text-center block"
                >
                  <Download className="w-4 h-4 inline mr-2" />
                  Download Artifact
                </a>
              </div>
            )}
          </div>
        </Drawer>
      )}

      {/* Whitepaper Builder Modal */}
      {showWhitepaperBuilder && (
        <WhitepaperBuilder
          artifacts={artifacts}
          isOpen={showWhitepaperBuilder}
          onClose={() => setShowWhitepaperBuilder(false)}
          selectedArtifacts={selectedArtifacts}
          onSelectionChange={setSelectedArtifacts}
        />
      )}
    </div>
  );
}