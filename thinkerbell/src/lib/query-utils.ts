import type { RunQuery, ArtifactQuery } from './schemas';

export function parseRunQueryFromURL(searchParams: URLSearchParams): RunQuery {
  const query: RunQuery = {};
  
  const type = searchParams.get('type');
  if (type && ['docs', 'train', 'export'].includes(type)) {
    query.type = type as RunQuery['type'];
  }
  
  const status = searchParams.get('status');
  if (status) {
    const statusArray = status.split(',').filter(s => 
      ['queued', 'running', 'success', 'failed', 'cancelled'].includes(s)
    ) as RunQuery['status'];
    if (statusArray.length > 0) {
      query.status = statusArray;
    }
  }
  
  const from = searchParams.get('from');
  if (from) query.from = from;
  
  const to = searchParams.get('to');
  if (to) query.to = to;
  
  const q = searchParams.get('q');
  if (q) query.q = q;
  
  const sort = searchParams.get('sort');
  if (sort && /^(startedAt|finishedAt|status|entropy|scup):(asc|desc)$/.test(sort)) {
    query.sort = sort;
  }
  
  return query;
}

export function runQueryToURLParams(query: RunQuery): URLSearchParams {
  const params = new URLSearchParams();
  
  if (query.type) params.set('type', query.type);
  if (query.status?.length) params.set('status', query.status.join(','));
  if (query.from) params.set('from', query.from);
  if (query.to) params.set('to', query.to);
  if (query.q) params.set('q', query.q);
  if (query.sort) params.set('sort', query.sort);
  
  return params;
}

export function parseArtifactQueryFromURL(searchParams: URLSearchParams): ArtifactQuery {
  const query: ArtifactQuery = {};
  
  const type = searchParams.get('type');
  if (type && ['model', 'report', 'zine', 'image', 'log'].includes(type)) {
    query.type = type as ArtifactQuery['type'];
  }
  
  const status = searchParams.get('status');
  if (status) {
    const statusArray = status.split(',').filter(s => 
      ['ready', 'processing', 'failed'].includes(s)
    ) as ArtifactQuery['status'];
    if (statusArray.length > 0) {
      query.status = statusArray;
    }
  }
  
  const sort = searchParams.get('sort');
  if (sort && /^(createdAt|size|status|name):(asc|desc)$/.test(sort)) {
    query.sort = sort;
  }
  
  return query;
}

export function artifactQueryToURLParams(query: ArtifactQuery): URLSearchParams {
  const params = new URLSearchParams();
  
  if (query.type) params.set('type', query.type);
  if (query.status?.length) params.set('status', query.status.join(','));
  if (query.sort) params.set('sort', query.sort);
  
  return params;
}

// Date range helpers
export function getDateRangeFilter(range: 'last24h' | 'last7d' | 'last30d' | 'custom'): { from?: string; to?: string } {
  const now = new Date();
  
  switch (range) {
    case 'last24h':
      return {
        from: new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString(),
        to: now.toISOString()
      };
    case 'last7d':
      return {
        from: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        to: now.toISOString()
      };
    case 'last30d':
      return {
        from: new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        to: now.toISOString()
      };
    default:
      return {};
  }
}