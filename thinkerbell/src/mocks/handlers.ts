import { http, HttpResponse } from 'msw';
import type { Run, Artifact, Health, LogLine, LogResponse } from '../lib/schemas';
import { mockTransformer } from './data-transformer';

// Generate rich mock datasets using our transformer
const mockRuns: Run[] = mockTransformer.generateRunDataset(50);
const mockArtifacts: Artifact[] = mockTransformer.generateArtifactDataset(30);

// Generate mock logs for a run with creative, realistic content
const generateMockLogs = (runId: string, count: number = 100): LogLine[] => {
  const logs: LogLine[] = [];
  const levels: LogLine['level'][] = ['INFO', 'WARN', 'ERROR', 'DEBUG'];
  
  // Creative, tech-themed log messages that match our energetic brand
  const messageTemplates = {
    INFO: [
      '🚀 Initializing Thinkerbell pipeline engine',
      '⚡ Neural network layers configured successfully',
      '🧠 Loading quantum training parameters',
      '✨ Batch processing at lightning speed',
      '🔥 Model convergence detected - accuracy: {accuracy}%',
      '💫 Checkpoint saved to cosmic storage',
      '🌟 Memory optimization complete: {memory}GB freed',
      '⭐ Data preprocessing finished in {time}ms',
      '🎯 Target metrics achieved successfully',
      '🎨 Artifact generation pipeline activated',
      '🏆 Training epoch {epoch}/100 completed',
      '🔋 System resources optimized',
      '📊 Performance metrics logged',
      '🌈 Color space transformation applied',
      '🎪 Creative synthesis mode enabled'
    ],
    DEBUG: [
      '🔍 Debugging neural pathway connections',
      '🛠️ Internal state inspection: {state}',
      '📡 Network topology analysis complete',
      '🔬 Microscopic parameter adjustment',
      '🧪 Experimental feature flag: {flag}',
      '📐 Geometric transformation applied',
      '🎲 Random seed initialized: {seed}',
      '🔎 Deep inspection of layer {layer}',
      '📈 Gradient flow analysis',
      '🎛️ Hyperparameter tuning session'
    ],
    WARN: [
      '⚠️ Memory usage approaching threshold',
      '🔶 Model drift detected in layer {layer}',
      '⏰ Training time exceeding estimates',
      '🌡️ GPU temperature rising: {temp}°C',
      '📉 Validation accuracy plateau detected',
      '🔄 Retry attempt {attempt}/3',
      '📦 Package version mismatch detected',
      '🎚️ Learning rate adjustment recommended',
      '🔋 Battery level low on remote workers',
      '📺 Display buffer overflow warning'
    ],
    ERROR: [
      '💥 CUDA out of memory error',
      '🚫 Model architecture validation failed',
      '❌ Data corruption detected in batch {batch}',
      '💔 Connection lost to training cluster',
      '🛑 Emergency stop triggered by user',
      '🔥 Critical failure in optimization step',
      '⛔ Permission denied accessing {resource}',
      '💀 Fatal error in neural network initialization',
      '🚨 System overload - emergency shutdown',
      '❗ Catastrophic model divergence detected'
    ]
  };

  for (let i = 0; i < count; i++) {
    const timestamp = new Date(Date.now() - (count - i) * 1000).toISOString();
    const level = levels[Math.floor(Math.random() * levels.length)];
    const templates = messageTemplates[level];
    let message = templates[Math.floor(Math.random() * templates.length)];
    
    // Replace placeholders with dynamic values
    message = message
      .replace('{accuracy}', (Math.random() * 20 + 80).toFixed(1))
      .replace('{memory}', (Math.random() * 3 + 1).toFixed(1))
      .replace('{time}', Math.floor(Math.random() * 1000 + 100).toString())
      .replace('{epoch}', Math.floor(Math.random() * 100 + 1).toString())
      .replace('{state}', Math.random().toString(36).substr(2, 8))
      .replace('{flag}', ['EXPERIMENTAL_MODE', 'TURBO_BOOST', 'QUANTUM_SYNC'][Math.floor(Math.random() * 3)])
      .replace('{seed}', Math.floor(Math.random() * 9999).toString())
      .replace('{layer}', Math.floor(Math.random() * 12 + 1).toString())
      .replace('{temp}', Math.floor(Math.random() * 30 + 70).toString())
      .replace('{attempt}', Math.floor(Math.random() * 3 + 1).toString())
      .replace('{batch}', Math.floor(Math.random() * 100 + 1).toString())
      .replace('{resource}', ['/data/models', '/cache/temp', '/logs/debug'][Math.floor(Math.random() * 3)]);
    
    const metadata = level === 'ERROR' 
      ? { 
          error_code: `E${Math.floor(Math.random() * 999 + 1).toString().padStart(3, '0')}`,
          stack_trace: 'thinkerbell.core.neural_engine.process_batch:142'
        }
      : level === 'WARN'
      ? { warning_type: 'performance', threshold_exceeded: true }
      : undefined;
    
    logs.push({
      timestamp,
      level,
      message,
      metadata
    });
  }
  
  return logs;
};

export const handlers = [
  // Enhanced GET /runs with filtering and sorting
  http.get('/v1/runs', ({ request }) => {
    const url = new URL(request.url);
    let filteredRuns = [...mockRuns];
    
    // Apply filters
    const type = url.searchParams.get('type');
    if (type) {
      filteredRuns = filteredRuns.filter(run => run.type === type);
    }
    
    const status = url.searchParams.get('status');
    if (status) {
      const statusList = status.split(',');
      filteredRuns = filteredRuns.filter(run => statusList.includes(run.status));
    }
    
    const from = url.searchParams.get('from');
    if (from) {
      filteredRuns = filteredRuns.filter(run => new Date(run.startedAt) >= new Date(from));
    }
    
    const to = url.searchParams.get('to');
    if (to) {
      filteredRuns = filteredRuns.filter(run => new Date(run.startedAt) <= new Date(to));
    }
    
    const q = url.searchParams.get('q');
    if (q) {
      const query = q.toLowerCase();
      filteredRuns = filteredRuns.filter(run => 
        run.id.toLowerCase().includes(query) ||
        run.type.toLowerCase().includes(query) ||
        run.status.toLowerCase().includes(query)
      );
    }
    
    // Apply sorting
    const sort = url.searchParams.get('sort');
    if (sort) {
      const [field, direction] = sort.split(':');
      filteredRuns.sort((a, b) => {
        let aVal, bVal;
        
        switch (field) {
          case 'startedAt':
          case 'finishedAt':
            aVal = new Date(a[field] || 0).getTime();
            bVal = new Date(b[field] || 0).getTime();
            break;
          case 'entropy':
          case 'scup':
            aVal = a.metrics?.[field] || 0;
            bVal = b.metrics?.[field] || 0;
            break;
          default:
            aVal = a[field as keyof Run] || '';
            bVal = b[field as keyof Run] || '';
        }
        
        if (direction === 'asc') {
          return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        } else {
          return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
        }
      });
    }
    
    console.log(`🔍 Filtered runs: ${filteredRuns.length}/${mockRuns.length} (filters: ${Object.fromEntries(url.searchParams)})`);
    return HttpResponse.json(filteredRuns);
  }),

  // GET /runs/:id
  http.get('/v1/runs/:id', ({ params }) => {
    const run = mockRuns.find(r => r.id === params.id);
    if (!run) {
      return new HttpResponse(null, { status: 404 });
    }
    return HttpResponse.json(run);
  }),

  // Enhanced GET /runs/:id/logs with filtering and pagination
  http.get('/v1/runs/:id/logs', ({ params, request }) => {
    const url = new URL(request.url);
    const runId = params.id as string;
    
    const offset = parseInt(url.searchParams.get('offset') || '0');
    const limit = parseInt(url.searchParams.get('limit') || '100');
    const level = url.searchParams.get('level') as LogLine['level'] | null;
    const q = url.searchParams.get('q');
    
    let allLogs = generateMockLogs(runId, 500); // Generate more logs for realistic pagination
    
    // Apply level filter
    if (level) {
      allLogs = allLogs.filter(log => log.level === level);
    }
    
    // Apply text search
    if (q) {
      const query = q.toLowerCase();
      allLogs = allLogs.filter(log => 
        log.message.toLowerCase().includes(query) ||
        log.level.toLowerCase().includes(query)
      );
    }
    
    // Apply pagination
    const total = allLogs.length;
    const logs = allLogs.slice(offset, offset + limit);
    const nextOffset = offset + limit < total ? offset + limit : null;
    
    const response: LogResponse = {
      lines: logs,
      nextOffset,
      total
    };
    
    console.log(`📜 Serving logs for ${runId}: ${logs.length}/${total} (offset: ${offset}, filters: level=${level}, q=${q})`);
    return HttpResponse.json(response);
  }),

  // POST /runs
  http.post('/v1/runs', async ({ request }) => {
    const body = await request.json() as { type: Run['type'] };
    const newRun = mockTransformer.generateRun({ 
      type: body.type, 
      status: 'queued',
      startedAt: new Date().toISOString()
    });
    mockRuns.unshift(newRun);
    
    console.log(`🚀 Started new ${body.type} run: ${newRun.id}`);
    return HttpResponse.json(newRun, { status: 201 });
  }),

  // Enhanced GET /artifacts with filtering and sorting
  http.get('/v1/artifacts', ({ request }) => {
    const url = new URL(request.url);
    let filteredArtifacts = [...mockArtifacts];
    
    // Apply filters
    const type = url.searchParams.get('type');
    if (type) {
      filteredArtifacts = filteredArtifacts.filter(artifact => artifact.type === type);
    }
    
    const status = url.searchParams.get('status');
    if (status) {
      const statusList = status.split(',');
      filteredArtifacts = filteredArtifacts.filter(artifact => statusList.includes(artifact.status));
    }
    
    // Apply sorting
    const sort = url.searchParams.get('sort');
    if (sort) {
      const [field, direction] = sort.split(':');
      filteredArtifacts.sort((a, b) => {
        let aVal, bVal;
        
        switch (field) {
          case 'createdAt':
            aVal = new Date(a.createdAt).getTime();
            bVal = new Date(b.createdAt).getTime();
            break;
          case 'size':
            aVal = a.size || 0;
            bVal = b.size || 0;
            break;
          default:
            aVal = a[field as keyof Artifact] || '';
            bVal = b[field as keyof Artifact] || '';
        }
        
        if (direction === 'asc') {
          return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        } else {
          return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
        }
      });
    }
    
    console.log(`🎨 Filtered artifacts: ${filteredArtifacts.length}/${mockArtifacts.length}`);
    return HttpResponse.json(filteredArtifacts);
  }),

  // GET /artifacts/:id
  http.get('/v1/artifacts/:id', ({ params }) => {
    const artifact = mockArtifacts.find(a => a.id === params.id);
    if (!artifact) {
      return new HttpResponse(null, { status: 404 });
    }
    return HttpResponse.json(artifact);
  }),

  // POST /artifacts/report - Create whitepaper from selected artifacts
  http.post('/v1/artifacts/report', async ({ request }) => {
    const body = await request.json() as { 
      sources: string[]; 
      title?: string;
      abstract?: string;
      author?: string;
    };
    
    const reportArtifact = mockTransformer.generateArtifact({
      type: 'report',
      name: `${body.title || mockTransformer.generateCreativeName()}.html`,
      status: 'ready'
    });
    
    mockArtifacts.unshift(reportArtifact);
    
    console.log(`📄 Generated whitepaper: ${reportArtifact.name} from ${body.sources.length} sources`);
    return HttpResponse.json(reportArtifact, { status: 201 });
  }),

  // GET /health
  http.get('/v1/health', () => {
    const health = mockTransformer.generateHealth();
    console.log(`💚 Health check: ${health.status} (dawn: ${health.dawn}, cursor: ${health.cursor})`);
    return HttpResponse.json(health);
  }),
];