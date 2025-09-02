import type { Run, Artifact, Health } from '../lib/schemas';

// Style Profiles - defining data patterns that match our design language
export const STYLE_PROFILES = {
  // Bold, energetic pipeline types
  PIPELINE_TYPES: {
    docs: { 
      emoji: '📚', 
      accent: 'tb-magenta', 
      label: 'DOCS', 
      description: 'Documentation generation with neural networks' 
    },
    train: { 
      emoji: '🧠', 
      accent: 'tb-green-500', 
      label: 'TRAIN', 
      description: 'Model training with advanced algorithms' 
    },
    export: { 
      emoji: '🚀', 
      accent: 'tb-pink-500', 
      label: 'EXPORT', 
      description: 'Production deployment and optimization' 
    }
  },

  // Status with vibrant, clear visual hierarchy
  RUN_STATUS: {
    queued: { 
      emoji: '⏳', 
      color: 'warning', 
      pulse: false, 
      label: 'QUEUED',
      message: 'Waiting in the pipeline...' 
    },
    running: { 
      emoji: '⚡', 
      color: 'tb-green-500', 
      pulse: true, 
      label: 'RUNNING',
      message: 'Processing at lightning speed!' 
    },
    success: { 
      emoji: '✨', 
      color: 'success', 
      pulse: false, 
      label: 'SUCCESS',
      message: 'Completed with excellence!' 
    },
    failed: { 
      emoji: '💥', 
      color: 'error', 
      pulse: false, 
      label: 'FAILED',
      message: 'Encountered an obstacle' 
    },
    cancelled: { 
      emoji: '🛑', 
      color: 'gray', 
      pulse: false, 
      label: 'CANCELLED',
      message: 'Stopped by user request' 
    }
  },

  // Artifact types with creative, bold styling
  ARTIFACT_TYPES: {
    model: { 
      emoji: '🧬', 
      accent: 'tb-magenta', 
      label: 'MODEL',
      description: 'AI/ML model artifacts',
      extensions: ['.pkl', '.onnx', '.pt', '.h5']
    },
    report: { 
      emoji: '📊', 
      accent: 'tb-green-500', 
      label: 'REPORT',
      description: 'Analysis and insights',
      extensions: ['.html', '.pdf', '.md']
    },
    zine: { 
      emoji: '📰', 
      accent: 'tb-pink-500', 
      label: 'ZINE',
      description: 'Creative publications',
      extensions: ['.html', '.pdf']
    },
    image: { 
      emoji: '🎨', 
      accent: 'tb-green-600', 
      label: 'IMAGE',
      description: 'Visual content',
      extensions: ['.png', '.jpg', '.svg', '.webp']
    },
    log: { 
      emoji: '📝', 
      accent: 'gray-600', 
      label: 'LOG',
      description: 'Execution logs',
      extensions: ['.log', '.txt']
    }
  }
} as const;

// Creative name generators for bold, energetic feel
const CREATIVE_ADJECTIVES = [
  'Blazing', 'Cosmic', 'Electric', 'Quantum', 'Stellar', 'Turbo', 'Ultra',
  'Mega', 'Super', 'Hyper', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Neural',
  'Smart', 'Deep', 'Fast', 'Bright', 'Sharp', 'Bold', 'Wild', 'Pure'
];

const TECH_NOUNS = [
  'Engine', 'Core', 'Matrix', 'Network', 'System', 'Framework', 'Pipeline',
  'Model', 'Agent', 'Bot', 'Scanner', 'Analyzer', 'Parser', 'Generator',
  'Transformer', 'Optimizer', 'Compiler', 'Renderer', 'Processor', 'Handler'
];

const PROJECT_THEMES = [
  'Phoenix', 'Thunder', 'Lightning', 'Storm', 'Nova', 'Meteor', 'Comet',
  'Fusion', 'Nexus', 'Vertex', 'Prism', 'Spark', 'Flash', 'Bolt', 'Pulse',
  'Wave', 'Flux', 'Zen', 'Arc', 'Edge', 'Peak', 'Summit', 'Flow'
];

// Mock data generators with style-conscious patterns
export class MockDataTransformer {
  private runCounter = 1;
  private artifactCounter = 1;

  generateCreativeName(): string {
    const adj = CREATIVE_ADJECTIVES[Math.floor(Math.random() * CREATIVE_ADJECTIVES.length)];
    const noun = TECH_NOUNS[Math.floor(Math.random() * TECH_NOUNS.length)];
    const theme = PROJECT_THEMES[Math.floor(Math.random() * PROJECT_THEMES.length)];
    return `${adj} ${theme} ${noun}`;
  }

  generateRun(overrides: Partial<Run> = {}): Run {
    const id = `run-${this.runCounter.toString().padStart(4, '0')}`;
    this.runCounter++;

    const types: Array<Run['type']> = ['docs', 'train', 'export'];
    const statuses: Array<Run['status']> = ['queued', 'running', 'success', 'failed', 'cancelled'];
    
    const type = overrides.type || types[Math.floor(Math.random() * types.length)];
    const status = overrides.status || statuses[Math.floor(Math.random() * statuses.length)];
    
    // Generate realistic timestamps
    const startedAt = new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString();
    const isCompleted = ['success', 'failed', 'cancelled'].includes(status);
    const finishedAt = isCompleted 
      ? new Date(new Date(startedAt).getTime() + Math.random() * 2 * 60 * 60 * 1000).toISOString()
      : null;

    // Generate metrics that match our visual style
    const metrics = type === 'train' ? {
      ticks: Math.floor(Math.random() * 10000) + 1000,
      entropy: Math.random() * 2.5 + 0.1,
      scup: Math.random() * 100 + 50
    } : undefined;

    return {
      id,
      type,
      status,
      startedAt,
      finishedAt,
      metrics,
      logsUrl: `http://localhost:8787/v1/runs/${id}/logs`,
      ...overrides
    };
  }

  generateArtifact(overrides: Partial<Artifact> = {}): Artifact {
    const id = `artifact-${this.artifactCounter.toString().padStart(4, '0')}`;
    this.artifactCounter++;

    const types: Array<Artifact['type']> = ['model', 'report', 'zine', 'image', 'log'];
    const statuses: Array<Artifact['status']> = ['ready', 'processing', 'failed'];
    
    const type = overrides.type || types[Math.floor(Math.random() * types.length)];
    const status = overrides.status || statuses[Math.floor(Math.random() * statuses.length)];
    const profile = STYLE_PROFILES.ARTIFACT_TYPES[type];
    
    // Generate creative names based on type
    const baseName = this.generateCreativeName();
    const extension = profile.extensions[Math.floor(Math.random() * profile.extensions.length)];
    const name = `${baseName}${extension}`;

    // Generate realistic file sizes based on type
    const sizeRanges = {
      model: [50 * 1024 * 1024, 500 * 1024 * 1024], // 50MB - 500MB
      report: [1 * 1024 * 1024, 10 * 1024 * 1024],   // 1MB - 10MB
      zine: [2 * 1024 * 1024, 15 * 1024 * 1024],     // 2MB - 15MB
      image: [100 * 1024, 5 * 1024 * 1024],          // 100KB - 5MB
      log: [10 * 1024, 1 * 1024 * 1024]              // 10KB - 1MB
    };
    
    const [minSize, maxSize] = sizeRanges[type];
    const size = Math.floor(Math.random() * (maxSize - minSize)) + minSize;

    const createdAt = new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString();
    const downloadUrl = status === 'ready' 
      ? `http://localhost:8787/v1/artifacts/${id}/download`
      : null;

    return {
      id,
      type,
      name,
      createdAt,
      size,
      status,
      downloadUrl,
      ...overrides
    };
  }

  generateHealth(): Health {
    // Occasionally show degraded status for realism
    const isDegraded = Math.random() < 0.1;
    const isDown = Math.random() < 0.02;
    
    return {
      status: isDown ? 'down' : (isDegraded ? 'degraded' : 'ok'),
      dawn: !isDown && Math.random() > 0.05,
      cursor: !isDown && Math.random() > 0.05
    };
  }

  // Generate datasets with realistic distributions
  generateRunDataset(count: number = 50): Run[] {
    const runs: Run[] = [];
    
    // Generate runs with realistic status distribution
    const statusDistribution = {
      success: 0.6,   // 60% success
      running: 0.15,  // 15% running
      failed: 0.15,   // 15% failed
      queued: 0.08,   // 8% queued
      cancelled: 0.02 // 2% cancelled
    };

    for (let i = 0; i < count; i++) {
      const rand = Math.random();
      let status: Run['status'] = 'success';
      let cumulative = 0;
      
      for (const [s, prob] of Object.entries(statusDistribution)) {
        cumulative += prob;
        if (rand <= cumulative) {
          status = s as Run['status'];
          break;
        }
      }
      
      runs.push(this.generateRun({ status }));
    }

    // Sort by most recent first
    return runs.sort((a, b) => new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime());
  }

  generateArtifactDataset(count: number = 30): Artifact[] {
    const artifacts: Artifact[] = [];
    
    // Generate artifacts with realistic type distribution
    const typeDistribution = {
      model: 0.3,   // 30% models
      report: 0.25, // 25% reports
      log: 0.2,     // 20% logs
      image: 0.15,  // 15% images
      zine: 0.1     // 10% zines
    };

    for (let i = 0; i < count; i++) {
      const rand = Math.random();
      let type: Artifact['type'] = 'model';
      let cumulative = 0;
      
      for (const [t, prob] of Object.entries(typeDistribution)) {
        cumulative += prob;
        if (rand <= cumulative) {
          type = t as Artifact['type'];
          break;
        }
      }
      
      artifacts.push(this.generateArtifact({ type }));
    }

    // Sort by most recent first
    return artifacts.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
  }

  // Get style profile for UI components
  getStyleProfile(type: string, category: 'pipeline' | 'run_status' | 'artifact') {
    switch (category) {
      case 'pipeline':
        return STYLE_PROFILES.PIPELINE_TYPES[type as keyof typeof STYLE_PROFILES.PIPELINE_TYPES];
      case 'run_status':
        return STYLE_PROFILES.RUN_STATUS[type as keyof typeof STYLE_PROFILES.RUN_STATUS];
      case 'artifact':
        return STYLE_PROFILES.ARTIFACT_TYPES[type as keyof typeof STYLE_PROFILES.ARTIFACT_TYPES];
      default:
        return null;
    }
  }
}

// Export singleton instance
export const mockTransformer = new MockDataTransformer();
