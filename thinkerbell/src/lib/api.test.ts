import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from '../mocks/handlers';
import { listRuns, getRun, startRun, listArtifacts, getHealth } from './api';

// Setup MSW server for testing
const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('API Client', () => {
  describe('listRuns', () => {
    it('should fetch and validate runs list', async () => {
      const runs = await listRuns();
      
      expect(Array.isArray(runs)).toBe(true);
      expect(runs.length).toBeGreaterThan(0);
      
      const firstRun = runs[0];
      expect(firstRun).toHaveProperty('id');
      expect(firstRun).toHaveProperty('type');
      expect(firstRun).toHaveProperty('status');
      expect(['docs', 'train', 'export']).toContain(firstRun.type);
      expect(['queued', 'running', 'success', 'failed', 'cancelled']).toContain(firstRun.status);
    });
  });

  describe('getRun', () => {
    it('should fetch a specific run', async () => {
      const runs = await listRuns();
      const runId = runs[0].id;
      
      const run = await getRun(runId);
      
      expect(run.id).toBe(runId);
      expect(run).toHaveProperty('type');
      expect(run).toHaveProperty('status');
    });
  });

  describe('startRun', () => {
    it('should start a new run', async () => {
      const newRun = await startRun('docs');
      
      expect(newRun).toHaveProperty('id');
      expect(newRun.type).toBe('docs');
      expect(newRun.status).toBe('queued');
      expect(newRun.startedAt).toBeDefined();
    });
  });

  describe('listArtifacts', () => {
    it('should fetch and validate artifacts list', async () => {
      const artifacts = await listArtifacts();
      
      expect(Array.isArray(artifacts)).toBe(true);
      expect(artifacts.length).toBeGreaterThan(0);
      
      const firstArtifact = artifacts[0];
      expect(firstArtifact).toHaveProperty('id');
      expect(firstArtifact).toHaveProperty('name');
      expect(firstArtifact).toHaveProperty('type');
      expect(['model', 'report', 'zine', 'image', 'log']).toContain(firstArtifact.type);
    });
  });

  describe('getHealth', () => {
    it('should fetch health status', async () => {
      const health = await getHealth();
      
      expect(health).toHaveProperty('status');
      expect(health).toHaveProperty('dawn');
      expect(health).toHaveProperty('cursor');
      expect(['ok', 'degraded', 'down']).toContain(health.status);
      expect(typeof health.dawn).toBe('boolean');
      expect(typeof health.cursor).toBe('boolean');
    });
  });
});
