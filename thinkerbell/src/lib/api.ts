import axios from "axios";
import type { 
  Run,
  Artifact,
  Health,
  RunQuery,
  ArtifactQuery,
  LogResponse,
  CreateReportRequest
} from "./schemas";

const api = axios.create({ 
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8787/v1" 
});

export async function listRuns(): Promise<Run[]> {
  const { data } = await api.get("/runs");
  return data;
}

export async function listRunsFiltered(query: RunQuery): Promise<Run[]> {
  const params = new URLSearchParams();
  
  if (query.type) params.append('type', query.type);
  if (query.status?.length) params.append('status', query.status.join(','));
  if (query.from) params.append('from', query.from);
  if (query.to) params.append('to', query.to);
  if (query.q) params.append('q', query.q);
  if (query.sort) params.append('sort', query.sort);
  
  const { data } = await api.get(`/runs?${params.toString()}`);
  return data;
}

export async function getRun(id: string): Promise<Run> {
  const { data } = await api.get(`/runs/${id}`);
  return data;
}

export async function getRunLogs(
  id: string, 
  opts?: { offset?: number; limit?: number; level?: "INFO"|"WARN"|"ERROR"|"DEBUG"; q?: string }
): Promise<LogResponse> {
  const params = new URLSearchParams();
  if (opts?.offset !== undefined) params.append('offset', opts.offset.toString());
  if (opts?.limit !== undefined) params.append('limit', opts.limit.toString());
  if (opts?.level) params.append('level', opts.level);
  if (opts?.q) params.append('q', opts.q);
  
  const { data } = await api.get(`/runs/${id}/logs?${params.toString()}`);
  return data;
}

export async function startRun(type: "docs" | "train" | "export"): Promise<Run> {
  const { data } = await api.post("/runs", { type });
  return data;
}

export async function listArtifacts(): Promise<Artifact[]> {
  const { data } = await api.get("/artifacts");
  return data;
}

export async function listArtifactsFiltered(query: ArtifactQuery): Promise<Artifact[]> {
  const params = new URLSearchParams();
  
  if (query.type) params.append('type', query.type);
  if (query.status?.length) params.append('status', query.status.join(','));
  if (query.sort) params.append('sort', query.sort);
  
  const { data } = await api.get(`/artifacts?${params.toString()}`);
  return data;
}

export async function getArtifact(id: string): Promise<Artifact> {
  const { data } = await api.get(`/artifacts/${id}`);
  return data;
}

export async function generateReport(): Promise<Artifact> {
  const { data } = await api.post("/artifacts/report");
  return data;
}

export async function createReport(request: CreateReportRequest): Promise<Artifact> {
  const { data } = await api.post("/artifacts/report", request);
  return data;
}

export async function getHealth(): Promise<Health> {
  const { data } = await api.get("/health");
  return data;
}
