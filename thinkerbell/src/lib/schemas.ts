// Simple TypeScript interfaces - no runtime validation needed for admin interface

export interface Run {
  id: string;
  type: "docs" | "train" | "export";
  status: "queued" | "running" | "success" | "failed" | "cancelled";
  startedAt: string;
  finishedAt: string | null;
  metrics?: {
    ticks?: number;
    entropy?: number;
    scup?: number;
  };
  logsUrl: string | null;
}

export interface Artifact {
  id: string;
  type: "model" | "report" | "zine" | "image" | "log";
  name: string;
  createdAt: string;
  size?: number;
  status: "ready" | "processing" | "failed";
  downloadUrl: string | null;
}

export interface Health {
  status: "ok" | "degraded" | "down";
  dawn: boolean;
  cursor: boolean;
}

export interface RunQuery {
  type?: "docs" | "train" | "export";
  status?: ("queued" | "running" | "success" | "failed" | "cancelled")[];
  from?: string;
  to?: string;
  q?: string;
  sort?: string;
}

export interface ArtifactQuery {
  type?: "model" | "report" | "zine" | "image" | "log";
  status?: ("ready" | "processing" | "failed")[];
  sort?: string;
}

export interface LogLine {
  timestamp: string;
  level: "INFO" | "WARN" | "ERROR" | "DEBUG";
  message: string;
  metadata?: Record<string, any>;
}

export interface LogResponse {
  lines: LogLine[];
  nextOffset: number | null;
  total: number;
}

export interface CreateReportRequest {
  sources: string[];
  title?: string;
  abstract?: string;
  author?: string;
}
