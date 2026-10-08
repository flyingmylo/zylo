// API 客户端：类型与 fetch 封装，字段与 src/runs.py、src/events.py 对齐

export type RunStatus =
  | "queued"
  | "running"
  | "waiting_for_human_review"
  | "revising"
  | "completed"
  | "failed"
  | "cancelled"
  | "partial";

export interface Run {
  id: string;
  topic: string;
  status: RunStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
}

export interface RunEvent {
  sequence: number;
  run_id: string;
  kind: "run" | "stage" | "agent" | "llm" | "tool";
  name: string;
  status: "started" | "completed" | "failed";
  payload: Record<string, unknown>;
  created_at: string;
}

export interface Article {
  title: string;
  markdown: string;
  review_score: number;
  revision_count: number;
}

export async function createRun(topic: string, sources: string[], instructions: string): Promise<Run> {
  const resp = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, sources, instructions }),
  });
  if (!resp.ok) {
    throw new Error(`创建失败 (${resp.status})`);
  }
  return resp.json();
}

export async function getRun(runId: string): Promise<Run> {
  const resp = await fetch(`/api/runs/${runId}`);
  if (!resp.ok) throw new Error(`查询失败 (${resp.status})`);
  return resp.json();
}

export async function getArticle(runId: string): Promise<Article> {
  const resp = await fetch(`/api/runs/${runId}/article`);
  if (!resp.ok) throw new Error(`文章尚未就绪 (${resp.status})`);
  return resp.json();
}

export const STATUS_LABEL: Record<RunStatus, string> = {
  queued: "排队中",
  running: "运行中",
  waiting_for_human_review: "等待人工审稿",
  revising: "修订中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  partial: "中断待恢复",
};
