import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { type Run, type RunEvent, STATUS_LABEL, getRun } from "./api";

// 运行状态 → 徽章配色：进行中用主色蓝，等待人工用黄，终态绿/红，排队灰
const STATUS_BADGE: Record<string, string> = {
  queued: "text-muted-foreground",
  running: "border-primary/40 bg-primary/10 text-primary",
  revising: "border-primary/40 bg-primary/10 text-primary",
  waiting_for_human_review: "border-warning/40 bg-warning/10 text-warning",
  completed: "border-success/40 bg-success/10 text-success",
  failed: "border-destructive/40 bg-destructive/10 text-destructive",
  cancelled: "border-destructive/40 bg-destructive/10 text-destructive",
};

// 事件状态 → 时间线左边框颜色（默认用中性边框色）
const EVENT_BAR: Record<string, string> = {
  started: "border-l-primary",
  completed: "border-l-success",
  failed: "border-l-destructive",
};

// 运行时间线：SSE 实时事件为主，1s 轮询作断线兜底；终态后给出下一步入口
export default function RunTimeline({
  runId,
  onViewArticle,
  onNewRun,
}: {
  runId: string;
  onViewArticle: () => void;
  onNewRun: () => void;
}) {
  const [run, setRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    let stopped = false;

    function applyEvent(event: RunEvent) {
      setEvents((prev) => {
        if (prev.some((e) => e.sequence === event.sequence)) return prev;
        return [...prev, event];
      });
    }

    // SSE：kind 即事件通道名（M1 只有 run 层事件）
    const es = new EventSource(`/api/runs/${runId}/events`);
    esRef.current = es;
    es.addEventListener("run", (e) => {
      applyEvent(JSON.parse((e as MessageEvent).data) as RunEvent);
    });

    // 轮询兜底：EventSource 断线时状态仍能推进
    const timer = setInterval(async () => {
      try {
        const latest = await getRun(runId);
        if (!stopped) setRun(latest);
      } catch {
        /* 运行查询失败不致命，等下一轮 */
      }
    }, 1000);
    void getRun(runId).then((r) => !stopped && setRun(r));

    return () => {
      stopped = true;
      clearInterval(timer);
      es.close();
      esRef.current = null;
    };
  }, [runId]);

  const status = run?.status ?? "queued";
  const terminal =
    run?.status === "completed" || run?.status === "failed" || run?.status === "cancelled";

  // 终态后关闭 SSE：在 effect 里做，避免渲染期访问 ref
  useEffect(() => {
    if (terminal) esRef.current?.close();
  }, [terminal]);

  return (
    <Card>
      <CardHeader className="flex-row flex-wrap items-center gap-3 space-y-0">
        <CardTitle>{run?.topic ?? runId}</CardTitle>
        <Badge variant="outline" className={STATUS_BADGE[status] ?? STATUS_BADGE.queued}>
          {STATUS_LABEL[status] ?? status}
        </Badge>
      </CardHeader>
      <CardContent>
        <ol className="my-4 list-none space-y-0 p-0">
          {events.map((event) => (
            <li
              key={event.sequence}
              className={`ml-2 flex items-baseline gap-3 border-l-2 py-2 pl-3 text-sm ${
                EVENT_BAR[event.status] ?? "border-l-border"
              }`}
            >
              <span className="text-muted-foreground tabular-nums">#{event.sequence}</span>
              <span className="font-semibold">{event.name}</span>
              <span className="text-muted-foreground">{event.status}</span>
              {event.payload.total_tokens !== undefined && (
                <span className="ml-auto text-warning">
                  {String(event.payload.total_tokens)} tokens
                </span>
              )}
              {event.payload.review_score !== undefined && (
                <span className="ml-auto text-warning">
                  评分 {String(event.payload.review_score)}
                </span>
              )}
              {typeof event.payload.error === "string" && (
                <span className="ml-auto text-destructive">{event.payload.error}</span>
              )}
            </li>
          ))}
          {events.length === 0 && (
            <li className="ml-2 border-l-2 border-l-border py-2 pl-3 text-sm text-muted-foreground">
              等待事件…
            </li>
          )}
        </ol>

        {run?.error && <p className="text-sm text-destructive">失败原因：{run.error}</p>}

        {terminal && (
          <div className="mt-4 flex gap-3">
            {run?.status === "completed" && (
              <Button onClick={onViewArticle}>查看文章 →</Button>
            )}
            {run?.status !== "completed" && (
              <Button variant="outline" onClick={onNewRun}>
                重新开始
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
