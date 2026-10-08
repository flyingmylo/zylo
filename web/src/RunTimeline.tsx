import { useEffect, useRef, useState } from "react";
import { type Run, type RunEvent, STATUS_LABEL, getRun } from "./api";

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

  const terminal = run?.status === "completed" || run?.status === "failed" || run?.status === "cancelled";
  if (terminal && esRef.current) esRef.current.close();

  return (
    <div className="card">
      <header className="timeline-header">
        <h2>{run?.topic ?? runId}</h2>
        <span className={`badge badge-${run?.status ?? "queued"}`}>
          {STATUS_LABEL[run?.status ?? "queued"]}
        </span>
      </header>

      <ol className="timeline">
        {events.map((event) => (
          <li key={event.sequence} className={`event event-${event.status}`}>
            <span className="event-seq">#{event.sequence}</span>
            <span className="event-name">{event.name}</span>
            <span className="event-status">{event.status}</span>
            {event.payload.total_tokens !== undefined && (
              <span className="event-meta">{String(event.payload.total_tokens)} tokens</span>
            )}
            {event.payload.review_score !== undefined && (
              <span className="event-meta">评分 {String(event.payload.review_score)}</span>
            )}
            {typeof event.payload.error === "string" && (
              <span className="event-meta error">{event.payload.error}</span>
            )}
          </li>
        ))}
        {events.length === 0 && <li className="event muted">等待事件…</li>}
      </ol>

      {run?.error && <p className="error">失败原因：{run.error}</p>}

      <div className="actions">
        {run?.status === "completed" && <button onClick={onViewArticle}>查看文章 →</button>}
        {terminal && run?.status !== "completed" && <button onClick={onNewRun}>重新开始</button>}
      </div>
    </div>
  );
}
