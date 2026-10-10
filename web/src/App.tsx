import { useState } from "react";
import ArticleView from "./ArticleView";
import CreateRunForm from "./CreateRunForm";
import RunTimeline from "./RunTimeline";

// M1 的极简视图状态机：表单 → 运行时间线 → 文章结果
type View =
  | { name: "form" }
  | { name: "run"; runId: string }
  | { name: "article"; runId: string };

export default function App() {
  const [view, setView] = useState<View>({ name: "form" });

  return (
    <main className="mx-auto max-w-215 px-5 pt-8 pb-20">
      <header className="mb-7">
        <h1 className="text-2xl font-bold">zylo 工作台</h1>
        <p className="mt-1 text-sm text-muted-foreground">多智能体深度技术写作 · 运行可观测</p>
      </header>

      {view.name === "form" && (
        <CreateRunForm onCreated={(runId) => setView({ name: "run", runId })} />
      )}

      {view.name === "run" && (
        <RunTimeline
          runId={view.runId}
          onViewArticle={() => setView({ name: "article", runId: view.runId })}
          onNewRun={() => setView({ name: "form" })}
        />
      )}

      {view.name === "article" && (
        <ArticleView runId={view.runId} onNewRun={() => setView({ name: "form" })} />
      )}
    </main>
  );
}
