import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { type Article, getArticle } from "./api";

// 文章结果页：元信息 + Markdown 渲染
export default function ArticleView({ runId, onNewRun }: { runId: string; onNewRun: () => void }) {
  const [article, setArticle] = useState<Article | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getArticle(runId).then(setArticle).catch((err) => setError(String(err)));
  }, [runId]);

  if (error) return <div className="card"><p className="error">{error}</p></div>;
  if (!article) return <div className="card"><p>加载文章中…</p></div>;

  return (
    <div className="card">
      <header className="timeline-header">
        <h2>{article.title}</h2>
      </header>
      <p className="meta">
        质检评分 <strong>{article.review_score.toFixed(1)}</strong> · 修订 {article.revision_count} 轮
      </p>
      <article className="markdown">
        <ReactMarkdown>{article.markdown}</ReactMarkdown>
      </article>
      <div className="actions">
        <button onClick={onNewRun}>再写一篇</button>
      </div>
    </div>
  );
}
