import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { type Article, getArticle } from "./api";

// 文章结果页：元信息 + Markdown 渲染（排版样式见 index.css 的 .markdown 全局规则）
export default function ArticleView({ runId, onNewRun }: { runId: string; onNewRun: () => void }) {
  const [article, setArticle] = useState<Article | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getArticle(runId).then(setArticle).catch((err) => setError(String(err)));
  }, [runId]);

  if (error)
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-sm text-destructive">{error}</p>
        </CardContent>
      </Card>
    );
  if (!article)
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-sm text-muted-foreground">加载文章中…</p>
        </CardContent>
      </Card>
    );

  return (
    <Card>
      <CardHeader>
        <CardTitle>{article.title}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="mb-4 text-sm text-muted-foreground">
          质检评分 <strong className="text-warning">{article.review_score.toFixed(1)}</strong> ·
          修订 {article.revision_count} 轮
        </p>
        <article className="markdown">
          <ReactMarkdown>{article.markdown}</ReactMarkdown>
        </article>
        <div className="mt-4 flex gap-3">
          <Button variant="outline" onClick={onNewRun}>
            再写一篇
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
