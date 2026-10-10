import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { createRun } from "./api";

// 创建任务表单：主题必填；来源支持逗号分隔的多个文件路径或 URL
export default function CreateRunForm({ onCreated }: { onCreated: (runId: string) => void }) {
  const [topic, setTopic] = useState("");
  const [sources, setSources] = useState("");
  const [instructions, setInstructions] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!topic.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const sourceList = sources
        .split(/[,，\n]/)
        .map((s) => s.trim())
        .filter(Boolean);
      const run = await createRun(topic.trim(), sourceList, instructions.trim());
      onCreated(run.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>🪶 新建写作任务</CardTitle>
        <CardDescription>来源支持逗号分隔的多个文件路径或 URL</CardDescription>
      </CardHeader>
      <CardContent>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <Label htmlFor="topic">
              技术主题 <span className="text-destructive">*</span>
            </Label>
            <Input
              id="topic"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="例如：大模型 KV-Cache 显存优化技术演进"
              autoFocus
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="sources">参考资料（可选，逗号分隔）</Label>
            <Input
              id="sources"
              value={sources}
              onChange={(e) => setSources(e.target.value)}
              placeholder="references/yoco.pdf, https://arxiv.org/abs/2405.05254"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="instructions">补充要求（可选）</Label>
            <Textarea
              id="instructions"
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              rows={3}
              placeholder="面向有部署经验的工程师，重点讲权衡"
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <Button type="submit" disabled={!topic.trim() || submitting}>
            {submitting ? "创建中..." : "开始写作"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
