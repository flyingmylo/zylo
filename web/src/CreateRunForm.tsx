import { useState } from "react";
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
    <form className="card" onSubmit={handleSubmit}>
      <h2>🪶 新建写作任务</h2>
      <label>
        技术主题 <span className="required">*</span>
        <input
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="例如：大模型 KV-Cache 显存优化技术演进"
          autoFocus
        />
      </label>
      <label>
        参考资料（可选，逗号分隔）
        <input
          value={sources}
          onChange={(e) => setSources(e.target.value)}
          placeholder="references/yoco.pdf, https://arxiv.org/abs/2405.05254"
        />
      </label>
      <label>
        补充要求（可选）
        <textarea
          value={instructions}
          onChange={(e) => setInstructions(e.target.value)}
          rows={3}
          placeholder="面向有部署经验的工程师，重点讲权衡"
        />
      </label>
      {error && <p className="error">{error}</p>}
      <button type="submit" disabled={!topic.trim() || submitting}>
        {submitting ? "创建中..." : "开始写作"}
      </button>
    </form>
  );
}
