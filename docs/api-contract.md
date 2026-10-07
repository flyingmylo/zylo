# zylo API 契约（M0 定稿，M1 实现）

> REST + SSE 的形态在 M0 固定下来，M1 的 FastAPI 按此实现。
> 数据模型字段以 `src/runs.py`、`src/events.py` 为代码事实来源，
> 本文是面向实现者与前端的一页速查。

## 通用约定

- 路径前缀 `/api`；JSON 编码 UTF-8。
- `run_id` 为 12 位十六进制短 ID（`src/runs.py: new_run_id`）。
- 时间戳一律 UTC ISO-8601 带时区。
- 错误响应：`{"error": {"code": "...", "message": "..."}}`。
- 密钥与完整 prompt 不出现在任何响应中（脱敏规则见 architecture.md）。

## REST 端点

| 方法 | 路径 | 里程碑 | 说明 |
|---|---|---|---|
| POST | `/api/runs` | M1 | 创建运行，body：`{topic, sources?, instructions?, config?}`；返回 `201` + Run |
| GET | `/api/runs` | M1 | 运行列表（分页字段 M1 再定，先支持 `?limit=`） |
| GET | `/api/runs/{id}` | M1 | 单个运行详情（含状态与时间戳） |
| GET | `/api/runs/{id}/article` | M1 | 最终文章（Markdown 文本 + Artifact 元信息） |
| GET | `/api/runs/{id}/revisions` | M3 | 稿件版本列表与 Diff 数据 |
| POST | `/api/runs/{id}/review-decisions` | M3 | 人在回路决策，body 为 ReviewDecision（可批量） |
| POST | `/api/runs/{id}/cancel` | M2 | 取消运行；终态运行返回 `409` |
| POST | `/api/runs/{id}/resume` | M2 | 从最新快照续跑 partial 运行 |
| GET | `/healthz` `/readyz` | M5 | 存活与就绪探针 |

创建示例：

```json
POST /api/runs
{
  "topic": "大模型 KV-Cache 显存优化技术演进",
  "sources": ["references/yoco.pdf", "https://arxiv.org/abs/2405.05254"],
  "instructions": "面向有推理部署经验的读者",
  "config": {"model": "deepseek-chat", "max_revisions": 2}
}
```

## SSE 事件流

`GET /api/runs/{id}/events`（M1 内存订阅，M3 历史回放）

- `id:` 字段 = 事件 `sequence`（run 内递增）。
- 客户端断线重连时携带 `Last-Event-ID`，服务端从该序号之后重放，
  序号溢出或事件已清理时返回快照重建提示。
- `event:` 字段 = `kind`（`run`/`stage`/`agent`/`llm`/`tool`），
  前端按 kind 分通道渲染时间线。
- `data:` 为 `RunEvent` 完整 JSON（`src/events.py`）。

```text
id: 1
event: run
data: {"sequence":1,"run_id":"a1b2c3d4e5f6","kind":"run","name":"writing","status":"started","span_id":"r0",...}

id: 2
event: stage
data: {"sequence":2,"run_id":"a1b2c3d4e5f6","kind":"stage","name":"researching","status":"started","span_id":"s1","parent_id":"r0",...}
```

## 状态码与状态机联动

- 创建成功即 `queued`；JobRunner 拉起后 `running`。
- `cancel` 对 `queued/running/waiting_for_human_review/revising/partial`
  生效，对终态返回 `409`（迁移合法性以 `src/runs.py` 状态机为准）。
- `resume` 仅对 `partial` 生效，其余返回 `409`。
