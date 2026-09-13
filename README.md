# zylo 🪶

> 基于纯手写（Zero-Framework）多智能体架构的**中文深度技术博客 / 长文写作 Agent**。

---

## 🌟 核心特性

- **纯手写轻量架构**：不依赖 LangGraph/CrewAI 等重量级外部框架，纯原生实现 Tool Calling 循环与中央 Orchestrator 状态流转。
- **本地 MPS 硬件加速**：针对 Apple Silicon (M1/M2/M3) 原生适配 `BAAI/bge-m3` 嵌入模型与可选 `bge-reranker-v2-m3` 重排模型，中英跨语言检索能力一流。
- **双语 Query 扩展 + 相对 Top-K**：破除跨语言检索相似度打折陷阱，使用同语言精确对齐，保留英文原著高价值论据。
- **专业术语双语对照规范**：英文文献概念首次出现强制标注文档：`中文译名（English Name）`，如 *倒数排名融合（Reciprocal Rank Fusion, RRF）*。
- **反思自愈审稿回路**：集成严苛的 Reviewer Agent，针对事实性、逻辑性与术语规范度多维度打分，最多 2 轮定向反思重写。
- **BYOK 与模型无关**：支持 OpenAI, DeepSeek, 通义千问, 智谱 GLM 以及本地 Ollama 等所有 OpenAI 标准接口。

---

## 🛠️ 快速开始

### 1. 配置环境变量

复制 `.env.example` 并填入你的 API Key：

```bash
cp .env.example .env
```

### 2. 运行 CLI 写作

激活虚拟环境后，直接使用 `zylo` 命令行：

```bash
# 1. 极简一键写作（支持直接传参，第一个参数为主题，后续自动嗅探本地文件或 arXiv 论文直链）
zylo "大模型 KV-Cache 显存优化技术演进"
zylo "YOCO: 你只需缓存一次的大模型架构" references/yoco.pdf
zylo "DeepSeek-V3 核心架构解析" https://arxiv.org/abs/2412.19437

# 2. 交互式向导（直接敲 zylo 回车，两步极简提示输入）
zylo

# 3. 经典参数模式（可选高级控制）
zylo write -t "FlashAttention 原理剖析" -f "references/fa.pdf" --rerank

# 4. 系统环境诊断与配置查看
zylo check
zylo config
```

> **💡 智能 Reranker 激活机制**：默认策略为 `ENABLE_RERANK=auto`。当检测到输入了本地文档或网页/论文 URL 时，系统会自动激活本地 `bge-reranker-v2-m3` 深度精排；纯纯网络检索时默认保持轻量极速。

生成的 Markdown 文件将自动归档至 `output/` 目录。

