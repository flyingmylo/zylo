import os
import re
from collections.abc import Callable
from datetime import UTC, datetime

from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.writer import WriterAgent
from src.embeddings.base import EmbeddingProvider
from src.embeddings.reranker_base import RerankerProvider
from src.llm.base import LLMProvider
from src.state import Stage, WritingState
from src.tools.knowledge_base import KnowledgeBase
from src.tools.search import SearchTool


class WritingOrchestrator:
    """
    中央写作编排器：
    管理状态机流转、多 Agent 协作调度、最大 2 轮审稿反思回路与终稿产出
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedding_provider: EmbeddingProvider | None = None,
        reranker_provider: RerankerProvider | None = None,
        tavily_api_key: str | None = None,
        progress_callback: Callable[[str, WritingState], None] | None = None,
    ):
        self.llm = llm
        self.embedding = embedding_provider
        self.reranker = reranker_provider
        self.tavily_api_key = tavily_api_key
        self.on_progress = progress_callback or (lambda msg, state: None)

    async def execute(
        self,
        topic: str,
        local_files: list[str] | None = None,
        extra_instructions: str = "",
        output_dir: str = "output",
    ) -> WritingState:
        state = WritingState(
            topic=topic,
            local_files=local_files or [],
            extra_instructions=extra_instructions,
        )

        # 1. 初始化专属文章级别的 KnowledgeBase
        kb = KnowledgeBase(
            embedding_provider=self.embedding,
            reranker_provider=self.reranker,
        )

        # 初始化 4 个 Agent
        search_tool = (
            SearchTool(api_key=self.tavily_api_key) if self.tavily_api_key else None
        )
        researcher = ResearcherAgent(
            self.llm, knowledge_base=kb, search_tool=search_tool
        )
        planner = PlannerAgent(self.llm)
        writer = WriterAgent(self.llm, knowledge_base=kb)
        reviewer = ReviewerAgent(self.llm)

        # ====== 阶段 1: 调研与知识库构建 ======
        self.on_progress(
            "🔍 启动 Researcher 进行文献解析、网络检索与知识库构建...", state
        )
        state = await researcher.run(state)
        self.on_progress(f"✅ 调研完成，入库向量片段 {kb.count()} 条。", state)

        # ====== 阶段 2: 大纲规划 ======
        self.on_progress("📐 启动 Planner 制定深度技术架构大纲与双语检索词...", state)
        state = await planner.run(state)
        self.on_progress(
            f"✅ 大纲制定完毕，共 {len(state.sections)} 个深度章节。", state
        )

        # ====== 阶段 3 & 4: 写作与审稿反思回路 ======
        while state.revision_count <= state.max_revisions:
            round_desc = (
                "初稿撰写"
                if state.revision_count == 0
                else f"第 {state.revision_count} 轮定向修改"
            )
            self.on_progress(f"✍️ 启动 Writer 执行小节向量检索与{round_desc}...", state)
            state = await writer.run(state)

            self.on_progress(
                "🧐 启动 Reviewer 审查技术事实、专业术语与行文逻辑...", state
            )
            state = await reviewer.run(state)

            self.on_progress(
                f"📊 审稿得分: {state.review_score:.1f}/100 | 是否通过: {state.review_passed}",
                state,
            )

            if state.review_passed:
                self.on_progress("🎉 审稿通过，符合发布质量！", state)
                break

            state.revision_count += 1
            if state.revision_count > state.max_revisions:
                self.on_progress(
                    f"⚠️ 已达到最大修改轮次 ({state.max_revisions} 轮)，强制定稿。",
                    state,
                )
                break

            self.on_progress(
                f"🔄 审稿未通过，存在 {len(state.actionable_revisions)} 处待改进项，进入下一轮迭代...",
                state,
            )

        # ====== 阶段 5: 定稿与导出 ======
        state.current_stage = Stage.COMPLETED
        state.final_markdown = self._format_final_markdown(state)
        self._export_to_file(state, output_dir=output_dir)
        self.on_progress("🚀 文章已生成并成功导出至 output 目录！", state)

        return state

    def _format_final_markdown(self, state: WritingState) -> str:
        header = f"""---
title: {state.outline_title}
date: {datetime.now(UTC).strftime("%Y-%m-%d")}
topic: {state.topic}
review_score: {state.review_score}
total_words: {len(state.full_draft)}
---

"""
        footer = f"""
---
### 审稿与生成元信息
- **综合质检评分**: `{state.review_score:.1f} / 100`
- **反思修订轮次**: `{state.revision_count}`
- **Token 消耗统计**: `Prompt: {state.token_usage.get("prompt_tokens", 0)} | Completion: {state.token_usage.get("completion_tokens", 0)} | Total: {state.token_usage.get("total_tokens", 0)}`
"""
        return header + state.full_draft + footer

    def _export_to_file(self, state: WritingState, output_dir: str = "output") -> str:
        os.makedirs(output_dir, exist_ok=True)
        safe_title = re.sub(r'[\\/*?:"<>| ]', "_", state.outline_title)[:40]
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_title}_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(state.final_markdown)

        return filepath
