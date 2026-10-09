import os
import re
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime

from src.agents.base import BaseAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.writer import WriterAgent
from src.budget import BudgetGuard
from src.embeddings.base import EmbeddingProvider
from src.embeddings.reranker_base import RerankerProvider
from src.events import EventStatus, NullTrace, RunTrace, SpanKind
from src.llm.base import LLMProvider
from src.state import Stage, WritingState
from src.tools.knowledge_base import KnowledgeBase
from src.tools.search import SearchTool


class HumanReviewRequired(Exception):
    """human_review 开启时，每轮审稿完成后抛出以暂停执行（M3-3）。

    携带停点 state：JobRunner 据此把 run 落位 WAITING_FOR_HUMAN_REVIEW
    并保留内存引用（快照已由 checkpoint 先行落盘，跨进程可从快照恢复）。
    人工决策注入 state 后经 resume_state 重入 execute 续跑。
    """

    def __init__(self, state: WritingState) -> None:
        super().__init__("等待人工审稿决策")
        self.state = state


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
        kb_persist_dir: str | None = None,
        budget: BudgetGuard | None = None,
        trace: RunTrace | None = None,
        human_review: bool = False,
    ):
        self.llm = llm
        self.embedding = embedding_provider
        self.reranker = reranker_provider
        self.tavily_api_key = tavily_api_key
        self.on_progress = progress_callback or (lambda msg, state: None)
        # None = 进程内内存库；指定目录则向量落盘且 resume 可恢复检索
        self.kb_persist_dir = kb_persist_dir
        # 无预算时也挂一个无上限 guard：照常记账（calls/tokens/cost），
        # 只是永不熔断；同一 run 的全部 Agent 共享此实例
        self.budget = budget or BudgetGuard()
        # 五层 Trace 的发射器（run 层由 JobRunner 负责，此处发
        # stage/agent 两层；llm/tool 在 BaseAgent 内发射）
        self.trace = trace or NullTrace()
        # 人审开关：开启后每轮审稿完成即暂停（抛 HumanReviewRequired），
        # 由上层落位 WAITING_FOR_HUMAN_REVIEW 并等待 POST /review-decisions
        self.human_review = human_review

    async def _run_agent(
        self, agent: BaseAgent, state: WritingState, parent_id: str | None = None
    ) -> WritingState:
        """以 agent span 包裹一次 agent.run，异常也落 failed 事件后上抛。

        parent_id 是所属 stage 的 span：stage → agent → llm/tool 三层
        父子链由此建立（前端按 parent_id 还原调用树）。
        """
        span = self.trace.start_span(
            SpanKind.AGENT, type(agent).__name__, parent_id=parent_id
        )
        agent.current_span_id = span
        try:
            state = await agent.run(state)
        except Exception as exc:
            self.trace.finish_span(
                span,
                EventStatus.FAILED,
                payload={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        finally:
            agent.current_span_id = None
        self.trace.finish_span(span, EventStatus.COMPLETED)
        return state

    def _stage_span(self, name: str, payload: dict | None = None) -> str:
        return self.trace.start_span(SpanKind.STAGE, name, payload=payload)

    async def execute(
        self,
        topic: str,
        local_files: list[str] | None = None,
        extra_instructions: str = "",
        output_dir: str = "output",
        resume_state: WritingState | None = None,
        on_checkpoint: Callable[[WritingState], None] | None = None,
    ) -> WritingState:
        """驱动一次完整写作；可从快照 resume，已完成阶段直接跳过。

        - resume_state：来自最新阶段快照的反序列化状态。阶段完成度以
          current_stage 判定（各 agent.run 完成后 stage 停留在本阶段），
          research/planner 已完成则不再执行，LLM 不重复扣费；
        - on_checkpoint：阶段边界回调（research/planner/每轮审稿后），
          调用方在此落快照；默认 no-op。
        注意：resume 时进程内知识库为空（检索向量需 M2-4 持久化后才能
        跨进程恢复），但 research_summary 等文本产出都在 state 里，写作
        不受影响。
        """
        checkpoint = on_checkpoint or (lambda s: None)
        state = resume_state or WritingState(
            topic=topic,
            local_files=local_files or [],
            extra_instructions=extra_instructions,
        )

        # 1. 初始化专属文章级别的 KnowledgeBase。
        # resume 时沿用快照里的 collection 名：配合持久目录可重开原库，
        # 调研产出的向量检索能力跨进程恢复
        kb = KnowledgeBase(
            collection_name=state.kb_collection_name or None,
            embedding_provider=self.embedding,
            reranker_provider=self.reranker,
            persist_dir=self.kb_persist_dir,
        )
        if not state.kb_collection_name:
            state.kb_collection_name = kb.collection_name

        # 初始化 4 个 Agent
        search_tool = (
            SearchTool(api_key=self.tavily_api_key) if self.tavily_api_key else None
        )
        researcher = ResearcherAgent(
            self.llm,
            knowledge_base=kb,
            search_tool=search_tool,
            budget=self.budget,
            trace=self.trace,
        )
        planner = PlannerAgent(self.llm, budget=self.budget, trace=self.trace)
        writer = WriterAgent(self.llm, knowledge_base=kb, budget=self.budget, trace=self.trace)
        reviewer = ReviewerAgent(self.llm, budget=self.budget, trace=self.trace)

        def completed(stage: Stage) -> bool:
            """该阶段是否已在快照中完成（Stage 枚举定义序即执行序）。"""
            order = [s for s in Stage]
            return order.index(state.current_stage) >= order.index(stage)

        # ====== 阶段 1: 调研与知识库构建 ======
        if not completed(Stage.RESEARCHING):
            self.on_progress(
                "🔍 启动 Researcher 进行文献解析、网络检索与知识库构建...", state
            )
            stage = self._stage_span("researching")
            state = await self._run_agent(researcher, state, parent_id=stage)
            self.trace.finish_span(stage, EventStatus.COMPLETED, payload={"chunks": kb.count()})
            self.on_progress(f"✅ 调研完成，入库向量片段 {kb.count()} 条。", state)
        else:
            self.on_progress("⏭️ 快照显示调研已完成，跳过 Researcher。", state)
        checkpoint(state)

        # ====== 阶段 2: 大纲规划 ======
        if not completed(Stage.PLANNING):
            self.on_progress("📐 启动 Planner 制定深度技术架构大纲与双语检索词...", state)
            stage = self._stage_span("planning")
            state = await self._run_agent(planner, state, parent_id=stage)
            self.trace.finish_span(
                stage, EventStatus.COMPLETED, payload={"sections": len(state.sections)}
            )
            self.on_progress(
                f"✅ 大纲制定完毕，共 {len(state.sections)} 个深度章节。", state
            )
        else:
            self.on_progress("⏭️ 快照显示大纲已就绪，跳过 Planner。", state)
        checkpoint(state)

        # ====== 阶段 3 & 4: 写作与审稿反思回路 ======
        best_snapshot: dict | None = None
        while state.revision_count <= state.max_revisions:
            # 人审决策重入（M3-3）：停点快照恢复时本轮写作/审稿均已完成，
            # 决策结果已注入 state（意见过滤 / 拍板置 passed），直接消费——
            # 跳过 Writer/Reviewer，避免重复扣费
            if state.awaiting_human:
                state.awaiting_human = False
                if state.review_passed:
                    state.selected_revision = state.revision_count
                    self.on_progress("✅ 人工决策：拍板定稿。", state)
                    break
                if state.revision_count >= state.max_revisions:
                    # 人刚逐条决策过，当前稿就是人的选择，不回退历史最优
                    state.selected_revision = state.revision_count
                    self.on_progress(
                        "⚠️ 人工决策后已达最大轮次，按当前稿定稿。", state
                    )
                    break
                state.revision_count += 1
                self.on_progress(
                    f"👤 人工决策已注入，进入第 {state.revision_count} 轮定向修改...",
                    state,
                )
            else:
                round_desc = (
                    "初稿撰写"
                    if state.revision_count == 0
                    else f"第 {state.revision_count} 轮定向修改"
                )
                self.on_progress(f"✍️ 启动 Writer 执行小节向量检索与{round_desc}...", state)
                write_stage = self._stage_span(
                    "writing", payload={"round": state.revision_count}
                )
                state = await self._run_agent(writer, state, parent_id=write_stage)
                self.trace.finish_span(write_stage, EventStatus.COMPLETED)

                self.on_progress(
                    "🧐 启动 Reviewer 审查技术事实、专业术语与行文逻辑...", state
                )
                review_stage = self._stage_span(
                    "reviewing", payload={"round": state.revision_count}
                )
                state = await self._run_agent(reviewer, state, parent_id=review_stage)
                self.trace.finish_span(
                    review_stage,
                    EventStatus.COMPLETED,
                    payload={"score": state.review_score, "passed": state.review_passed},
                )

                self.on_progress(
                    f"📊 审稿得分: {state.review_score:.1f}/100 | 是否通过: {state.review_passed}",
                    state,
                )

                # 记录历史最优稿：审稿无单调保证，强制定稿时回填最优轮，
                # 避免「最后一轮」覆盖「最好一轮」
                if best_snapshot is None or state.review_score > best_snapshot["score"]:
                    best_snapshot = {
                        "round": state.revision_count,
                        "score": state.review_score,
                        "section_drafts": dict(state.section_drafts),
                        "full_draft": state.full_draft,
                        "review_passed": state.review_passed,
                        "critiques": list(state.critiques),
                        "actionable_revisions": deepcopy(state.actionable_revisions),
                    }

                # 审稿完成为最细粒度快照点：resume 自此重入修订轮，
                # 已生成正文都在 section_drafts 里，重写仅限被点名小节
                checkpoint(state)

                # 人审停点：审稿完成后不自动继续，state 与快照均已就绪，
                # 等待 POST /review-decisions 注入决策后续跑
                if self.human_review:
                    state.awaiting_human = True
                    checkpoint(state)
                    self.on_progress("⏸️ 审稿完成，等待人工决策...", state)
                    raise HumanReviewRequired(state)

                if self._advance_after_review(state, best_snapshot):
                    break
                state.revision_count += 1

        # ====== 阶段 5: 定稿与导出 ======
        export_stage = self._stage_span("exporting")
        state.current_stage = Stage.COMPLETED
        state.final_markdown = self._format_final_markdown(state)
        self._export_to_file(state, output_dir=output_dir)
        self.trace.finish_span(export_stage, EventStatus.COMPLETED)
        self.on_progress("🚀 文章已生成并成功导出至 output 目录！", state)
        checkpoint(state)

        return state

    def _advance_after_review(
        self, state: WritingState, best_snapshot: dict | None
    ) -> bool:
        """自动路径（人审关闭）的审稿后推进决策：返回 True 表示定稿退出回路。

        - 审稿通过：当前轮即定稿；
        - 达到最大轮次：回退历史最优稿定稿；
        - 否则返回 False，由调用方进入下一轮修订。
        """
        if state.review_passed:
            state.selected_revision = state.revision_count
            self.on_progress("🎉 审稿通过，符合发布质量！", state)
            return True

        if state.revision_count >= state.max_revisions:
            # best_snapshot 为 None：resume 重入后本进程尚未记录过最优稿，
            # 当前稿是唯一候选（原实现此处直接下标访问会 TypeError）
            if best_snapshot is not None and best_snapshot["score"] > state.review_score:
                state.review_score = best_snapshot["score"]
                state.section_drafts = best_snapshot["section_drafts"]
                state.full_draft = best_snapshot["full_draft"]
                state.review_passed = best_snapshot["review_passed"]
                state.critiques = best_snapshot["critiques"]
                state.actionable_revisions = best_snapshot["actionable_revisions"]
                state.selected_revision = best_snapshot["round"]
                best_round_desc = (
                    "初稿"
                    if best_snapshot["round"] == 0
                    else f"第 {best_snapshot['round']} 轮修改稿"
                )
                self.on_progress(
                    f"🏅 当前终稿非历史最优，已回退保留{best_round_desc}"
                    f"（{best_snapshot['score']:.1f} 分）...",
                    state,
                )
            else:
                state.selected_revision = state.revision_count
            self.on_progress(
                f"⚠️ 已达到最大修改轮次 ({state.max_revisions} 轮)，强制定稿。",
                state,
            )
            return True

        self.on_progress(
            f"🔄 审稿未通过，存在 {len(state.actionable_revisions)} 处待改进项，进入下一轮迭代...",
            state,
        )
        return False

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
- **最终采用版本**: `{"初稿" if state.selected_revision == 0 else f"第 {state.selected_revision} 轮修改稿"}`
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
