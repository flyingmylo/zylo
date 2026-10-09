from uuid import uuid4

from src.budget import BudgetGuard
from src.events import RunTrace
from src.llm.base import LLMProvider
from src.prompts import REVIEWER_SYSTEM_PROMPT
from src.schemas import ReviewReport, StructuredOutputError
from src.state import GLOBAL_SCOPE, Stage, WritingState

from .base import BaseAgent


class ReviewerAgent(BaseAgent):
    """
    审稿人 Agent：
    1. 全局比对原始大纲要求与调研事实，审查技术论据的客观性与深度
    2. 严格核验专业英文术语是否遵循「中文译名（English Name）」规范
    3. 输出结构化评审报告（得分、通过判定、针对性修改清单），
       经 Pydantic 严格校验 + section 取值域校验，失败带错误重试一次
    """

    def __init__(
        self,
        llm: LLMProvider,
        budget: BudgetGuard | None = None,
        trace: RunTrace | None = None,
    ):
        super().__init__(
            name="Reviewer",
            llm=llm,
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            budget=budget,
            trace=trace,
        )

    async def run(self, state: WritingState) -> WritingState:
        state.current_stage = Stage.REVIEWING

        # 合法取值显式告知：模型不再需要猜 section 该写什么格式
        titles = [s.title for s in state.sections]
        legal_values = [*titles, GLOBAL_SCOPE]

        user_content = f"""【文章标题】：{state.outline_title}
【预期大纲要求】：
{titles}

【待评审草稿全文】：
{state.full_draft}

请依据审查标准，严格评估并输出符合规范的 JSON 评审报告。
actionable_revisions 中每条 section 字段只允许取以下值之一（逐字复制，禁止使用正文子标题）：
{legal_values}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            report = await self._chat_structured(
                messages,
                ReviewReport,
                state,
                temperature=0.2,
                validator=self._make_scope_validator(titles),
            )
        except StructuredOutputError as exc:
            # review_degraded（fail-closed）：绝不默认通过。判定为未通过，
            # 交给修订回路与强制定稿回退兜底；降级事实显式进 errors 与意见列表
            self.logger.warning("审稿输出降级：%s", exc)
            state.errors.append(f"review_degraded: {exc}")
            state.review_passed = False
            state.critiques = [f"（review_degraded）审稿输出不合规，本轮判定未通过：{exc}"]
            state.actionable_revisions = []
            return state

        state.review_score = report.score
        state.review_passed = report.passed
        state.critiques = report.critiques
        # critique_id 是人工决策的定位锚点：POST /review-decisions 按 ID
        # 逐条处置（采纳/拒绝/修改），降级路径无意见故无需生成
        state.actionable_revisions = [
            {
                "critique_id": uuid4().hex[:8],
                "section": rev.section,
                "advice": rev.advice,
            }
            for rev in report.actionable_revisions
        ]
        return state

    def _make_scope_validator(self, titles: list[str]):
        """构造取值域校验器：section 必须落在大纲标题或 GLOBAL_SCOPE。

        能模糊匹配的先归一化为精确标题（保留容错），完全定位不了的
        返回错误信息——由 _chat_structured 回灌给模型重试修正，
        而不是像旧逻辑那样静默归入全局（基线 17/17 定位失败的根因）。
        """

        def validate(report: ReviewReport) -> str | None:
            for i, rev in enumerate(report.actionable_revisions):
                scope = rev.section.strip()
                if scope == GLOBAL_SCOPE or scope in titles:
                    if scope != rev.section:
                        rev.section = scope
                    continue
                matched = self._match_title(scope, titles) or self._match_title(
                    rev.advice, titles
                )
                if matched:
                    rev.section = matched
                    continue
                return (
                    f"actionable_revisions[{i}].section 的值「{rev.section}」"
                    f"不在合法取值内，必须逐字复制 {titles} 之一，"
                    f"或不针对具体小节时填 \"{GLOBAL_SCOPE}\""
                )
            return None

        return validate

    @staticmethod
    def _match_title(text: str, titles: list[str]) -> str:
        """精确匹配优先，其次做包含式匹配（短于 4 字的片段不参与包含，避免误匹配）。"""
        text = text.strip()
        if not text:
            return ""
        for title in titles:
            if text == title:
                return title
            if len(title) >= 4 and title in text:
                return title
            if len(text) >= 4 and text in title:
                return title
        return ""
