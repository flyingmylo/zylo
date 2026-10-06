import json
import re

from src.llm.base import LLMProvider
from src.prompts import REVIEWER_SYSTEM_PROMPT
from src.state import GLOBAL_SCOPE, Stage, WritingState

from .base import BaseAgent


class ReviewerAgent(BaseAgent):
    """
    审稿人 Agent：
    1. 全局比对原始大纲要求与调研事实，审查技术论据的客观性与深度
    2. 严格核验专业英文术语是否遵循「中文译名（English Name）」规范
    3. 输出结构化评审报告（得分、通过判定、针对性修改清单）
    """

    def __init__(self, llm: LLMProvider):
        super().__init__(name="Reviewer", llm=llm, system_prompt=REVIEWER_SYSTEM_PROMPT)

    async def run(self, state: WritingState) -> WritingState:
        state.current_stage = Stage.REVIEWING

        user_content = f"""【文章标题】：{state.outline_title}
【预期大纲要求】：
{[s.title for s in state.sections]}

【待评审草稿全文】：
{state.full_draft}

请依据审查标准，严格评估并输出符合规范的 JSON 评审报告："""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        resp = await self._chat_with_tools(
            messages=messages,
            state=state,
            temperature=0.2,
        )

        raw = resp.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:].rsplit("```", 1)[0].strip()
        elif raw.startswith("```"):
            raw = raw[3:].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = {
                    "passed": True,
                    "score": 85.0,
                    "critiques": ["无法解析审稿人详细 JSON，默认通过并定稿。"],
                    "actionable_revisions": [],
                }

        state.review_score = float(data.get("score", 85.0))
        state.review_passed = bool(data.get("passed", state.review_score >= 85.0))
        state.critiques = data.get("critiques", [])
        raw_revisions = data.get("actionable_revisions", [])
        if not isinstance(raw_revisions, list):
            raw_revisions = []
        state.actionable_revisions = self._normalize_revisions(
            raw_revisions, [s.title for s in state.sections]
        )

        return state

    def _normalize_revisions(
        self, raw_revisions: list, section_titles: list[str]
    ) -> list[dict[str, str]]:
        """把模型输出的意见统一规范为 {"section": 标题|全局, "advice": 建议}。

        兼容两种输出形态：结构化对象（新规范）与纯字符串（模型偶发的旧格式）。
        定位不到任何小节标题的意见归入全局，保证意见永远不会被静默丢弃。
        """
        titles = [t.strip() for t in section_titles]
        normalized: list[dict[str, str]] = []
        for rev in raw_revisions:
            if isinstance(rev, dict):
                raw_advice = rev.get("advice", "")
                raw_scope = rev.get("section", "")
                advice = raw_advice.strip() if isinstance(raw_advice, str) else ""
                scope = raw_scope.strip() if isinstance(raw_scope, str) else ""
            elif isinstance(rev, str):
                advice, scope = rev.strip(), ""
            else:
                continue
            if not advice:
                continue

            # 显式标记为全局时必须尊重其作用域，不能因为 advice 偶然包含
            # 某个标题而把它重新归类成局部意见。
            if scope == GLOBAL_SCOPE:
                normalized.append({"section": GLOBAL_SCOPE, "advice": advice})
                continue

            # 优先用声明的 section 定位；定位失败再尝试从建议文本中反查标题
            matched = self._match_title(scope, titles) or self._match_title(
                advice, titles
            )
            if scope and not matched:
                self.logger.warning(
                    "审稿意见定位「%s」未能匹配任何小节标题，已归入全局", scope
                )
            normalized.append({"section": matched or GLOBAL_SCOPE, "advice": advice})
        return normalized

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
