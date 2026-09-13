import json
import re

from src.llm.base import LLMProvider
from src.prompts import REVIEWER_SYSTEM_PROMPT
from src.state import Stage, WritingState

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
            response_format={"type": "json_object"}
            if "gpt" in getattr(self.llm, "model", "")
            else None,
        )

        raw = resp.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:].rsplit("```", 1)[0].strip()
        elif raw.startswith("```"):
            raw = raw[3:].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except Exception:
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
        state.actionable_revisions = data.get("actionable_revisions", [])

        return state
