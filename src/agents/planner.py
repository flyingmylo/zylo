import json
import re

from src.llm.base import LLMProvider
from src.prompts import PLANNER_SYSTEM_PROMPT
from src.state import SectionSpec, Stage, WritingState

from .base import BaseAgent


class PlannerAgent(BaseAgent):
    """
    大纲规划师 Agent：
    综合技术主题、用户额外指令与调研综述，制定深度技术博客的全局架构与章节划分
    关键职责：为每一个章节分别精准生成中英双语的向量检索关键词（retrieval_query_zh & retrieval_query_en）
    """

    def __init__(self, llm: LLMProvider):
        super().__init__(name="Planner", llm=llm, system_prompt=PLANNER_SYSTEM_PROMPT)

    async def run(self, state: WritingState) -> WritingState:
        state.current_stage = Stage.PLANNING

        user_content = f"""【写作主题】：{state.topic}
【补充要求】：{state.extra_instructions or "无特殊要求，注重技术深度与清晰度"}
【前期调研综述】：
{state.research_summary or "未执行外部调研，请基于通用技术深度知识进行架构规划。"}

请生成完整的大纲 JSON 结构。必须严格为每个 section 配置用于检索英文文献与中文资料的双语检索词。"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        resp = await self._chat_with_tools(
            messages=messages,
            state=state,
            temperature=0.4,
        )

        raw = resp.content.strip()
        # 清理可能附带的 markdown 标记
        if raw.startswith("```json"):
            raw = raw[7:].rsplit("```", 1)[0].strip()
        elif raw.startswith("```"):
            raw = raw[3:].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except Exception:
            # 正则容错提取第一个 json object
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = {
                    "outline_title": f"{state.topic} 深度技术解析与实践指南",
                    "target_total_words": 3000,
                    "sections": [
                        {
                            "title": "一、背景与核心痛点剖析",
                            "target_words": 700,
                            "focus_points": ["核心问题背景", "传统架构局限性"],
                            "retrieval_query_zh": f"{state.topic} 痛点与背景",
                            "retrieval_query_en": f"{state.topic} challenges and limitations",
                        },
                        {
                            "title": "二、核心架构与技术原理",
                            "target_words": 1200,
                            "focus_points": ["核心设计机制", "关键算法与交互链路"],
                            "retrieval_query_zh": f"{state.topic} 核心架构 原理",
                            "retrieval_query_en": f"{state.topic} architecture and mechanisms",
                        },
                        {
                            "title": "三、工程落地与最佳实践",
                            "target_words": 800,
                            "focus_points": ["典型生产环境配置", "性能优化与避坑指南"],
                            "retrieval_query_zh": f"{state.topic} 实践 优化",
                            "retrieval_query_en": f"{state.topic} best practices production deployment",
                        },
                        {
                            "title": "四、总结与未来展望",
                            "target_words": 300,
                            "focus_points": ["技术趋势", "演进方向"],
                            "retrieval_query_zh": f"{state.topic} 趋势",
                            "retrieval_query_en": f"{state.topic} future trends",
                        },
                    ],
                }

        state.outline_title = data.get("outline_title", f"{state.topic} 深度解析")
        state.target_total_words = data.get("target_total_words", 3000)

        sections = []
        for s in data.get("sections", []):
            sections.append(
                SectionSpec(
                    title=s.get("title", "未命名小节"),
                    target_words=s.get("target_words", 600),
                    focus_points=s.get("focus_points", []),
                    retrieval_query_zh=s.get("retrieval_query_zh", state.topic),
                    retrieval_query_en=s.get("retrieval_query_en", ""),
                )
            )
        state.sections = sections
        return state
