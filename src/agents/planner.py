from src.budget import BudgetGuard
from src.llm.base import LLMProvider
from src.prompts import PLANNER_SYSTEM_PROMPT
from src.schemas import PlannerOutline, StructuredOutputError
from src.state import SectionSpec, Stage, WritingState

from .base import BaseAgent


class PlannerAgent(BaseAgent):
    """
    大纲规划师 Agent：
    综合技术主题、用户额外指令与调研综述，制定深度技术博客的全局架构与章节划分
    关键职责：为每一个章节分别精准生成中英双语的向量检索关键词（retrieval_query_zh & retrieval_query_en）
    """

    def __init__(self, llm: LLMProvider, budget: BudgetGuard | None = None):
        super().__init__(
            name="Planner", llm=llm, system_prompt=PLANNER_SYSTEM_PROMPT, budget=budget
        )

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

        try:
            outline = await self._chat_structured(
                messages, PlannerOutline, state, temperature=0.4
            )
        except StructuredOutputError as exc:
            # planning_degraded：重试后仍不合规。落模板大纲继续写作并显式留痕，
            # 绝不静默（修订回路与终稿溯源都需要知道大纲不是模型产出）
            self.logger.warning("大纲输出降级：%s", exc)
            state.errors.append(f"planning_degraded: {exc}")
            state.sections = _fallback_sections(state.topic)
            state.outline_title = f"{state.topic} 深度技术解析与实践指南"
            state.target_total_words = 3000
            return state

        state.outline_title = outline.outline_title
        state.target_total_words = outline.target_total_words
        state.sections = [
            SectionSpec(
                title=s.title,
                target_words=s.target_words,
                focus_points=s.focus_points,
                retrieval_query_zh=s.retrieval_query_zh or state.topic,
                retrieval_query_en=s.retrieval_query_en,
            )
            for s in outline.sections
        ]
        return state


def _fallback_sections(topic: str) -> list[SectionSpec]:
    """planning_degraded 时的模板大纲：保证修订回路有可定位的标题可用。"""
    return [
        SectionSpec(
            title="一、背景与核心痛点剖析",
            target_words=700,
            focus_points=["核心问题背景", "传统架构局限性"],
            retrieval_query_zh=f"{topic} 痛点与背景",
            retrieval_query_en=f"{topic} challenges and limitations",
        ),
        SectionSpec(
            title="二、核心架构与技术原理",
            target_words=1200,
            focus_points=["核心设计机制", "关键算法与交互链路"],
            retrieval_query_zh=f"{topic} 核心架构 原理",
            retrieval_query_en=f"{topic} architecture and mechanisms",
        ),
        SectionSpec(
            title="三、工程落地与最佳实践",
            target_words=800,
            focus_points=["典型生产环境配置", "性能优化与避坑指南"],
            retrieval_query_zh=f"{topic} 实践 优化",
            retrieval_query_en=f"{topic} best practices production deployment",
        ),
        SectionSpec(
            title="四、总结与未来展望",
            target_words=300,
            focus_points=["技术趋势", "演进方向"],
            retrieval_query_zh=f"{topic} 趋势",
            retrieval_query_en=f"{topic} future trends",
        ),
    ]
