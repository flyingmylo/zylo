from src.llm.base import LLMProvider
from src.prompts import WRITER_SYSTEM_PROMPT
from src.state import Stage, WritingState
from src.tools.knowledge_base import KnowledgeBase

from .base import BaseAgent


class WriterAgent(BaseAgent):
    """
    技术写作者 Agent：
    1. 逐章节执行靶向双语向量检索（兼顾中文资料与英文论文）
    2. 严格执行跨语言写作规范：输出通顺纯正的中文技术表达
    3. 专业概念强制执行「中文译名（English Name）」双语对照
    4. 支持审稿反思回路下的局部定向重写与润色
    """

    def __init__(self, llm: LLMProvider, knowledge_base: KnowledgeBase):
        super().__init__(name="Writer", llm=llm, system_prompt=WRITER_SYSTEM_PROMPT)
        self.kb = knowledge_base

    async def run(self, state: WritingState) -> WritingState:
        state.current_stage = (
            Stage.WRITING if state.revision_count == 0 else Stage.REVISING
        )

        section_drafts = {}
        for idx, sec in enumerate(state.sections):
            # 1. 执行精准双语检索
            retrieved_chunks = self.kb.retrieve(
                query_zh=sec.retrieval_query_zh,
                query_en=sec.retrieval_query_en,
                top_k=4,
            )

            context_str = ""
            if retrieved_chunks:
                parts = []
                for i, c in enumerate(retrieved_chunks):
                    parts.append(
                        f"【参考片段 {i + 1} | 来源: {c.get('source', '')} 第{c.get('page', '1')}页】:\n{c.get('text', '')}"
                    )
                context_str = "\n\n".join(parts)
            else:
                context_str = "本节无特定检索参考资料，请根据总体技术脉络严谨推演编写。"

            # 2. 审稿反思建议注入（如果处于修改阶段）
            revision_note = ""
            if state.actionable_revisions and state.revision_count > 0:
                matching_revisions = [
                    rev
                    for rev in state.actionable_revisions
                    if sec.title in rev or f"第{idx + 1}节" in rev or "全局" in rev
                ]
                if matching_revisions:
                    revision_note = (
                        f"\n【上一轮审稿人对本节的修改意见，请务必针对性优化】:\n"
                        + "\n".join([f"- {r}" for r in matching_revisions])
                    )

            user_prompt = f"""【文章全局大标题】：{state.outline_title}
【当前撰写小节】：{sec.title}
【本节规划字数】：约 {sec.target_words} 字
【本节必须涵盖要点】：{", ".join(sec.focus_points)}
{revision_note}

【检索到的参考资料片段（包含中/英文背景）】：
{context_str}

请开始撰写本小节的完整 Markdown 正文（包含小节二级/三级标题）："""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            resp = await self._chat_with_tools(
                messages=messages,
                state=state,
                temperature=0.7,
            )
            section_drafts[sec.title] = resp.content.strip()

        state.section_drafts = section_drafts

        # 拼接整合为完整 Markdown 草稿
        full_content = [f"# {state.outline_title}\n"]
        for sec in state.sections:
            if sec.title in state.section_drafts:
                full_content.append(state.section_drafts[sec.title])
                full_content.append("\n---\n")

        state.full_draft = "\n\n".join(full_content)
        return state
