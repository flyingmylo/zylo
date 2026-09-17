from src.llm.base import LLMProvider
from src.prompts import WRITER_SYSTEM_PROMPT
from src.state import GLOBAL_SCOPE, Stage, WritingState
from src.tools.knowledge_base import KnowledgeBase

from .base import BaseAgent


class WriterAgent(BaseAgent):
    """
    技术写作者 Agent：
    1. 逐章节执行靶向双语向量检索（兼顾中文资料与英文论文）
    2. 严格执行跨语言写作规范：输出通顺纯正的中文技术表达
    3. 专业概念强制执行「中文译名（English Name）」双语对照
    4. 修订轮在上一版正文基础上定向重写被点名的小节，未点名小节原样保留
    """

    def __init__(self, llm: LLMProvider, knowledge_base: KnowledgeBase):
        super().__init__(name="Writer", llm=llm, system_prompt=WRITER_SYSTEM_PROMPT)
        self.kb = knowledge_base

    async def run(self, state: WritingState) -> WritingState:
        state.current_stage = (
            Stage.WRITING if state.revision_count == 0 else Stage.REVISING
        )

        previous_drafts = dict(state.section_drafts)
        is_revision = state.revision_count > 0 and bool(previous_drafts)

        # 1. 意见分组：定位到小节的意见进入 section_notes，其余（含全局）进入 global_notes
        all_titles = {sec.title for sec in state.sections}
        section_notes: dict[str, list[str]] = {}
        global_notes: list[str] = []
        if is_revision:
            for rev in state.actionable_revisions:
                advice = str(rev.get("advice", "")).strip()
                scope = str(rev.get("section", "")).strip()
                if not advice:
                    continue
                if scope in all_titles:
                    section_notes.setdefault(scope, []).append(advice)
                else:
                    global_notes.append(advice)

        # 2. 圈定本轮需要重写的小节：定向点名优先；
        #    只有全局意见（或没有意见）时全量重写，但同样基于上一版正文做修订
        if is_revision and section_notes:
            targeted_titles = set(section_notes)
            # 初稿轮被跳过的小节没有旧稿可复用，必须趁修订轮补写
            targeted_titles |= {
                t for t in all_titles if t not in previous_drafts
            }
        else:
            targeted_titles = set(all_titles)

        section_drafts = {}
        for sec in state.sections:
            # 3. 未被点名且已有上一版的小节：原样保留，零 LLM 成本，修复自动累积
            if is_revision and sec.title not in targeted_titles:
                section_drafts[sec.title] = previous_drafts[sec.title]
                continue

            # 4. 执行精准双语检索
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

            # 5. 修订轮注入：本节意见 + 全局意见 + 上一版正文
            revision_note = ""
            if is_revision:
                notes = section_notes.get(sec.title, []) + global_notes
                if notes:
                    revision_note = (
                        "\n【上一轮审稿人的修改意见，请务必针对性优化】:\n"
                        + "\n".join([f"- {n}" for n in notes])
                    )

            previous_block = ""
            if is_revision and sec.title in previous_drafts:
                previous_block = (
                    "【本节上一版正文（请在此基础上修订，未涉及部分保持稳定，"
                    "不要整节推倒重写）】：\n"
                    f"{previous_drafts[sec.title]}\n\n"
                )

            closing_line = (
                "请基于上一版正文完成修订，直接输出本小节修订后的完整 Markdown 正文（包含小节二级/三级标题）："
                if is_revision and sec.title in previous_drafts
                else "请开始撰写本小节的完整 Markdown 正文（包含小节二级/三级标题）："
            )

            user_prompt = f"""【文章全局大标题】：{state.outline_title}
【当前撰写小节】：{sec.title}
【本节规划字数】：约 {sec.target_words} 字
【本节必须涵盖要点】：{", ".join(sec.focus_points)}
{revision_note}

{previous_block}【检索到的参考资料片段（包含中/英文背景）】：
{context_str}

{closing_line}"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # 修订是编辑任务，用低温度收敛；初稿才需要发散
            resp = await self._chat_with_tools(
                messages=messages,
                state=state,
                temperature=0.3 if is_revision else 0.7,
            )
            content = resp.content.strip()
            if not content:
                if is_revision and sec.title in previous_drafts:
                    # 定向重写失败绝不能丢掉上一版内容
                    state.errors.append(
                        f"小节「{sec.title}」修订生成内容为空"
                        f"（finish_reason={resp.finish_reason}），已保留上一版。"
                    )
                    self.logger.warning("小节「%s」修订内容为空，保留上一版", sec.title)
                    section_drafts[sec.title] = previous_drafts[sec.title]
                else:
                    # 工具调用未收敛等情况下会拿到空正文，宁可留白也不写入空小节
                    state.errors.append(
                        f"小节「{sec.title}」生成内容为空"
                        f"（finish_reason={resp.finish_reason}），已跳过。"
                    )
                    self.logger.warning("小节「%s」生成内容为空，已跳过", sec.title)
                continue
            section_drafts[sec.title] = content

        state.section_drafts = section_drafts

        # 拼接整合为完整 Markdown 草稿
        full_content = [f"# {state.outline_title}\n"]
        for sec in state.sections:
            if sec.title in state.section_drafts:
                full_content.append(state.section_drafts[sec.title])
                full_content.append("\n---\n")

        state.full_draft = "\n\n".join(full_content)
        return state
