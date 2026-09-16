import json

from src.llm.base import LLMProvider
from src.prompts import RESEARCHER_SYSTEM_PROMPT
from src.state import Stage, WritingState
from src.tools.knowledge_base import KnowledgeBase
from src.tools.pdf_reader import DocumentReader
from src.tools.search import SearchTool

from .base import BaseAgent


class ResearcherAgent(BaseAgent):
    """
    调研员 Agent：
    1. 解析本地参考文档（英文论文 / PDF / Markdown）切片入库
    2. 针对技术主题生成搜索词，调用 Tavily 补充网络最新资料
    3. 提取核心事实，生成调研综述，构建当前写作专属的向量知识库

    检索走确定性流程（生成检索词 → 逐条搜索），不经过 Function Calling，
    因此不向基类注册 tools；LLM 交互统一经 _chat 以便统计 Token。
    """

    def __init__(
        self,
        llm: LLMProvider,
        knowledge_base: KnowledgeBase,
        search_tool: SearchTool | None = None,
    ):
        super().__init__(
            name="Researcher",
            llm=llm,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
        )
        self.kb = knowledge_base
        self.doc_reader = DocumentReader()
        self.search_tool = search_tool

    async def run(self, state: WritingState) -> WritingState:
        state.current_stage = Stage.RESEARCHING

        all_chunks: list[dict[str, str]] = []

        # 1. 解析参考文档（支持本地文件与在线论文/网页 URL）
        for source in state.local_files:
            try:
                chunks = await self.doc_reader.read_source(source)
                all_chunks.extend(chunks)
            except Exception as e:
                state.errors.append(f"读取参考资料失败 {source}: {str(e)}")

        # 2. 联网补充搜索（针对主题生成中英文搜索词）
        if self.search_tool:
            kw_prompt = [
                {
                    "role": "system",
                    "content": '你是一位技术调研员。请为给定的技术主题生成最核心的 2 个中文检索词和 2 个英文检索词。直接输出 JSON 数组，如 ["词1", "词2"]。',
                },
                {"role": "user", "content": f"技术主题: {state.topic}"},
            ]
            kw_resp = await self._chat(kw_prompt, state, temperature=0.3)
            queries = []
            try:
                content = kw_resp.content.strip()
                if content.startswith("```json"):
                    content = content[7:].rsplit("```", 1)[0].strip()
                elif content.startswith("```"):
                    content = content[3:].rsplit("```", 1)[0].strip()
                queries = json.loads(content)
            except Exception:
                queries = [state.topic]

            for q in queries[:4]:
                search_results = await self.search_tool.search(q, max_results=3)
                for r in search_results:
                    if r.get("content"):
                        all_chunks.append(
                            {
                                "text": f"来源标题: {r.get('title')}\nURL: {r.get('url')}\n内容: {r.get('content')}",
                                "source": r.get("url") or "web_search",
                                "page": "1",
                            }
                        )

        # 3. 全部数据持久化入向量库
        if all_chunks:
            self.kb.add_documents(all_chunks)
            state.kb_collection_name = self.kb.collection_name

        # 4. 生成调研综述供 Planner 大纲规划使用
        sample_context = "\n\n---\n\n".join([c["text"][:600] for c in all_chunks[:6]])
        summary_prompt = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"请针对主题【{state.topic}】，结合以下抓取到的代表性核心资料片段，撰写一份系统性调研综述：\n\n{sample_context}",
            },
        ]
        summary_resp = await self._chat(summary_prompt, state, temperature=0.5)
        state.research_summary = summary_resp.content

        return state
