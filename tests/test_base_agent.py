import json
import logging

import pytest

from src.agents.base import BaseAgent
from src.agents.researcher import ResearcherAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.writer import WriterAgent
from src.llm.base import LLMProvider, LLMResponse, ToolCall
from src.state import SectionSpec, WritingState
from src.tools.base import Tool
from src.tools.search import SearchTool
from src.tools.web_reader import WebReader


class ScriptedLLM(LLMProvider):
    """按脚本依次返回预设响应，并记录每次调用收到的 messages 与 tools。"""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None, temperature=0.7):
        # messages 会被工具循环就地追加，这里做快照以便断言调用当时的协议形态
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "temperature": temperature,
            }
        )
        if not self.responses:
            raise AssertionError("ScriptedLLM 响应脚本已耗尽，说明 LLM 调用次数超出预期。")
        return self.responses.pop(0)


class StubTool:
    """最小 Tool 实现，满足 src.tools.base.Tool 协议且不触发网络请求。"""

    def __init__(self, result=None, error: Exception | None = None):
        self.result = result if result is not None else {"ok": True}
        self.error = error
        self.received: list[dict] = []
        self.schema = {
            "type": "function",
            "function": {
                "name": "stub_tool",
                "description": "测试用工具",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }

    async def execute(self, **kwargs):
        self.received.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class EchoAgent(BaseAgent):
    """测试用最小 Agent，直接暴露基类的 LLM 交互方法。"""

    def __init__(self, llm: LLMProvider, tools=None):
        super().__init__(name="Echo", llm=llm, system_prompt="test", tools=tools)

    async def run(self, state: WritingState) -> WritingState:
        return state


def _tool_call(
    call_id: str, name: str = "stub_tool", args: dict | None = None
) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(args or {}))


# --------------------------------------------------------------------------
# Token 统计
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_accumulates_token_usage():
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="hi",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ]
    )
    agent = EchoAgent(llm)
    state = WritingState(topic="t")

    resp = await agent._chat([{"role": "user", "content": "x"}], state)

    assert resp.content == "hi"
    assert state.token_usage["prompt_tokens"] == 10
    assert state.token_usage["completion_tokens"] == 5
    assert state.token_usage["total_tokens"] == 15


# --------------------------------------------------------------------------
# 工具调用循环
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_with_tools_degrades_without_tools():
    """无工具时必须退化为一次普通调用，并向 provider 传 tools=None 而非空列表。"""
    llm = ScriptedLLM([LLMResponse(content="直接回答", usage={"total_tokens": 7})])
    agent = EchoAgent(llm)
    state = WritingState(topic="t")

    resp = await agent._chat_with_tools([{"role": "user", "content": "x"}], state)

    assert resp.content == "直接回答"
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] is None
    assert state.token_usage["total_tokens"] == 7


@pytest.mark.asyncio
async def test_tool_loop_pairs_tool_calls_and_accumulates_usage():
    tool = StubTool(result={"hit": "KV-Cache"})
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[_tool_call("call_1", args={"q": "KV-Cache"})],
                usage={"total_tokens": 100},
            ),
            LLMResponse(content="最终答案", usage={"total_tokens": 40}),
        ]
    )
    agent = EchoAgent(llm, tools=[tool])
    state = WritingState(topic="t")

    resp = await agent._chat_with_tools([{"role": "user", "content": "x"}], state)

    assert resp.content == "最终答案"
    assert len(llm.calls) == 2
    # Token 必须跨轮次累计，否则终稿的消耗统计会偏低
    assert state.token_usage["total_tokens"] == 140

    # Schema 被透传给 provider，且工具名以 LLM 侧名字为准
    assert llm.calls[0]["tools"][0]["function"]["name"] == "stub_tool"
    assert tool.received == [{"q": "KV-Cache"}]

    # assistant 的 tool_calls 必须与 tool 结果按 id 严格配对
    second_round = llm.calls[1]["messages"]
    assistant_msg, tool_msg = second_round[-2], second_round[-1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == ""
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][0]["type"] == "function"
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    # 工具结果以 JSON 文本回灌，中文不被转义
    assert json.loads(tool_msg["content"]) == {"hit": "KV-Cache"}


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_response_each_get_a_result_message():
    tool = StubTool()
    llm = ScriptedLLM(
        [
            LLMResponse(
                content="并行调用",
                tool_calls=[_tool_call("c1"), _tool_call("c2")],
            ),
            LLMResponse(content="done"),
        ]
    )
    agent = EchoAgent(llm, tools=[tool])
    state = WritingState(topic="t")

    await agent._chat_with_tools([{"role": "user", "content": "x"}], state)

    second_round = llm.calls[1]["messages"]
    tool_msgs = [m for m in second_round if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    # 每个 tool 结果都必须紧跟在其对应的 assistant 消息之后，顺序不可错乱
    assert [m["role"] for m in second_round[-3:]] == ["assistant", "tool", "tool"]
    assert len(tool.received) == 2


@pytest.mark.asyncio
async def test_empty_tool_calls_list_breaks_loop():
    """tool_calls 为空列表时也应当作无工具调用处理（None 与 [] 的边界）。"""
    llm = ScriptedLLM(
        [LLMResponse(content="答案", tool_calls=[], usage={"total_tokens": 3})]
    )
    agent = EchoAgent(llm, tools=[StubTool()])
    state = WritingState(topic="t")

    resp = await agent._chat_with_tools([{"role": "user", "content": "x"}], state)

    assert resp.content == "答案"
    assert len(llm.calls) == 1
    assert state.token_usage["total_tokens"] == 3


@pytest.mark.asyncio
async def test_tool_exception_becomes_error_text_and_keeps_traceback(caplog):
    """工具异常必须转成错误文本回灌给模型，同时把完整 traceback 留在日志里。"""
    tool = StubTool(error=ValueError("boom"))
    llm = ScriptedLLM(
        [
            LLMResponse(content="", tool_calls=[_tool_call("c1")]),
            LLMResponse(content="已根据错误信息修正"),
        ]
    )
    agent = EchoAgent(llm, tools=[tool])
    state = WritingState(topic="t")

    with caplog.at_level(logging.ERROR):
        resp = await agent._chat_with_tools([{"role": "user", "content": "x"}], state)

    # 循环未被异常打断，模型看到了错误文本
    assert resp.content == "已根据错误信息修正"
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"].startswith("Error executing tool 'stub_tool'")
    assert "ValueError" in tool_msg["content"]
    assert "boom" in tool_msg["content"]

    records = [r for r in caplog.records if "执行失败" in r.getMessage()]
    assert records, "工具执行失败必须写日志"
    assert records[0].exc_info is not None, "日志必须携带完整 traceback"


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_text_without_raising():
    llm = ScriptedLLM(
        [
            LLMResponse(content="", tool_calls=[_tool_call("c1", name="not_registered")]),
            LLMResponse(content="ok"),
        ]
    )
    agent = EchoAgent(llm, tools=[StubTool()])
    state = WritingState(topic="t")

    await agent._chat_with_tools([{"role": "user", "content": "x"}], state)

    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["content"] == "Error: Tool 'not_registered' not found."


@pytest.mark.asyncio
async def test_tool_loop_warns_when_iteration_cap_reached(caplog):
    """跑满上限仍未收敛时必须告警，并原样返回未收敛的响应供上层判断。"""
    llm = ScriptedLLM(
        [LLMResponse(content="", tool_calls=[_tool_call(f"c{i}")]) for i in range(2)]
    )
    agent = EchoAgent(llm, tools=[StubTool()])
    state = WritingState(topic="t")

    with caplog.at_level(logging.WARNING):
        resp = await agent._chat_with_tools(
            [{"role": "user", "content": "x"}], state, max_tool_iters=2
        )

    assert len(llm.calls) == 2
    assert resp.tool_calls is not None
    assert resp.content == ""  # 上层 Writer 必须自己处理空正文
    assert any("仍未收敛" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_zero_iteration_cap_raises_runtime_error():
    agent = EchoAgent(ScriptedLLM([]), tools=[StubTool()])

    with pytest.raises(RuntimeError, match="未获得任何 LLM 响应"):
        await agent._chat_with_tools(
            [{"role": "user", "content": "x"}],
            WritingState(topic="t"),
            max_tool_iters=0,
        )


# --------------------------------------------------------------------------
# 工具注册表
# --------------------------------------------------------------------------


def test_tool_not_satisfying_protocol_is_ignored_with_warning(caplog):
    """缺 schema 的对象不满足 Tool 协议，应被忽略而不是推迟到调用时才炸。"""

    class NoSchemaTool:
        async def execute(self, **kwargs):
            return None

    with caplog.at_level(logging.WARNING):
        agent = EchoAgent(ScriptedLLM([]), tools=[NoSchemaTool()])

    assert agent._tool_map == {}
    assert agent._get_tool_schemas() is None
    assert any("不满足 Tool 协议" in r.getMessage() for r in caplog.records)


def test_tool_with_malformed_schema_is_ignored_without_crashing(caplog):
    """schema 形状非法时也走 warning 分支，不能让 Agent 构造失败。"""

    class MalformedTool:
        def __init__(self):
            self.schema = "not-a-dict"

        async def execute(self, **kwargs):
            return None

    with caplog.at_level(logging.WARNING):
        agent = EchoAgent(ScriptedLLM([]), tools=[MalformedTool()])

    assert agent._tool_map == {}
    assert agent._get_tool_schemas() is None
    assert any("缺少 function.name" in r.getMessage() for r in caplog.records)


def test_duplicate_tool_name_warns_and_last_registration_wins(caplog):
    first, second = StubTool(), StubTool()

    with caplog.at_level(logging.WARNING):
        agent = EchoAgent(ScriptedLLM([]), tools=[first, second])

    assert len(agent._get_tool_schemas()) == 1
    assert agent._tool_map["stub_tool"] is second
    assert any("重复注册" in r.getMessage() for r in caplog.records)


def test_schema_snapshot_prevents_advertise_dispatch_split():
    """注册后篡改工具自身的 schema，不得让「宣告的工具」与「可调度的实现」分裂。"""
    tool = StubTool()
    agent = EchoAgent(ScriptedLLM([]), tools=[tool])

    tool.schema["function"]["name"] = "renamed_after_registration"

    assert agent._get_tool_schemas()[0]["function"]["name"] == "stub_tool"
    assert list(agent._tool_map) == ["stub_tool"]


# --------------------------------------------------------------------------
# SearchTool 与 Tool 协议一致性
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_tool_conforms_to_tool_protocol(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool = SearchTool()

    # 真正的协议一致性检查，而不只是「看起来像」
    assert isinstance(tool, Tool)
    assert callable(tool.execute)
    assert tool.schema["type"] == "function"
    assert tool.schema["function"]["name"] == "web_search"
    assert isinstance(tool.schema["function"]["parameters"], dict)
    assert tool.client is None  # 无 key 时不发起真实请求

    results = await tool.execute(query="KV-Cache")
    assert "Mock Search" in results[0]["title"]


@pytest.mark.asyncio
async def test_search_tool_execute_ignores_undeclared_params(monkeypatch):
    """execute 只接受 Schema 声明过的参数，未声明的键不得影响检索成本。"""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool = SearchTool()
    captured: dict = {}

    async def spy_search(query, search_depth="basic", max_results=5):
        captured.update(
            query=query, search_depth=search_depth, max_results=max_results
        )
        return [{"title": "ok"}]

    monkeypatch.setattr(tool, "search", spy_search)

    await tool.execute(query="KV-Cache", max_results=1000, bogus="x")

    assert captured == {
        "query": "KV-Cache",
        "search_depth": "basic",
        "max_results": 5,  # 模型要求 1000，被白名单挡回默认值
    }


@pytest.mark.asyncio
async def test_search_tool_execute_rejects_invalid_depth_and_missing_query(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool = SearchTool()
    captured: dict = {}

    async def spy_search(query, search_depth="basic", max_results=5):
        captured.update(search_depth=search_depth)
        return []

    monkeypatch.setattr(tool, "search", spy_search)

    await tool.execute(query="q", search_depth="evil")
    assert captured["search_depth"] == "basic"  # 非法枚举值回落到默认

    with pytest.raises(ValueError, match="query"):
        await tool.execute()


# --------------------------------------------------------------------------
# Researcher token 统计回归
# --------------------------------------------------------------------------


class StubSearchTool:
    """Researcher 的确定性检索替身，不满足 Tool 协议也不需要满足。"""

    def __init__(self, results: list[dict] | None = None):
        self.queries: list[str] = []
        self._results = results or [
            {"title": "T", "url": "https://example.com", "content": "检索到的片段"}
        ]

    async def search(self, query, max_results=5):
        self.queries.append(query)
        return [dict(r) for r in self._results]


class FakeKnowledgeBase:
    collection_name = "fake_collection"

    def __init__(self):
        self.documents: list[dict] = []

    def add_documents(self, documents):
        self.documents.extend(documents)

    def retrieve(self, **kwargs):
        return []

    def count(self):
        return len(self.documents)


@pytest.mark.asyncio
async def test_researcher_token_usage_regression(monkeypatch):
    """回归：调研阶段两次 LLM 调用都必须计入 token_usage。

    修复前 Researcher 直接调 llm.chat，绕过了基类的统计入口，
    导致终稿展示的总消耗漏掉整个调研阶段。
    """

    async def no_fetch(url, timeout=10.0):
        return ""  # 原文抓取一律失败，走摘要兜底，测试不发起真实网络请求

    monkeypatch.setattr(WebReader, "fetch_and_clean", no_fetch)
    search_tool = StubSearchTool()
    llm = ScriptedLLM(
        [
            LLMResponse(
                content='["KV-Cache 优化"]',
                usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            ),
            LLMResponse(
                content="调研综述正文",
                usage={
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                },
            ),
        ]
    )
    kb = FakeKnowledgeBase()
    agent = ResearcherAgent(llm, knowledge_base=kb, search_tool=search_tool)
    state = WritingState(topic="KV-Cache 显存优化")

    state = await agent.run(state)

    assert state.research_summary == "调研综述正文"
    assert search_tool.queries == ["KV-Cache 优化"]
    assert len(llm.calls) == 2
    assert state.token_usage["prompt_tokens"] == 220
    assert state.token_usage["completion_tokens"] == 110
    assert state.token_usage["total_tokens"] == 330
    # 调研阶段不注册工具，检索走确定性流程
    assert agent._get_tool_schemas() is None


@pytest.mark.asyncio
async def test_researcher_ingests_fulltext_with_snippet_fallback(monkeypatch):
    """搜索结果优先抓原文切块入库；失败/过短/重复 URL 一律降级为摘要。"""
    search_tool = StubSearchTool(
        results=[
            {"title": "好文", "url": "https://good.com", "content": "好文摘要"},
            {"title": "坏文", "url": "https://bad.com", "content": "坏文摘要"},
            {"title": "短文", "url": "https://short.com", "content": "短文摘要"},
            {"title": "重复", "url": "https://good.com", "content": "重复摘要"},
        ]
    )
    fetched: list[str] = []

    async def fake_fetch(url, timeout=10.0):
        fetched.append(url)
        if url == "https://good.com":
            return "这是抓取到的完整正文，包含充分的技术细节。" * 30
        if url == "https://short.com":
            return "太短的正文"
        return ""

    monkeypatch.setattr(WebReader, "fetch_and_clean", fake_fetch)

    llm = ScriptedLLM(
        [
            LLMResponse(content='["检索词"]'),
            LLMResponse(content="调研综述正文"),
        ]
    )
    kb = FakeKnowledgeBase()
    agent = ResearcherAgent(llm, knowledge_base=kb, search_tool=search_tool)

    state = await agent.run(WritingState(topic="t"))

    # URL 去重：重复的 good.com 只抓取一次
    assert sorted(fetched) == ["https://bad.com", "https://good.com", "https://short.com"]

    good = [d for d in kb.documents if d["source"] == "https://good.com"]
    assert good, "抓到原文的 URL 必须以全文切块入库"
    assert all("完整正文" in d["text"] for d in good)
    assert all("好文摘要" not in d["text"] for d in good), "全文成功后摘要不应再入库"

    for url, marker in [
        ("https://bad.com", "坏文摘要"),
        ("https://short.com", "短文摘要"),
    ]:
        fallback = [d for d in kb.documents if d["source"] == url]
        assert len(fallback) == 1 and marker in fallback[0]["text"]

    assert state.research_summary == "调研综述正文"


@pytest.mark.asyncio
async def test_researcher_fulltext_fetch_is_capped(monkeypatch):
    """原文抓取有上限，超出上限的搜索结果仍以摘要兜底入库，不静默丢弃。"""
    results = [
        {"title": f"t{i}", "url": f"https://x.com/{i}", "content": f"摘要{i}"}
        for i in range(10)
    ]
    search_tool = StubSearchTool(results=results)
    fetched: list[str] = []

    async def no_fetch(url, timeout=10.0):
        fetched.append(url)
        return ""

    monkeypatch.setattr(WebReader, "fetch_and_clean", no_fetch)

    llm = ScriptedLLM(
        [
            LLMResponse(content='["检索词"]'),
            LLMResponse(content="调研综述正文"),
        ]
    )
    kb = FakeKnowledgeBase()
    agent = ResearcherAgent(llm, knowledge_base=kb, search_tool=search_tool)

    await agent.run(WritingState(topic="t"))

    assert len(fetched) == 8  # 只尝试前 8 条
    assert len(kb.documents) == 10  # 10 条全部以摘要形式入库


def test_researcher_summary_sampling_round_robins_across_sources():
    """调研综述必须优先覆盖不同来源，而不是被首个网页的切片垄断。"""
    chunks = [
        {"text": "a1", "source": "a", "page": "1"},
        {"text": "a2", "source": "a", "page": "1"},
        {"text": "a3", "source": "a", "page": "1"},
        {"text": "b1", "source": "b", "page": "1"},
        {"text": "b2", "source": "b", "page": "1"},
        {"text": "c1", "source": "c", "page": "1"},
    ]

    selected = ResearcherAgent._select_diverse_chunks(chunks, limit=5)

    assert [chunk["text"] for chunk in selected] == ["a1", "b1", "c1", "a2", "b2"]


@pytest.mark.asyncio
async def test_researcher_fetch_exception_falls_back_to_snippet(monkeypatch):
    async def raising_fetch(url, timeout=10.0):
        raise RuntimeError("network failed")

    monkeypatch.setattr(WebReader, "fetch_and_clean", raising_fetch)
    agent = ResearcherAgent(
        ScriptedLLM([]), knowledge_base=FakeKnowledgeBase(), search_tool=None
    )
    candidate = {
        "url": "https://example.com",
        "snippet": {"text": "摘要", "source": "https://example.com", "page": "1"},
    }

    result = await agent._fetch_one(candidate, __import__("asyncio").Semaphore(1))

    assert result == [candidate["snippet"]]


# --------------------------------------------------------------------------
# Writer 空正文处理
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_skips_empty_section_and_records_error(caplog):
    """空正文不得写进草稿，必须留下可追溯的错误记录。"""
    llm = ScriptedLLM([LLMResponse(content="   ", finish_reason="tool_calls")])
    agent = WriterAgent(llm, knowledge_base=FakeKnowledgeBase())
    state = WritingState(topic="t", outline_title="标题")
    state.sections = [SectionSpec(title="一、原理", target_words=100)]

    with caplog.at_level(logging.WARNING):
        state = await agent.run(state)

    assert state.section_drafts == {}
    assert len(state.errors) == 1
    assert "一、原理" in state.errors[0]
    assert "生成内容为空" in state.errors[0]
    assert "tool_calls" in state.errors[0]  # finish_reason 一并暴露，便于定位
    assert "一、原理" not in state.full_draft
    assert any("生成内容为空" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# Reviewer 结构化意见解析与规范化
# --------------------------------------------------------------------------


def _review_state(titles: list[str]) -> WritingState:
    state = WritingState(topic="t", outline_title="标题", full_draft="全文正文")
    state.sections = [SectionSpec(title=t, target_words=100) for t in titles]
    return state


def _review_json(revisions) -> str:
    return json.dumps(
        {
            "passed": False,
            "score": 78.5,
            "critiques": ["c"],
            "actionable_revisions": revisions,
        }
    )


@pytest.mark.asyncio
async def test_reviewer_parses_structured_revisions():
    llm = ScriptedLLM(
        [
            LLMResponse(
                content=_review_json(
                    [
                        {"section": "一、原理", "advice": "补充显存对比数据"},
                        {"section": "全局", "advice": "统一术语双语对照"},
                    ]
                )
            )
        ]
    )
    agent = ReviewerAgent(llm)
    state = _review_state(["一、原理", "二、实践"])

    state = await agent.run(state)

    assert state.review_score == 78.5
    assert state.review_passed is False
    assert state.actionable_revisions == [
        {"section": "一、原理", "advice": "补充显存对比数据"},
        {"section": "全局", "advice": "统一术语双语对照"},
    ]


@pytest.mark.asyncio
async def test_reviewer_preserves_explicit_global_scope_when_advice_names_section():
    """显式全局意见不能因为正文中提到某个小节标题而被错误降级为局部意见。"""
    llm = ScriptedLLM(
        [
            LLMResponse(
                content=_review_json(
                    [
                        {
                            "section": "全局",
                            "advice": "统一一、原理与二、实践之间的术语表达",
                        }
                    ]
                )
            )
        ]
    )
    state = await ReviewerAgent(llm).run(_review_state(["一、原理", "二、实践"]))

    assert state.actionable_revisions == [
        {
            "section": "全局",
            "advice": "统一一、原理与二、实践之间的术语表达",
        }
    ]


@pytest.mark.asyncio
async def test_reviewer_normalizes_legacy_string_revisions():
    """模型偶发输出纯字符串旧格式：标题出现在文本里则归属该节，否则归全局。"""
    llm = ScriptedLLM(
        [
            LLMResponse(
                content=_review_json(
                    [
                        "一、核心原理：请务必增加键值缓存（KV-Cache）双语对照。",
                        "语言整体偏生硬。",
                    ]
                )
            )
        ]
    )
    agent = ReviewerAgent(llm)
    state = _review_state(["一、核心原理", "二、实践"])

    state = await agent.run(state)

    assert state.actionable_revisions == [
        {
            "section": "一、核心原理",
            "advice": "一、核心原理：请务必增加键值缓存（KV-Cache）双语对照。",
        },
        {"section": "全局", "advice": "语言整体偏生硬。"},
    ]


@pytest.mark.asyncio
async def test_reviewer_fuzzy_scope_match_and_global_fallback(caplog):
    """section 与标题不完全一致时按包含关系匹配；完全匹配不上必须归全局并告警。"""
    llm = ScriptedLLM(
        [
            LLMResponse(
                content=_review_json(
                    [
                        {"section": "核心原理", "advice": "补充公式推导"},
                        {"section": "第二节", "advice": "调整详略"},
                    ]
                )
            )
        ]
    )
    agent = ReviewerAgent(llm)
    state = _review_state(["一、核心原理", "二、实践"])

    with caplog.at_level(logging.WARNING):
        state = await agent.run(state)

    assert state.actionable_revisions[0]["section"] == "一、核心原理"
    assert state.actionable_revisions[1]["section"] == "全局"
    assert any("归入全局" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# Writer 修订轮定向重写
# --------------------------------------------------------------------------


def _revision_state() -> WritingState:
    state = WritingState(topic="t", outline_title="标题", full_draft="旧全文")
    state.sections = [
        SectionSpec(title="一、原理", target_words=100),
        SectionSpec(title="二、实践", target_words=100),
    ]
    state.section_drafts = {"一、原理": "旧版原理", "二、实践": "旧版实践"}
    state.revision_count = 1
    return state


@pytest.mark.asyncio
async def test_revision_round_rewrites_only_targeted_sections():
    """被点名的小节才走 LLM，未点名小节原样保留上一版，修复自动累积。"""
    state = _revision_state()
    state.actionable_revisions = [{"section": "一、原理", "advice": "补充数据"}]
    llm = ScriptedLLM([LLMResponse(content="新版原理")])
    agent = WriterAgent(llm, knowledge_base=FakeKnowledgeBase())

    state = await agent.run(state)

    assert state.section_drafts == {"一、原理": "新版原理", "二、实践": "旧版实践"}
    assert len(llm.calls) == 1
    assert "新版原理" in state.full_draft
    assert "旧版实践" in state.full_draft


@pytest.mark.asyncio
async def test_mixed_global_and_local_notes_rewrite_every_section():
    """混合意见中只要存在全局项，就必须重写所有章节。"""
    state = _revision_state()
    state.actionable_revisions = [
        {"section": "一、原理", "advice": "补充显存对比数据"},
        {"section": "全局", "advice": "统一术语对照"},
    ]
    llm = ScriptedLLM(
        [LLMResponse(content="新版原理"), LLMResponse(content="新版实践")]
    )
    agent = WriterAgent(llm, knowledge_base=FakeKnowledgeBase())

    await agent.run(state)

    assert len(llm.calls) == 2
    first_prompt = llm.calls[0]["messages"][1]["content"]
    second_prompt = llm.calls[1]["messages"][1]["content"]
    assert "旧版原理" in first_prompt
    assert "补充显存对比数据" in first_prompt
    assert "统一术语对照" in first_prompt
    assert "旧版实践" in second_prompt
    assert "补充显存对比数据" not in second_prompt
    assert "统一术语对照" in second_prompt
    assert all(call["temperature"] == 0.3 for call in llm.calls)


@pytest.mark.asyncio
async def test_global_only_notes_trigger_full_rewrite_with_previous_draft():
    """只有全局意见时全量重写，但每节都基于上一版修订而非盲写重抽。"""
    state = _revision_state()
    state.actionable_revisions = [{"section": "全局", "advice": "统一术语对照"}]
    llm = ScriptedLLM([LLMResponse(content="新原理"), LLMResponse(content="新实践")])
    agent = WriterAgent(llm, knowledge_base=FakeKnowledgeBase())

    state = await agent.run(state)

    assert len(llm.calls) == 2
    assert state.section_drafts == {"一、原理": "新原理", "二、实践": "新实践"}
    for call in llm.calls:
        body = call["messages"][1]["content"]
        assert "统一术语对照" in body
        assert "在此基础上修订" in body
        assert call["temperature"] == 0.3


@pytest.mark.asyncio
async def test_revision_empty_content_keeps_previous_version():
    """定向重写拿到空正文时必须保留上一版，修订轮绝不丢内容。"""
    state = _revision_state()
    state.actionable_revisions = [{"section": "一、原理", "advice": "补充数据"}]
    llm = ScriptedLLM([LLMResponse(content="   ", finish_reason="tool_calls")])
    agent = WriterAgent(llm, knowledge_base=FakeKnowledgeBase())

    state = await agent.run(state)

    assert state.section_drafts == {"一、原理": "旧版原理", "二、实践": "旧版实践"}
    assert any("保留上一版" in e for e in state.errors)


@pytest.mark.asyncio
async def test_initial_write_uses_high_temperature():
    """初稿轮行为不变：全量撰写、temperature 0.7。"""
    state = WritingState(topic="t", outline_title="标题")
    state.sections = [SectionSpec(title="一、原理", target_words=100)]
    llm = ScriptedLLM([LLMResponse(content="初稿正文")])
    agent = WriterAgent(llm, knowledge_base=FakeKnowledgeBase())

    state = await agent.run(state)

    assert llm.calls[0]["temperature"] == 0.7
    assert "旧版" not in llm.calls[0]["messages"][1]["content"]
    assert state.section_drafts == {"一、原理": "初稿正文"}
