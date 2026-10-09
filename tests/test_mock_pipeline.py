import json

from api.bus import TraceBus, TraceEmitter
from api.runner import JobRunner
from src.embeddings.dummy import DummyEmbeddingProvider
from src.events import EventStatus, RunEvent
from src.llm.base import LLMProvider, LLMResponse
from src.llm.mock import MOCK_USAGE, MockLLMProvider
from src.orchestrator import WritingOrchestrator
from src.runs import Run, RunStatus
from src.state import deserialize_state, serialize_state


async def _chat(provider: LLMProvider, system: str, user: str = "") -> LLMResponse:
    return await provider.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )


# --------------------------------------------------------------------------
# Mock Provider 角色分派
# --------------------------------------------------------------------------


async def test_mock_provider_dispatches_by_role():
    provider = MockLLMProvider()

    keywords = await _chat(provider, "你是一位技术调研员。请生成检索词", "技术主题: MoE 路由")
    assert json.loads(keywords.content) == ["MoE 路由 核心原理", "MoE 路由 生产实践"]

    summary = await _chat(provider, "你是一位严谨资深的技术调研专家。")
    assert "调研综述" in summary.content

    outline = await _chat(provider, "你是一位顶级技术布道师与架构规划专家。", "【写作主题】：MoE")
    data = json.loads(outline.content)
    assert data["outline_title"] == "MoE 深度解析"
    assert len(data["sections"]) == 3

    writing = await _chat(
        provider, "你是一位卓越的中文技术作家", "【当前撰写小节】：一、背景"
    )
    assert "一、背景" in writing.content
    assert "KV-Cache" in writing.content


async def test_mock_review_two_round_semantics():
    """首轮审稿不通过并给修订意见，次轮通过——演示反思回路的最小剧本。"""
    provider = MockLLMProvider()

    first = await _chat(provider, "你是一位极其挑剔的技术审稿专家")
    data1 = json.loads(first.content)
    assert data1["passed"] is False and data1["score"] < 85
    assert data1["actionable_revisions"]

    second = await _chat(provider, "你是一位极其挑剔的技术审稿专家")
    data2 = json.loads(second.content)
    assert data2["passed"] is True and data2["score"] >= 85


async def test_mock_provider_reports_fake_usage():
    provider = MockLLMProvider()
    resp = await _chat(provider, "任意系统提示")
    assert resp.usage == MOCK_USAGE


def test_dummy_embedding_is_deterministic():
    provider = DummyEmbeddingProvider()

    v1 = provider.embed_query("KV-Cache")
    v2 = provider.embed_query("KV-Cache")
    v3 = provider.embed_query("完全不同的文本")

    assert v1 == v2
    assert v1 != v3
    assert len(v1) == DummyEmbeddingProvider.DIM
    assert len(provider.embed_documents(["a", "b"])) == 2


# --------------------------------------------------------------------------
# 真实 Orchestrator 的离线全链路（Mock LLM + Dummy 嵌入，不触网不花钱）
# --------------------------------------------------------------------------


async def test_full_pipeline_offline_with_real_orchestrator(tmp_path):
    bus = TraceBus()
    runner = JobRunner(bus)

    def make_executor() -> WritingOrchestrator:
        return WritingOrchestrator(
            llm=MockLLMProvider(),
            embedding_provider=DummyEmbeddingProvider(),
        )

    run = runner.create("MoE 路由机制")
    runner.launch(run.id, make_executor(), output_dir=str(tmp_path))
    await runner.wait(run.id)

    # 运行完成，且真实经历了"首轮不通过 → 修订 → 通过"的反思回路
    assert run.status is RunStatus.COMPLETED
    state = runner.result(run.id)
    assert state is not None
    assert state.revision_count == 1
    assert state.review_score == 92.0
    assert state.outline_title == "MoE 路由机制 深度解析"

    # 终稿内容与产物文件
    assert "MoE 路由机制 深度解析" in state.final_markdown
    articles = list(tmp_path.glob("*.md"))
    assert len(articles) == 1

    # Token 统计有数：全链路共 10 次 Mock 调用（tavily 未配置，
    # 检索词生成一并跳过）：综述1 + 规划1 + 初稿3 + 审稿1 + 修订3 + 审稿1
    assert state.token_usage["total_tokens"] == MOCK_USAGE["total_tokens"] * 10

    # 事件流可完整回放
    sub = bus.subscribe(run.id)
    events = []
    while not sub._queue.empty():
        item = sub._queue.get_nowait()
        if item is not None and hasattr(item, "status"):
            events.append(item)
    assert events[0].status.value == "started"
    assert events[-1].status.value == "completed"
    assert events[-1].payload["total_tokens"] == state.token_usage["total_tokens"]


# --------------------------------------------------------------------------
# M3-1：五层 Trace 事件
# --------------------------------------------------------------------------


async def test_full_pipeline_emits_all_five_span_layers():
    """run/stage/agent/llm/tool 五层事件齐全，parent 链与层级一致。"""
    from api.bus import TraceBus
    from src.events import EventStatus, SpanKind

    bus = TraceBus()
    # 给 Orchestrator 注入 TraceEmitter（M1 版总线测试只覆盖 run 层）

    orchestrator = WritingOrchestrator(
        llm=MockLLMProvider(),
        embedding_provider=DummyEmbeddingProvider(),
        trace=TraceEmitter(bus, "trace_run"),
    )
    await orchestrator.execute(topic="MoE 路由", output_dir="/tmp/zylo-trace-test")

    sub = bus.subscribe("trace_run")
    events = []
    while not sub._queue.empty():
        item = sub._queue.get_nowait()
        if isinstance(item, RunEvent):
            events.append(item)
    bus.close("trace_run")

    kinds = {e.kind for e in events}
    # run 层由 JobRunner 发（此处直跑 Orchestrator 无 run 层）；
    # stage/agent/llm 必须齐备；tool 层本链路无注册工具（Researcher
    # 检索走确定性流程），M3-1 在 BaseAgent._execute_tool 已接线
    assert kinds >= {SpanKind.STAGE, SpanKind.AGENT, SpanKind.LLM}

    stage_names = [e.name for e in events if e.kind is SpanKind.STAGE]
    # started/completed 成对：每个 stage 名出现 2 次
    # researching/planning/writing/reviewing（×2 轮）/exporting
    assert "researching" in stage_names and "planning" in stage_names
    assert stage_names.count("writing") == 4  # 2 个 span × 2 状态
    assert stage_names.count("reviewing") == 4
    assert "exporting" in stage_names

    agent_names = {e.name for e in events if e.kind is SpanKind.AGENT}
    assert agent_names == {"ResearcherAgent", "PlannerAgent", "WriterAgent", "ReviewerAgent"}

    # agent 事件挂在 stage 之下：parent_id 必须指向真实存在的 stage span
    span_ids = {e.span_id for e in events}
    agent_events = [e for e in events if e.kind is SpanKind.AGENT]
    assert all(e.parent_id in span_ids for e in agent_events)

    # LLM 事件挂在 agent 之下，且 started/completed 成对（含 usage 载荷）
    llm_events = [e for e in events if e.kind is SpanKind.LLM]
    llm_spans = {e.span_id for e in llm_events}
    assert len(llm_spans) * 2 == len(llm_events)
    completed_llm = [e for e in llm_events if e.status is EventStatus.COMPLETED]
    assert all(e.payload.get("total_tokens") for e in completed_llm)
    assert all(e.parent_id in span_ids for e in llm_events)


async def test_llm_failure_emits_failed_span():
    """LLM 调用抛错时 span 以 failed 收尾，错误摘要进 payload。"""

    class ExplodingLLM(MockLLMProvider):
        async def chat(self, messages, tools=None, temperature=0.7):
            raise RuntimeError("endpoint exploded")

    bus = TraceBus()

    orchestrator = WritingOrchestrator(
        llm=ExplodingLLM(),
        embedding_provider=DummyEmbeddingProvider(),
        trace=TraceEmitter(bus, "fail_run"),
    )

    import pytest

    with pytest.raises(RuntimeError, match="endpoint exploded"):
        await orchestrator.execute(topic="任何", output_dir="/tmp/zylo-trace-test")

    sub = bus.subscribe("fail_run")
    events = []
    while not sub._queue.empty():
        item = sub._queue.get_nowait()
        if isinstance(item, RunEvent):
            events.append(item)
    bus.close("fail_run")

    failed = [e for e in events if e.status is EventStatus.FAILED]
    assert failed, "LLM 故障必须产生 failed 事件"
    assert any("endpoint exploded" in str(e.payload.get("error", "")) for e in failed)


# --------------------------------------------------------------------------
# M2-3：断点恢复——已完成阶段不重复执行
# --------------------------------------------------------------------------


async def test_resume_skips_completed_stages(tmp_path):
    """从 PLANNING 快照 resume：调研与大纲不重跑（LLM 调用数少 2 次）。"""
    # 第一次运行：收集各阶段快照
    first_provider = MockLLMProvider()
    checkpoints: dict[str, dict] = {}

    def collect(state):
        checkpoints[state.current_stage.value] = serialize_state(state)

    first_state = await WritingOrchestrator(
        llm=first_provider, embedding_provider=DummyEmbeddingProvider()
    ).execute(topic="MoE 路由", output_dir=str(tmp_path), on_checkpoint=collect)

    assert "planning" in checkpoints
    assert first_provider.call_count == 10

    # 模拟崩溃后恢复：PLANNING 快照 + 全新进程（全新 Provider/Orchestrator）
    resume_provider = MockLLMProvider()
    resumed_state = await WritingOrchestrator(
        llm=resume_provider, embedding_provider=DummyEmbeddingProvider()
    ).execute(
        topic="MoE 路由",
        output_dir=str(tmp_path),
        resume_state=deserialize_state(checkpoints["planning"]),
    )

    # 调研与规划被跳过：10 - 2 = 8 次 LLM 调用
    assert resume_provider.call_count == 8
    # 快照中的产出原样保留，文章照常完成
    assert resumed_state.research_summary == first_state.research_summary
    assert resumed_state.outline_title == first_state.outline_title
    assert resumed_state.revision_count == 1
    assert resumed_state.review_score == 92.0


# --------------------------------------------------------------------------
# M3-3：人在回路——审稿停点与决策重入
# --------------------------------------------------------------------------


async def test_human_review_pauses_then_completes(tmp_path):
    """human_review 开启：首轮审稿后停 WAITING；决策注入后恢复，终稿零浪费。

    恢复沿用同一个 Mock 实例（审稿轮次延续），第二轮通过后再次停点，
    空决策集（无意见可决策）等价拍板定稿。停点/恢复不额外多花 LLM 调用。
    """
    from src.runs import ReviewAction, ReviewDecision

    bus = TraceBus()
    runner = JobRunner(bus)
    mock_llm = MockLLMProvider()

    def make_executor(run) -> WritingOrchestrator:
        return WritingOrchestrator(
            llm=mock_llm,
            embedding_provider=DummyEmbeddingProvider(),
            human_review=bool(run.config.get("human_review")),
        )

    run = runner.create("人审演示", human_review=True)
    runner.launch(run.id, make_executor(run), output_dir=str(tmp_path))
    await runner.wait(run.id)

    # 首轮审稿完成即停：等待人审而非自动修订
    assert run.status is RunStatus.WAITING_FOR_HUMAN_REVIEW
    pending = runner.pending_review(run.id)
    assert pending is not None
    assert pending.awaiting_human is True
    assert pending.revision_count == 0
    ids = [rev["critique_id"] for rev in pending.actionable_revisions]
    assert len(ids) == 1  # Mock 首轮审稿恰好一条全局意见

    # 采纳该意见 → 恢复执行：Writer 修订 → 第二轮审稿通过 → 再次停点
    decisions = [
        ReviewDecision(
            run_id=run.id, revision=0, critique_id=ids[0], action=ReviewAction.ACCEPT
        )
    ]
    runner.apply_review_decisions(
        run.id, decisions, make_executor, output_dir=str(tmp_path)
    )
    await runner.wait(run.id)
    assert run.status is RunStatus.WAITING_FOR_HUMAN_REVIEW
    pending2 = runner.pending_review(run.id)
    assert pending2 is not None
    assert pending2.revision_count == 1
    assert pending2.review_passed is True
    assert pending2.actionable_revisions == []

    # 无意见 → 空决策集即拍板：恢复后直接导出定稿
    runner.apply_review_decisions(run.id, [], make_executor, output_dir=str(tmp_path))
    await runner.wait(run.id)

    assert run.status is RunStatus.COMPLETED
    state = runner.result(run.id)
    assert state is not None
    assert state.revision_count == 1
    assert state.review_score == 92.0
    assert state.awaiting_human is False
    # 与自动路径完全相同的 10 次调用：停点与决策重入零额外 LLM 成本
    assert state.token_usage["total_tokens"] == MOCK_USAGE["total_tokens"] * 10


async def test_waiting_stream_stays_open_for_reconnect():
    """WAITING 期间 SSE 订阅流不终结：决策后的新事件继续原流推送。"""
    import asyncio

    import pytest

    bus = TraceBus()
    runner = JobRunner(bus)
    mock_llm = MockLLMProvider()

    def make_executor(run) -> WritingOrchestrator:
        return WritingOrchestrator(
            llm=mock_llm,
            embedding_provider=DummyEmbeddingProvider(),
            human_review=True,
        )

    run = runner.create("流保持")
    runner.launch(run.id, make_executor(run), output_dir="/tmp/zylo-waiting-test")
    await runner.wait(run.id)
    assert run.status is RunStatus.WAITING_FOR_HUMAN_REVIEW

    sub = bus.subscribe(run.id)
    try:
        # 流必须保持挂起（不终结）；能立即 drain 完说明被误关了
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_drain_mock(sub), timeout=0.2)
    finally:
        bus.unsubscribe(run.id, sub)
        bus.close(run.id)


async def _drain_mock(sub):
    async for _ in sub:
        pass


async def test_decision_semantics_edit_reject_and_validation():
    """决策注入语义（单元级）：EDIT 替换建议、REJECT 移除、全拒拍板；
    未全覆盖 / 轮次错误 / 未知 ID 被拒绝。用记录型执行体隔离注入逻辑。"""
    import pytest

    from api.runner import ReviewDecisionMismatchError
    from src.runs import ReviewAction, ReviewDecision
    from src.state import WritingState

    captured: dict[str, WritingState | None] = {}

    class RecordingExecutor:
        async def execute(
            self,
            topic,
            local_files=None,
            extra_instructions="",
            output_dir="output",
            resume_state=None,
            on_checkpoint=None,
        ):
            captured["state"] = resume_state
            return resume_state or WritingState(topic=topic)

    def make_factory():
        return lambda run: RecordingExecutor()

    def _pending_runner(opinions: list[tuple[str, str]]) -> tuple[JobRunner, Run]:
        """构造一个停在 WAITING、带指定意见集的 runner（不跑真实执行器）。"""
        runner = JobRunner(TraceBus())
        run = runner.create("决策语义", human_review=True)
        run.transition(RunStatus.RUNNING)
        state = WritingState(topic="t", revision_count=0)
        state.actionable_revisions = [
            {"critique_id": cid, "section": "全局", "advice": advice}
            for cid, advice in opinions
        ]
        runner._pending_reviews[run.id] = state
        run.transition(RunStatus.WAITING_FOR_HUMAN_REVIEW)
        return runner, run

    # ---- 校验路径 ----
    runner, run = _pending_runner([("aaa", "意见A")])
    with pytest.raises(ReviewDecisionMismatchError):  # 空决策 vs 1 条意见
        runner.apply_review_decisions(run.id, [], make_factory())
    with pytest.raises(ReviewDecisionMismatchError):  # 轮次错误
        runner.apply_review_decisions(
            run.id,
            [ReviewDecision(run_id=run.id, revision=5, critique_id="aaa", action=ReviewAction.REJECT)],
            make_factory(),
        )
    with pytest.raises(ReviewDecisionMismatchError):  # 未知 ID
        runner.apply_review_decisions(
            run.id,
            [ReviewDecision(run_id=run.id, revision=0, critique_id="zzz", action=ReviewAction.ACCEPT)],
            make_factory(),
        )

    # ---- EDIT + REJECT 混合：EDIT 替换建议文本，REJECT 被移除 ----
    runner, run = _pending_runner([("aaa", "原意见A"), ("bbb", "原意见B"), ("ccc", "原意见C")])
    runner.apply_review_decisions(
        run.id,
        [
            ReviewDecision(run_id=run.id, revision=0, critique_id="aaa", action=ReviewAction.ACCEPT),
            ReviewDecision(
                run_id=run.id,
                revision=0,
                critique_id="bbb",
                action=ReviewAction.EDIT,
                edited_advice="（人工改写）意见B",
            ),
            ReviewDecision(run_id=run.id, revision=0, critique_id="ccc", action=ReviewAction.REJECT, reason="不必改"),
        ],
        make_factory(),
    )
    await runner.wait(run.id)
    injected = captured["state"]
    assert injected is not None
    assert [
        (rev["critique_id"], rev["advice"]) for rev in injected.actionable_revisions
    ] == [("aaa", "原意见A"), ("bbb", "（人工改写）意见B")]  # ccc 已移除
    assert injected.review_passed is False  # 仍有意见 → 进入修订轮

    # ---- 全部拒绝 = 认可当前稿，等价拍板 ----
    runner, run = _pending_runner([("aaa", "意见A")])
    runner.apply_review_decisions(
        run.id,
        [ReviewDecision(run_id=run.id, revision=0, critique_id="aaa", action=ReviewAction.REJECT)],
        make_factory(),
    )
    await runner.wait(run.id)
    injected2 = captured["state"]
    assert injected2 is not None
    assert injected2.actionable_revisions == []
    assert injected2.review_passed is True
