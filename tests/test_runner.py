import asyncio

import pytest

from api.bus import TraceBus
from api.runner import JobRunner, RunExecutor, RunNotFoundError
from src.runs import RunStatus
from src.state import WritingState


class StubExecutor:
    """满足 RunExecutor 协议的替身：可控制产出或抛错。"""

    def __init__(self, state: WritingState | None = None, error: Exception | None = None):
        self.state = state or WritingState(topic="t")
        self.error = error
        self.calls: list[dict] = []

    async def execute(self, topic, local_files=None, extra_instructions="", output_dir="output"):
        self.calls.append(
            {
                "topic": topic,
                "local_files": local_files,
                "extra_instructions": extra_instructions,
                "output_dir": output_dir,
            }
        )
        if self.error:
            raise self.error
        self.state.final_markdown = "# 成稿"
        return self.state


def _drain_events(bus: TraceBus, run_id: str) -> list:
    """订阅后关闭，取回全部已缓冲事件。"""
    sub = bus.subscribe(run_id)
    bus.close(run_id)
    events = []
    while not sub._queue.empty():
        events.append(sub._queue.get_nowait())
    return [e for e in events if e is not None and hasattr(e, "status")]


async def test_lifecycle_queued_to_completed_with_events():
    bus = TraceBus()
    runner = JobRunner(bus)
    state = WritingState(topic="KV-Cache")
    state.review_score = 88.0
    state.token_usage["total_tokens"] = 330
    executor = StubExecutor(state=state)

    run = runner.create("KV-Cache 显存优化", sources=["references/yoco.pdf"], instructions="面向工程师")
    assert run.status is RunStatus.QUEUED

    runner.launch(run.id, executor, output_dir="output/x")
    await runner.wait(run.id)

    assert run.status is RunStatus.COMPLETED
    assert run.finished_at is not None
    # executor 收到的参数来自 Run 登记信息
    assert executor.calls[0]["topic"] == "KV-Cache 显存优化"
    assert executor.calls[0]["local_files"] == ["references/yoco.pdf"]
    assert executor.calls[0]["extra_instructions"] == "面向工程师"
    assert runner.result(run.id) is state

    events = _drain_events(bus, run.id)
    statuses = [(e.status, e.payload) for e in events]
    assert statuses[0][0].value == "started"
    assert statuses[-1][0].value == "completed"
    assert statuses[-1][1]["total_tokens"] == 330


async def test_executor_failure_marks_run_failed():
    bus = TraceBus()
    runner = JobRunner(bus)
    executor = StubExecutor(error=RuntimeError("LLM 端点不可用"))

    run = runner.create("任何主题")
    runner.launch(run.id, executor)
    await runner.wait(run.id)

    assert run.status is RunStatus.FAILED
    assert run.error is not None
    assert "LLM 端点不可用" in run.error
    assert runner.result(run.id) is None

    events = _drain_events(bus, run.id)
    last = events[-1]
    assert last.status.value == "failed"
    assert "RuntimeError" in last.payload["error"]


async def test_unknown_run_raises():
    runner = JobRunner(TraceBus())

    with pytest.raises(RunNotFoundError):
        runner.get("nope")

    with pytest.raises(RunNotFoundError):
        runner.launch("nope", StubExecutor())


async def test_double_launch_is_rejected():
    bus = TraceBus()
    runner = JobRunner(bus)

    run = runner.create("主题")
    slow = asyncio.Event()

    class SlowExecutor(StubExecutor):
        async def execute(self, *args, **kwargs):
            await slow.wait()
            return await super().execute(*args, **kwargs)

    runner.launch(run.id, SlowExecutor())
    try:
        with pytest.raises(ValueError, match="已在执行中"):
            runner.launch(run.id, StubExecutor())
    finally:
        slow.set()
        await runner.wait(run.id)


def test_executor_protocol_accepts_orchestrator_shape():
    """WritingOrchestrator 的 execute 签名必须满足 RunExecutor 协议（静态检查）。"""
    from src.orchestrator import WritingOrchestrator

    executor: RunExecutor = WritingOrchestrator(llm=None)  # pyright: ignore[reportArgumentType]
    assert callable(executor.execute)
