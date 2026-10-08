import asyncio

from api.bus import TraceBus
from src.events import EventStatus, RunEvent, SpanKind


async def _emit(bus: TraceBus, run_id: str, name: str, **kwargs) -> RunEvent:
    return await bus.emit(
        run_id,
        kind=kwargs.pop("kind", SpanKind.STAGE),
        name=name,
        status=kwargs.pop("status", EventStatus.COMPLETED),
        **kwargs,
    )


async def test_sequence_increments_per_run_independently():
    bus = TraceBus()

    e1 = await _emit(bus, "run_a", "researching")
    e2 = await _emit(bus, "run_a", "planning")
    e3 = await _emit(bus, "run_b", "researching")

    assert (e1.sequence, e2.sequence) == (1, 2)
    assert e3.sequence == 1


async def test_subscriber_receives_events_emitted_after_subscription():
    bus = TraceBus()
    sub = bus.subscribe("run_a")
    received: list[RunEvent] = []

    async def consume():
        async for event in sub:
            received.append(event)

    task = asyncio.create_task(consume())
    await _emit(bus, "run_a", "researching")
    await _emit(bus, "run_a", "planning")
    await asyncio.sleep(0)
    bus.close("run_a")
    await asyncio.wait_for(task, timeout=1)

    assert [e.name for e in received] == ["researching", "planning"]


async def test_late_subscriber_replays_history_then_continues():
    """Last-Event-ID 语义：after_sequence 之后的历史先补发，再接续实时。"""
    bus = TraceBus()
    await _emit(bus, "run_a", "e1")
    await _emit(bus, "run_a", "e2")
    await _emit(bus, "run_a", "e3")

    sub = bus.subscribe("run_a", after_sequence=1)
    received = []

    async def consume():
        async for event in sub:
            received.append(event)

    task = asyncio.create_task(consume())
    # 让消费任务先跑到第一个 await（队列等待）挂起点
    await asyncio.sleep(0)
    await _emit(bus, "run_a", "e4")
    bus.close("run_a")
    await asyncio.wait_for(task, timeout=1)

    assert [e.sequence for e in received] == [2, 3, 4]


async def test_two_subscribers_are_independent():
    bus = TraceBus()
    sub1 = bus.subscribe("run_a")
    sub2 = bus.subscribe("run_a", after_sequence=0)

    await _emit(bus, "run_a", "e1")

    q1_events = []
    q2_events = []
    for sub, sink in ((sub1, q1_events), (sub2, q2_events)):
        while not sub._queue.empty():
            sink.append(sub._queue.get_nowait())

    assert len(q1_events) == 1 and len(q2_events) == 1


async def test_emit_sanitizes_payload_on_the_way_out():
    bus = TraceBus()

    event = await _emit(bus, "run_a", "llm_call", payload={"api_key": "sk-x", "total_tokens": 5})

    assert event.payload["api_key"] == "[REDACTED]"
    assert event.payload["total_tokens"] == 5


async def test_unsubscribe_stops_delivery():
    bus = TraceBus()
    sub = bus.subscribe("run_a")

    bus.unsubscribe("run_a", sub)
    await _emit(bus, "run_a", "e1")

    assert sub._queue.empty()
    bus.close("run_a")


async def test_subscribing_after_close_replays_then_terminates():
    """回归：run 结束后才连接的 SSE 客户端，回放历史后必须收到流终止。

    修复前后来者永远等不到结束信号，HTTP 客户端会无限挂起。
    """
    bus = TraceBus()
    await _emit(bus, "run_a", "e1")
    await _emit(bus, "run_a", "e2")
    bus.close("run_a")

    late_sub = bus.subscribe("run_a", after_sequence=0)
    received = await asyncio.wait_for(
        _drain(late_sub), timeout=1
    )

    assert [e.sequence for e in received if e is not None] == [1, 2]


async def _drain(sub):
    events = []
    async for event in sub:
        events.append(event)
    return events
