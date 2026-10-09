import asyncio

from api.bus import TraceBus
from api.store import RunStore
from src.events import EventStatus, RunEvent, SpanKind
from src.runs import Run, RunStatus


def _emit(bus: TraceBus, run_id: str, name: str, **kwargs) -> RunEvent:
    return bus.emit(
        run_id,
        kind=kwargs.pop("kind", SpanKind.STAGE),
        name=name,
        status=kwargs.pop("status", EventStatus.COMPLETED),
        **kwargs,
    )


async def test_sequence_increments_per_run_independently():
    bus = TraceBus()

    e1 = _emit(bus, "run_a", "researching")
    e2 = _emit(bus, "run_a", "planning")
    e3 = _emit(bus, "run_b", "researching")

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
    _emit(bus, "run_a", "researching")
    _emit(bus, "run_a", "planning")
    await asyncio.sleep(0)
    bus.close("run_a")
    await asyncio.wait_for(task, timeout=1)

    assert [e.name for e in received] == ["researching", "planning"]


async def test_late_subscriber_replays_history_then_continues():
    """Last-Event-ID 语义：after_sequence 之后的历史先补发，再接续实时。"""
    bus = TraceBus()
    _emit(bus, "run_a", "e1")
    _emit(bus, "run_a", "e2")
    _emit(bus, "run_a", "e3")

    sub = bus.subscribe("run_a", after_sequence=1)
    received = []

    async def consume():
        async for event in sub:
            received.append(event)

    task = asyncio.create_task(consume())
    # 让消费任务先跑到第一个 await（队列等待）挂起点
    await asyncio.sleep(0)
    _emit(bus, "run_a", "e4")
    bus.close("run_a")
    await asyncio.wait_for(task, timeout=1)

    assert [e.sequence for e in received] == [2, 3, 4]


async def test_two_subscribers_are_independent():
    bus = TraceBus()
    sub1 = bus.subscribe("run_a")
    sub2 = bus.subscribe("run_a", after_sequence=0)

    _emit(bus, "run_a", "e1")

    q1_events = []
    q2_events = []
    for sub, sink in ((sub1, q1_events), (sub2, q2_events)):
        while not sub._queue.empty():
            sink.append(sub._queue.get_nowait())

    assert len(q1_events) == 1 and len(q2_events) == 1


async def test_emit_sanitizes_payload_on_the_way_out():
    bus = TraceBus()

    event = _emit(bus, "run_a", "llm_call", payload={"api_key": "sk-x", "total_tokens": 5})

    assert event.payload["api_key"] == "[REDACTED]"
    assert event.payload["total_tokens"] == 5


async def test_unsubscribe_stops_delivery():
    bus = TraceBus()
    sub = bus.subscribe("run_a")

    bus.unsubscribe("run_a", sub)
    _emit(bus, "run_a", "e1")

    assert sub._queue.empty()
    bus.close("run_a")


async def test_subscribing_after_close_replays_then_terminates():
    """回归：run 结束后才连接的 SSE 客户端，回放历史后必须收到流终止。

    修复前后来者永远等不到结束信号，HTTP 客户端会无限挂起。
    """
    bus = TraceBus()
    _emit(bus, "run_a", "e1")
    _emit(bus, "run_a", "e2")
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


# --------------------------------------------------------------------------
# M3-2：SQLite 事实来源——序号续接与跨进程回放
# --------------------------------------------------------------------------


def _seed_run(store: RunStore, run_id: str, status: RunStatus) -> None:
    """FK 约束要求 run_events 的 run_id 先在 runs 表落位。"""
    store.upsert_run(Run(id=run_id, topic="t", status=status))


async def test_new_bus_continues_sequence_from_sqlite(tmp_path):
    """回归（resume 覆盖 bug）：新 bus 实例不得从 1 重计序号。

    修复前 resume 换新 bus 后，事件按 (run_id, sequence) 主键 REPLACE
    掉第一轮的同序号事件，历史时间线被篡改。
    """
    store = RunStore(tmp_path / "zylo.db")
    _seed_run(store, "run_r", RunStatus.PARTIAL)
    bus1 = TraceBus(store=store)
    for i in range(3):
        _emit(bus1, "run_r", f"e{i}")

    # resume 场景：换一个全新 bus（内存计数器为空），只有同一个 SQLite 文件
    bus2 = TraceBus(store=store)
    resumed = _emit(bus2, "run_r", "resumed")

    assert resumed.sequence == 4
    assert [e.name for e in store.get_events("run_r")] == ["e0", "e1", "e2", "resumed"]


async def test_subscribe_replays_from_sqlite_and_ends_for_stopped_run(tmp_path):
    """跨进程回放：新 bus 对已停摆 run 订阅，历史来自 SQLite 且流正常终结。

    修复前新 bus 的内存缓冲与 _closed 都是空的：回放为空 + 永不终结，
    SSE 客户端既看不到历史又无限挂死。
    """
    store = RunStore(tmp_path / "zylo.db")
    _seed_run(store, "run_x", RunStatus.PARTIAL)
    bus1 = TraceBus(store=store)
    _emit(bus1, "run_x", "e1")
    _emit(bus1, "run_x", "e2")

    # 全新进程视角：内存一无所知，只有同一个 SQLite 文件
    bus2 = TraceBus(store=store)
    sub = bus2.subscribe("run_x", after_sequence=0)
    # _drain 能在 timeout 内返回本身就证明流正常终结（挂死会超时失败）
    received = await asyncio.wait_for(_drain(sub), timeout=1)

    assert [e.sequence for e in received] == [1, 2]
