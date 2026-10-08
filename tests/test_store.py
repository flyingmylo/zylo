import sqlite3

import pytest

from api.store import RunStore
from src.events import EventStatus, RunEvent, SpanKind
from src.runs import Run, RunStatus


def _make_store(tmp_path) -> RunStore:
    return RunStore(db_path=tmp_path / "zylo.db")


def _make_event(run_id: str, sequence: int) -> RunEvent:
    return RunEvent(
        sequence=sequence,
        run_id=run_id,
        kind=SpanKind.RUN,
        name="writing",
        status=EventStatus.COMPLETED,
        span_id=f"run_{run_id}",
        payload={"total_tokens": 180},
    )


def test_run_roundtrip_preserves_all_fields(tmp_path):
    store = _make_store(tmp_path)
    run = Run(topic="KV-Cache 显存优化", config={"sources": ["a.pdf"], "instructions": "面向工程师"})

    store.upsert_run(run)
    loaded = store.get_run(run.id)
    assert loaded is not None
    assert loaded == run
    assert loaded.config["sources"] == ["a.pdf"]


def test_upsert_updates_status_without_duplicating(tmp_path):
    store = _make_store(tmp_path)
    run = Run(topic="主题")

    store.upsert_run(run)
    run.transition(RunStatus.RUNNING)
    run.transition(RunStatus.COMPLETED)
    store.upsert_run(run)

    loaded = store.get_run(run.id)
    assert loaded is not None
    assert loaded.status is RunStatus.COMPLETED
    assert loaded.started_at is not None
    assert loaded.finished_at is not None

    all_runs = store.list_runs()
    assert len(all_runs) == 1


def test_get_missing_run_returns_none(tmp_path):
    store = _make_store(tmp_path)
    assert store.get_run("nope") is None


def test_list_runs_orders_by_creation_desc_with_pagination(tmp_path):
    store = _make_store(tmp_path)
    ids = []
    for i in range(5):
        run = Run(topic=f"主题{i}")
        store.upsert_run(run)
        ids.append(run.id)

    page1 = store.list_runs(limit=3)
    page2 = store.list_runs(limit=3, offset=3)

    assert [r.id for r in page1] == list(reversed(ids))[:3]
    assert [r.id for r in page2] == [ids[1], ids[0]]


def test_event_roundtrip_and_after_filter(tmp_path):
    store = _make_store(tmp_path)
    run = Run(topic="主题")
    store.upsert_run(run)

    for seq in (1, 2, 3):
        store.save_event(_make_event(run.id, seq))

    all_events = store.get_events(run.id)
    assert [e.sequence for e in all_events] == [1, 2, 3]
    assert all_events[-1].payload["total_tokens"] == 180

    tail = store.get_events(run.id, after_sequence=1)
    assert [e.sequence for e in tail] == [2, 3]


def test_events_are_idempotent_on_same_sequence(tmp_path):
    """(run_id, sequence) 主键 + REPLACE：崩溃重放场景下重复落盘不产生重复行。"""
    store = _make_store(tmp_path)
    run = Run(topic="主题")
    store.upsert_run(run)

    store.save_event(_make_event(run.id, 1))
    store.save_event(_make_event(run.id, 1))

    assert len(store.get_events(run.id)) == 1


def test_event_for_unknown_run_is_rejected(tmp_path):
    """外键约束：事件必须挂在已登记的 run 上，防止孤儿事件污染回放。"""
    store = _make_store(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        store.save_event(_make_event("ghost_run", 1))


def test_data_survives_reconnect(tmp_path):
    """模拟进程重启：新连接同一 db 文件，数据必须完整（WAL 持久化）。"""
    db_path = tmp_path / "zylo.db"
    store = RunStore(db_path=db_path)
    run = Run(topic="重启幸存者")
    store.upsert_run(run)
    store.save_event(_make_event(run.id, 1))
    store.close()

    reopened = RunStore(db_path=db_path)
    loaded = reopened.get_run(run.id)
    events = reopened.get_events(run.id)

    assert loaded is not None and loaded.topic == "重启幸存者"
    assert len(events) == 1
    reopened.close()
