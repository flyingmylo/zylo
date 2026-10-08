import sqlite3

import pytest

from api.store import RunStore
from src.events import EventStatus, RunEvent, SpanKind
from src.runs import Run, RunStatus
from src.state import (
    SNAPSHOT_SCHEMA_VERSION,
    SectionSpec,
    Stage,
    WritingState,
    deserialize_state,
    serialize_state,
)


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


# --------------------------------------------------------------------------
# M2-3：WritingState 序列化与阶段快照
# --------------------------------------------------------------------------


def _rich_state() -> WritingState:
    state = WritingState(topic="KV-Cache 优化")
    state.current_stage = Stage.PLANNING
    state.research_summary = "调研综述内容"
    state.outline_title = "KV-Cache 深度解析"
    state.sections = [
        SectionSpec(
            title="一、原理",
            target_words=500,
            focus_points=["要点"],
            retrieval_query_zh="原理",
            retrieval_query_en="principle",
        )
    ]
    state.section_drafts = {"一、原理": "正文"}
    state.review_score = 88.0
    state.token_usage["total_tokens"] = 330
    return state


def test_state_serialization_roundtrip():
    state = _rich_state()
    payload = serialize_state(state)

    import json

    json.dumps(payload)  # 必须可 JSON 化（SQLite 落盘前提）
    restored = deserialize_state(payload)

    assert restored.current_stage is Stage.PLANNING
    assert restored.sections[0].retrieval_query_en == "principle"
    assert restored.section_drafts == {"一、原理": "正文"}
    assert restored.token_usage["total_tokens"] == 330
    assert restored == state


def test_deserialize_rejects_wrong_version_and_corruption():
    with pytest.raises(ValueError, match="版本不兼容"):
        deserialize_state({"schema_version": 99, "state": {}})

    with pytest.raises(ValueError, match="结构损坏"):
        deserialize_state(
            {"schema_version": SNAPSHOT_SCHEMA_VERSION, "state": {"sections": "not-a-list"}}
        )


def test_state_snapshot_save_and_latest(tmp_path):
    store = _make_store(tmp_path)
    run = Run(topic="快照运行")
    store.upsert_run(run)
    state = _rich_state()

    store.save_state_snapshot(run.id, "researching", serialize_state(state))
    state.current_stage = Stage.WRITING
    store.save_state_snapshot(run.id, "writing", serialize_state(state))

    latest = store.latest_state_snapshot(run.id)
    assert latest is not None
    restored = deserialize_state(latest)
    assert restored.current_stage is Stage.WRITING

    assert store.latest_state_snapshot("nope") is None


# --------------------------------------------------------------------------
# 服务启动恢复：遗留 RUNNING → PARTIAL
# --------------------------------------------------------------------------


def test_recover_stale_runs_marks_running_as_partial(tmp_path):
    from api.runner import recover_stale_runs

    store = _make_store(tmp_path)
    live = Run(topic="还在跑")
    live.transition(RunStatus.RUNNING)
    done = Run(topic="已完成")
    done.transition(RunStatus.RUNNING)
    done.transition(RunStatus.COMPLETED)
    store.upsert_run(live)
    store.upsert_run(done)

    recovered = recover_stale_runs(store)

    assert [r.id for r in recovered] == [live.id]
    live_loaded = store.get_run(live.id)
    done_loaded = store.get_run(done.id)
    assert live_loaded is not None and live_loaded.status is RunStatus.PARTIAL
    assert done_loaded is not None and done_loaded.status is RunStatus.COMPLETED
    # 幂等：再次启动不再有可恢复对象
    assert recover_stale_runs(store) == []
