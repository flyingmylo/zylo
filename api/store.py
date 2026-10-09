"""SQLite RunStore：运行与事件的持久化事实来源（ADR-002）。

设计约束：
- 标准库 sqlite3 + WAL，不引 ORM：单机单人场景下写频率极低，
  同步方法的延迟（微秒级）不值得引入异步驱动的复杂度；
- 表结构随功能小步落地：本模块先承接 runs 与 run_events，
  快照/修订/产物/来源表在对应里程碑步骤中追加；
- 时间戳一律 UTC ISO-8601 字符串，与 API 契约一致；
- 幂等性：upsert_run 可重复调用，事件以 (run_id, sequence) 主键去重。
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from src.events import RunEvent
from src.runs import ReviewAction, ReviewDecision, Run, RunStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id             TEXT PRIMARY KEY,
    topic          TEXT NOT NULL,
    status         TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    config_json    TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT,
    error          TEXT
);

CREATE TABLE IF NOT EXISTS run_events (
    run_id        TEXT NOT NULL REFERENCES runs(id),
    sequence      INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    name          TEXT NOT NULL,
    status        TEXT NOT NULL,
    span_id       TEXT NOT NULL,
    parent_id     TEXT,
    payload_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS state_snapshots (
    run_id         TEXT NOT NULL REFERENCES runs(id),
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stage          TEXT NOT NULL,
    state_json     TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_state_snapshots_run
    ON state_snapshots(run_id, id);

CREATE TABLE IF NOT EXISTS review_decisions (
    run_id        TEXT NOT NULL REFERENCES runs(id),
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    revision      INTEGER NOT NULL,
    critique_id   TEXT NOT NULL,
    action        TEXT NOT NULL,
    edited_advice TEXT,
    reason        TEXT,
    decided_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_decisions_run
    ON review_decisions(run_id, revision);
"""


class RunStore:
    """运行注册表与事件流的持久化实现。"""

    def __init__(self, db_path: str | Path = "data/zylo.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        # WAL：读写不互斥，进程崩溃后自动恢复已提交事务
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- runs ----

    def upsert_run(self, run: Run) -> None:
        """插入或更新运行记录；调用方负责状态迁移合法性（状态机在 Run 上）。"""
        self._conn.execute(
            """
            INSERT INTO runs (id, topic, status, schema_version, config_json,
                              created_at, started_at, finished_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                error = excluded.error
            """,
            (
                run.id,
                run.topic,
                run.status.value,
                run.schema_version,
                json.dumps(run.config, ensure_ascii=False),
                run.created_at.isoformat(),
                run.started_at.isoformat() if run.started_at else None,
                run.finished_at.isoformat() if run.finished_at else None,
                run.error,
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Run | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["config"] = json.loads(data.pop("config_json"))
        return Run.model_validate(data)

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[Run]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def runs_in_status(self, status: RunStatus) -> list[Run]:
        """按状态过滤（服务启动时扫描遗留 running 的恢复入口）。"""
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE status = ? ORDER BY created_at ASC",
            (status.value,),
        ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        data = dict(row)
        data["config"] = json.loads(data.pop("config_json"))
        return Run.model_validate(data)

    # ---- run_events ----

    def save_event(self, event: RunEvent) -> None:
        """落盘单个事件；(run_id, sequence) 主键天然去重，重复写不报错由调用方保证语义。"""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO run_events
                (run_id, sequence, schema_version, kind, name, status,
                 span_id, parent_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.run_id,
                event.sequence,
                event.schema_version,
                event.kind.value,
                event.name,
                event.status.value,
                event.span_id,
                event.parent_id,
                json.dumps(event.payload, ensure_ascii=False),
                event.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def last_sequence(self, run_id: str) -> int | None:
        """该 run 已落盘的最大 sequence；无事件返回 None。

        resume 换新 bus 实例时，内存计数器归零，必须以此续接序号，
        否则 (run_id, sequence) 主键会把第一轮事件 REPLACE 覆盖。
        """
        row = self._conn.execute(
            "SELECT MAX(sequence) AS max_seq FROM run_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return row["max_seq"] if row is not None else None

    def get_events(self, run_id: str, after_sequence: int = 0) -> list[RunEvent]:
        """按 sequence 升序返回事件，供 SSE 历史回放与 resume 重建上下文。"""
        rows = self._conn.execute(
            """
            SELECT * FROM run_events WHERE run_id = ? AND sequence > ?
            ORDER BY sequence ASC
            """,
            (run_id, after_sequence),
        ).fetchall()
        events = []
        for row in rows:
            data = dict(row)
            data["payload"] = json.loads(data.pop("payload_json"))
            events.append(RunEvent.model_validate(data))
        return events

    # ---- state_snapshots ----

    def save_state_snapshot(self, run_id: str, stage: str, state_payload: dict) -> None:
        """追加一条阶段快照；只增不改，resume 取最新一条即可。"""
        self._conn.execute(
            """
            INSERT INTO state_snapshots (run_id, stage, state_json, schema_version, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                stage,
                json.dumps(state_payload, ensure_ascii=False),
                state_payload.get("schema_version", 1),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def latest_state_snapshot(self, run_id: str) -> dict | None:
        """最新快照的完整载荷（含 schema_version）；无快照返回 None。"""
        row = self._conn.execute(
            """
            SELECT state_json FROM state_snapshots WHERE run_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["state_json"])

    # ---- review_decisions ----

    def save_review_decisions(self, decisions: list[ReviewDecision]) -> None:
        """批量落盘一轮人工决策（M3-3）；只增不改，作为后续评估数据。"""
        rows = [
            (
                d.run_id,
                d.revision,
                d.critique_id,
                d.action.value,
                d.edited_advice,
                d.reason,
                d.decided_at.isoformat(),
            )
            for d in decisions
        ]
        self._conn.executemany(
            """
            INSERT INTO review_decisions
                (run_id, revision, critique_id, action, edited_advice, reason, decided_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def list_review_decisions(self, run_id: str) -> list[ReviewDecision]:
        """该 run 的全部人工决策，按落库顺序返回。"""
        rows = self._conn.execute(
            "SELECT * FROM review_decisions WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [
            ReviewDecision(
                run_id=row["run_id"],
                revision=row["revision"],
                critique_id=row["critique_id"],
                action=ReviewAction(row["action"]),
                edited_advice=row["edited_advice"],
                reason=row["reason"],
                decided_at=datetime.fromisoformat(row["decided_at"]),
            )
            for row in rows
        ]

    # ---- 生命周期 ----

    def close(self) -> None:
        self._conn.close()
