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
from pathlib import Path

from src.events import RunEvent
from src.runs import Run

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
        runs = []
        for row in rows:
            data = dict(row)
            data["config"] = json.loads(data.pop("config_json"))
            runs.append(Run.model_validate(data))
        return runs

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

    # ---- 生命周期 ----

    def close(self) -> None:
        self._conn.close()
