"""进程内事件总线：RunEvent 的实时分发与短期回放。

事实来源是 SQLite run_events 表：bus 挂载 RunStore 后每次 emit 自动
双写（实时广播 + 落盘），SSE 断线重连与 zylo trace 都以此为准。
emit 刻意设计为同步方法——内部只有内存操作与微秒级的 SQLite 写，
同步化让 span 的 start/finish 可以在任意调用点发射，不必把 async
传染到 Agent 内部。
"""

import asyncio
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

from src.events import (
    EventStatus,
    RunEvent,
    SpanKind,
    sanitize_payload,
)
from src.runs import RunStatus

if TYPE_CHECKING:
    from api.store import RunStore

# 每 run 的内存缓冲上限：单机单人场景一次运行的事件量远低于此，
# 超出时丢弃最旧事件（回放只能覆盖近期），防止长驻进程内存缓慢增长
HISTORY_LIMIT = 2000

_CLOSE_SENTINEL = object()


class Subscription:
    """单个订阅者的异步迭代器；bus.close(run_id) 后收尾结束。"""

    def __init__(self, queue: asyncio.Queue[RunEvent | object]):
        self._queue = queue

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> RunEvent:
        item = await self._queue.get()
        # 队列里除事件外只有关闭哨兵；isinstance 能让类型检查器把
        # RunEvent | object 收窄成 RunEvent，is 比对做不到
        if not isinstance(item, RunEvent):
            raise StopAsyncIteration from None
        return item


class TraceBus:
    """按 run 隔离的事件发布/订阅中心，sequence 分配的唯一权威。"""

    def __init__(
        self,
        history_limit: int = HISTORY_LIMIT,
        store: "RunStore | None" = None,
    ) -> None:
        self._history: dict[str, deque[RunEvent]] = {}
        self._next_sequence: dict[str, int] = {}
        self._subscribers: dict[str, list[asyncio.Queue[RunEvent | object]]] = {}
        self._closed: set[str] = set()
        self._history_limit = history_limit
        self._store = store

    def emit(
        self,
        run_id: str,
        kind: SpanKind,
        name: str,
        status: EventStatus,
        parent_id: str | None = None,
        span_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        """发布事件：分配 sequence、生成 span_id、统一脱敏后广播并落盘。

        所有事件出口（SSE、JSONL 导出、zylo trace）都经过这里的
        sanitize_payload，保证脱敏规则只有一处实现。
        """
        event = RunEvent(
            sequence=self._base_sequence(run_id),
            run_id=run_id,
            kind=kind,
            name=name,
            status=status,
            span_id=span_id or uuid.uuid4().hex[:8],
            parent_id=parent_id,
            payload=sanitize_payload(payload or {}),
        )
        self._next_sequence[run_id] = event.sequence + 1

        history = self._history.setdefault(
            run_id, deque(maxlen=self._history_limit)
        )
        history.append(event)
        for queue in self._subscribers.get(run_id, []):
            # 单机个位数订阅者、每事件一次入队，同步 put 不会阻塞事件循环
            queue.put_nowait(event)
        if self._store is not None:
            self._store.save_event(event)
        return event

    def _base_sequence(self, run_id: str) -> int:
        """该 run 的下一个 sequence：内存计数器优先，首次见到该 run 时
        以 SQLite 已落盘的最大序号续接。

        resume 场景的 bus 是全新实例，若计数器从 1 重计，落盘会按
        (run_id, sequence) 主键 REPLACE 掉第一轮的同序号事件——历史被篡改。
        """
        if run_id in self._next_sequence:
            return self._next_sequence[run_id]
        last = self._store.last_sequence(run_id) if self._store else None
        self._next_sequence[run_id] = last + 1 if last is not None else 1
        return self._next_sequence[run_id]

    def subscribe(self, run_id: str, after_sequence: int = 0) -> Subscription:
        """订阅某 run 的事件；先回放 sequence 之后的历史，再接续实时流。

        回放的事实来源是 SQLite（挂载 store 时，ADR-002）：跨进程重连与
        内存缓冲溢出都由它兜底；未挂 store 的纯内存用法退回内存缓冲。
        「登记订阅者 + 回放历史」与 emit 同为无 await 的同步方法，
        单线程事件循环里天然互斥，回放与实时流之间不存在缝隙。
        """
        queue: asyncio.Queue[RunEvent | object] = asyncio.Queue()
        subscribers = self._subscribers.setdefault(run_id, [])
        subscribers.append(queue)
        if self._store is not None:
            replay = self._store.get_events(run_id, after_sequence)
        else:
            replay = [
                e
                for e in self._history.get(run_id, ())
                if e.sequence > after_sequence
            ]
        for event in replay:
            queue.put_nowait(event)
        if self._stream_ended(run_id):
            queue.put_nowait(_CLOSE_SENTINEL)
        return Subscription(queue)

    def _stream_ended(self, run_id: str) -> bool:
        """本次订阅是否应在回放后立即终结。

        两种情况：本进程 close 过该 run；或 SQLite 里 run 已落入终态/PARTIAL
        ——跨进程视角（服务重启后 _closed 为空），只能靠持久化状态判断。
        PARTIAL 也视为流终结：当前执行已停摆，resume 追加的新事件由客户端
        下次带 Last-Event-ID 重连时从 SQLite 增量拉取。
        """
        if run_id in self._closed:
            return True
        if self._store is None:
            return False
        run = self._store.get_run(run_id)
        return run is not None and (
            run.status.is_terminal or run.status is RunStatus.PARTIAL
        )

    def unsubscribe(self, run_id: str, subscription: Subscription) -> None:
        """移除订阅者；SSE 连接断开时必须调用，避免队列滞留。"""
        subscribers = self._subscribers.get(run_id, [])
        self._subscribers[run_id] = [
            q for q in subscribers if q is not subscription._queue
        ]

    def close(self, run_id: str) -> None:
        """通知该 run 的所有订阅者流已终结（历史缓冲保留供后到者回放）。"""
        for queue in self._subscribers.get(run_id, []):
            queue.put_nowait(_CLOSE_SENTINEL)
        self._subscribers[run_id] = []
        self._closed.add(run_id)


class TraceEmitter:
    """RunTrace 协议的总线实现：以 span 为单位向 bus 发射五层事件。

    span 采用显式 start/finish 而非上下文管理器：Agent 内部的
    _chat/_execute_tool 需要在 await 点之间持有 span_id，显式传递
    比嵌套作用域更直白。
    """

    def __init__(self, bus: TraceBus, run_id: str) -> None:
        self._bus = bus
        self._run_id = run_id
        # span_id -> (kind, name, parent_id)：finish 事件必须携带与 start
        # 完全一致的 kind/name/parent，SSE 消费方按 kind 分通道渲染、
        # 按 parent_id 还原调用树，缺失任何一项都会破坏时间线结构
        self._spans: dict[str, tuple[SpanKind, str, str | None]] = {}

    def start_span(
        self,
        kind: SpanKind,
        name: str,
        parent_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        span_id = uuid.uuid4().hex[:8]
        self._spans[span_id] = (kind, name, parent_id)
        self._bus.emit(
            self._run_id,
            kind=kind,
            name=name,
            status=EventStatus.STARTED,
            span_id=span_id,
            parent_id=parent_id,
            payload=payload,
        )
        return span_id

    def finish_span(
        self,
        span_id: str,
        status: EventStatus = EventStatus.COMPLETED,
        payload: dict[str, Any] | None = None,
    ) -> None:
        kind, name, parent_id = self._spans.pop(
            span_id, (SpanKind.RUN, "unknown", None)
        )
        self._bus.emit(
            self._run_id,
            kind=kind,
            name=name,
            status=status,
            span_id=span_id,
            parent_id=parent_id,
            payload=payload,
        )
