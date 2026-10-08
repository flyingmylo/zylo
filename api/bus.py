"""进程内事件总线：RunEvent 的实时分发与短期回放。

事实来源是（M2 的）SQLite run_events 表；本总线只负责把新事件即时
推给在线订阅者，并按 run 保留一段内存缓冲，供 SSE 断线重连时以
Last-Event-ID（即 sequence）回放。进程重启后缓冲即失——历史回放的
权威实现由 M2 持久化接管，本模块不做落盘。
"""

import asyncio
import uuid
from collections import deque
from typing import Any

from src.events import (
    EventStatus,
    RunEvent,
    SpanKind,
    sanitize_payload,
)

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

    def __init__(self, history_limit: int = HISTORY_LIMIT) -> None:
        self._history: dict[str, deque[RunEvent]] = {}
        self._next_sequence: dict[str, int] = {}
        self._subscribers: dict[str, list[asyncio.Queue[RunEvent | object]]] = {}
        self._closed: set[str] = set()
        self._history_limit = history_limit

    async def emit(
        self,
        run_id: str,
        kind: SpanKind,
        name: str,
        status: EventStatus,
        parent_id: str | None = None,
        span_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        """发布事件：分配 sequence、生成 span_id、统一脱敏后广播。

        所有事件出口（SSE、未来的 JSONL 导出）都经过这里的
        sanitize_payload，保证脱敏规则只有一处实现。
        """
        event = RunEvent(
            sequence=self._next_sequence.setdefault(run_id, 1),
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
        return event

    def subscribe(self, run_id: str, after_sequence: int = 0) -> Subscription:
        """订阅某 run 的事件；先回放 sequence 之后的历史，再接续实时流。

        「登记订阅者 + 回放历史」在同一个无 await 的同步段内完成，
        而 emit 也全程同步，两者天然互斥，不存在回放与实时之间的缝隙。
        """
        queue: asyncio.Queue[RunEvent | object] = asyncio.Queue()
        subscribers = self._subscribers.setdefault(run_id, [])
        subscribers.append(queue)
        for event in self._history.get(run_id, deque()):
            if event.sequence > after_sequence:
                queue.put_nowait(event)
        # 流已终结的 run：回放完历史后立即结束，否则 SSE 客户端会永远等待
        if run_id in self._closed:
            queue.put_nowait(_CLOSE_SENTINEL)
        return Subscription(queue)

    def unsubscribe(self, run_id: str, subscription: Subscription) -> None:
        """移除订阅者；SSE 连接断开时必须调用，避免队列滞留。"""
        subscribers = self._subscribers.get(run_id, [])
        self._subscribers[run_id] = [q for q in subscribers if q is not subscription._queue]

    def close(self, run_id: str) -> None:
        """通知该 run 的所有订阅者流已终结（历史缓冲保留供后到者回放）。"""
        for queue in self._subscribers.get(run_id, []):
            queue.put_nowait(_CLOSE_SENTINEL)
        self._subscribers[run_id] = []
        self._closed.add(run_id)
