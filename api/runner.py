"""asyncio JobRunner：把写作运行变成可查询、可订阅的后台作业。

挂载 RunStore 后，运行与事件在内存之外同步落盘（SQLite 是事实来源，
内存只是加速层）：进程重启后 run 列表与详情仍可查询。
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Protocol

from api.bus import TraceBus
from api.store import RunStore
from src.budget import BudgetExceededError
from src.events import EventStatus, RunEvent, SpanKind
from src.runs import Run, RunStatus
from src.state import WritingState, serialize_state

logger = logging.getLogger("src.api.runner")


class RunNotFoundError(KeyError):
    """查询的 run_id 不存在。"""


class RunExecutor(Protocol):
    """执行体的最小契约：现有 WritingOrchestrator 天然满足。

    参数排布必须与 WritingOrchestrator.execute 完全一致——
    pyright 的协议匹配按位置参数对齐，多一个可选参数就会错位。
    """

    async def execute(
        self,
        topic: str,
        local_files: list[str] | None = None,
        extra_instructions: str = "",
        output_dir: str = "output",
        resume_state: WritingState | None = None,
        on_checkpoint: Callable[[WritingState], None] | None = None,
    ) -> WritingState: ...


class JobRunner:
    """运行注册表 + 后台执行调度，状态迁移与事件发布的唯一入口。"""

    def __init__(self, bus: TraceBus, store: RunStore | None = None) -> None:
        self.bus: TraceBus = bus
        self.store = store
        self._runs: dict[str, Run] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._results: dict[str, WritingState] = {}

    # ---- 注册与查询 ----

    def create(
        self,
        topic: str,
        sources: list[str] | None = None,
        instructions: str = "",
    ) -> Run:
        """登记一个 queued 状态的新运行（尚未启动）。"""
        run = Run(
            topic=topic,
            config={"sources": sources or [], "instructions": instructions},
        )
        self._runs[run.id] = run
        self._persist(run)
        return run

    def get(self, run_id: str) -> Run:
        """内存优先，miss 时回落 SQLite（重启后的历史 run）并回填内存。"""
        if run_id in self._runs:
            return self._runs[run_id]
        if self.store:
            persisted = self.store.get_run(run_id)
            if persisted is not None:
                self._runs[run_id] = persisted
                return persisted
        raise RunNotFoundError(run_id)

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[Run]:
        """运行列表：有 store 时以持久层为准（跨重启），否则退回内存倒序。"""
        if self.store:
            return self.store.list_runs(limit=limit, offset=offset)
        ordered = sorted(
            self._runs.values(), key=lambda r: (r.created_at, r.id), reverse=True
        )
        return ordered[offset : offset + limit]

    def result(self, run_id: str) -> WritingState | None:
        """运行成功后的终态 WritingState；未完成或失败时为 None。"""
        return self._results.get(run_id)

    def _persist(self, run: Run) -> None:
        if self.store:
            self.store.upsert_run(run)

    # ---- 执行 ----

    def launch(
        self, run_id: str, executor: RunExecutor, output_dir: str = "output"
    ) -> None:
        """把 queued 运行派发给后台任务；同一 run 重复 launch 视为编程错误。"""
        run = self.get(run_id)
        if run.id in self._tasks:
            raise ValueError(f"run {run_id} 已在执行中")
        self._tasks[run.id] = asyncio.create_task(
            self._execute(run, executor, output_dir)
        )

    async def _execute(self, run: Run, executor: RunExecutor, output_dir: str) -> None:
        def on_checkpoint(state: WritingState) -> None:
            """阶段边界落快照：崩溃后 resume 从此续跑，已付的 LLM 费用不打水漂。"""
            if self.store:
                self.store.save_state_snapshot(
                    run.id, state.current_stage.value, serialize_state(state)
                )

        try:
            run.transition(RunStatus.RUNNING)
            self._persist(run)
            await self._emit(run, EventStatus.STARTED)
            state = await executor.execute(
                topic=run.topic,
                local_files=run.config.get("sources"),
                extra_instructions=str(run.config.get("instructions", "")),
                output_dir=output_dir,
                on_checkpoint=on_checkpoint,
            )
            self._results[run.id] = state
            run.transition(RunStatus.COMPLETED)
            self._persist(run)
            await self._emit(
                run,
                EventStatus.COMPLETED,
                payload={
                    "review_score": state.review_score,
                    "total_tokens": state.token_usage.get("total_tokens", 0),
                },
            )
        except asyncio.CancelledError:
            run.error = "任务被取消"
            run.transition(RunStatus.CANCELLED)
            self._persist(run)
            await self._emit(run, EventStatus.FAILED, payload={"error": "cancelled"})
            raise
        except BudgetExceededError as exc:
            # 预算耗尽是"资源不足"而非"执行出错"：落位 PARTIAL，
            # 快照已保留，调大预算后 zylo resume 即可续跑
            logger.warning("run %s 预算耗尽：%s", run.id, exc)
            run.error = f"预算耗尽：{exc}"
            run.transition(RunStatus.PARTIAL)
            self._persist(run)
            await self._emit(
                run,
                EventStatus.FAILED,
                payload={"error": f"budget_exceeded: {exc}"},
            )
        except Exception as exc:  # 作业边界：任何异常都转为 FAILED 终态
            logger.exception("run %s 执行失败", run.id)
            run.error = str(exc)
            run.transition(RunStatus.FAILED)
            self._persist(run)
            await self._emit(
                run,
                EventStatus.FAILED,
                payload={"error": f"{type(exc).__name__}: {exc}"},
            )
        finally:
            # 无论成败都终结订阅流，SSE 客户端据此收尾
            self.bus.close(run.id)

    async def _emit(
        self, run: Run, status: EventStatus, payload: dict[str, Any] | None = None
    ) -> RunEvent:
        event = await self.bus.emit(
            run.id,
            kind=SpanKind.RUN,
            name="writing",
            status=status,
            span_id=f"run_{run.id}",
            payload=payload,
        )
        # 事件双写：SQLite 是事实来源（重启后可回放），总线只负责实时分发。
        # 同步写 SQLite 在微秒级，不值得为此引入异步驱动
        if self.store:
            self.store.save_event(event)
        return event

    async def wait(self, run_id: str) -> None:
        """等待后台任务结束（测试与优雅停机用）。"""
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait_for(task, timeout=300)


def recover_stale_runs(store: RunStore) -> list[Run]:
    """服务启动恢复：上一进程遗留的 RUNNING 已无宿主任务，统一落位 PARTIAL。

    返回被恢复的 run 列表供上层提示用户；PARTIAL 可经 zylo resume 续跑。
    """
    recovered: list[Run] = []
    for stale in store.runs_in_status(RunStatus.RUNNING):
        stale.transition(RunStatus.PARTIAL)
        store.upsert_run(stale)
        recovered.append(stale)
    return recovered
