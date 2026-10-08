"""asyncio JobRunner：把写作运行变成可查询、可订阅的后台作业。

M1 是单机内存版：注册表与结果都只在进程内，重启即失；
M2 的 RunStore（SQLite）接入后接口保持不变。
"""

import asyncio
import logging
from typing import Any, Protocol

from api.bus import TraceBus
from src.events import EventStatus, RunEvent, SpanKind
from src.runs import Run, RunStatus
from src.state import WritingState

logger = logging.getLogger("src.api.runner")


class RunNotFoundError(KeyError):
    """查询的 run_id 不存在。"""


class RunExecutor(Protocol):
    """执行体的最小契约：现有 WritingOrchestrator 天然满足。"""

    async def execute(
        self,
        topic: str,
        local_files: list[str] | None = None,
        extra_instructions: str = "",
        output_dir: str = "output",
    ) -> WritingState: ...


class JobRunner:
    """运行注册表 + 后台执行调度，状态迁移与事件发布的唯一入口。"""

    def __init__(self, bus: TraceBus) -> None:
        self.bus: TraceBus = bus
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
        return run

    def get(self, run_id: str) -> Run:
        try:
            return self._runs[run_id]
        except KeyError:
            raise RunNotFoundError(run_id) from None

    def result(self, run_id: str) -> WritingState | None:
        """运行成功后的终态 WritingState；未完成或失败时为 None。"""
        return self._results.get(run_id)

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
        try:
            run.transition(RunStatus.RUNNING)
            await self._emit(run, EventStatus.STARTED)
            state = await executor.execute(
                topic=run.topic,
                local_files=run.config.get("sources"),
                extra_instructions=str(run.config.get("instructions", "")),
                output_dir=output_dir,
            )
            self._results[run.id] = state
            run.transition(RunStatus.COMPLETED)
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
            await self._emit(run, EventStatus.FAILED, payload={"error": "cancelled"})
            raise
        except Exception as exc:  # 作业边界：任何异常都转为 FAILED 终态
            logger.exception("run %s 执行失败", run.id)
            run.error = str(exc)
            run.transition(RunStatus.FAILED)
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
        return await self.bus.emit(
            run.id,
            kind=SpanKind.RUN,
            name="writing",
            status=status,
            span_id=f"run_{run.id}",
            payload=payload,
        )

    async def wait(self, run_id: str) -> None:
        """等待后台任务结束（测试与优雅停机用）。"""
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait_for(task, timeout=300)
