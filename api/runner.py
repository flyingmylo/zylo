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
from src.orchestrator import HumanReviewRequired
from src.runs import ReviewAction, ReviewDecision, Run, RunStatus
from src.state import WritingState, deserialize_state, serialize_state

logger = logging.getLogger("src.api.runner")


class RunNotFoundError(KeyError):
    """查询的 run_id 不存在。"""


class ReviewNotPendingError(ValueError):
    """决策到达时 run 并不处于等待人审状态。"""


class ReviewDecisionMismatchError(ValueError):
    """决策集与当前轮审稿意见不吻合（未全覆盖 / 含未知或重复 ID / 轮次不符）。"""


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
        # 人审停点时的 state 内存引用（快照已先行落盘，此为免反序列化加速层）
        self._pending_reviews: dict[str, WritingState] = {}

    # ---- 注册与查询 ----

    def create(
        self,
        topic: str,
        sources: list[str] | None = None,
        instructions: str = "",
        human_review: bool = False,
    ) -> Run:
        """登记一个 queued 状态的新运行（尚未启动）。"""
        run = Run(
            topic=topic,
            config={
                "sources": sources or [],
                "instructions": instructions,
                # executor_factory 据此为该 run 构造开启人审停点的编排器
                "human_review": human_review,
            },
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
        self,
        run_id: str,
        executor: RunExecutor,
        output_dir: str = "output",
        resume_state: WritingState | None = None,
    ) -> None:
        """把 queued 运行派发给后台任务；同一 run 重复 launch 视为编程错误。

        resume_state：人审决策注入后的 state（M3-3）——执行器据此跳过
        已完成阶段，直接消费决策结果续跑。
        """
        run = self.get(run_id)
        if run.id in self._tasks:
            raise ValueError(f"run {run_id} 已在执行中")
        self._tasks[run.id] = asyncio.create_task(
            self._execute(run, executor, output_dir, resume_state)
        )

    async def _execute(
        self,
        run: Run,
        executor: RunExecutor,
        output_dir: str,
        resume_state: WritingState | None = None,
    ) -> None:
        def on_checkpoint(state: WritingState) -> None:
            """阶段边界落快照：崩溃后 resume 从此续跑，已付的 LLM 费用不打水漂。"""
            if self.store:
                self.store.save_state_snapshot(
                    run.id, state.current_stage.value, serialize_state(state)
                )

        try:
            run.transition(RunStatus.RUNNING)
            self._persist(run)
            self._emit(run, EventStatus.STARTED)
            state = await executor.execute(
                topic=run.topic,
                local_files=run.config.get("sources"),
                extra_instructions=str(run.config.get("instructions", "")),
                output_dir=output_dir,
                resume_state=resume_state,
                on_checkpoint=on_checkpoint,
            )
            self._results[run.id] = state
            run.transition(RunStatus.COMPLETED)
            self._persist(run)
            self._emit(
                run,
                EventStatus.COMPLETED,
                payload={
                    "review_score": state.review_score,
                    "total_tokens": state.token_usage.get("total_tokens", 0),
                },
            )
        except HumanReviewRequired as exc:
            # 人审停点（M3-3）：不是失败，state 已随 checkpoint 落快照。
            # 流不终结——订阅者保持挂起，决策恢复后的新事件继续原流推送
            run.transition(RunStatus.WAITING_FOR_HUMAN_REVIEW)
            self._persist(run)
            self._pending_reviews[run.id] = exc.state
            self._emit(
                run,
                EventStatus.WAITING,
                payload={
                    "revision": exc.state.revision_count,
                    "score": exc.state.review_score,
                    "critiques": len(exc.state.actionable_revisions),
                },
            )
        except asyncio.CancelledError:
            run.error = "任务被取消"
            run.transition(RunStatus.CANCELLED)
            self._persist(run)
            self._emit(run, EventStatus.FAILED, payload={"error": "cancelled"})
            raise
        except BudgetExceededError as exc:
            # 预算耗尽是"资源不足"而非"执行出错"：落位 PARTIAL，
            # 快照已保留，调大预算后 zylo resume 即可续跑
            logger.warning("run %s 预算耗尽：%s", run.id, exc)
            run.error = f"预算耗尽：{exc}"
            run.transition(RunStatus.PARTIAL)
            self._persist(run)
            self._emit(
                run,
                EventStatus.FAILED,
                payload={"error": f"budget_exceeded: {exc}"},
            )
        except Exception as exc:  # 作业边界：任何异常都转为 FAILED 终态
            logger.exception("run %s 执行失败", run.id)
            run.error = str(exc)
            run.transition(RunStatus.FAILED)
            self._persist(run)
            self._emit(
                run,
                EventStatus.FAILED,
                payload={"error": f"{type(exc).__name__}: {exc}"},
            )
        finally:
            # 任务结束即从调度表摘除：人审决策恢复要对同一 run 重新 launch
            self._tasks.pop(run.id, None)
            # 等待人审不是流终结（run 仍活着，决策后原订阅继续收新事件）；
            # 其余终态才 close，SSE 客户端据此收尾
            if run.status is not RunStatus.WAITING_FOR_HUMAN_REVIEW:
                self.bus.close(run.id)

    def _emit(
        self, run: Run, status: EventStatus, payload: dict[str, Any] | None = None
    ) -> RunEvent:
        # 落盘由 bus 挂载的 store 自动完成（emit 内统一双写），
        # 这里只负责 run 层事件本身
        return self.bus.emit(
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

    # ---- 人工审稿决策（M3-3） ----

    def pending_review(self, run_id: str) -> WritingState | None:
        """等待人审的 run 的停点 state：内存优先，miss 回落最新快照。"""
        state = self._pending_reviews.get(run_id)
        if state is not None:
            return state
        if self.store:
            payload = self.store.latest_state_snapshot(run_id)
            if payload is not None:
                return deserialize_state(payload)
        return None

    def apply_review_decisions(
        self,
        run_id: str,
        decisions: list[ReviewDecision],
        executor_factory: Callable[[Run], RunExecutor],
        output_dir: str = "output",
    ) -> Run:
        """校验并注入人工决策，随后恢复执行（POST /review-decisions 的核心）。

        决策必须恰好覆盖当前轮全部意见（防止漏决策静默通过）；
        注入规则：ACCEPT 保留、EDIT 换写建议、REJECT 移除、
        APPROVE_FINAL 或全拒 → 视为拍板定稿（review_passed=True）。
        """
        run = self.get(run_id)
        if run.status is not RunStatus.WAITING_FOR_HUMAN_REVIEW:
            raise ReviewNotPendingError(f"运行不在等待人审状态（当前 {run.status.value}）")

        state = self.pending_review(run_id)
        if state is None:
            raise ReviewNotPendingError("找不到停点状态（快照缺失）")

        current_ids = [rev["critique_id"] for rev in state.actionable_revisions]
        decision_ids = [d.critique_id for d in decisions]
        if (
            sorted(decision_ids) != sorted(current_ids)
            or len(set(decision_ids)) != len(decision_ids)
            or any(d.revision != state.revision_count for d in decisions)
        ):
            raise ReviewDecisionMismatchError(
                f"决策须恰好覆盖第 {state.revision_count} 轮的 "
                f"{len(current_ids)} 条意见（收到 {len(decisions)} 条，"
                "每条意见一个决策，不允许遗漏或重复）"
            )

        by_id = {d.critique_id: d for d in decisions}
        if any(d.action is ReviewAction.APPROVE_FINAL for d in decisions):
            state.review_passed = True
        else:
            kept: list[dict[str, str]] = []
            for rev in state.actionable_revisions:
                decision = by_id[rev["critique_id"]]
                if decision.action is ReviewAction.ACCEPT:
                    kept.append(rev)
                elif decision.action is ReviewAction.EDIT:
                    kept.append({**rev, "advice": decision.edited_advice or rev["advice"]})
                # REJECT：丢弃该意见（理由已随决策落库）
            state.actionable_revisions = kept
            # 全部拒绝 = 人认可当前稿，等价拍板定稿
            state.review_passed = not kept

        if self.store:
            self.store.save_review_decisions(decisions)

        self._pending_reviews.pop(run_id, None)
        # WAITING→REVISING→(launch 后)RUNNING：轨迹贴合 PLAN 状态图
        run.transition(RunStatus.REVISING)
        self._persist(run)
        self.launch(
            run_id, executor_factory(run), output_dir=output_dir, resume_state=state
        )
        return run


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
