"""FastAPI 应用工厂：REST 端点与 SSE 事件流（契约见 docs/api-contract.md）。

CLI 与 API 共用同一个 Orchestrator：应用工厂只做装配，
executor 由外部注入（M1-4 起接真实/Mock Orchestrator）。
"""

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api.bus import TraceBus
from api.runner import (
    JobRunner,
    ReviewDecisionMismatchError,
    ReviewNotPendingError,
    RunExecutor,
    RunNotFoundError,
)
from api.schema import (
    ArticleResponse,
    ReviewDecisionsRequest,
    ReviewOpinion,
    ReviewResponse,
    RunCreateRequest,
)
from src.runs import ReviewDecision, Run, RunStatus

API_PREFIX = "/api"


def create_app(
    bus: TraceBus,
    runner: JobRunner,
    executor_factory: Callable[[Run], RunExecutor],
) -> FastAPI:
    """装配应用；bus/runner/executor_factory 挂到 app.state 供路由访问。

    executor_factory 以 Run 为参数：执行体可按 run 决定持久化目录
    （如 data/chroma/{run.id}），实现按 run 隔离的知识库。
    """
    app = FastAPI(title="zylo", version="0.1.0")
    app.state.bus = bus
    app.state.runner = runner
    app.state.executor_factory = executor_factory

    @app.exception_handler(RunNotFoundError)
    async def _run_not_found(_: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "run_not_found", "message": f"运行 {exc.args[0]} 不存在"}},
        )

    @app.exception_handler(ReviewNotPendingError)
    async def _review_not_pending(_: Request, exc: ReviewNotPendingError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": {"code": "review_not_pending", "message": str(exc)}},
        )

    @app.exception_handler(ReviewDecisionMismatchError)
    async def _decision_mismatch(_: Request, exc: ReviewDecisionMismatchError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "decision_mismatch", "message": str(exc)}},
        )

    @app.post(f"{API_PREFIX}/runs", status_code=201)
    async def create_run(body: RunCreateRequest, request: Request) -> dict:
        runner_: JobRunner = request.app.state.runner
        run = runner_.create(
            topic=body.topic,
            sources=body.sources,
            instructions=body.instructions,
            human_review=body.human_review,
        )
        runner_.launch(run.id, request.app.state.executor_factory(run))
        return run.model_dump(mode="json")

    @app.get(f"{API_PREFIX}/runs")
    async def list_runs(request: Request, limit: int = 50, offset: int = 0) -> dict:
        runner_: JobRunner = request.app.state.runner
        runs = runner_.list_runs(limit=limit, offset=offset)
        return {
            "items": [run.model_dump(mode="json") for run in runs],
            "limit": limit,
            "offset": offset,
        }

    @app.get(f"{API_PREFIX}/runs/{{run_id}}")
    async def get_run(run_id: str, request: Request) -> dict:
        run = request.app.state.runner.get(run_id)
        return run.model_dump(mode="json")

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/article")
    async def get_article(run_id: str, request: Request) -> ArticleResponse:
        runner_ = request.app.state.runner
        run = runner_.get(run_id)
        if run.status is not RunStatus.COMPLETED:
            # 409 的错误体走统一 error 契约，与声明的 ArticleResponse 不同源；
            # 忽略注释必须放在 return 行——诊断定位在表达式起始处
            return JSONResponse(  # pyright: ignore[reportReturnType]
                status_code=409,
                content={
                    "error": {
                        "code": "run_not_completed",
                        "message": f"运行尚未完成（当前状态 {run.status.value}）",
                    }
                },
            )
        state = runner_.result(run_id)
        # 状态机保证 COMPLETED 必有结果；缺失说明内部状态被绕路篡改
        assert state is not None
        return ArticleResponse(
            title=state.outline_title or run.topic,
            markdown=state.final_markdown,
            review_score=state.review_score,
            revision_count=state.revision_count,
        )

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/review")
    async def get_review(run_id: str, request: Request) -> ReviewResponse:
        """当前轮审稿结果与待决策意见（仅 WAITING 状态可查）。"""
        runner_: JobRunner = request.app.state.runner
        run = runner_.get(run_id)  # 404 校验
        if run.status is not RunStatus.WAITING_FOR_HUMAN_REVIEW:
            return JSONResponse(  # pyright: ignore[reportReturnType]
                status_code=409,
                content={
                    "error": {
                        "code": "review_not_pending",
                        "message": f"运行不在等待人审状态（当前 {run.status.value}）",
                    }
                },
            )
        state = runner_.pending_review(run_id)
        assert state is not None, "WAITING 状态必有停点 state（内存或快照）"
        return ReviewResponse(
            revision=state.revision_count,
            score=state.review_score,
            passed=state.review_passed,
            opinions=[
                ReviewOpinion(
                    critique_id=rev["critique_id"],
                    scope=rev["section"],
                    advice=rev["advice"],
                )
                for rev in state.actionable_revisions
            ],
        )

    @app.post(f"{API_PREFIX}/runs/{{run_id}}/review-decisions", status_code=202)
    async def submit_review_decisions(
        run_id: str, body: ReviewDecisionsRequest, request: Request
    ) -> dict:
        """注入人工决策并恢复执行；决策须恰好覆盖当前轮全部意见。"""
        runner_: JobRunner = request.app.state.runner
        pending = runner_.pending_review(run_id)
        revision = pending.revision_count if pending else 0
        decisions = [
            ReviewDecision(
                run_id=run_id,
                revision=revision,
                critique_id=item.critique_id,
                action=item.action,
                edited_advice=item.edited_advice,
                reason=item.reason,
            )
            for item in body.items
        ]
        run = runner_.apply_review_decisions(
            run_id, decisions, request.app.state.executor_factory
        )
        return {"run_id": run.id, "status": run.status.value}

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/events")
    async def stream_events(run_id: str, request: Request, after: int = 0) -> StreamingResponse:
        bus: TraceBus = request.app.state.bus
        runner_: JobRunner = request.app.state.runner
        runner_.get(run_id)  # 404 校验
        # SSE 断线重连标准头优先，query 参数作显式覆盖；
        # 畸形 last-event-id 不按 500 处理，回退到 query 参数的回放起点
        last_event_id = request.headers.get("last-event-id")
        try:
            after_sequence = int(last_event_id) if last_event_id else after
        except ValueError:
            after_sequence = after
        subscription = bus.subscribe(run_id, after_sequence=after_sequence)

        async def event_stream():
            try:
                async for event in subscription:
                    yield (
                        f"id: {event.sequence}\n"
                        f"event: {event.kind.value}\n"
                        f"data: {event.to_json()}\n\n"
                    )
            finally:
                # 客户端断开或流结束时回收订阅，避免队列滞留
                bus.unsubscribe(run_id, subscription)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app
