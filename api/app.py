"""FastAPI 应用工厂：REST 端点与 SSE 事件流（契约见 docs/api-contract.md）。

CLI 与 API 共用同一个 Orchestrator：应用工厂只做装配，
executor 由外部注入（M1-4 起接真实/Mock Orchestrator）。
"""

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api.bus import TraceBus
from api.runner import JobRunner, RunExecutor, RunNotFoundError
from api.schema import ArticleResponse, RunCreateRequest
from src.runs import RunStatus

API_PREFIX = "/api"


def create_app(
    bus: TraceBus,
    runner: JobRunner,
    executor_factory: Callable[[], RunExecutor],
) -> FastAPI:
    """装配应用；bus/runner/executor_factory 挂到 app.state 供路由访问。"""
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

    @app.post(f"{API_PREFIX}/runs", status_code=201)
    async def create_run(body: RunCreateRequest, request: Request) -> dict:
        runner_: JobRunner = request.app.state.runner
        run = runner_.create(
            topic=body.topic, sources=body.sources, instructions=body.instructions
        )
        runner_.launch(run.id, request.app.state.executor_factory())
        return run.model_dump(mode="json")

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

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/events")
    async def stream_events(run_id: str, request: Request, after: int = 0) -> StreamingResponse:
        bus: TraceBus = request.app.state.bus
        runner_: JobRunner = request.app.state.runner
        runner_.get(run_id)  # 404 校验
        # SSE 断线重连标准头优先，query 参数作显式覆盖
        last_event_id = request.headers.get("last-event-id")
        after_sequence = int(last_event_id) if last_event_id else after
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
