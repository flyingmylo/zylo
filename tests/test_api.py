import asyncio

from httpx import ASGITransport, AsyncClient

from api.app import create_app
from api.bus import TraceBus
from api.runner import JobRunner
from src.state import WritingState


class StubExecutor:
    def __init__(self, state: WritingState | None = None):
        self.state = state or WritingState(topic="t")

    async def execute(self, topic, local_files=None, extra_instructions="", output_dir="output"):
        self.state.outline_title = f"关于{topic}的深度解析"
        self.state.final_markdown = f"# 关于{topic}的深度解析\n\n正文"
        self.state.review_score = 90.0
        self.state.revision_count = 0
        return self.state


def _make_client(state: WritingState | None = None):
    bus = TraceBus()
    runner = JobRunner(bus)
    app = create_app(bus=bus, runner=runner, executor_factory=lambda: StubExecutor(state))
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), runner


async def test_create_run_returns_201_and_run_payload():
    client, runner = _make_client()

    resp = await client.post(
        "/api/runs",
        json={"topic": "KV-Cache", "sources": ["references/yoco.pdf"], "instructions": "面向工程师"},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] in ("queued", "running")
    assert body["config"]["sources"] == ["references/yoco.pdf"]
    await runner.wait(body["id"])


async def test_create_run_rejects_empty_topic():
    client, runner = _make_client()

    resp = await client.post("/api/runs", json={"topic": ""})

    assert resp.status_code == 422
    await runner.wait((await client.post("/api/runs", json={"topic": "x"})).json()["id"])


async def test_get_run_and_404():
    client, runner = _make_client()

    created = (await client.post("/api/runs", json={"topic": "x"})).json()
    await runner.wait(created["id"])

    ok = await client.get(f"/api/runs/{created['id']}")
    assert ok.status_code == 200
    assert ok.json()["status"] == "completed"

    missing = await client.get("/api/runs/nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "run_not_found"


async def test_article_after_completion():
    client, runner = _make_client()

    created = (await client.post("/api/runs", json={"topic": "MoE 路由"})).json()
    await runner.wait(created["id"])

    resp = await client.get(f"/api/runs/{created['id']}/article")

    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "关于MoE 路由的深度解析"
    assert body["markdown"].startswith("# 关于MoE 路由")
    assert body["review_score"] == 90.0


async def test_article_rejects_incomplete_run():
    bus = TraceBus()
    runner = JobRunner(bus)
    release = asyncio.Event()

    class SlowExecutor(StubExecutor):
        async def execute(self, *args, **kwargs):
            await release.wait()
            return await super().execute(*args, **kwargs)

    app = create_app(bus=bus, runner=runner, executor_factory=SlowExecutor)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = (await client.post("/api/runs", json={"topic": "慢任务"})).json()
        try:
            resp = await client.get(f"/api/runs/{created['id']}/article")
            assert resp.status_code == 409
            assert resp.json()["error"]["code"] == "run_not_completed"
        finally:
            release.set()
            await runner.wait(created["id"])


async def test_sse_replays_events_with_id_and_data():
    client, runner = _make_client()

    created = (await client.post("/api/runs", json={"topic": "SSE 演示"})).json()
    await runner.wait(created["id"])

    resp = await client.get(f"/api/runs/{created['id']}/events")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    # M0 契约：id=sequence、event=span kind、data=完整 RunEvent JSON
    assert "id: 1" in text and "id: 2" in text
    assert "event: run" in text
    assert '"status": "started"' in text or '"status":"started"' in text
    assert '"kind": "run"' in text or '"kind":"run"' in text


async def test_sse_respects_after_parameter():
    client, runner = _make_client()

    created = (await client.post("/api/runs", json={"topic": "回放"})).json()
    await runner.wait(created["id"])

    resp = await client.get(f"/api/runs/{created['id']}/events?after=1")

    # 只回放 sequence > 1 的事件
    assert "id: 1\n" not in resp.text
    assert "id: 2" in resp.text


async def test_sse_unknown_run_404():
    client, _ = _make_client()

    resp = await client.get("/api/runs/nope/events")

    assert resp.status_code == 404


async def test_healthz():
    client, _runner = _make_client()
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
