import asyncio

from httpx import ASGITransport, AsyncClient

from api.app import create_app
from api.bus import TraceBus
from api.runner import JobRunner
from api.store import RunStore
from src.state import WritingState


class StubExecutor:
    def __init__(self, state: WritingState | None = None):
        self.state = state or WritingState(topic="t")

    async def execute(
        self,
        topic,
        local_files=None,
        extra_instructions="",
        output_dir="output",
        resume_state=None,
        on_checkpoint=None,
    ):
        self.state.outline_title = f"关于{topic}的深度解析"
        self.state.final_markdown = f"# 关于{topic}的深度解析\n\n正文"
        self.state.review_score = 90.0
        self.state.revision_count = 0
        return self.state


def _make_client(state: WritingState | None = None):
    bus = TraceBus()
    runner = JobRunner(bus)
    app = create_app(bus=bus, runner=runner, executor_factory=lambda run: StubExecutor(state))
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

    app = create_app(bus=bus, runner=runner, executor_factory=lambda run: SlowExecutor())
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


# --------------------------------------------------------------------------
# M2-2：SQLite 事实来源——服务重启后数据仍在
# --------------------------------------------------------------------------


async def test_runs_survive_service_restart(tmp_path):
    """重启模拟：完成一次运行后，用同一 db 文件装配全新的 bus/runner/app，
    历史运行的列表、详情与事件历史都必须可见。"""
    db_path = tmp_path / "zylo.db"

    # bus 挂载 store：事件在 emit 内统一双写（实时广播 + 落盘）
    bus1 = TraceBus(store=RunStore(db_path))
    runner1 = JobRunner(bus1, store=RunStore(db_path))
    app1 = create_app(bus=bus1, runner=runner1, executor_factory=lambda run: StubExecutor())
    async with AsyncClient(transport=ASGITransport(app=app1), base_url="http://t1") as c1:
        created = (await c1.post("/api/runs", json={"topic": "重启幸存者"})).json()
        await runner1.wait(created["id"])
        assert (await c1.get(f"/api/runs/{created['id']}")).json()["status"] == "completed"

    # 全新进程的内存状态：只有同一个 SQLite 文件
    bus2 = TraceBus()
    runner2 = JobRunner(bus2, store=RunStore(db_path))
    app2 = create_app(bus=bus2, runner=runner2, executor_factory=lambda run: StubExecutor())
    async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t2") as c2:
        listing = (await c2.get("/api/runs")).json()
        assert [item["id"] for item in listing["items"]] == [created["id"]]

        detail = (await c2.get(f"/api/runs/{created['id']}")).json()
        assert detail["status"] == "completed"
        assert detail["topic"] == "重启幸存者"

        # 事件历史可从 SQLite 回放（即使内存总线里没有这个 run）
        store2 = RunStore(db_path)
        events = store2.get_events(created["id"])
        assert [e.sequence for e in events] == [1, 2]
        assert events[-1].status.value == "completed"


# --------------------------------------------------------------------------
# M3-2：SSE 历史回放接 SQLite——断线重连跨进程可续
# --------------------------------------------------------------------------


def _sse_ids(text: str) -> list[int]:
    """从 SSE 响应体提取全部事件 id（= sequence）。"""
    return [
        int(line.split(": ")[1])
        for line in text.splitlines()
        if line.startswith("id: ")
    ]


async def test_sse_replay_after_restart_with_last_event_id(tmp_path):
    """重启后带 Last-Event-ID 重连：增量历史从 SQLite 回放，流正常终结。

    修复前新进程的内存缓冲与 _closed 都是空的——客户端收不到历史，
    且因为 run 已终结、永远不会有人 close，连接无限挂死。
    """
    db_path = tmp_path / "zylo.db"
    store1 = RunStore(db_path)
    bus1 = TraceBus(store=store1)
    runner1 = JobRunner(bus1, store=store1)
    app1 = create_app(bus=bus1, runner=runner1, executor_factory=lambda run: StubExecutor())
    async with AsyncClient(transport=ASGITransport(app=app1), base_url="http://t1") as c1:
        created = (await c1.post("/api/runs", json={"topic": "断线重连"})).json()
        await runner1.wait(created["id"])
        first = await c1.get(f"/api/runs/{created['id']}/events")
        total = max(_sse_ids(first.text))

    # 全新进程的内存状态：只有同一个 SQLite 文件；模拟客户端已收到 total-1 条
    store2 = RunStore(db_path)
    bus2 = TraceBus(store=store2)
    runner2 = JobRunner(bus2, store=store2)
    app2 = create_app(bus=bus2, runner=runner2, executor_factory=lambda run: StubExecutor())
    async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t2") as c2:
        resp = await c2.get(
            f"/api/runs/{created['id']}/events",
            headers={"last-event-id": str(total - 1)},
        )
        assert resp.status_code == 200
        # 增量恰好一条、无重复；请求正常返回本身证明流已终结（挂死会读超时）
        assert _sse_ids(resp.text) == [total]


async def test_sse_malformed_last_event_id_falls_back_to_full_replay():
    """畸形 last-event-id 不触发 500，按无回放起点全量回放。"""
    client, runner = _make_client()

    created = (await client.post("/api/runs", json={"topic": "畸形头"})).json()
    await runner.wait(created["id"])

    resp = await client.get(
        f"/api/runs/{created['id']}/events",
        headers={"last-event-id": "not-a-number"},
    )

    assert resp.status_code == 200
    assert _sse_ids(resp.text)[0] == 1


# --------------------------------------------------------------------------
# M3-3：人在回路——审稿停点与决策端点
# --------------------------------------------------------------------------


async def test_human_review_flow_over_http(tmp_path):
    """HTTP 全流程：创建（human_review）→ 停 WAITING → 查意见 → 决策 → 完成。"""
    from src.embeddings.dummy import DummyEmbeddingProvider
    from src.llm.mock import MockLLMProvider
    from src.orchestrator import WritingOrchestrator

    store = RunStore(tmp_path / "zylo.db")
    bus = TraceBus(store=store)
    runner = JobRunner(bus, store=store)
    mock_llm = MockLLMProvider()

    def make_executor(run):
        return WritingOrchestrator(
            llm=mock_llm,
            embedding_provider=DummyEmbeddingProvider(),
            human_review=bool(run.config.get("human_review")),
        )

    app = create_app(bus=bus, runner=runner, executor_factory=make_executor)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        created = (
            await client.post("/api/runs", json={"topic": "人审演示", "human_review": True})
        ).json()
        run_id = created["id"]
        await runner.wait(run_id)

        # 停在等待人审
        detail = (await client.get(f"/api/runs/{run_id}")).json()
        assert detail["status"] == "waiting_for_human_review"

        # 待决策意见可见：稳定 ID + 作用域 + 建议
        review = (await client.get(f"/api/runs/{run_id}/review")).json()
        assert review["revision"] == 0
        assert review["passed"] is False
        assert len(review["opinions"]) == 1
        opinion = review["opinions"][0]
        assert opinion["critique_id"] and opinion["scope"] and opinion["advice"]

        # 非等待状态查意见 → 409（用另一个已完成的 run 验证）
        other = (await client.post("/api/runs", json={"topic": "全自动"})).json()
        await runner.wait(other["id"])
        resp409 = await client.get(f"/api/runs/{other['id']}/review")
        assert resp409.status_code == 409

        # 部分决策（漏掉意见）→ 422，run 仍 WAITING
        partial = await client.post(
            f"/api/runs/{run_id}/review-decisions", json={"items": []}
        )
        assert partial.status_code == 422
        assert partial.json()["error"]["code"] == "decision_mismatch"

        # 采纳全部意见 → 202，恢复执行到第二轮审稿（通过）后再次停点
        accepted = await client.post(
            f"/api/runs/{run_id}/review-decisions",
            json={
                "items": [
                    {
                        "critique_id": opinion["critique_id"],
                        "action": "accept",
                    }
                ]
            },
        )
        assert accepted.status_code == 202
        await runner.wait(run_id)
        review2 = (await client.get(f"/api/runs/{run_id}/review")).json()
        assert review2["revision"] == 1
        assert review2["passed"] is True
        assert review2["opinions"] == []

        # 无意见 → 空决策集拍板 → 完成导出
        final = await client.post(
            f"/api/runs/{run_id}/review-decisions", json={"items": []}
        )
        assert final.status_code == 202
        await runner.wait(run_id)

        assert (await client.get(f"/api/runs/{run_id}")).json()["status"] == "completed"
        article = (await client.get(f"/api/runs/{run_id}/article")).json()
        assert article["review_score"] == 92.0

        # 人工决策已落库（评估数据）：一条 accept，revision 0
        decisions = store.list_review_decisions(run_id)
        assert len(decisions) == 1
        assert decisions[0].action.value == "accept"
        assert decisions[0].revision == 0
