import pytest

from api.bus import TraceBus
from api.runner import JobRunner
from src.budget import BudgetExceededError, BudgetGuard
from src.embeddings.dummy import DummyEmbeddingProvider
from src.llm.mock import MockLLMProvider
from src.orchestrator import WritingOrchestrator
from src.runs import RunStatus
from src.state import WritingState, deserialize_state, serialize_state

# --------------------------------------------------------------------------
# BudgetGuard 账本
# --------------------------------------------------------------------------


def test_budget_blocks_on_call_limit():
    guard = BudgetGuard(max_calls=2)

    guard.check_before_call()  # 0 < 2：允许
    guard.settle({"total_tokens": 10})
    guard.check_before_call()  # 1 < 2：允许
    guard.settle({"total_tokens": 10})

    with pytest.raises(BudgetExceededError, match="调用次数"):
        guard.check_before_call()  # 2 >= 2：熔断


def test_budget_blocks_on_token_limit():
    guard = BudgetGuard(max_total_tokens=100)

    guard.settle({"total_tokens": 90})
    guard.check_before_call()  # 90 < 100 尚可

    guard.settle({"total_tokens": 20})
    with pytest.raises(BudgetExceededError, match="Token"):
        guard.check_before_call()


def test_cost_guard_requires_explicit_prices():
    """未知价格不默认为零：缺单价时金额维度显式失效（记录告警）。"""
    guard = BudgetGuard(max_cost_usd=1.0)
    assert guard.max_cost_usd is None

    priced = BudgetGuard(
        max_cost_usd=1.0,
        price_input_per_mtok=1.0,
        price_output_per_mtok=2.0,
    )
    priced.settle({"prompt_tokens": 1_000_000, "completion_tokens": 500_000, "total_tokens": 1_500_000})
    assert priced.cost_usd == pytest.approx(1.0 * 1.0 + 0.5 * 2.0)
    with pytest.raises(BudgetExceededError, match="金额"):
        priced.check_before_call()


def test_budget_from_env(monkeypatch):
    monkeypatch.setenv("ZYLO_BUDGET_MAX_TOKENS", "1000")
    monkeypatch.setenv("ZYLO_BUDGET_MAX_CALLS", "5")
    monkeypatch.delenv("ZYLO_BUDGET_MAX_COST_USD", raising=False)

    guard = BudgetGuard.from_env()

    assert guard.max_total_tokens == 1000
    assert guard.max_calls == 5
    assert guard.max_cost_usd is None


# --------------------------------------------------------------------------
# BaseAgent._chat 接线
# --------------------------------------------------------------------------


async def test_chat_settles_budget():
    from src.agents.base import BaseAgent
    from src.llm.base import LLMResponse
    from tests.test_base_agent import ScriptedLLM

    class EchoAgent(BaseAgent):
        async def run(self, state):
            return state

    guard = BudgetGuard()
    llm = ScriptedLLM(
        [LLMResponse(content="hi", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})]
    )
    agent = EchoAgent("Echo", llm, "sys", budget=guard)
    state = WritingState(topic="t")

    await agent._chat([{"role": "user", "content": "x"}], state)

    assert guard.calls == 1
    assert guard.total_tokens == 15
    assert state.token_usage["total_tokens"] == 15


async def test_chat_blocks_before_call_when_budget_exhausted():
    from src.agents.base import BaseAgent
    from tests.test_base_agent import ScriptedLLM

    class EchoAgent(BaseAgent):
        async def run(self, state):
            return state

    guard = BudgetGuard(max_calls=1)
    guard.settle({"total_tokens": 1})  # 已耗尽
    agent = EchoAgent("Echo", ScriptedLLM([]), "sys", budget=guard)

    with pytest.raises(BudgetExceededError):
        await agent._chat([{"role": "user", "content": "x"}], WritingState(topic="t"))


# --------------------------------------------------------------------------
# 熔断 → PARTIAL → 调大预算 → resume 完成（端到端闭环）
# --------------------------------------------------------------------------


async def test_budget_exhaustion_marks_partial_then_resumes(tmp_path):
    checkpoints: list[dict] = []

    def make_orchestrator(max_calls: int):
        return WritingOrchestrator(
            llm=MockLLMProvider(),
            embedding_provider=DummyEmbeddingProvider(),
            kb_persist_dir=str(tmp_path / "chroma"),
            budget=BudgetGuard(max_calls=max_calls),
        )

    # 1) 预算只够 5 次调用：中途熔断
    tight = make_orchestrator(max_calls=5)

    def on_checkpoint(state):
        checkpoints.append(serialize_state(state))

    with pytest.raises(BudgetExceededError):
        await tight.execute(topic="MoE 路由", output_dir=str(tmp_path), on_checkpoint=on_checkpoint)

    assert tight.budget.calls == 5
    assert checkpoints  # 熔断前的阶段快照已留存

    # 2) JobRunner 层：BudgetExceededError → run 落位 PARTIAL
    bus = TraceBus()
    runner = JobRunner(bus)

    class ExplodingExecutor:
        async def execute(self, *args, **kwargs):
            raise BudgetExceededError("Token 总量已达上限：100/100")

    run = runner.create("预算测试")
    runner.launch(run.id, ExplodingExecutor())
    await runner.wait(run.id)
    assert run.status is RunStatus.PARTIAL
    assert "预算耗尽" in (run.error or "")

    # 3) 调大预算后从最新快照 resume：任务完成
    latest = checkpoints[-1]
    resumed = await make_orchestrator(max_calls=99).execute(
        topic="MoE 路由",
        output_dir=str(tmp_path),
        resume_state=deserialize_state(latest),
    )
    assert resumed.review_score == 92.0
