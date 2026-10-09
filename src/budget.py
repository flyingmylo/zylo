"""BudgetGuard：LLM 调用的三层预算熔断（Token 数 / 调用次数 / 金额）。

语义约定：
- 调用前预检（check_before_call）：任一维度已达上限即抛 BudgetExceededError；
- 调用后结算（settle）：按真实 usage 累计，杜绝按预估漂移；
- 金额维度必须显式提供每百万 token 单价，未知价格不默认为零——
  未提供单价时金额熔断自动失效并记录一次警告（宁可失守也不假装在守）；
- 预算耗尽对作业意味着"资源不足"而非"执行出错"：上游应把 run 落位
  PARTIAL，调大预算后可从最新快照 resume 续跑。
"""

import logging

logger = logging.getLogger("src.budget")


class BudgetExceededError(RuntimeError):
    """预算耗尽；message 说明触发的维度与当前用量。"""


class BudgetGuard:
    """进程内单 run 共享的预算账本；同一实例被该 run 的全部 Agent 持有。"""

    def __init__(
        self,
        max_total_tokens: int | None = None,
        max_calls: int | None = None,
        max_cost_usd: float | None = None,
        price_input_per_mtok: float | None = None,
        price_output_per_mtok: float | None = None,
    ) -> None:
        self.max_total_tokens = max_total_tokens
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.price_input_per_mtok = price_input_per_mtok
        self.price_output_per_mtok = price_output_per_mtok

        self.calls = 0
        self.total_tokens = 0
        self.cost_usd = 0.0

        if max_cost_usd is not None and (
            price_input_per_mtok is None or price_output_per_mtok is None
        ):
            # 未知价格不默认为零：显式放弃金额维度，避免虚假的安全感
            logger.warning(
                "设置了金额上限但未提供 token 单价，金额熔断不生效"
                "（需要 price_input_per_mtok / price_output_per_mtok）"
            )
            self.max_cost_usd = None

    @classmethod
    def from_env(cls) -> "BudgetGuard":
        """从环境变量构造（serve/write/resume 的统一入口）。

        ZYLO_BUDGET_MAX_TOKENS / ZYLO_BUDGET_MAX_CALLS / ZYLO_BUDGET_MAX_COST_USD
        ZYLO_PRICE_INPUT_PER_MTOK / ZYLO_PRICE_OUTPUT_PER_MTOK
        """
        import os

        def _int(name: str) -> int | None:
            raw = os.getenv(name, "").strip()
            return int(raw) if raw else None

        def _float(name: str) -> float | None:
            raw = os.getenv(name, "").strip()
            return float(raw) if raw else None

        return cls(
            max_total_tokens=_int("ZYLO_BUDGET_MAX_TOKENS"),
            max_calls=_int("ZYLO_BUDGET_MAX_CALLS"),
            max_cost_usd=_float("ZYLO_BUDGET_MAX_COST_USD"),
            price_input_per_mtok=_float("ZYLO_PRICE_INPUT_PER_MTOK"),
            price_output_per_mtok=_float("ZYLO_PRICE_OUTPUT_PER_MTOK"),
        )

    def check_before_call(self) -> None:
        """调用前预检；任何维度已达上限立即熔断。"""
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceededError(
                f"LLM 调用次数已达上限：{self.calls}/{self.max_calls}"
            )
        if (
            self.max_total_tokens is not None
            and self.total_tokens >= self.max_total_tokens
        ):
            raise BudgetExceededError(
                f"Token 总量已达上限：{self.total_tokens}/{self.max_total_tokens}"
            )
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(
                f"金额已达上限：${self.cost_usd:.4f}/${self.max_cost_usd:.2f}"
            )

    def settle(self, usage: dict[str, int]) -> None:
        """调用成功后按真实 usage 结算。"""
        self.calls += 1
        self.total_tokens += usage.get("total_tokens", 0)
        if self.max_cost_usd is not None:
            cost = (
                usage.get("prompt_tokens", 0) / 1_000_000 * (self.price_input_per_mtok or 0)
                + usage.get("completion_tokens", 0)
                / 1_000_000
                * (self.price_output_per_mtok or 0)
            )
            self.cost_usd += cost

    def snapshot(self) -> dict[str, float | int]:
        """当前用量，供事件 payload 与前端展示。"""
        return {
            "budget_calls": self.calls,
            "budget_total_tokens": self.total_tokens,
            "budget_cost_usd": round(self.cost_usd, 6),
        }
