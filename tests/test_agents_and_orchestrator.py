import json

import pytest

from src.embeddings.base import EmbeddingProvider
from src.llm.base import LLMProvider, LLMResponse
from src.orchestrator import WritingOrchestrator
from src.state import Stage


class MockLLMProvider(LLMProvider):
    """用于测试的 Mock LLM，根据 Prompt 角色模拟返回不同 Agent 的响应"""

    async def chat(self, messages, tools=None, temperature=0.7):
        system_msg = messages[0]["content"] if messages else ""
        user_msg = messages[-1]["content"] if messages else ""

        # Researcher keywords
        if "最核心的 2 个中文检索词" in system_msg:
            return LLMResponse(
                content='["KV-Cache优化", "PageAttention", "KV-Cache memory", "vLLM"]'
            )

        # Researcher summary
        if "技术调研专家" in system_msg:
            return LLMResponse(
                content="KV-Cache 是 LLM 推理加速的核心技术，但在长文本下带来显著显存压力。主要优化方案包含 PagedAttention 与量化。"
            )

        # Planner outline
        if "架构规划专家" in system_msg:
            outline_json = {
                "outline_title": "大模型 KV-Cache 显存优化技术演进",
                "target_total_words": 1500,
                "sections": [
                    {
                        "title": "一、KV-Cache 物理瓶颈与痛点",
                        "target_words": 500,
                        "focus_points": ["显存爆炸", "动态分配瓶颈"],
                        "retrieval_query_zh": "KV-Cache 显存瓶颈",
                        "retrieval_query_en": "KV-Cache memory footprint bottleneck",
                    },
                    {
                        "title": "二、分页注意力机制原理剖析",
                        "target_words": 700,
                        "focus_points": ["虚拟内存思想", "块碎片消除"],
                        "retrieval_query_zh": "PagedAttention 原理",
                        "retrieval_query_en": "PagedAttention virtual memory fragmentation",
                    },
                ],
            }
            return LLMResponse(content=json.dumps(outline_json))

        # Writer
        if "技术作家" in system_msg:
            assert user_msg, "Writer 的用户消息不应为空。"
            return LLMResponse(
                content="### 正文小节\n\n在大模型推理阶段，键值缓存（KV-Cache）是核心技术。通过引入分页注意力（PagedAttention），系统可以大幅减少内存碎片。"
            )

        # Reviewer
        if "审稿专家" in system_msg:
            review_json = {
                "passed": True,
                "score": 92.0,
                "critiques": ["术语对照严格，技术推导清晰。"],
                "actionable_revisions": [],
            }
            return LLMResponse(content=json.dumps(review_json))

        return LLMResponse(content="默认测试响应")


class DummyEmbeddingProvider(EmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.05] * 8 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.05] * 8


@pytest.mark.asyncio
async def test_full_pipeline_orchestration(tmp_path):
    mock_llm = MockLLMProvider()
    dummy_embed = DummyEmbeddingProvider()

    # 准备一个临时英文参考文档
    ref_file = tmp_path / "paper_sample.md"
    ref_file.write_text(
        "PagedAttention allows storing continuous keys and values in non-contiguous memory spaces."
    )

    out_dir = tmp_path / "output"

    orchestrator = WritingOrchestrator(
        llm=mock_llm,
        embedding_provider=dummy_embed,
    )

    state = await orchestrator.execute(
        topic="KV-Cache 显存优化",
        local_files=[str(ref_file)],
        output_dir=str(out_dir),
    )

    assert state.current_stage == Stage.COMPLETED
    assert state.outline_title == "大模型 KV-Cache 显存优化技术演进"
    assert len(state.sections) == 2
    assert state.review_passed is True
    assert state.review_score == 92.0
    assert "键值缓存（KV-Cache）" in state.full_draft
    assert len(list(out_dir.glob("*.md"))) == 1


class MultiRoundMockLLM(LLMProvider):
    """模拟第一轮审查不通过、第二轮审查通过的多轮回路"""

    def __init__(self):
        self.review_round = 0

    async def chat(self, messages, tools=None, temperature=0.7):
        system_msg = messages[0]["content"] if messages else ""

        if "技术调研专家" in system_msg:
            return LLMResponse(content="调研综述：大模型长上下文显存开销。")

        if "架构规划专家" in system_msg:
            outline_json = {
                "outline_title": "长上下文显存优化",
                "target_total_words": 1000,
                "sections": [
                    {
                        "title": "一、核心原理",
                        "target_words": 500,
                        "focus_points": ["核心要点"],
                        "retrieval_query_zh": "显存优化 原理",
                        "retrieval_query_en": "memory optimization principles",
                    }
                ],
            }
            return LLMResponse(content=json.dumps(outline_json))

        if "技术作家" in system_msg:
            return LLMResponse(content="### 一、核心原理\n\n正文阐述显存管理机制。")

        if "审稿专家" in system_msg:
            self.review_round += 1
            if self.review_round == 1:
                return LLMResponse(
                    content=json.dumps(
                        {
                            "passed": False,
                            "score": 75.0,
                            "critiques": ["第一节缺少具体的术语英文对照。"],
                            "actionable_revisions": [
                                "一、核心原理：请务必增加键值缓存（KV-Cache）双语对照。"
                            ],
                        }
                    )
                )
            else:
                return LLMResponse(
                    content=json.dumps(
                        {
                            "passed": True,
                            "score": 95.0,
                            "critiques": ["修订版已完美补充双语对照。"],
                            "actionable_revisions": [],
                        }
                    )
                )

        return LLMResponse(content="默认响应")


@pytest.mark.asyncio
async def test_revision_loop_execution(tmp_path):
    multi_mock = MultiRoundMockLLM()
    dummy_embed = DummyEmbeddingProvider()
    out_dir = tmp_path / "out_revision"

    orchestrator = WritingOrchestrator(
        llm=multi_mock,
        embedding_provider=dummy_embed,
    )

    state = await orchestrator.execute(
        topic="长上下文显存优化",
        output_dir=str(out_dir),
    )

    # 验证经历了 1 轮修改后成功通过
    assert state.revision_count == 1
    assert state.review_passed is True
    assert state.review_score == 95.0
    assert multi_mock.review_round == 2


class UsageReportingMock(MockLLMProvider):
    """在既有 Mock 基础上为每次响应附加 usage，并记录调用次数。"""

    PER_CALL_TOKENS = 15

    def __init__(self):
        self.call_count = 0

    async def chat(self, messages, tools=None, temperature=0.7):
        resp = await super().chat(messages, tools, temperature)
        self.call_count += 1
        resp.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": self.PER_CALL_TOKENS,
        }
        return resp


@pytest.mark.asyncio
async def test_token_usage_is_counted_for_every_llm_call(tmp_path):
    """每一次 LLM 调用都必须被计入 token_usage，并反映到终稿 footer。

    回归保护：Researcher 曾直接调用 llm.chat 绕过统计，导致终稿的总消耗
    系统性漏掉整个调研阶段。
    """
    mock_llm = UsageReportingMock()

    orchestrator = WritingOrchestrator(
        llm=mock_llm,
        embedding_provider=DummyEmbeddingProvider(),
    )

    state = await orchestrator.execute(
        topic="KV-Cache 显存优化",
        output_dir=str(tmp_path / "out_tokens"),
    )

    # 调研 1 + 规划 1 + 逐节写作 2 + 审稿 1 = 5 次调用，每次都恰好被统计一次。
    # 绝对次数是必要的：只用相对等式的话，整个调研环节被删掉后测试仍会通过。
    assert mock_llm.call_count == 5
    assert state.token_usage["total_tokens"] == mock_llm.call_count * 15
    assert state.token_usage["prompt_tokens"] == mock_llm.call_count * 10
    assert f"Total: {state.token_usage['total_tokens']}" in state.final_markdown


def test_cli_positional_logic():
    # 测试参数提取逻辑
    inputs = [
        "KV-Cache 显存优化",
        "references/yoco.pdf",
        "https://arxiv.org/abs/2405.05254",
    ]
    topic = inputs[0]
    sources = inputs[1:]
    assert topic == "KV-Cache 显存优化"
    assert len(sources) == 2
    assert "yoco.pdf" in sources[0]
    assert sources[1].startswith("https://")
