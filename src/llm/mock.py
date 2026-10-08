"""离线演示用 Mock LLM：按 Agent 角色返回结构化假响应。

设计目标是让 `zylo serve --mock` 不配任何 API Key、不加载本地
嵌入权重即可完整演示调研→规划→写作→审稿→修订的全链路：
- 审稿带状态：首轮不通过（附一条修订意见），次轮起通过，
  用于演示反思回路与修订 Diff；
- 每次调用返回仿真 usage，终稿的 Token 统计有数可看。
"""

import json
import re
from typing import Any

from .base import LLMProvider, LLMResponse

# 每次假调用的仿真用量：让演示里的成本统计面板有真实感
MOCK_USAGE = {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180}


def _extract_topic(messages: list[dict[str, Any]]) -> str:
    """从用户消息里粗提取主题；提不到就用占位主题。"""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        m = re.search(r"技术主题[:：]\s*(.+)", content)
        if m:
            return m.group(1).strip()
        m = re.search(r"【写作主题】：(.+)", content)
        if m:
            return m.group(1).strip()
        # Researcher 综述请求格式：请针对主题【X】，……
        m = re.search(r"主题【(.+?)】", content)
        if m:
            return m.group(1).strip()
    return "示例技术主题"


class MockLLMProvider(LLMProvider):
    """角色感知的 Mock：依据系统提示词分派到各 Agent 的预设响应。"""

    def __init__(self) -> None:
        # 审稿轮次计数。executor_factory 每次 POST 新建一个 Mock 实例，
        # 因此计数天然按 run 隔离，无需显式按 run_id 分桶
        self._review_rounds = 0
        self.call_count = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.call_count += 1
        system = str(messages[0].get("content", "")) if messages else ""
        user = str(messages[-1].get("content", "")) if messages else ""
        topic = _extract_topic(messages)

        # 角色判据必须用系统提示词里独有的词：PLANNER_SYSTEM_PROMPT 的
        # JSON 示例中也包含"检索词"字样，按子串"检索词"分派会误路由
        if "技术调研员" in system:
            content = json.dumps([f"{topic} 核心原理", f"{topic} 生产实践"], ensure_ascii=False)
        elif "技术调研专家" in system:
            content = (
                f"# {topic} 调研综述\n\n"
                "（Mock 数据）该主题的核心机制包括注意力优化、显存管理与调度策略；"
                "主流方案已在工业界大规模验证。"
            )
        elif "架构规划" in system:
            content = json.dumps(
                {
                    "outline_title": f"{topic} 深度解析",
                    "target_total_words": 1200,
                    "sections": [
                        {
                            "title": "一、背景与核心痛点",
                            "target_words": 400,
                            "focus_points": ["问题背景"],
                            "retrieval_query_zh": f"{topic} 背景",
                            "retrieval_query_en": f"{topic} background",
                        },
                        {
                            "title": "二、核心机制解析",
                            "target_words": 500,
                            "focus_points": ["关键设计"],
                            "retrieval_query_zh": f"{topic} 原理",
                            "retrieval_query_en": f"{topic} mechanism",
                        },
                        {
                            "title": "三、工程实践与展望",
                            "target_words": 300,
                            "focus_points": ["落地建议"],
                            "retrieval_query_zh": f"{topic} 实践",
                            "retrieval_query_en": f"{topic} practice",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        elif "技术作家" in system:
            section = "本节正文"
            m = re.search(r"【当前撰写小节】：(.+)", user)
            if m:
                section = m.group(1).strip()
            content = (
                f"### {section}\n\n"
                f"这是关于 {topic} 的（Mock）正文内容。键值缓存（KV-Cache）与"
                "分页注意力（PagedAttention）是本节的代表性术语对照示例。"
            )
        elif "审稿专家" in system:
            self._review_rounds += 1
            if self._review_rounds == 1:
                content = json.dumps(
                    {
                        "passed": False,
                        "score": 78.0,
                        "critiques": ["（Mock）首轮审稿：术语对照可以更完整。"],
                        "actionable_revisions": [
                            {"section": "全局", "advice": "（Mock）统一术语双语对照格式。"}
                        ],
                    },
                    ensure_ascii=False,
                )
            else:
                content = json.dumps(
                    {
                        "passed": True,
                        "score": 92.0,
                        "critiques": ["（Mock）修订版已达到发布标准。"],
                        "actionable_revisions": [],
                    },
                    ensure_ascii=False,
                )
        else:
            content = "（Mock 默认响应）"

        return LLMResponse(content=content, usage=dict(MOCK_USAGE))
