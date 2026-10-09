"""运行事件契约：TraceBus、SQLite run_events 表与 SSE 共用的事件形态。

对应 PLAN ADR-002：事件是事实来源，SSE 只做投递；sequence 在单个 run
内递增，SSE 的 id 字段直接使用它，以支持 Last-Event-ID 历史回放。
"""

import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

# 事件结构版本：字段增删或语义变更时递增，消费方据此决定兼容策略
EVENT_SCHEMA_VERSION = 1

# payload 内任何字符串的最大长度：防止一段正文或错误堆栈把事件撑爆
MAX_PAYLOAD_STRING = 500

# 密钥类键：精确匹配（大小写不敏感），值整体替换为占位符。
# 注意不能按子串匹配，否则 token_usage 这类统计数据会被误伤。
SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
    }
)

# 内容类键：完整 prompt 与文档正文不进事件，只保留长度线索供调试
CONTENT_KEYS = frozenset({"prompt", "messages", "content", "text", "document"})


class SpanKind(str, Enum):
    """五层 span：run、stage、agent、llm、tool（M3 Trace 的层级结构）。"""

    RUN = "run"
    STAGE = "stage"
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"


class EventStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


def sanitize_payload(value: Any) -> Any:
    """递归脱敏事件属性：密钥替换、内容省略、超长字符串截断。

    规则按精确键名匹配（大小写不敏感），键的集合在本模块集中维护，
    TraceBus 与未来的 JSONL 导出都只经此一处出口。
    """
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in SENSITIVE_KEYS:
                cleaned[str(key)] = "[REDACTED]"
            elif lowered in CONTENT_KEYS:
                cleaned[str(key)] = f"[OMITTED {len(str(item))} chars]"
            else:
                cleaned[str(key)] = sanitize_payload(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, str) and len(value) > MAX_PAYLOAD_STRING:
        return value[:MAX_PAYLOAD_STRING] + f"...[truncated {len(value)} chars]"
    return value


class RunTrace(Protocol):
    """运行时事件发射器的最小契约（五层 span：run/stage/agent/llm/tool）。

    放在 src 层使 Agent 依赖抽象而非 api 层实现；具体实现见
    api.bus.TraceEmitter（总线+落盘），无观测需求时用 NullTrace。
    """

    def start_span(
        self,
        kind: SpanKind,
        name: str,
        parent_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str: ...

    def finish_span(
        self,
        span_id: str,
        status: EventStatus = EventStatus.COMPLETED,
        payload: dict[str, Any] | None = None,
    ) -> None: ...


class NullTrace:
    """零开销空实现：未注入观测时 Agent 代码路径完全无感。"""

    def start_span(
        self,
        kind: SpanKind,
        name: str,
        parent_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        return ""

    def finish_span(
        self,
        span_id: str,
        status: EventStatus = EventStatus.COMPLETED,
        payload: dict[str, Any] | None = None,
    ) -> None:
        return None


class RunEvent(BaseModel):
    """单个运行事件的不可变记录，字段与 M2 的 run_events 表对齐。"""

    sequence: int = Field(ge=1)
    run_id: str
    schema_version: int = EVENT_SCHEMA_VERSION
    kind: SpanKind
    name: str
    status: EventStatus
    span_id: str
    parent_id: str | None = None  # run 层事件的父 span 为空
    # 脱敏后的属性（Token 用量、耗时、错误摘要等）；写入前必须过 sanitize_payload
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_json(self) -> str:
        """SSE data 与 JSONL 导出共用的序列化形态。"""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> "RunEvent":
        """回放与测试用：从 JSONL/SSE 载荷还原事件。"""
        data = json.loads(raw)
        return cls(**data)
