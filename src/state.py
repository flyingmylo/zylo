from dataclasses import asdict, dataclass, field
from enum import Enum

# 快照结构版本：序列化字段发生不兼容变更时递增，resume 时据此拒绝旧快照
SNAPSHOT_SCHEMA_VERSION = 1

# 审稿意见的全局作用域标记：不针对任何具体小节的意见统一归入该值
GLOBAL_SCOPE = "全局"


class Stage(str, Enum):
    INIT = "init"
    RESEARCHING = "researching"
    PLANNING = "planning"
    WRITING = "writing"
    REVIEWING = "reviewing"
    REVISING = "revising"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SectionSpec:
    title: str
    target_words: int
    focus_points: list[str] = field(default_factory=list)
    retrieval_query_zh: str = ""  # 中文检索词
    retrieval_query_en: str = ""  # 英文检索词（解决跨语言相似度衰减）


@dataclass
class WritingState:
    topic: str
    extra_instructions: str = ""
    local_files: list[str] = field(default_factory=list)  # 本地英文/中文文档路径

    # 调研产出
    kb_collection_name: str = ""
    research_summary: str = ""
    external_links: list[dict[str, str]] = field(default_factory=list)

    # 规划大纲产出
    outline_title: str = ""
    target_total_words: int = 3000
    sections: list[SectionSpec] = field(default_factory=list)

    # 写作草稿产出
    section_drafts: dict[str, str] = field(default_factory=dict)  # title -> markdown
    full_draft: str = ""

    # 审稿与反思回路
    review_passed: bool = False
    review_score: float = 0.0  # 0 - 100
    critiques: list[str] = field(default_factory=list)
    # 每条为 {"section": 小节标题或 GLOBAL_SCOPE, "advice": 修改建议}
    actionable_revisions: list[dict[str, str]] = field(default_factory=list)
    revision_count: int = 0
    max_revisions: int = 2
    selected_revision: int = 0  # 最终导出采用的版本轮次，0 表示初稿
    # 人审停点标志（M3-3）：True 表示本轮审稿已完成、正等待人工决策，
    # resume 重入编排器时据此跳过已完成的写作/审稿直接消费决策结果。
    # 带默认值的新字段对旧快照向后兼容，无需递增 SNAPSHOT_SCHEMA_VERSION
    awaiting_human: bool = False

    # 终稿与日志
    final_markdown: str = ""
    current_stage: Stage = Stage.INIT
    errors: list[str] = field(default_factory=list)
    token_usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


def serialize_state(state: WritingState) -> dict:
    """WritingState -> 可 JSON 序列化的 dict（含版本号）。

    Stage 枚举落值为字符串；嵌套的 SectionSpec 递归展开。
    敏感信息不在 state 中（密钥只在 LLMConfig），序列化天然脱敏。
    """
    data = asdict(state)
    data["current_stage"] = state.current_stage.value
    return {"schema_version": SNAPSHOT_SCHEMA_VERSION, "state": data}


def deserialize_state(payload: dict) -> WritingState:
    """serialize_state 的逆操作；版本不匹配或结构损坏时抛 ValueError。

    resume 的入口防线：宁可显式失败，也不要用半损坏的状态静默续跑。
    """
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"快照版本不兼容：期望 {SNAPSHOT_SCHEMA_VERSION}，"
            f"实际 {payload.get('schema_version')}"
        )
    data = dict(payload["state"])
    try:
        data["current_stage"] = Stage(data["current_stage"])
        data["sections"] = [SectionSpec(**s) for s in data["sections"]]
        return WritingState(**data)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"快照结构损坏，无法恢复: {exc}") from exc
