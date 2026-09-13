from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    actionable_revisions: list[str] = field(default_factory=list)
    revision_count: int = 0
    max_revisions: int = 2

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
