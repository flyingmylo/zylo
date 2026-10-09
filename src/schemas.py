"""Planner/Reviewer 结构化输出的严格契约（Pydantic）。

这些模型是 LLM 输出与运行时之间的边界校验层：
- 类型不再宽容：passed 必须是真布尔、score 必须 0-100、advice 非空；
- 校验失败由 BaseAgent._chat_structured 带错误信息重试一次，
  仍失败则走 planning_degraded / review_degraded 降级路径，
  绝不默认通过（M0 基线教训：fail-open 曾放行低分稿）。
"""

from pydantic import BaseModel, Field, field_validator


class StructuredOutputError(ValueError):
    """结构化输出解析/校验失败（含重试后仍失败）；调用方决定降级语义。"""


class PlannerSection(BaseModel):
    title: str = Field(min_length=1)
    target_words: int = Field(default=600, ge=50)
    focus_points: list[str] = Field(default_factory=list)
    retrieval_query_zh: str = ""
    retrieval_query_en: str = ""


class PlannerOutline(BaseModel):
    outline_title: str = Field(min_length=1)
    target_total_words: int = Field(default=3000, ge=100)
    sections: list[PlannerSection] = Field(min_length=1)

    @field_validator("sections")
    @classmethod
    def _titles_must_be_unique(cls, sections: list[PlannerSection]) -> list[PlannerSection]:
        titles = [s.title for s in sections]
        if len(titles) != len(set(titles)):
            raise ValueError("小节标题存在重复，修订回路按标题定位会失效")
        return sections


class ReviewRevision(BaseModel):
    # 取值域校验（必须是大纲标题或 GLOBAL_SCOPE）在 Reviewer 侧做，
    # 因为合法集合来自当前运行的 sections 上下文
    section: str = Field(min_length=1)
    advice: str = Field(min_length=1)


class ReviewReport(BaseModel):
    passed: bool
    score: float = Field(ge=0, le=100)
    critiques: list[str] = Field(default_factory=list)
    actionable_revisions: list[ReviewRevision] = Field(default_factory=list)
