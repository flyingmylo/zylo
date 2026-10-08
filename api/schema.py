"""API 请求/响应模型：只做传输形态定义，业务模型复用 src/runs.py。"""

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    """POST /api/runs 请求体，字段与 docs/api-contract.md 契约一致。"""

    topic: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)
    instructions: str = ""


class ArticleResponse(BaseModel):
    """GET /api/runs/{id}/article 响应体。"""

    title: str
    markdown: str
    review_score: float
    revision_count: int
