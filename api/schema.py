"""API 请求/响应模型：只做传输形态定义，业务模型复用 src/runs.py。"""

from pydantic import BaseModel, Field

from src.runs import ReviewAction


class RunCreateRequest(BaseModel):
    """POST /api/runs 请求体，字段与 docs/api-contract.md 契约一致。"""

    topic: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)
    instructions: str = ""
    # 开启后每轮审稿完成即停（waiting_for_human_review），
    # 等待 POST /review-decisions 注入人工决策后继续
    human_review: bool = False


class ArticleResponse(BaseModel):
    """GET /api/runs/{id}/article 响应体。"""

    title: str
    markdown: str
    review_score: float
    revision_count: int


class ReviewOpinion(BaseModel):
    """单条待决策审稿意见（稳定 ID 是人工决策的定位锚点）。"""

    critique_id: str
    scope: str
    advice: str


class ReviewResponse(BaseModel):
    """GET /api/runs/{id}/review 响应体：当前轮审稿结果与待决策意见。"""

    revision: int
    score: float
    passed: bool
    opinions: list[ReviewOpinion]


class ReviewDecisionItem(BaseModel):
    """POST /review-decisions 请求体中的单条决策（run_id/revision/时间由服务端补齐）。"""

    critique_id: str
    action: ReviewAction
    edited_advice: str | None = None
    reason: str | None = None


class ReviewDecisionsRequest(BaseModel):
    """POST /api/runs/{id}/review-decisions 请求体。"""

    items: list[ReviewDecisionItem] = Field(min_length=0)
