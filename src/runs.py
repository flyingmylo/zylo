"""运行（Run）产品契约的核心：状态机与运行实体。

M0 阶段只定义契约与校验，不做持久化（M2 的 RunStore）
与 API 暴露（M1 的 FastAPI），三者的数据形态都以本模块为准。
"""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.state import GLOBAL_SCOPE

# 契约版本号：模型结构发生不兼容变更时递增，供 RunStore 快照与 SSE 事件消费方判断
SCHEMA_VERSION = 1


def new_run_id() -> str:
    """短随机 run_id：足够区分单机单人场景下的运行，且对 URL 友好。"""
    return uuid.uuid4().hex[:12]


class RunStatus(str, Enum):
    """一次写作运行的全部状态。

    PARTIAL 是"进程中断但存在可用快照"的停摆态：只由服务恢复逻辑
    落位（M2），之后仅允许被 resume 拉起或显式取消。
    """

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_HUMAN_REVIEW = "waiting_for_human_review"
    REVISING = "revising"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"

    @property
    def is_terminal(self) -> bool:
        """终态不会再自发迁移；PARTIAL 不算终态，因为可被 resume 续跑。"""
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


# 合法状态迁移表。RUNNING→REVISING 允许无人介入的自动修订（当前
# Orchestrator 行为）；WAITING_FOR_HUMAN_REVIEW→REVISING 是人审采纳后的路径。
_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.WAITING_FOR_HUMAN_REVIEW,
            RunStatus.REVISING,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            # 仅服务恢复逻辑使用：进程重启后把遗留的 RUNNING 落位为 PARTIAL
            RunStatus.PARTIAL,
        }
    ),
    RunStatus.WAITING_FOR_HUMAN_REVIEW: frozenset(
        {RunStatus.REVISING, RunStatus.COMPLETED, RunStatus.CANCELLED}
    ),
    RunStatus.REVISING: frozenset(
        {
            RunStatus.WAITING_FOR_HUMAN_REVIEW,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.PARTIAL: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


class RunTransitionError(ValueError):
    """非法状态迁移，消息中携带 from -> to 便于日志定位。"""


class Run(BaseModel):
    """一次写作运行的实体，字段与 M2 计划的 runs 表一一对应。"""

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=new_run_id)
    topic: str = Field(min_length=1)
    status: RunStatus = RunStatus.QUEUED
    schema_version: int = SCHEMA_VERSION
    # 脱敏后的运行配置（模型名、参数等）；密钥绝不进入本字段
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    def transition(self, to: RunStatus) -> None:
        """按状态机迁移；非法迁移抛 RunTransitionError。

        时间戳是迁移的副作用而非调用方责任：进入 RUNNING 落
        started_at（保留首次启动时间），进入终态落 finished_at。
        """
        if to not in _ALLOWED_TRANSITIONS[self.status]:
            raise RunTransitionError(
                f"非法状态迁移: {self.status.value} -> {to.value}"
            )
        self.status = to
        if to is RunStatus.RUNNING and self.started_at is None:
            self.started_at = datetime.now(UTC)
        if to.is_terminal:
            self.finished_at = datetime.now(UTC)


class Revision(BaseModel):
    """一轮稿件快照，字段与 M2 计划的 revision_snapshots 表对齐。

    review 保存该轮审稿结果的结构化 JSON（得分、意见列表），
    是修订 Diff 与人工审稿的数据基础。
    """

    run_id: str
    revision: int = Field(ge=0)  # 0 表示初稿
    full_draft: str
    section_drafts: dict[str, str] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ArtifactKind(str, Enum):
    """运行产物类型；path 一律相对仓库根（output/、data/），保证可迁移。"""

    ARTICLE = "article"
    TRACE = "trace"


class Artifact(BaseModel):
    """导出产物登记，字段与 M2 计划的 artifacts 表对齐。"""

    run_id: str
    kind: ArtifactKind
    # 相对路径或引用；绝对路径会让仓库搬迁后记录全部失效
    path_or_ref: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("path_or_ref")
    @classmethod
    def _reject_absolute_path(cls, value: str) -> str:
        if value.startswith(("/", "\\")):
            raise ValueError("产物路径必须使用相对路径")
        return value


class SourceStatus(str, Enum):
    """抓取来源的状态，与 Researcher 的实际降级路径一一对应。"""

    PENDING = "pending"
    FETCHED = "fetched"  # 原文抓取成功并入库
    DEGRADED = "degraded"  # 原文不可用，以搜索摘要兜底入库
    FAILED = "failed"  # 彻底失败，未入库


class Source(BaseModel):
    """一次运行实际使用的资料来源，字段与 M2 计划的 run_sources 表对齐。

    source_id 是稳定标识，M4 引用协议用它把文章论断反查回来源。
    """

    source_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    run_id: str
    url_or_path: str
    status: SourceStatus = SourceStatus.PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Critique(BaseModel):
    """单条审稿意见：稳定 ID + 作用域 + 建议，供人工逐条决策。"""

    critique_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    # 小节标题或 GLOBAL_SCOPE；为保持与现有 Writer 修订逻辑同词表，复用 state 的常量
    scope: str = GLOBAL_SCOPE
    advice: str = Field(min_length=1)


class ReviewAction(str, Enum):
    """人对单条审稿意见的处置动作（M3 人在回路）。"""

    ACCEPT = "accept"  # 采纳，按意见修订
    REJECT = "reject"  # 拒绝，记录理由后跳过该意见
    EDIT = "edit"  # 修改后采纳，advice 以 edited_advice 为准
    APPROVE_FINAL = "approve_final"  # 直接拍板定稿，不再修订


class ReviewDecision(BaseModel):
    """一条人工审稿决策，字段设计对齐 M3 的 POST /review-decisions。"""

    run_id: str
    revision: int = Field(ge=0)  # 针对第几轮审稿
    critique_id: str
    action: ReviewAction
    edited_advice: str | None = None
    reason: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _edit_requires_advice(self) -> "ReviewDecision":
        # EDIT 语义是"改写建议后执行"，缺了改写文本就无法落到 Writer 的修订指令
        if self.action is ReviewAction.EDIT and not (
            self.edited_advice and self.edited_advice.strip()
        ):
            raise ValueError("EDIT 决策必须提供非空 edited_advice")
        return self
