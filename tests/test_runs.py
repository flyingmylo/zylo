import pytest

from src.runs import (
    SCHEMA_VERSION,
    Artifact,
    ArtifactKind,
    Critique,
    ReviewAction,
    ReviewDecision,
    Revision,
    Run,
    RunStatus,
    RunTransitionError,
    Source,
    SourceStatus,
)


def _new_run() -> Run:
    return Run(topic="KV-Cache 显存优化")


def test_new_run_defaults():
    run = _new_run()

    assert run.status is RunStatus.QUEUED
    assert run.schema_version == SCHEMA_VERSION
    assert len(run.id) == 12
    assert run.started_at is None
    assert run.finished_at is None
    assert run.error is None


def test_empty_topic_is_rejected():
    with pytest.raises(ValueError):
        Run(topic="")


def test_full_happy_path_with_human_review():
    """人在回路主路径：排队 → 运行 → 待人审 → 修订 → 再审 → 完成。"""
    run = _new_run()

    run.transition(RunStatus.RUNNING)
    run.transition(RunStatus.WAITING_FOR_HUMAN_REVIEW)
    run.transition(RunStatus.REVISING)
    run.transition(RunStatus.WAITING_FOR_HUMAN_REVIEW)
    run.transition(RunStatus.COMPLETED)

    assert run.status is RunStatus.COMPLETED
    assert run.finished_at is not None


def test_autonomous_revision_path():
    """无人工介入的自动修订（当前 Orchestrator 行为）也是合法路径。"""
    run = _new_run()
    run.transition(RunStatus.RUNNING)
    run.transition(RunStatus.REVISING)

    assert run.status is RunStatus.REVISING


def test_cancel_from_any_live_state():
    for builder in (
        lambda: _new_run(),
        lambda: _transitioned(RunStatus.RUNNING),
        lambda: _transitioned(RunStatus.RUNNING, RunStatus.WAITING_FOR_HUMAN_REVIEW),
        lambda: _transitioned(RunStatus.RUNNING, RunStatus.REVISING),
        lambda: _transitioned(RunStatus.RUNNING, RunStatus.PARTIAL),
    ):
        run = builder()
        run.transition(RunStatus.CANCELLED)
        assert run.status is RunStatus.CANCELLED


def _transitioned(*steps: RunStatus) -> Run:
    run = _new_run()
    for step in steps:
        run.transition(step)
    return run


def test_illegal_transitions_raise():
    # 未运行不能直接完成；终态不能复活
    with pytest.raises(RunTransitionError, match="queued -> completed"):
        _new_run().transition(RunStatus.COMPLETED)

    done = _transitioned(RunStatus.RUNNING, RunStatus.COMPLETED)
    with pytest.raises(RunTransitionError, match="completed -> running"):
        done.transition(RunStatus.RUNNING)

    # PARTIAL 只能被 resume 拉起或取消，不能被"审"或直接完成
    partial = _transitioned(RunStatus.RUNNING, RunStatus.PARTIAL)
    with pytest.raises(RunTransitionError, match="partial -> completed"):
        partial.transition(RunStatus.COMPLETED)

    # QUEUED 从未开始过，不存在"部分完成"，不允许落入 PARTIAL
    with pytest.raises(RunTransitionError, match="queued -> partial"):
        _new_run().transition(RunStatus.PARTIAL)


def test_partial_can_be_resumed():
    run = _transitioned(RunStatus.RUNNING, RunStatus.PARTIAL)

    run.transition(RunStatus.RUNNING)

    assert run.status is RunStatus.RUNNING


def test_timestamps_are_transition_side_effects():
    run = _new_run()
    assert run.started_at is None
    assert run.finished_at is None

    run.transition(RunStatus.RUNNING)
    assert run.started_at is not None

    run.transition(RunStatus.FAILED)
    assert run.finished_at is not None


def test_resume_keeps_original_start_time():
    """PARTIAL 被 resume 拉起时保留首次启动时间，不覆盖。"""
    run = _transitioned(RunStatus.RUNNING)
    original_start = run.started_at
    assert original_start is not None

    run.transition(RunStatus.PARTIAL)
    run.transition(RunStatus.RUNNING)

    assert run.started_at == original_start


def test_terminal_states_are_closed():
    for status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
        assert status.is_terminal
        assert len(RunStatus) >= 8  # 全集存在性 sanity
    assert not RunStatus.PARTIAL.is_terminal


# --------------------------------------------------------------------------
# Revision / Artifact / Source / Critique / ReviewDecision 契约
# --------------------------------------------------------------------------


def test_revision_snapshot_roundtrip():
    revision = Revision(
        run_id="abc123",
        revision=1,
        full_draft="# 标题\n正文",
        section_drafts={"一、原理": "正文"},
        review={"score": 78.5, "passed": False},
    )

    assert revision.revision == 1
    assert Revision(run_id="abc123", revision=0, full_draft="x").review == {}


def test_revision_number_must_be_non_negative():
    with pytest.raises(ValueError):
        Revision(run_id="abc123", revision=-1, full_draft="x")


def test_artifact_rejects_absolute_path():
    ok = Artifact(run_id="r", kind=ArtifactKind.ARTICLE, path_or_ref="output/a.md")
    assert ok.path_or_ref == "output/a.md"

    with pytest.raises(ValueError, match="相对路径"):
        Artifact(run_id="r", kind=ArtifactKind.ARTICLE, path_or_ref="/etc/passwd")


def test_source_defaults_and_status_set():
    source = Source(run_id="r", url_or_path="https://arxiv.org/abs/2405.05254")

    assert source.status is SourceStatus.PENDING
    assert len(source.source_id) == 8

    source.status = SourceStatus.DEGRADED
    assert source.status is SourceStatus.DEGRADED


def test_critique_has_stable_id_and_global_scope_default():
    critique = Critique(advice="补充显存对比数据")

    assert len(critique.critique_id) == 8
    assert critique.scope == "全局"
    with pytest.raises(ValueError):
        Critique(advice="")


def test_review_decision_edit_requires_advice():
    with pytest.raises(ValueError, match="edited_advice"):
        ReviewDecision(run_id="r", revision=0, critique_id="c1", action=ReviewAction.EDIT)

    ok = ReviewDecision(
        run_id="r",
        revision=0,
        critique_id="c1",
        action=ReviewAction.EDIT,
        edited_advice="改写后的建议",
    )
    assert ok.edited_advice == "改写后的建议"

    # 拒绝与采纳不强制 advice，但可附理由
    reject = ReviewDecision(
        run_id="r", revision=0, critique_id="c1", action=ReviewAction.REJECT, reason="与事实不符"
    )
    assert reject.reason == "与事实不符"
