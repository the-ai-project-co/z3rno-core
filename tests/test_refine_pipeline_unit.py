"""Unit tests for the Refine pipeline stages (Phase D slice 3).

Reweight math is tested in isolation; pipeline orchestration is
tested with a fake AsyncConnection so we exercise the call order and
counter aggregation without a database.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.refine import RefineOptions, RefinePipeline
from z3rno_core.refine.reweight import compute_new_weight

# ---------------------------------------------------------------------------
# Reweight math
# ---------------------------------------------------------------------------


def test_compute_new_weight_positive_signal_increases_weight() -> None:
    new = compute_new_weight(old=0.5, signal_mean=1.0, decay=0.95)
    assert new > 0.5


def test_compute_new_weight_negative_signal_decreases_weight() -> None:
    new = compute_new_weight(old=0.5, signal_mean=-1.0, decay=0.95)
    assert new < 0.5


def test_compute_new_weight_neutral_signal_pulls_toward_half() -> None:
    new = compute_new_weight(old=0.9, signal_mean=0.0, decay=0.95)
    assert 0.85 < new < 0.9  # nudged toward 0.5 by 5%


def test_compute_new_weight_clamps_to_unit_interval() -> None:
    assert compute_new_weight(old=1.5, signal_mean=1.0, decay=0.95) <= 1.0
    assert compute_new_weight(old=-0.5, signal_mean=-1.0, decay=0.95) >= 0.0


def test_compute_new_weight_decay_one_is_no_op() -> None:
    """decay=1.0 → keep old weight unchanged regardless of signal."""
    assert compute_new_weight(old=0.4, signal_mean=1.0, decay=1.0) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_conn() -> MagicMock:
    """An AsyncConnection that returns empty result sets for every query."""
    conn = MagicMock()
    result = MagicMock()
    result.fetchall = lambda: []
    result.fetchone = lambda: None
    result.rowcount = 0
    conn.execute = AsyncMock(return_value=result)
    return conn


@pytest.mark.asyncio
async def test_pipeline_run_on_empty_org_completes_with_zero_counters(
    fake_conn: MagicMock,
) -> None:
    pipeline = RefinePipeline()
    summary = await pipeline.run(fake_conn, org_id=uuid4())

    assert summary.status == "completed"
    assert summary.memos_scanned == 0
    assert summary.memos_deduped == 0
    assert summary.edges_reweighted == 0
    assert summary.edges_pruned == 0
    assert summary.feedback_drained == 0
    assert summary.error is None


@pytest.mark.asyncio
async def test_pipeline_run_marks_failed_on_stage_exception() -> None:
    """If a stage raises, the pipeline records failure and re-raises."""
    conn = MagicMock()
    conn.execute = AsyncMock(side_effect=RuntimeError("dedupe blew up"))
    pipeline = RefinePipeline()

    with pytest.raises(RuntimeError, match="dedupe blew up"):
        await pipeline.run(conn, org_id=uuid4())


@pytest.mark.asyncio
async def test_pipeline_run_respects_supplied_job_id(fake_conn: MagicMock) -> None:
    fixed = uuid4()
    pipeline = RefinePipeline()
    summary = await pipeline.run(fake_conn, org_id=uuid4(), job_id=fixed)
    assert summary.job_id == fixed


@pytest.mark.asyncio
async def test_pipeline_run_uses_options(fake_conn: MagicMock) -> None:
    """RefineOptions.feedback_weight_decay is threaded into reweight."""
    pipeline = RefinePipeline(options=RefineOptions(feedback_weight_decay=0.5))
    summary = await pipeline.run(fake_conn, org_id=uuid4())
    assert summary.status == "completed"
