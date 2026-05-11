"""Phase D slice 4 pipeline-orchestration tests.

Adds coverage for the new infer + summarize stages plugged into
:class:`RefinePipeline`. DB is mocked; LLM gateway is a Stub.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.distill import StubLLMGateway
from z3rno_core.refine import RefineOptions, RefinePipeline


@pytest.fixture
def fake_conn() -> MagicMock:
    conn = MagicMock()
    result = MagicMock()
    result.fetchall = lambda: []
    result.fetchone = lambda: None
    result.rowcount = 0
    conn.execute = AsyncMock(return_value=result)
    return conn


@pytest.mark.asyncio
async def test_pipeline_skips_infer_when_disabled(fake_conn: MagicMock) -> None:
    pipeline = RefinePipeline(
        options=RefineOptions(infer_enabled=False, summarize_enabled=False),
        gateway=StubLLMGateway(model="x"),
    )
    summary = await pipeline.run(fake_conn, org_id=uuid4())
    assert summary.status == "completed"
    assert summary.infer is None
    assert summary.summarize is None


@pytest.mark.asyncio
async def test_pipeline_skips_infer_when_no_gateway(fake_conn: MagicMock) -> None:
    """Flag on, gateway None → no-op."""
    pipeline = RefinePipeline(
        options=RefineOptions(infer_enabled=True, summarize_enabled=True),
        gateway=None,
    )
    summary = await pipeline.run(fake_conn, org_id=uuid4())
    assert summary.status == "completed"
    assert summary.infer is None
    assert summary.summarize is None


@pytest.mark.asyncio
async def test_pipeline_runs_infer_and_summarize_with_gateway(fake_conn: MagicMock) -> None:
    """Both stages execute (no candidates → 0 work, but stages ran)."""
    pipeline = RefinePipeline(
        options=RefineOptions(infer_enabled=True, summarize_enabled=True),
        gateway=StubLLMGateway(model="x"),
    )
    summary = await pipeline.run(fake_conn, org_id=uuid4())
    assert summary.status == "completed"
    assert summary.infer is not None
    assert summary.summarize is not None
    assert summary.infer.candidates_examined == 0
    assert summary.summarize.clusters_examined == 0
