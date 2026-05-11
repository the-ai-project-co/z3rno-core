"""Unit tests for z3rno_core.refine.feedback (Phase D slice 2).

DB call is mocked. Live RLS / CHECK-constraint enforcement is covered
by tests/test_phase_d_schema_integration.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.refine import record_feedback
from z3rno_core.refine.feedback import decay_weights


@pytest.fixture
def fake_conn() -> MagicMock:
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=MagicMock())
    return conn


@pytest.mark.asyncio
async def test_record_feedback_with_memory_id_inserts_row(fake_conn: MagicMock) -> None:
    fid = await record_feedback(
        fake_conn,
        org_id=uuid4(),
        agent_id=uuid4(),
        signal=1,
        memory_id=uuid4(),
        reason="helpful",
    )
    assert fid is not None
    fake_conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_feedback_with_edge_id_inserts_row(fake_conn: MagicMock) -> None:
    fid = await record_feedback(
        fake_conn,
        org_id=uuid4(),
        agent_id=uuid4(),
        signal=-1,
        edge_id="e:works_for:42",
    )
    assert fid is not None
    fake_conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_feedback_rejects_signal_out_of_range(fake_conn: MagicMock) -> None:
    with pytest.raises(ValueError, match="signal must be"):
        await record_feedback(
            fake_conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            signal=2,
            memory_id=uuid4(),
        )
    fake_conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_record_feedback_rejects_no_target(fake_conn: MagicMock) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        await record_feedback(
            fake_conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            signal=1,
        )
    fake_conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_record_feedback_rejects_both_targets(fake_conn: MagicMock) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        await record_feedback(
            fake_conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            signal=1,
            memory_id=uuid4(),
            edge_id="e:1",
        )
    fake_conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_record_feedback_uses_provided_id(fake_conn: MagicMock) -> None:
    fixed = uuid4()
    returned = await record_feedback(
        fake_conn,
        org_id=uuid4(),
        agent_id=uuid4(),
        signal=0,
        memory_id=uuid4(),
        feedback_id=fixed,
    )
    assert returned == fixed


@pytest.mark.asyncio
async def test_decay_weights_stub_returns_zero(fake_conn: MagicMock) -> None:
    """Slice 3 fills this in; for now confirm the contract is callable."""
    count = await decay_weights(fake_conn, org_id=uuid4(), decay_factor=0.95)
    assert count == 0
