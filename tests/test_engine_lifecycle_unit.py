"""Unit tests for z3rno_core.engine.lifecycle — pure logic only (no DB).

Tests decay_importance formula, sweep_expired_memories, enforce_retention_cap,
transition_memory, and ensure_audit_partitions using mocked connections.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from z3rno_core.engine.lifecycle import (
    PartitionResult,
    TransitionResult,
    decay_importance,
    enforce_retention_cap,
    ensure_audit_partitions,
    sweep_expired_memories,
    transition_memory,
)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


class TestPartitionResult:
    """Test PartitionResult frozen dataclass."""

    def test_construction(self) -> None:
        result = PartitionResult(created_count=2, partition_names=["a", "b"])
        assert result.created_count == 2
        assert result.partition_names == ["a", "b"]

    def test_frozen(self) -> None:
        result = PartitionResult(created_count=0, partition_names=[])
        with pytest.raises(AttributeError):
            result.created_count = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers for mocking async DB calls
# ---------------------------------------------------------------------------


def _make_mock_conn() -> AsyncMock:
    """Create a mock AsyncConnection."""
    conn = AsyncMock()
    result_mock = MagicMock()
    result_mock.fetchall.return_value = []
    result_mock.fetchone.return_value = None
    result_mock.scalar.return_value = 0
    conn.execute.return_value = result_mock
    return conn


# ---------------------------------------------------------------------------
# sweep_expired_memories
# ---------------------------------------------------------------------------


class TestSweepExpiredMemories:
    """Test TTL expiry sweep."""

    async def test_no_expired_memories(self) -> None:
        """Returns zero count when no memories have expired."""
        org_id = uuid4()
        conn = _make_mock_conn()

        result = await sweep_expired_memories(conn, org_id=org_id)

        assert result.expired_count == 0
        assert result.memory_ids == []

    async def test_expired_memories_soft_deleted(self) -> None:
        """Expired memories are soft-deleted and audit entries created."""
        org_id = uuid4()
        mid1 = uuid4()
        mid2 = uuid4()

        conn = AsyncMock()
        # First call: SELECT expired IDs
        select_result = MagicMock()
        select_result.fetchall.return_value = [(mid1,), (mid2,)]
        # Second call: UPDATE soft delete
        update_result = MagicMock()
        update_result.rowcount = 2
        conn.execute.side_effect = [select_result, update_result]

        with patch(
            "z3rno_core.engine.lifecycle.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            result = await sweep_expired_memories(conn, org_id=org_id)

        assert result.expired_count == 2
        assert mid1 in result.memory_ids
        assert mid2 in result.memory_ids
        # Audit entry per expired memory
        assert mock_audit.call_count == 2

    async def test_sweep_audit_includes_ttl_reason(self) -> None:
        """Audit entries for TTL sweep have reason='ttl_expiry'."""
        org_id = uuid4()
        mid = uuid4()

        conn = AsyncMock()
        select_result = MagicMock()
        select_result.fetchall.return_value = [(mid,)]
        update_result = MagicMock()
        update_result.rowcount = 1
        conn.execute.side_effect = [select_result, update_result]

        with patch(
            "z3rno_core.engine.lifecycle.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            await sweep_expired_memories(conn, org_id=org_id)

        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["details"]["reason"] == "ttl_expiry"
        assert call_kwargs["operation"] == "forget"


# ---------------------------------------------------------------------------
# enforce_retention_cap
# ---------------------------------------------------------------------------


class TestEnforceRetentionCap:
    """Test retention cap enforcement."""

    async def test_under_cap_no_action(self) -> None:
        """No deletions when count is at or below the cap."""
        org_id = uuid4()
        agent_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 50  # At cap
        conn.execute.return_value = count_result

        result = await enforce_retention_cap(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            memory_type="episodic",
            max_count=50,
        )

        assert result.expired_count == 0
        assert result.memory_ids == []

    async def test_over_cap_deletes_excess(self) -> None:
        """Excess memories are soft-deleted when over cap."""
        org_id = uuid4()
        agent_id = uuid4()
        excess_id = uuid4()

        conn = AsyncMock()
        # First call: COUNT = 51 (1 over cap of 50)
        count_result = MagicMock()
        count_result.scalar.return_value = 51
        # Second call: SELECT excess IDs
        select_result = MagicMock()
        select_result.fetchall.return_value = [(excess_id,)]
        # Third call: UPDATE soft delete
        update_result = MagicMock()
        update_result.rowcount = 1
        conn.execute.side_effect = [count_result, select_result, update_result]

        with patch(
            "z3rno_core.engine.lifecycle.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            result = await enforce_retention_cap(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_type="working",
                max_count=50,
            )

        assert result.expired_count == 1
        assert excess_id in result.memory_ids
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["details"]["reason"] == "retention_cap"

    async def test_over_cap_but_no_unpinned_memories(self) -> None:
        """If all excess memories are pinned, nothing is deleted."""
        org_id = uuid4()
        agent_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 60
        select_result = MagicMock()
        select_result.fetchall.return_value = []  # All pinned
        conn.execute.side_effect = [count_result, select_result]

        result = await enforce_retention_cap(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            memory_type="semantic",
            max_count=50,
        )

        assert result.expired_count == 0
        assert result.memory_ids == []

    async def test_count_zero_scalar_none_fallback(self) -> None:
        """scalar() returning None is treated as 0."""
        org_id = uuid4()
        agent_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = None  # None from empty DB
        conn.execute.return_value = count_result

        result = await enforce_retention_cap(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            memory_type="working",
            max_count=100,
        )

        assert result.expired_count == 0


# ---------------------------------------------------------------------------
# transition_memory
# ---------------------------------------------------------------------------


class TestTransitionMemory:
    """Test memory type transitions."""

    async def test_successful_transition(self) -> None:
        """A valid transition changes the type and creates audit."""
        org_id = uuid4()
        mid = uuid4()
        agent_id = uuid4()

        conn = AsyncMock()
        # First call: SELECT current type
        select_result = MagicMock()
        select_result.fetchone.return_value = ("working",)
        # Second call: UPDATE type
        update_result = MagicMock()
        update_result.rowcount = 1
        conn.execute.side_effect = [select_result, update_result]

        with patch(
            "z3rno_core.engine.lifecycle.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            result = await transition_memory(
                conn,
                org_id=org_id,
                memory_id=mid,
                new_type="episodic",
                agent_id=agent_id,
            )

        assert isinstance(result, TransitionResult)
        assert result.memory_id == mid
        assert result.old_type == "working"
        assert result.new_type == "episodic"

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["operation"] == "update"
        assert "working -> episodic" in call_kwargs["details"]["transition"]

    async def test_transition_not_found_raises(self) -> None:
        """Transitioning a non-existent memory raises ValueError."""
        org_id = uuid4()
        mid = uuid4()

        conn = AsyncMock()
        select_result = MagicMock()
        select_result.fetchone.return_value = None
        conn.execute.return_value = select_result

        with pytest.raises(ValueError, match="not found"):
            await transition_memory(
                conn,
                org_id=org_id,
                memory_id=mid,
                new_type="semantic",
            )


# ---------------------------------------------------------------------------
# decay_importance — formula tests
# ---------------------------------------------------------------------------


class TestDecayImportance:
    """Test importance decay formula: max(floor, score * e^(-rate * days))."""

    async def test_decay_formula_correctness(self) -> None:
        """Verify the decay formula produces expected results."""
        org_id = uuid4()
        mid = uuid4()
        now = datetime.now(tz=UTC)
        last_access = now - timedelta(days=10)

        conn = AsyncMock()
        # SELECT memories: id, importance_score, last_recalled_at, created_at
        select_result = MagicMock()
        select_result.fetchall.return_value = [
            (mid, 0.8, last_access, now - timedelta(days=20)),
        ]
        # UPDATE result (single batch call)
        update_result = MagicMock()
        update_result.rowcount = 1
        conn.execute.side_effect = [select_result, update_result]

        result = await decay_importance(conn, org_id=org_id, decay_rate=0.05, min_threshold=0.05)

        assert result.decayed_count == 1
        # Verify the batch UPDATE SQL contains the expected score
        update_call = conn.execute.call_args_list[1]
        sql_text = str(update_call[0][0].text)
        expected = round(0.8 * math.exp(-0.05 * 10), 6)
        assert str(expected) in sql_text
        assert str(mid) in sql_text

    async def test_decay_respects_floor(self) -> None:
        """Score never drops below the decay_floor."""
        org_id = uuid4()
        mid = uuid4()
        now = datetime.now(tz=UTC)
        # Very old access -> heavy decay
        last_access = now - timedelta(days=1000)

        conn = AsyncMock()
        select_result = MagicMock()
        select_result.fetchall.return_value = [
            (mid, 0.5, last_access, now - timedelta(days=1000)),
        ]
        update_result = MagicMock()
        conn.execute.side_effect = [select_result, update_result]

        result = await decay_importance(conn, org_id=org_id, decay_rate=0.1, decay_floor=0.1)

        assert result.decayed_count == 1
        # Verify the batch UPDATE SQL contains the floor-clamped score
        update_call = conn.execute.call_args_list[1]
        sql_text = str(update_call[0][0].text)
        assert str(mid) in sql_text
        # The score in the SQL should be >= 0.1 (the floor)
        assert "0.1" in sql_text

    async def test_no_decay_when_score_unchanged(self) -> None:
        """Memories with negligible score change are not updated."""
        org_id = uuid4()
        mid = uuid4()
        now = datetime.now(tz=UTC)
        # Very recent access -> negligible decay
        last_access = now - timedelta(seconds=1)

        conn = AsyncMock()
        select_result = MagicMock()
        select_result.fetchall.return_value = [
            (mid, 0.5, last_access, now),
        ]
        conn.execute.side_effect = [select_result]

        result = await decay_importance(conn, org_id=org_id)

        # Score change < 0.0001 so no update
        assert result.decayed_count == 0
        # Only the SELECT call, no UPDATE
        assert conn.execute.call_count == 1

    async def test_below_threshold_counted(self) -> None:
        """Memories decaying below min_threshold are counted."""
        org_id = uuid4()
        mid = uuid4()
        now = datetime.now(tz=UTC)
        last_access = now - timedelta(days=100)

        conn = AsyncMock()
        select_result = MagicMock()
        select_result.fetchall.return_value = [
            (mid, 0.1, last_access, now - timedelta(days=100)),
        ]
        update_result = MagicMock()
        conn.execute.side_effect = [select_result, update_result]

        result = await decay_importance(conn, org_id=org_id, decay_rate=0.05, min_threshold=0.05)

        assert result.below_threshold_count >= 1

    async def test_fallback_to_created_at_when_no_last_recalled(self) -> None:
        """When last_recalled_at is None, uses created_at for decay calc."""
        org_id = uuid4()
        mid = uuid4()
        now = datetime.now(tz=UTC)
        created = now - timedelta(days=5)

        conn = AsyncMock()
        select_result = MagicMock()
        select_result.fetchall.return_value = [
            (mid, 0.8, None, created),  # last_recalled_at is None
        ]
        update_result = MagicMock()
        conn.execute.side_effect = [select_result, update_result]

        result = await decay_importance(conn, org_id=org_id, decay_rate=0.05)

        assert result.decayed_count == 1
        # Should use created_at (5 days ago) for decay
        update_call = conn.execute.call_args_list[1]
        sql_text = str(update_call[0][0].text)
        expected = round(0.8 * math.exp(-0.05 * 5), 6)
        assert str(expected) in sql_text

    async def test_no_memories_returns_zero(self) -> None:
        """No active memories results in zero counts."""
        org_id = uuid4()
        conn = _make_mock_conn()

        result = await decay_importance(conn, org_id=org_id)

        assert result.decayed_count == 0
        assert result.below_threshold_count == 0

    async def test_multiple_memories_batch_update(self) -> None:
        """Multiple memories are updated in a single batch UPDATE."""
        org_id = uuid4()
        mid1 = uuid4()
        mid2 = uuid4()
        now = datetime.now(tz=UTC)
        old_time = now - timedelta(days=30)

        conn = AsyncMock()
        select_result = MagicMock()
        select_result.fetchall.return_value = [
            (mid1, 0.8, old_time, old_time),
            (mid2, 0.6, old_time, old_time),
        ]
        update_result = MagicMock()
        conn.execute.side_effect = [select_result, update_result]

        result = await decay_importance(conn, org_id=org_id, decay_rate=0.05)

        assert result.decayed_count == 2
        # 1 SELECT + 1 batch UPDATE (not N individual UPDATEs)
        assert conn.execute.call_count == 2
        # Verify both memory IDs appear in the batch SQL
        update_call = conn.execute.call_args_list[1]
        sql_text = str(update_call[0][0].text)
        assert str(mid1) in sql_text
        assert str(mid2) in sql_text


# ---------------------------------------------------------------------------
# ensure_audit_partitions
# ---------------------------------------------------------------------------


class TestEnsureAuditPartitions:
    """Test audit log partition management."""

    async def test_creates_missing_partitions(self) -> None:
        """Creates partitions for months where they don't exist."""
        conn = AsyncMock()
        # For each month check, return None (partition doesn't exist)
        check_result = MagicMock()
        check_result.fetchone.return_value = None
        # For creates, return a normal result
        create_result = MagicMock()

        # months_ahead=0 means just the current month
        # So: 1 check + 1 create
        conn.execute.side_effect = [check_result, create_result]

        result = await ensure_audit_partitions(conn, months_ahead=0)

        assert result.created_count == 1
        assert len(result.partition_names) == 1
        assert result.partition_names[0].startswith("audit_log_y")

    async def test_skips_existing_partitions(self) -> None:
        """Existing partitions are not recreated."""
        conn = AsyncMock()
        # Partition exists
        check_result = MagicMock()
        check_result.fetchone.return_value = (1,)
        conn.execute.return_value = check_result

        result = await ensure_audit_partitions(conn, months_ahead=0)

        assert result.created_count == 0
        assert result.partition_names == []

    async def test_multiple_months_ahead(self) -> None:
        """Creates partitions for multiple future months."""
        conn = AsyncMock()
        # All partitions missing: for months_ahead=2, 3 months (current + 2)
        # Each month: 1 check (None) + 1 create
        check_result_missing = MagicMock()
        check_result_missing.fetchone.return_value = None
        create_result = MagicMock()

        side_effects = []
        for _ in range(3):
            side_effects.extend([check_result_missing, create_result])
        conn.execute.side_effect = side_effects

        result = await ensure_audit_partitions(conn, months_ahead=2)

        assert result.created_count == 3
        assert len(result.partition_names) == 3

    async def test_partition_name_format(self) -> None:
        """Partition names follow the y{year}m{month:02d} format."""
        conn = AsyncMock()
        check_result = MagicMock()
        check_result.fetchone.return_value = None
        create_result = MagicMock()
        conn.execute.side_effect = [check_result, create_result]

        result = await ensure_audit_partitions(conn, months_ahead=0)

        name = result.partition_names[0]
        assert name.startswith("audit_log_y")
        # Should match pattern like audit_log_y2026m04
        assert "m" in name
        parts = name.replace("audit_log_y", "").split("m")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    async def test_month_overflow_wraps_to_next_year(self) -> None:
        """When month arithmetic exceeds 12, year increments correctly.

        Covers lines 380-381 (while month > 12 loop) and lines 387-388
        (next_month > 12 boundary).

        Current month is April 2026, so months_ahead=9 reaches January 2027,
        which means month=13 triggers the while loop (line 380-381) and
        December's next_month=13 triggers the if branch (line 387-388).
        """
        conn = AsyncMock()
        # months_ahead=9 means 10 months: April..January = offsets 0..9
        side_effects = []
        for _ in range(10):
            check_missing = MagicMock()
            check_missing.fetchone.return_value = None
            create_result = MagicMock()
            side_effects.extend([check_missing, create_result])
        conn.execute.side_effect = side_effects

        result = await ensure_audit_partitions(conn, months_ahead=9)

        now = datetime.now(UTC)
        # The exact count depends on current month but should be 10
        assert result.created_count == 10
        names = result.partition_names

        # Verify year-crossing partitions exist
        # From April 2026 + 9 months = January 2027
        next_year = now.year + 1 if now.month + 9 > 12 else now.year
        assert any(f"y{next_year}" in n for n in names)

        # December partition must appear (exercises next_month > 12 branch)
        assert f"audit_log_y{now.year}m12" in names

        # Verify the SQL for December partition uses next year's January
        # Find the CREATE call for December partition
        for _i, call in enumerate(conn.execute.call_args_list):
            sql_text = str(call[0][0].text) if hasattr(call[0][0], "text") else str(call[0][0])
            if f"audit_log_y{now.year}m12" in sql_text and "CREATE" in sql_text.upper():
                assert f"{next_year}-01-01" in sql_text
                break

    async def test_mixed_existing_and_missing(self) -> None:
        """Mix of existing and missing partitions."""
        conn = AsyncMock()
        # Month 1: exists, Month 2: missing
        check_exists = MagicMock()
        check_exists.fetchone.return_value = (1,)
        check_missing = MagicMock()
        check_missing.fetchone.return_value = None
        create_result = MagicMock()

        conn.execute.side_effect = [check_exists, check_missing, create_result]

        result = await ensure_audit_partitions(conn, months_ahead=1)

        assert result.created_count == 1
