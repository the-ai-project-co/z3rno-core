"""Unit tests for z3rno_core.engine.forget — pure logic only (no DB).

Tests the forget function's soft delete, hard delete, cascade,
and error handling using mocked connections.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from z3rno_core.engine.forget import ForgetError, ForgetResult, forget

# ---------------------------------------------------------------------------
# ForgetResult dataclass
# ---------------------------------------------------------------------------


class TestForgetResult:
    """Test ForgetResult frozen dataclass."""

    def test_construction(self) -> None:
        mid = uuid4()
        result = ForgetResult(
            deleted_count=1,
            hard_deleted=False,
            cascade_count=0,
            memory_ids=[mid],
        )
        assert result.deleted_count == 1
        assert result.hard_deleted is False
        assert result.cascade_count == 0
        assert result.memory_ids == [mid]

    def test_frozen(self) -> None:
        result = ForgetResult(
            deleted_count=1,
            hard_deleted=False,
            cascade_count=0,
            memory_ids=[uuid4()],
        )
        with pytest.raises(AttributeError):
            result.deleted_count = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ForgetError
# ---------------------------------------------------------------------------


class TestForgetError:
    """Test ForgetError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(ForgetError, Exception)

    def test_message(self) -> None:
        err = ForgetError("must provide memory_id or memory_ids")
        assert "memory_id" in str(err)

    def test_raises(self) -> None:
        with pytest.raises(ForgetError, match="must provide"):
            raise ForgetError("must provide memory_id or memory_ids")


# ---------------------------------------------------------------------------
# Helpers for mocking async DB calls
# ---------------------------------------------------------------------------


def _make_mock_conn(rowcount: int = 1) -> AsyncMock:
    """Create a mock AsyncConnection for forget operations."""
    conn = AsyncMock()
    result_mock = MagicMock()
    result_mock.rowcount = rowcount
    result_mock.fetchall.return_value = []
    conn.execute.return_value = result_mock
    # Mock run_sync for graph operations
    conn.run_sync = AsyncMock()
    return conn


# ---------------------------------------------------------------------------
# forget() — validation
# ---------------------------------------------------------------------------


class TestForgetValidation:
    """Test forget() input validation."""

    async def test_no_ids_raises_error(self) -> None:
        """forget() raises ForgetError when no IDs are provided."""
        conn = _make_mock_conn()
        org_id = uuid4()
        agent_id = uuid4()

        with pytest.raises(ForgetError, match="must provide memory_id or memory_ids"):
            await forget(conn, org_id=org_id, agent_id=agent_id)

    async def test_empty_memory_ids_raises_error(self) -> None:
        """forget() raises ForgetError when memory_ids is an empty list."""
        conn = _make_mock_conn()
        org_id = uuid4()
        agent_id = uuid4()

        with pytest.raises(ForgetError, match="must provide"):
            await forget(conn, org_id=org_id, agent_id=agent_id, memory_ids=[])


# ---------------------------------------------------------------------------
# forget() — soft delete
# ---------------------------------------------------------------------------


class TestForgetSoftDelete:
    """Test soft delete path."""

    async def test_soft_delete_single_memory(self) -> None:
        """Soft delete of a single memory returns expected result."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        conn = _make_mock_conn(rowcount=1)

        with patch(
            "z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            result = await forget(conn, org_id=org_id, agent_id=agent_id, memory_id=mid)

        assert result.deleted_count == 1
        assert result.hard_deleted is False
        assert result.cascade_count == 0
        assert mid in result.memory_ids

        # Verify audit entry created
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["operation"] == "forget"

    async def test_soft_delete_multiple_memories(self) -> None:
        """Soft delete of multiple memories works correctly."""
        org_id = uuid4()
        agent_id = uuid4()
        mids = [uuid4(), uuid4(), uuid4()]
        conn = _make_mock_conn(rowcount=3)

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            result = await forget(conn, org_id=org_id, agent_id=agent_id, memory_ids=mids)

        assert result.deleted_count == 3
        assert result.hard_deleted is False

    async def test_soft_delete_both_id_and_ids(self) -> None:
        """memory_id and memory_ids are merged together."""
        org_id = uuid4()
        agent_id = uuid4()
        single = uuid4()
        batch = [uuid4(), uuid4()]
        conn = _make_mock_conn(rowcount=3)

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            result = await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=single,
                memory_ids=batch,
            )

        # All three should be targeted
        assert len(result.memory_ids) == 3

    async def test_soft_delete_updates_deleted_at(self) -> None:
        """Soft delete SQL should set deleted_at and updated_at."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        conn = _make_mock_conn(rowcount=1)

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            await forget(conn, org_id=org_id, agent_id=agent_id, memory_id=mid)

        # First execute call should be the UPDATE for soft delete
        call_args = conn.execute.call_args_list[0]
        sql_text = str(call_args[0][0].text)
        assert "UPDATE memories" in sql_text
        assert "deleted_at = now()" in sql_text


# ---------------------------------------------------------------------------
# forget() — hard delete
# ---------------------------------------------------------------------------


class TestForgetHardDelete:
    """Test hard delete path."""

    async def test_hard_delete_single_memory(self) -> None:
        """Hard delete permanently removes the memory."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        conn = _make_mock_conn(rowcount=1)

        with patch(
            "z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            result = await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                hard_delete=True,
            )

        assert result.hard_deleted is True
        assert result.deleted_count == 1

        # Verify audit entry uses gdpr_delete operation
        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["operation"] == "gdpr_delete"

    async def test_hard_delete_calls_graph_cleanup(self) -> None:
        """Hard delete attempts to remove graph vertices."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        conn = _make_mock_conn(rowcount=1)

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                hard_delete=True,
            )

        # run_sync should be called for graph cleanup
        conn.run_sync.assert_called_once()

    async def test_hard_delete_graph_failure_swallowed(self) -> None:
        """Hard delete continues even if graph cleanup fails."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        conn = _make_mock_conn(rowcount=1)
        conn.run_sync.side_effect = RuntimeError("AGE not loaded")

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            result = await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                hard_delete=True,
            )

        # Should still complete successfully
        assert result.hard_deleted is True

    async def test_hard_delete_removes_relationships_first(self) -> None:
        """Hard delete deletes relationships before memories (FK constraint)."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        conn = _make_mock_conn(rowcount=1)

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                hard_delete=True,
            )

        # Execute should be called: relationships DELETE, then memories DELETE
        execute_calls = conn.execute.call_args_list
        rel_sql = str(execute_calls[0][0][0].text)
        mem_sql = str(execute_calls[1][0][0].text)
        assert "memory_relationships" in rel_sql
        assert "DELETE FROM memories" in mem_sql

    async def test_hard_delete_gdpr_audit_omits_content(self) -> None:
        """GDPR audit entries for hard delete should NOT contain memory_id in details."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        conn = _make_mock_conn(rowcount=1)

        with patch(
            "z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                hard_delete=True,
            )

        call_kwargs = mock_audit.call_args[1]
        details = call_kwargs["details"]
        # For hard_delete, memory_id should NOT be in the details dict
        assert "memory_id" not in details


# ---------------------------------------------------------------------------
# forget() — cascade
# ---------------------------------------------------------------------------


class TestForgetCascade:
    """Test cascade delete path."""

    async def test_cascade_finds_related_memories(self) -> None:
        """Cascade mode finds and includes 1-hop related memories."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        related_id = uuid4()

        conn = AsyncMock()
        # First call: _find_related_memories SELECT
        related_result = MagicMock()
        related_result.fetchall.return_value = [(related_id,)]
        # Second call: soft delete UPDATE
        delete_result = MagicMock()
        delete_result.rowcount = 2
        conn.execute.side_effect = [related_result, delete_result]

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            result = await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                cascade=True,
            )

        assert result.cascade_count == 1
        assert related_id in result.memory_ids

    async def test_cascade_no_related_memories(self) -> None:
        """Cascade with no related memories has cascade_count=0."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()

        conn = AsyncMock()
        # First call: _find_related_memories returns empty
        related_result = MagicMock()
        related_result.fetchall.return_value = []
        # Second call: soft delete
        delete_result = MagicMock()
        delete_result.rowcount = 1
        conn.execute.side_effect = [related_result, delete_result]

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            result = await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                cascade=True,
            )

        assert result.cascade_count == 0

    async def test_cascade_multi_hop_traversal(self) -> None:
        """Cascade with depth=2 traverses two hops of relationships."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        hop1_id = uuid4()
        hop2_id = uuid4()

        conn = AsyncMock()
        # First hop: finds hop1_id related to mid
        hop1_result = MagicMock()
        hop1_result.fetchall.return_value = [(hop1_id,)]
        # Second hop: finds hop2_id related to hop1_id
        hop2_result = MagicMock()
        hop2_result.fetchall.return_value = [(hop2_id,)]
        # Soft delete UPDATE
        delete_result = MagicMock()
        delete_result.rowcount = 3
        conn.execute.side_effect = [hop1_result, hop2_result, delete_result]

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            result = await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                cascade=True,
                cascade_depth=2,
            )

        assert result.cascade_count == 2
        assert hop1_id in result.memory_ids
        assert hop2_id in result.memory_ids

    async def test_cascade_stops_when_frontier_empty(self) -> None:
        """Cascade with depth=3 breaks early when frontier empties after hop 2.

        Hop 1 finds hop1_id, hop 2 returns no new neighbors so frontier
        becomes empty, and the third iteration hits the `break` at line 222.
        Only 2 execute calls happen for _find_related_memories (not 3).
        """
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        hop1_id = uuid4()

        conn = AsyncMock()
        # Hop 1: finds hop1_id
        hop1_result = MagicMock()
        hop1_result.fetchall.return_value = [(hop1_id,)]
        # Hop 2: no new neighbors → frontier becomes empty
        hop2_result = MagicMock()
        hop2_result.fetchall.return_value = []
        # Hop 3 never runs because frontier is empty → break
        # Soft delete
        delete_result = MagicMock()
        delete_result.rowcount = 2
        conn.execute.side_effect = [hop1_result, hop2_result, delete_result]

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            result = await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                cascade=True,
                cascade_depth=3,
            )

        assert result.cascade_count == 1
        assert hop1_id in result.memory_ids

    async def test_cascade_deduplicates_ids(self) -> None:
        """If a related memory is already in the target list, it's deduplicated."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()

        conn = AsyncMock()
        # Related query returns the same ID that's already targeted
        related_result = MagicMock()
        related_result.fetchall.return_value = [(mid,)]  # same as target
        delete_result = MagicMock()
        delete_result.rowcount = 1
        conn.execute.side_effect = [related_result, delete_result]

        with patch("z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock):
            result = await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                cascade=True,
            )

        # The related ID equals the target, so _find_related_memories filters it out
        assert result.cascade_count == 0


# ---------------------------------------------------------------------------
# forget() — audit context
# ---------------------------------------------------------------------------


class TestForgetAuditContext:
    """Test audit entry creation in forget."""

    async def test_soft_delete_includes_reason(self) -> None:
        """Custom reason is passed to audit entry."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        conn = _make_mock_conn(rowcount=1)

        with patch(
            "z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                reason="user requested removal",
            )

        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["details"]["reason"] == "user requested removal"

    async def test_hard_delete_default_reason_is_gdpr(self) -> None:
        """Hard delete without explicit reason uses GDPR reason."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        conn = _make_mock_conn(rowcount=1)

        with patch(
            "z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
                hard_delete=True,
            )

        call_kwargs = mock_audit.call_args[1]
        assert "GDPR" in call_kwargs["details"]["reason"]

    async def test_soft_delete_details_include_memory_id(self) -> None:
        """Soft delete audit details include the memory_id."""
        org_id = uuid4()
        agent_id = uuid4()
        mid = uuid4()
        conn = _make_mock_conn(rowcount=1)

        with patch(
            "z3rno_core.engine.forget.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            await forget(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                memory_id=mid,
            )

        call_kwargs = mock_audit.call_args[1]
        assert "memory_id" in call_kwargs["details"]
