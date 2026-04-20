"""Unit tests for z3rno_core.engine.audit_query — pure logic only (no DB).

Tests the audit query function's pagination, filter building,
activity summary aggregation, and memory lifecycle queries.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from z3rno_core.engine.audit_query import (
    ActivitySummary,
    AuditEntry,
    AuditPage,
    audit,
    get_agent_activity_summary,
    get_memory_lifecycle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audit_row(
    *,
    entry_id: int = 1,
    org_id: Any = None,
    agent_id: Any = None,
    user_id: Any = None,
    operation: str = "store",
    memory_id: Any = None,
    memory_type: str | None = None,
    details: dict[str, Any] | None = None,
    prev_hash: bytes | None = None,
    row_hash: bytes = b"\x00" * 32,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
    created_at: datetime | None = None,
) -> tuple[Any, ...]:
    """Create a mock audit log row."""
    if org_id is None:
        org_id = uuid4()
    if created_at is None:
        created_at = datetime.now(tz=UTC)
    return (
        entry_id,
        org_id,
        agent_id,
        user_id,
        operation,
        memory_id,
        memory_type,
        details,
        prev_hash,
        row_hash,
        ip_address,
        user_agent,
        request_id,
        created_at,
    )


# ---------------------------------------------------------------------------
# audit() — pagination
# ---------------------------------------------------------------------------


class TestAuditPagination:
    """Test audit query pagination logic."""

    async def test_basic_pagination(self) -> None:
        """Returns a page of audit entries."""
        org_id = uuid4()
        row = _make_audit_row(org_id=org_id)

        conn = AsyncMock()
        # First call: COUNT
        count_result = MagicMock()
        count_result.scalar.return_value = 1
        # Second call: SELECT
        select_result = MagicMock()
        select_result.fetchall.return_value = [row]
        conn.execute.side_effect = [count_result, select_result]

        page = await audit(conn, org_id=org_id)

        assert isinstance(page, AuditPage)
        assert page.total == 1
        assert page.page == 1
        assert len(page.entries) == 1
        assert page.has_next is False

    async def test_has_next_true_when_more_pages(self) -> None:
        """has_next is True when there are more pages."""
        org_id = uuid4()
        row = _make_audit_row(org_id=org_id)

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 100  # Total entries
        select_result = MagicMock()
        select_result.fetchall.return_value = [row] * 10
        conn.execute.side_effect = [count_result, select_result]

        page = await audit(conn, org_id=org_id, page=1, page_size=10)

        assert page.has_next is True

    async def test_has_next_false_on_last_page(self) -> None:
        """has_next is False on the last page."""
        org_id = uuid4()
        row = _make_audit_row(org_id=org_id)

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 10
        select_result = MagicMock()
        select_result.fetchall.return_value = [row] * 10
        conn.execute.side_effect = [count_result, select_result]

        page = await audit(conn, org_id=org_id, page=1, page_size=10)

        assert page.has_next is False

    async def test_page_size_capped_at_100(self) -> None:
        """page_size is capped at 100."""
        org_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        page = await audit(conn, org_id=org_id, page_size=200)

        assert page.page_size == 100

    async def test_empty_results(self) -> None:
        """Returns empty page when no entries match."""
        org_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        page = await audit(conn, org_id=org_id)

        assert page.total == 0
        assert page.entries == []
        assert page.has_next is False

    async def test_offset_calculation(self) -> None:
        """Offset is calculated from page and page_size."""
        org_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        await audit(conn, org_id=org_id, page=3, page_size=25)

        # Second execute call (SELECT) should have offset = (3-1)*25 = 50
        call_args = conn.execute.call_args_list[1]
        params = call_args[0][1]
        assert params["offset"] == 50
        assert params["limit"] == 25

    async def test_total_none_fallback_to_zero(self) -> None:
        """scalar() returning None is treated as 0."""
        org_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = None
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        page = await audit(conn, org_id=org_id)

        assert page.total == 0


# ---------------------------------------------------------------------------
# audit() — filter building
# ---------------------------------------------------------------------------


class TestAuditFilterBuilding:
    """Test audit query filter construction."""

    async def test_agent_id_filter(self) -> None:
        """agent_id filter is added to the query."""
        org_id = uuid4()
        agent_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        await audit(conn, org_id=org_id, agent_id=agent_id)

        # Check params include agent_id
        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert "agent_id" in params
        assert params["agent_id"] == str(agent_id)

    async def test_user_id_filter(self) -> None:
        """user_id filter is added to the query."""
        org_id = uuid4()
        user_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        await audit(conn, org_id=org_id, user_id=user_id)

        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert "user_id" in params

    async def test_operation_filter(self) -> None:
        """operation filter is added to the query."""
        org_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        await audit(conn, org_id=org_id, operation="store")

        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert params["operation"] == "store"

    async def test_memory_id_filter(self) -> None:
        """memory_id filter is added to the query."""
        org_id = uuid4()
        mid = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        await audit(conn, org_id=org_id, memory_id=mid)

        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert "memory_id" in params

    async def test_memory_type_filter(self) -> None:
        """memory_type filter is added to the query."""
        org_id = uuid4()

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        await audit(conn, org_id=org_id, memory_type="semantic")

        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert params["memory_type"] == "semantic"

    async def test_time_range_filter(self) -> None:
        """time_range filter adds start and end conditions."""
        org_id = uuid4()
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 12, 31, tzinfo=UTC)

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        await audit(conn, org_id=org_id, time_range=(start, end))

        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert "time_start" in params
        assert "time_end" in params

    async def test_all_filters_combined(self) -> None:
        """All filters can be combined."""
        org_id = uuid4()
        agent_id = uuid4()
        user_id = uuid4()
        mid = uuid4()
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 12, 31, tzinfo=UTC)

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        select_result = MagicMock()
        select_result.fetchall.return_value = []
        conn.execute.side_effect = [count_result, select_result]

        await audit(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            user_id=user_id,
            operation="recall",
            memory_id=mid,
            memory_type="episodic",
            time_range=(start, end),
        )

        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert len(params) >= 7  # org_id + 6 filters


# ---------------------------------------------------------------------------
# audit() — AuditEntry construction
# ---------------------------------------------------------------------------


class TestAuditEntryConstruction:
    """Test AuditEntry construction from rows."""

    async def test_null_details_becomes_empty_dict(self) -> None:
        """Row with None details is converted to empty dict."""
        org_id = uuid4()
        row = _make_audit_row(org_id=org_id, details=None)

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 1
        select_result = MagicMock()
        select_result.fetchall.return_value = [row]
        conn.execute.side_effect = [count_result, select_result]

        page = await audit(conn, org_id=org_id)

        assert page.entries[0].details == {}

    async def test_populated_details_preserved(self) -> None:
        """Row with populated details preserves the dict."""
        org_id = uuid4()
        details = {"key": "value", "count": 42}
        row = _make_audit_row(org_id=org_id, details=details)

        conn = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 1
        select_result = MagicMock()
        select_result.fetchall.return_value = [row]
        conn.execute.side_effect = [count_result, select_result]

        page = await audit(conn, org_id=org_id)

        assert page.entries[0].details == details


# ---------------------------------------------------------------------------
# get_agent_activity_summary
# ---------------------------------------------------------------------------


class TestGetAgentActivitySummary:
    """Test agent activity summary aggregation."""

    async def test_basic_summary(self) -> None:
        """Returns operation counts for an agent."""
        org_id = uuid4()
        agent_id = uuid4()

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = [
            ("store", 10),
            ("recall", 25),
            ("forget", 3),
        ]
        conn.execute.return_value = result_mock

        summary = await get_agent_activity_summary(conn, org_id=org_id, agent_id=agent_id)

        assert isinstance(summary, ActivitySummary)
        assert summary.agent_id == agent_id
        assert summary.operation_counts == {"store": 10, "recall": 25, "forget": 3}
        assert summary.total == 38

    async def test_empty_summary(self) -> None:
        """Returns zero total when no activity."""
        org_id = uuid4()
        agent_id = uuid4()

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        conn.execute.return_value = result_mock

        summary = await get_agent_activity_summary(conn, org_id=org_id, agent_id=agent_id)

        assert summary.total == 0
        assert summary.operation_counts == {}

    async def test_summary_with_time_range(self) -> None:
        """Time range filter is applied to the query."""
        org_id = uuid4()
        agent_id = uuid4()
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 6, 30, tzinfo=UTC)

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = [("store", 5)]
        conn.execute.return_value = result_mock

        summary = await get_agent_activity_summary(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            time_range=(start, end),
        )

        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert "time_start" in params
        assert "time_end" in params
        assert summary.total == 5


# ---------------------------------------------------------------------------
# get_memory_lifecycle
# ---------------------------------------------------------------------------


class TestGetMemoryLifecycle:
    """Test memory lifecycle audit history."""

    async def test_returns_ordered_entries(self) -> None:
        """Returns audit entries for a specific memory."""
        org_id = uuid4()
        mid = uuid4()
        row1 = _make_audit_row(org_id=org_id, memory_id=mid, operation="store", entry_id=1)
        row2 = _make_audit_row(org_id=org_id, memory_id=mid, operation="recall", entry_id=2)

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = [row1, row2]
        conn.execute.return_value = result_mock

        entries = await get_memory_lifecycle(conn, org_id=org_id, memory_id=mid)

        assert len(entries) == 2
        assert all(isinstance(e, AuditEntry) for e in entries)
        assert entries[0].operation == "store"
        assert entries[1].operation == "recall"

    async def test_empty_lifecycle(self) -> None:
        """Returns empty list for a memory with no audit history."""
        org_id = uuid4()
        mid = uuid4()

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        conn.execute.return_value = result_mock

        entries = await get_memory_lifecycle(conn, org_id=org_id, memory_id=mid)

        assert entries == []

    async def test_null_details_in_lifecycle(self) -> None:
        """Null details in lifecycle entries become empty dict."""
        org_id = uuid4()
        mid = uuid4()
        row = _make_audit_row(org_id=org_id, memory_id=mid, details=None)

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = [row]
        conn.execute.return_value = result_mock

        entries = await get_memory_lifecycle(conn, org_id=org_id, memory_id=mid)

        assert entries[0].details == {}
