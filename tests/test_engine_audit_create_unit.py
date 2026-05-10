"""Unit tests for z3rno_core.engine.audit — async functions (no DB).

Covers:
- ``get_latest_hash``
- ``enqueue_audit_entry`` (the v0.7.0 fast path; writes to audit_log_pending,
  one execute call, no chain lookup, fully parallel-safe)
- ``create_audit_entry`` (backwards-compat alias for ``enqueue_audit_entry``)

The drainer (``drain_audit_chain`` / ``flush_audit_chain``) needs the real
audit_log + audit_log_pending tables to exercise meaningfully and is
covered by the integration suite.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from z3rno_core.engine.audit import (
    create_audit_entry,
    enqueue_audit_entry,
    get_latest_hash,
)

# ---------------------------------------------------------------------------
# get_latest_hash
# ---------------------------------------------------------------------------


class TestGetLatestHash:
    """Test get_latest_hash async function."""

    async def test_returns_none_for_first_entry(self) -> None:
        """Returns None when no previous hash exists."""
        org_id = uuid4()
        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = None
        conn.execute.return_value = result_mock

        result = await get_latest_hash(conn, org_id)

        assert result is None

    async def test_returns_hash_bytes(self) -> None:
        """Returns the row_hash bytes from the latest entry."""
        org_id = uuid4()
        expected_hash = b"\xab" * 32
        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = (expected_hash,)
        conn.execute.return_value = result_mock

        result = await get_latest_hash(conn, org_id)

        assert result == expected_hash

    async def test_queries_correct_org_id(self) -> None:
        """Passes org_id as parameter to the query."""
        org_id = uuid4()
        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = None
        conn.execute.return_value = result_mock

        await get_latest_hash(conn, org_id)

        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert params["org_id"] == str(org_id)


# ---------------------------------------------------------------------------
# enqueue_audit_entry
# ---------------------------------------------------------------------------


class TestEnqueueAuditEntry:
    """Tests for the fast write path that lands a row in audit_log_pending."""

    async def test_single_execute_call(self) -> None:
        """One INSERT into audit_log_pending — no SELECT prev_hash, no chain."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = AsyncMock()

        await enqueue_audit_entry(
            conn, org_id=org_id, operation="store", agent_id=agent_id
        )

        assert conn.execute.call_count == 1

    async def test_inserts_required_fields(self) -> None:
        """org_id, operation, agent_id round-trip into the parameter dict."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = AsyncMock()

        await enqueue_audit_entry(
            conn, org_id=org_id, operation="store", agent_id=agent_id
        )

        insert_call = conn.execute.call_args_list[0]
        params = insert_call[0][1]
        assert params["org_id"] == str(org_id)
        assert params["operation"] == "store"
        assert params["agent_id"] == str(agent_id)

    async def test_optional_fields_default_none(self) -> None:
        """Unset optional fields stay None in the params dict."""
        org_id = uuid4()
        conn = AsyncMock()

        await enqueue_audit_entry(conn, org_id=org_id, operation="store")

        params = conn.execute.call_args_list[0][0][1]
        assert params["agent_id"] is None
        assert params["user_id"] is None
        assert params["memory_id"] is None
        assert params["memory_type"] is None
        assert params["api_key_id"] is None
        assert params["ip_address"] is None
        assert params["user_agent"] is None
        assert params["request_id"] is None

    async def test_all_optional_fields_populated(self) -> None:
        """Every optional field round-trips when supplied."""
        org_id = uuid4()
        agent_id = uuid4()
        user_id = uuid4()
        memory_id = uuid4()
        api_key_id = uuid4()
        conn = AsyncMock()

        await enqueue_audit_entry(
            conn,
            org_id=org_id,
            operation="forget",
            agent_id=agent_id,
            user_id=user_id,
            memory_id=memory_id,
            memory_type="episodic",
            details={"reason": "test"},
            api_key_id=api_key_id,
            ip_address="127.0.0.1",
            user_agent="test-agent/1.0",
            request_id="req-123",
        )

        params = conn.execute.call_args_list[0][0][1]
        assert params["agent_id"] == str(agent_id)
        assert params["user_id"] == str(user_id)
        assert params["memory_id"] == str(memory_id)
        assert params["memory_type"] == "episodic"
        assert params["api_key_id"] == str(api_key_id)
        assert params["ip_address"] == "127.0.0.1"
        assert params["user_agent"] == "test-agent/1.0"
        assert params["request_id"] == "req-123"

    async def test_empty_details_serialized(self) -> None:
        """None / empty details serializes as the empty JSON object."""
        org_id = uuid4()
        conn = AsyncMock()

        await enqueue_audit_entry(conn, org_id=org_id, operation="store")

        params = conn.execute.call_args_list[0][0][1]
        assert params["details"] == "{}"


# ---------------------------------------------------------------------------
# Backwards compatibility — create_audit_entry is an alias for enqueue
# ---------------------------------------------------------------------------


class TestCreateAuditEntryAlias:
    """create_audit_entry remains importable; redirects to enqueue."""

    async def test_alias_points_to_enqueue(self) -> None:
        assert create_audit_entry is enqueue_audit_entry

    async def test_alias_call_does_one_execute(self) -> None:
        org_id = uuid4()
        conn = AsyncMock()

        await create_audit_entry(conn, org_id=org_id, operation="store")

        assert conn.execute.call_count == 1
