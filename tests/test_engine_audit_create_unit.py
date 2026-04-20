"""Unit tests for z3rno_core.engine.audit — async functions (no DB).

Tests get_latest_hash and create_audit_entry using mocked connections.
The compute_row_hash tests are in test_engine_audit.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from z3rno_core.engine.audit import create_audit_entry, get_latest_hash

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
# create_audit_entry
# ---------------------------------------------------------------------------


class TestCreateAuditEntry:
    """Test create_audit_entry async function."""

    async def test_inserts_audit_entry(self) -> None:
        """Creates an audit log entry with correct parameters."""
        org_id = uuid4()
        agent_id = uuid4()

        conn = AsyncMock()
        # First call: get_latest_hash SELECT
        hash_result = MagicMock()
        hash_result.fetchone.return_value = None
        # Second call: INSERT
        insert_result = MagicMock()
        conn.execute.side_effect = [hash_result, insert_result]

        await create_audit_entry(
            conn,
            org_id=org_id,
            operation="store",
            agent_id=agent_id,
        )

        # Second call should be the INSERT
        assert conn.execute.call_count == 2
        insert_call = conn.execute.call_args_list[1]
        params = insert_call[0][1]
        assert params["org_id"] == str(org_id)
        assert params["operation"] == "store"
        assert params["agent_id"] == str(agent_id)

    async def test_chains_from_previous_hash(self) -> None:
        """Uses previous hash for chain computation."""
        org_id = uuid4()
        prev_hash = b"\xde\xad" * 16

        conn = AsyncMock()
        hash_result = MagicMock()
        hash_result.fetchone.return_value = (prev_hash,)
        insert_result = MagicMock()
        conn.execute.side_effect = [hash_result, insert_result]

        await create_audit_entry(
            conn,
            org_id=org_id,
            operation="recall",
        )

        insert_call = conn.execute.call_args_list[1]
        params = insert_call[0][1]
        assert params["prev_hash"] == prev_hash
        assert params["row_hash"] is not None
        assert isinstance(params["row_hash"], bytes)

    async def test_null_prev_hash_for_first_entry(self) -> None:
        """First entry has None prev_hash."""
        org_id = uuid4()

        conn = AsyncMock()
        hash_result = MagicMock()
        hash_result.fetchone.return_value = None
        insert_result = MagicMock()
        conn.execute.side_effect = [hash_result, insert_result]

        await create_audit_entry(
            conn,
            org_id=org_id,
            operation="store",
        )

        insert_call = conn.execute.call_args_list[1]
        params = insert_call[0][1]
        assert params["prev_hash"] is None

    async def test_optional_fields_none_by_default(self) -> None:
        """Optional fields default to None."""
        org_id = uuid4()

        conn = AsyncMock()
        hash_result = MagicMock()
        hash_result.fetchone.return_value = None
        insert_result = MagicMock()
        conn.execute.side_effect = [hash_result, insert_result]

        await create_audit_entry(
            conn,
            org_id=org_id,
            operation="store",
        )

        insert_call = conn.execute.call_args_list[1]
        params = insert_call[0][1]
        assert params["agent_id"] is None
        assert params["user_id"] is None
        assert params["memory_id"] is None
        assert params["memory_type"] is None
        assert params["api_key_id"] is None
        assert params["ip_address"] is None
        assert params["user_agent"] is None
        assert params["request_id"] is None

    async def test_all_optional_fields_populated(self) -> None:
        """All optional fields are correctly passed."""
        org_id = uuid4()
        agent_id = uuid4()
        user_id = uuid4()
        memory_id = uuid4()
        api_key_id = uuid4()

        conn = AsyncMock()
        hash_result = MagicMock()
        hash_result.fetchone.return_value = None
        insert_result = MagicMock()
        conn.execute.side_effect = [hash_result, insert_result]

        await create_audit_entry(
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

        insert_call = conn.execute.call_args_list[1]
        params = insert_call[0][1]
        assert params["agent_id"] == str(agent_id)
        assert params["user_id"] == str(user_id)
        assert params["memory_id"] == str(memory_id)
        assert params["memory_type"] == "episodic"
        assert params["api_key_id"] == str(api_key_id)
        assert params["ip_address"] == "127.0.0.1"
        assert params["user_agent"] == "test-agent/1.0"
        assert params["request_id"] == "req-123"

    async def test_empty_details_serialized(self) -> None:
        """Empty/None details is serialized as empty JSON object."""
        org_id = uuid4()

        conn = AsyncMock()
        hash_result = MagicMock()
        hash_result.fetchone.return_value = None
        insert_result = MagicMock()
        conn.execute.side_effect = [hash_result, insert_result]

        await create_audit_entry(
            conn,
            org_id=org_id,
            operation="store",
        )

        insert_call = conn.execute.call_args_list[1]
        params = insert_call[0][1]
        assert params["details"] == "{}"
