"""Unit tests for z3rno_core.engine.update — pure logic only (no DB).

Tests UpdateError, UpdateResult, and update_memory function using mocked connections.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.engine.update import UpdateError, UpdateResult, update_memory

# ---------------------------------------------------------------------------
# UpdateError
# ---------------------------------------------------------------------------


class TestUpdateError:
    """Test UpdateError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(UpdateError, Exception)

    def test_message(self) -> None:
        err = UpdateError("bad input")
        assert str(err) == "bad input"


# ---------------------------------------------------------------------------
# UpdateResult dataclass
# ---------------------------------------------------------------------------


class TestUpdateResult:
    """Test UpdateResult frozen dataclass."""

    def test_construction(self) -> None:
        mid = uuid4()
        now = datetime.now(tz=UTC)
        r = UpdateResult(
            memory_id=mid,
            content="hello",
            metadata={"key": "val"},
            importance_score=0.5,
            updated_at=now,
        )
        assert r.memory_id == mid
        assert r.content == "hello"
        assert r.metadata == {"key": "val"}
        assert r.importance_score == 0.5
        assert r.updated_at == now

    def test_frozen(self) -> None:
        r = UpdateResult(
            memory_id=uuid4(),
            content="hi",
            metadata={},
            importance_score=0.5,
            updated_at=datetime.now(tz=UTC),
        )
        with pytest.raises(AttributeError):
            r.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestUpdateMemoryValidation:
    """Test update_memory() validation logic."""

    @pytest.mark.asyncio
    async def test_no_fields_raises(self) -> None:
        conn = AsyncMock()
        with pytest.raises(UpdateError, match="At least one"):
            await update_memory(conn, org_id=uuid4(), memory_id=uuid4())

    @pytest.mark.asyncio
    async def test_empty_content_raises(self) -> None:
        conn = AsyncMock()
        with pytest.raises(UpdateError, match="non-empty"):
            await update_memory(conn, org_id=uuid4(), memory_id=uuid4(), content="")

    @pytest.mark.asyncio
    async def test_whitespace_content_raises(self) -> None:
        conn = AsyncMock()
        with pytest.raises(UpdateError, match="non-empty"):
            await update_memory(conn, org_id=uuid4(), memory_id=uuid4(), content="   ")

    @pytest.mark.asyncio
    async def test_importance_below_zero_raises(self) -> None:
        conn = AsyncMock()
        with pytest.raises(UpdateError, match=r"between 0\.0 and 1\.0"):
            await update_memory(conn, org_id=uuid4(), memory_id=uuid4(), importance=-0.1)

    @pytest.mark.asyncio
    async def test_importance_above_one_raises(self) -> None:
        conn = AsyncMock()
        with pytest.raises(UpdateError, match=r"between 0\.0 and 1\.0"):
            await update_memory(conn, org_id=uuid4(), memory_id=uuid4(), importance=1.5)


# ---------------------------------------------------------------------------
# Successful update
# ---------------------------------------------------------------------------


def _make_mock_conn(
    update_row: tuple[Any, ...] | None = None,
    agent_row: tuple[Any, ...] | None = None,
) -> AsyncMock:
    """Build a mock AsyncConnection that returns the given rows."""
    conn = AsyncMock()
    update_result = MagicMock()
    update_result.fetchone.return_value = update_row

    agent_result = MagicMock()
    agent_result.fetchone.return_value = agent_row

    conn.execute = AsyncMock(side_effect=[update_result, agent_result])
    return conn


class TestUpdateMemoryExecution:
    """Test update_memory() SQL execution paths."""

    @pytest.mark.asyncio
    async def test_not_found_raises(self) -> None:
        conn = _make_mock_conn(update_row=None)
        from z3rno_core.engine.get import MemoryNotFoundError

        with pytest.raises(MemoryNotFoundError, match="not found"):
            await update_memory(conn, org_id=uuid4(), memory_id=uuid4(), content="new content")

    @pytest.mark.asyncio
    async def test_content_update_calls_execute(self) -> None:
        mid = uuid4()
        oid = uuid4()
        aid = uuid4()
        now = datetime.now(tz=UTC)

        conn = AsyncMock()
        # execute calls: 1) UPDATE, 2) audit get_latest_hash, 3) audit INSERT, 4) get_memory
        update_result = MagicMock()
        update_result.fetchone.return_value = (mid,)

        agent_result = MagicMock()
        agent_result.fetchone.return_value = (aid,)

        audit_hash_result = MagicMock()
        audit_hash_result.fetchone.return_value = None

        audit_insert_result = MagicMock()

        get_result = MagicMock()
        get_result.fetchone.return_value = (
            mid,  # id
            oid,  # org_id
            aid,  # agent_id
            None,  # user_id
            "episodic",  # memory_type
            "new content",  # content
            None,  # summary
            {"key": "val"},  # memory_metadata
            0.5,  # importance_score
            0,  # recall_count
            None,  # last_recalled_at
            None,  # embedding_model
            False,  # pinned
            now,  # created_at
            now,  # valid_from
            None,  # valid_to
            None,  # deleted_at
        )

        conn.execute = AsyncMock(
            side_effect=[
                update_result,
                agent_result,
                audit_hash_result,
                audit_insert_result,
                get_result,
            ]
        )

        result = await update_memory(conn, org_id=oid, memory_id=mid, content="new content")

        assert result.id == mid
        assert conn.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_metadata_update(self) -> None:
        mid = uuid4()
        oid = uuid4()
        aid = uuid4()
        now = datetime.now(tz=UTC)

        conn = AsyncMock()
        update_result = MagicMock()
        update_result.fetchone.return_value = (mid,)

        # agent_id provided, so no agent lookup needed
        audit_hash_result = MagicMock()
        audit_hash_result.fetchone.return_value = None

        audit_insert_result = MagicMock()

        get_result = MagicMock()
        get_result.fetchone.return_value = (
            mid,
            oid,
            aid,
            None,
            "episodic",
            "content",
            None,
            {"new": "meta"},
            0.5,
            0,
            None,
            None,
            False,
            now,
            now,
            None,
            None,
        )

        conn.execute = AsyncMock(
            side_effect=[update_result, audit_hash_result, audit_insert_result, get_result]
        )

        result = await update_memory(
            conn,
            org_id=oid,
            memory_id=mid,
            metadata={"new": "meta"},
            agent_id=aid,
        )

        assert result.id == mid

    @pytest.mark.asyncio
    async def test_importance_update(self) -> None:
        mid = uuid4()
        oid = uuid4()
        aid = uuid4()
        now = datetime.now(tz=UTC)

        conn = AsyncMock()
        update_result = MagicMock()
        update_result.fetchone.return_value = (mid,)

        audit_hash_result = MagicMock()
        audit_hash_result.fetchone.return_value = None

        audit_insert_result = MagicMock()

        get_result = MagicMock()
        get_result.fetchone.return_value = (
            mid,
            oid,
            aid,
            None,
            "episodic",
            "content",
            None,
            {},
            0.8,
            0,
            None,
            None,
            False,
            now,
            now,
            None,
            None,
        )

        conn.execute = AsyncMock(
            side_effect=[update_result, audit_hash_result, audit_insert_result, get_result]
        )

        result = await update_memory(
            conn,
            org_id=oid,
            memory_id=mid,
            importance=0.8,
            agent_id=aid,
        )

        assert result.id == mid
