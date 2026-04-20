"""Unit tests for z3rno_core.engine.get — pure logic only (no DB).

Tests MemoryNotFoundError, MemoryDetail construction, and the
get_memory function using mocked connections.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.engine.get import MemoryDetail, MemoryNotFoundError, get_memory

# ---------------------------------------------------------------------------
# MemoryNotFoundError
# ---------------------------------------------------------------------------


class TestMemoryNotFoundError:
    """Test MemoryNotFoundError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(MemoryNotFoundError, Exception)

    def test_message_includes_id(self) -> None:
        mid = uuid4()
        err = MemoryNotFoundError(f"Memory {mid} not found")
        assert str(mid) in str(err)

    def test_raises(self) -> None:
        with pytest.raises(MemoryNotFoundError, match="not found"):
            raise MemoryNotFoundError("Memory xyz not found")


# ---------------------------------------------------------------------------
# MemoryDetail dataclass
# ---------------------------------------------------------------------------


class TestMemoryDetail:
    """Test MemoryDetail frozen dataclass."""

    def test_construction(self) -> None:
        mid = uuid4()
        org_id = uuid4()
        agent_id = uuid4()
        now = datetime.now(tz=UTC)
        detail = MemoryDetail(
            id=mid,
            org_id=org_id,
            agent_id=agent_id,
            user_id=None,
            memory_type="episodic",
            content="test content",
            summary=None,
            metadata={"key": "val"},
            importance_score=0.7,
            recall_count=5,
            last_recalled_at=None,
            embedding_model="text-embedding-3-small",
            pinned=False,
            created_at=now,
            valid_from=now,
            valid_to=None,
            deleted_at=None,
        )
        assert detail.id == mid
        assert detail.content == "test content"
        assert detail.importance_score == 0.7

    def test_frozen(self) -> None:
        now = datetime.now(tz=UTC)
        detail = MemoryDetail(
            id=uuid4(),
            org_id=uuid4(),
            agent_id=uuid4(),
            user_id=None,
            memory_type="semantic",
            content="x",
            summary=None,
            metadata={},
            importance_score=0.5,
            recall_count=0,
            last_recalled_at=None,
            embedding_model=None,
            pinned=False,
            created_at=now,
            valid_from=now,
            valid_to=None,
            deleted_at=None,
        )
        with pytest.raises(AttributeError):
            detail.content = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory_row(
    *,
    memory_id=None,
    org_id=None,
    agent_id=None,
    user_id=None,
    memory_type: str = "episodic",
    content: str = "test content",
    summary: str | None = None,
    metadata: dict | None = None,  # type: ignore[type-arg]
    importance_score: float = 0.7,
    recall_count: int = 3,
    last_recalled_at=None,
    embedding_model: str | None = "text-embedding-3-small",
    pinned: bool = False,
    created_at: datetime | None = None,
    valid_from: datetime | None = None,
    valid_to=None,
    deleted_at=None,
) -> tuple:  # type: ignore[type-arg]
    """Create a mock DB row matching the get_memory SELECT columns."""
    if memory_id is None:
        memory_id = uuid4()
    if org_id is None:
        org_id = uuid4()
    if agent_id is None:
        agent_id = uuid4()
    if created_at is None:
        created_at = datetime.now(tz=UTC)
    if valid_from is None:
        valid_from = created_at
    return (
        memory_id,
        org_id,
        agent_id,
        user_id,
        memory_type,
        content,
        summary,
        metadata,
        importance_score,
        recall_count,
        last_recalled_at,
        embedding_model,
        pinned,
        created_at,
        valid_from,
        valid_to,
        deleted_at,
    )


# ---------------------------------------------------------------------------
# get_memory()
# ---------------------------------------------------------------------------


class TestGetMemory:
    """Test the get_memory function."""

    async def test_returns_memory_detail(self) -> None:
        """Returns a MemoryDetail when the memory exists."""
        org_id = uuid4()
        mid = uuid4()
        row = _make_memory_row(memory_id=mid, org_id=org_id)

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = row
        conn.execute.return_value = result_mock

        detail = await get_memory(conn, org_id=org_id, memory_id=mid)

        assert isinstance(detail, MemoryDetail)
        assert detail.id == mid
        assert detail.org_id == org_id
        assert detail.content == "test content"

    async def test_raises_not_found(self) -> None:
        """Raises MemoryNotFoundError when the memory doesn't exist."""
        org_id = uuid4()
        mid = uuid4()

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = None
        conn.execute.return_value = result_mock

        with pytest.raises(MemoryNotFoundError, match=str(mid)):
            await get_memory(conn, org_id=org_id, memory_id=mid)

    async def test_include_deleted_changes_sql(self) -> None:
        """include_deleted=True omits the deleted_at filter."""
        org_id = uuid4()
        mid = uuid4()
        row = _make_memory_row(memory_id=mid, org_id=org_id)

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = row
        conn.execute.return_value = result_mock

        await get_memory(conn, org_id=org_id, memory_id=mid, include_deleted=True)

        call_args = conn.execute.call_args_list[0]
        sql_text = str(call_args[0][0].text)
        assert "deleted_at IS NULL" not in sql_text

    async def test_default_excludes_deleted(self) -> None:
        """Default include_deleted=False includes deleted_at filter."""
        org_id = uuid4()
        mid = uuid4()
        row = _make_memory_row(memory_id=mid, org_id=org_id)

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = row
        conn.execute.return_value = result_mock

        await get_memory(conn, org_id=org_id, memory_id=mid)

        call_args = conn.execute.call_args_list[0]
        sql_text = str(call_args[0][0].text)
        assert "deleted_at IS NULL" in sql_text

    async def test_null_metadata_becomes_empty_dict(self) -> None:
        """Row with None metadata is converted to empty dict."""
        org_id = uuid4()
        mid = uuid4()
        row = _make_memory_row(memory_id=mid, org_id=org_id, metadata=None)

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = row
        conn.execute.return_value = result_mock

        detail = await get_memory(conn, org_id=org_id, memory_id=mid)

        assert detail.metadata == {}

    async def test_all_fields_populated(self) -> None:
        """All fields are correctly mapped from the row."""
        org_id = uuid4()
        mid = uuid4()
        agent_id = uuid4()
        user_id = uuid4()
        now = datetime.now(tz=UTC)
        meta = {"project": "z3rno"}
        row = _make_memory_row(
            memory_id=mid,
            org_id=org_id,
            agent_id=agent_id,
            user_id=user_id,
            memory_type="semantic",
            content="deep knowledge",
            summary="summary text",
            metadata=meta,
            importance_score=0.95,
            recall_count=42,
            last_recalled_at=now,
            embedding_model="text-embedding-3-small",
            pinned=True,
            created_at=now,
            valid_from=now,
            valid_to=None,
            deleted_at=None,
        )

        conn = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = row
        conn.execute.return_value = result_mock

        detail = await get_memory(conn, org_id=org_id, memory_id=mid)

        assert detail.agent_id == agent_id
        assert detail.user_id == user_id
        assert detail.memory_type == "semantic"
        assert detail.summary == "summary text"
        assert detail.metadata == meta
        assert detail.importance_score == 0.95
        assert detail.recall_count == 42
        assert detail.pinned is True
        assert detail.embedding_model == "text-embedding-3-small"
