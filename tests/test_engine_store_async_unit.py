"""Unit tests for z3rno_core.engine.store async paths — no DB.

Tests the store() function's validation, embedding generation,
TTL computation, and relationship insertion using mocked connections.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from z3rno_core.engine.store import DuplicateMemoryError, StoreError, StoreResult, store
from z3rno_core.models.enums import MemoryType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_conn() -> AsyncMock:
    """Create a mock AsyncConnection for store operations."""
    conn = AsyncMock()
    result_mock = MagicMock()
    result_mock.rowcount = 1
    conn.execute.return_value = result_mock
    return conn


# ---------------------------------------------------------------------------
# store() — validation
# ---------------------------------------------------------------------------


class TestStoreValidation:
    """Test store() input validation."""

    async def test_empty_content_raises(self) -> None:
        """Empty content raises StoreError."""
        conn = _make_mock_conn()

        with pytest.raises(StoreError, match="non-empty"):
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="",
                memory_type=MemoryType.EPISODIC,
            )

    async def test_whitespace_only_content_raises(self) -> None:
        """Whitespace-only content raises StoreError."""
        conn = _make_mock_conn()

        with pytest.raises(StoreError, match="non-empty"):
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="   \n\t  ",
                memory_type=MemoryType.WORKING,
            )

    async def test_importance_below_zero_raises(self) -> None:
        """Negative importance score raises StoreError."""
        conn = _make_mock_conn()

        with pytest.raises(StoreError, match="importance"):
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="valid content",
                memory_type=MemoryType.SEMANTIC,
                importance=-0.1,
            )

    async def test_importance_above_one_raises(self) -> None:
        """Importance score > 1.0 raises StoreError."""
        conn = _make_mock_conn()

        with pytest.raises(StoreError, match="importance"):
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="valid content",
                memory_type=MemoryType.PROCEDURAL,
                importance=1.1,
            )


# ---------------------------------------------------------------------------
# store() — successful creation
# ---------------------------------------------------------------------------


class TestStoreSuccess:
    """Test successful store() operations."""

    async def test_basic_store(self) -> None:
        """Basic store without embedding returns StoreResult."""
        conn = _make_mock_conn()

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            result = await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="test content",
                memory_type=MemoryType.EPISODIC,
            )

        assert isinstance(result, StoreResult)
        assert result.memory_id is not None
        assert result.embedding_model is None
        assert result.importance_score == 0.5  # Default

    async def test_store_with_custom_importance(self) -> None:
        """Custom importance score is used."""
        conn = _make_mock_conn()

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            result = await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="important",
                memory_type=MemoryType.SEMANTIC,
                importance=0.9,
            )

        assert result.importance_score == 0.9

    async def test_store_with_embedding_provider(self) -> None:
        """Store with an embedding provider generates and stores embedding."""
        conn = _make_mock_conn()
        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = [0.1, 0.2, 0.3]
        embed_provider.model_name = "text-embedding-3-small"

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            result = await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="embed me",
                memory_type=MemoryType.EPISODIC,
                embedding_provider=embed_provider,
            )

        assert result.embedding_model == "text-embedding-3-small"
        embed_provider.embed_text.assert_called_once_with("embed me")

    async def test_store_with_noop_embedding_provider(self) -> None:
        """NoOp provider returns empty list, embedding is None."""
        conn = _make_mock_conn()
        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = []  # NoOp
        embed_provider.model_name = "noop"

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            result = await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="no embedding",
                memory_type=MemoryType.WORKING,
                embedding_provider=embed_provider,
            )

        assert result.embedding_model == "noop"
        # Embedding is None (empty list treated as no embedding)
        insert_call = conn.execute.call_args_list[0]
        params = insert_call[0][1]
        assert params["embedding"] is None

    async def test_store_with_ttl(self) -> None:
        """TTL creates ttl_expires_at timestamp."""
        conn = _make_mock_conn()

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="temporary",
                memory_type=MemoryType.WORKING,
                ttl_seconds=3600,
            )

        insert_call = conn.execute.call_args_list[0]
        params = insert_call[0][1]
        assert params["ttl_expires_at"] is not None

    async def test_store_without_ttl(self) -> None:
        """No TTL means ttl_expires_at is None."""
        conn = _make_mock_conn()

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="permanent",
                memory_type=MemoryType.SEMANTIC,
            )

        insert_call = conn.execute.call_args_list[0]
        params = insert_call[0][1]
        assert params["ttl_expires_at"] is None

    async def test_store_with_metadata(self) -> None:
        """Metadata dict is serialized to JSON."""
        conn = _make_mock_conn()
        meta = {"project": "z3rno", "tags": ["test"]}

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="with metadata",
                memory_type=MemoryType.EPISODIC,
                metadata=meta,
            )

        insert_call = conn.execute.call_args_list[0]
        params = insert_call[0][1]
        assert '"project"' in params["metadata"]

    async def test_store_creates_audit_entry(self) -> None:
        """Store always creates an audit entry."""
        conn = _make_mock_conn()
        org_id = uuid4()

        with patch(
            "z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            await store(
                conn,
                org_id=org_id,
                agent_id=uuid4(),
                content="audit me",
                memory_type=MemoryType.EPISODIC,
            )

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["operation"] == "store"
        assert call_kwargs["org_id"] == org_id


# ---------------------------------------------------------------------------
# store() — relationships
# ---------------------------------------------------------------------------


class TestStoreRelationships:
    """Test relationship creation during store."""

    async def test_store_with_relationship(self) -> None:
        """Relationships are inserted alongside the memory."""
        from z3rno_core.engine.store import RelationshipInput

        conn = _make_mock_conn()
        target = uuid4()

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="linked memory",
                memory_type=MemoryType.EPISODIC,
                relationships=[
                    RelationshipInput(
                        target_memory_id=target,
                        relationship_type="derived_from",
                    )
                ],
            )

        # 1 memory INSERT + 1 relationship INSERT + 3 from memo_versioning
        # (Phase F slice 3): SELECT prior, UPDATE close (the mock returns
        # a truthy fetchone so the close-prior branch fires), INSERT new.
        assert conn.execute.call_count == 5
        rel_call = conn.execute.call_args_list[1]
        params = rel_call[0][1]
        assert params["target"] == str(target)
        assert params["rel_type"] == "derived_from"

    async def test_store_without_relationships(self) -> None:
        """No relationship insertion when none provided."""
        conn = _make_mock_conn()

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="standalone",
                memory_type=MemoryType.SEMANTIC,
            )

        # 1 memory INSERT + 3 from memo_versioning (Phase F slice 3:
        # SELECT prior, UPDATE close, INSERT new — the mock returns a
        # truthy fetchone so the close-prior branch fires).
        assert conn.execute.call_count == 4


# ---------------------------------------------------------------------------
# store() — near-duplicate detection
# ---------------------------------------------------------------------------


def _make_dup_mock_conn(
    dup_row: tuple | None = None,  # type: ignore[type-arg]
) -> AsyncMock:
    """Create a mock AsyncConnection that returns dup_row for the first
    execute (duplicate check) and succeeds for subsequent calls (INSERT)."""
    conn = AsyncMock()

    # First call: duplicate check SELECT
    dup_result = MagicMock()
    dup_result.fetchone.return_value = dup_row

    # Subsequent calls: INSERT (memory + relationships)
    insert_result = MagicMock()
    insert_result.rowcount = 1

    conn.execute.side_effect = [dup_result, insert_result, insert_result]
    return conn


class TestStoreDuplicateDetection:
    """Test near-duplicate detection in store()."""

    async def test_duplicate_detected_raises(self) -> None:
        """When similarity > threshold, DuplicateMemoryError is raised."""
        existing_id = uuid4()
        conn = _make_dup_mock_conn(dup_row=(existing_id, 0.98))

        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = [0.1, 0.2, 0.3]
        embed_provider.model_name = "text-embedding-3-small"

        with pytest.raises(DuplicateMemoryError) as exc_info:
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="duplicate content",
                memory_type=MemoryType.EPISODIC,
                embedding_provider=embed_provider,
                check_duplicates=True,
            )

        assert exc_info.value.existing_memory_id == existing_id
        assert exc_info.value.similarity == 0.98

    async def test_no_duplicate_stores_normally(self) -> None:
        """When similarity <= threshold, memory is stored."""
        conn = _make_dup_mock_conn(dup_row=(uuid4(), 0.50))

        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = [0.1, 0.2, 0.3]
        embed_provider.model_name = "text-embedding-3-small"

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            result = await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="unique content",
                memory_type=MemoryType.EPISODIC,
                embedding_provider=embed_provider,
                check_duplicates=True,
            )

        assert isinstance(result, StoreResult)

    async def test_no_existing_memories_stores_normally(self) -> None:
        """When no existing memories, store proceeds."""
        conn = _make_dup_mock_conn(dup_row=None)

        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = [0.1, 0.2, 0.3]
        embed_provider.model_name = "text-embedding-3-small"

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            result = await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="first memory",
                memory_type=MemoryType.SEMANTIC,
                embedding_provider=embed_provider,
                check_duplicates=True,
            )

        assert isinstance(result, StoreResult)

    async def test_custom_threshold(self) -> None:
        """Custom duplicate_threshold is respected."""
        existing_id = uuid4()
        # Similarity is 0.90, default threshold 0.95 would pass, but 0.85 catches it
        conn = _make_dup_mock_conn(dup_row=(existing_id, 0.90))

        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = [0.1, 0.2, 0.3]
        embed_provider.model_name = "text-embedding-3-small"

        with pytest.raises(DuplicateMemoryError):
            await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="close enough",
                memory_type=MemoryType.EPISODIC,
                embedding_provider=embed_provider,
                check_duplicates=True,
                duplicate_threshold=0.85,
            )

    async def test_check_duplicates_false_skips_check(self) -> None:
        """When check_duplicates=False (default), no duplicate check is done."""
        conn = _make_mock_conn()

        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = [0.1, 0.2, 0.3]
        embed_provider.model_name = "text-embedding-3-small"

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            result = await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="no dup check",
                memory_type=MemoryType.EPISODIC,
                embedding_provider=embed_provider,
                check_duplicates=False,
            )

        assert isinstance(result, StoreResult)
        # 1 INSERT + 3 from memo_versioning (mock fetchone is truthy
        # so the close-prior UPDATE branch fires), no duplicate check.
        assert conn.execute.call_count == 4

    async def test_check_duplicates_without_embedding_skips(self) -> None:
        """When no embedding is generated, duplicate check is skipped."""
        conn = _make_mock_conn()

        with patch("z3rno_core.engine.store.create_audit_entry", new_callable=AsyncMock):
            result = await store(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                content="no embedding provider",
                memory_type=MemoryType.WORKING,
                check_duplicates=True,
            )

        assert isinstance(result, StoreResult)
        # 1 INSERT + 3 from memo_versioning (mock fetchone is truthy
        # so close-prior UPDATE branch fires), no duplicate check.
        assert conn.execute.call_count == 4
