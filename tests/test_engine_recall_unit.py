"""Unit tests for z3rno_core.engine.recall — pure logic only (no DB).

Tests the recall function's query building, fused relevance scoring,
similarity threshold filtering, and edge cases using mocked connections.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from z3rno_core.engine.recall import RecallError, RecallResult, _fallback_query, recall

# ---------------------------------------------------------------------------
# _fallback_query helper
# ---------------------------------------------------------------------------


class TestFallbackQuery:
    """Test the _fallback_query SQL builder."""

    def test_returns_string(self) -> None:
        result = _fallback_query("org_id = :org_id")
        assert isinstance(result, str)

    def test_contains_where_clause(self) -> None:
        result = _fallback_query("org_id = :org_id AND agent_id = :agent_id")
        assert "org_id = :org_id AND agent_id = :agent_id" in result

    def test_selects_null_similarity(self) -> None:
        result = _fallback_query("1=1")
        assert "NULL AS similarity" in result

    def test_orders_by_importance_then_recency(self) -> None:
        result = _fallback_query("1=1")
        assert "ORDER BY importance_score DESC, created_at DESC" in result

    def test_has_limit_placeholder(self) -> None:
        result = _fallback_query("1=1")
        assert ":top_k" in result


# ---------------------------------------------------------------------------
# RecallError
# ---------------------------------------------------------------------------


class TestRecallError:
    """Test RecallError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(RecallError, Exception)

    def test_message(self) -> None:
        err = RecallError("something went wrong")
        assert str(err) == "something went wrong"


# ---------------------------------------------------------------------------
# RecallResult dataclass
# ---------------------------------------------------------------------------


class TestRecallResult:
    """Test RecallResult frozen dataclass."""

    def test_construction(self) -> None:
        mid = uuid4()
        now = datetime.now(tz=UTC)
        r = RecallResult(
            memory_id=mid,
            content="hello",
            summary=None,
            memory_type="episodic",
            similarity_score=0.9,
            importance_score=0.7,
            relevance_score=0.8,
            recall_count=5,
            created_at=now,
            valid_from=now,
            metadata={"key": "val"},
        )
        assert r.memory_id == mid
        assert r.content == "hello"
        assert r.relevance_score == 0.8

    def test_frozen(self) -> None:
        now = datetime.now(tz=UTC)
        r = RecallResult(
            memory_id=uuid4(),
            content="x",
            summary=None,
            memory_type="episodic",
            similarity_score=0.9,
            importance_score=0.7,
            relevance_score=0.8,
            recall_count=0,
            created_at=now,
            valid_from=now,
            metadata={},
        )
        with pytest.raises(AttributeError):
            r.content = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers for mocking async DB calls
# ---------------------------------------------------------------------------


def _make_mock_conn(
    rows: list[tuple],  # type: ignore[type-arg]
    *,
    scalar_value: int | None = None,
) -> AsyncMock:
    """Create a mock AsyncConnection that returns the given rows."""
    conn = AsyncMock()
    result_mock = MagicMock()
    result_mock.fetchall.return_value = rows
    result_mock.fetchone.return_value = rows[0] if rows else None
    if scalar_value is not None:
        result_mock.scalar.return_value = scalar_value
    conn.execute.return_value = result_mock
    return conn


def _make_memory_row(
    *,
    memory_id: Any = None,
    content: str = "test content",
    summary: str | None = None,
    memory_type: str = "episodic",
    importance_score: float = 0.7,
    recall_count: int = 3,
    created_at: datetime | None = None,
    valid_from: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    similarity: float | None = None,
) -> tuple[Any, ...]:
    """Create a mock DB row matching the recall() SELECT columns."""
    if memory_id is None:
        memory_id = uuid4()
    if created_at is None:
        created_at = datetime.now(tz=UTC) - timedelta(hours=1)
    if valid_from is None:
        valid_from = created_at
    if metadata is None:
        metadata = {}
    return (
        memory_id,
        content,
        summary,
        memory_type,
        importance_score,
        recall_count,
        created_at,
        valid_from,
        metadata,
        similarity,
    )


# ---------------------------------------------------------------------------
# recall() — query building and filter tests
# ---------------------------------------------------------------------------


class TestRecallQueryBuilding:
    """Test that recall() builds the correct SQL conditions."""

    async def test_basic_recall_no_query(self) -> None:
        """Without a query, recall uses the fallback query."""
        org_id = uuid4()
        agent_id = uuid4()
        row = _make_memory_row()
        conn = _make_mock_conn([row])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(conn, org_id=org_id, agent_id=agent_id)

        assert len(results) == 1
        # Verify execute was called
        conn.execute.assert_called()

    async def test_memory_type_filter(self) -> None:
        """memory_type filter adds a condition to the SQL."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([_make_memory_row()])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            await recall(conn, org_id=org_id, agent_id=agent_id, memory_type="semantic")

        # Check the SQL includes the memory_type filter
        call_args = conn.execute.call_args_list[0]
        sql_text = str(call_args[0][0].text)
        assert "memory_type" in sql_text

    async def test_time_range_filter(self) -> None:
        """time_range filter adds created_at conditions."""
        org_id = uuid4()
        agent_id = uuid4()
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 12, 31, tzinfo=UTC)
        conn = _make_mock_conn([_make_memory_row()])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            await recall(conn, org_id=org_id, agent_id=agent_id, time_range=(start, end))

        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert "time_start" in params
        assert "time_end" in params

    async def test_metadata_filter(self) -> None:
        """filters dict adds JSONB containment condition."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([_make_memory_row()])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            await recall(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                filters={"tag": "important"},
            )

        call_args = conn.execute.call_args_list[0]
        sql_text = str(call_args[0][0].text)
        assert "@>" in sql_text

    async def test_as_of_temporal_filter(self) -> None:
        """as_of replaces the default 'valid_to IS NULL' with point-in-time query."""
        org_id = uuid4()
        agent_id = uuid4()
        as_of = datetime(2025, 6, 15, tzinfo=UTC)
        conn = _make_mock_conn([_make_memory_row()])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            await recall(conn, org_id=org_id, agent_id=agent_id, as_of=as_of)

        call_args = conn.execute.call_args_list[0]
        sql_text = str(call_args[0][0].text)
        params = call_args[0][1]
        assert "as_of" in params
        assert "valid_from" in sql_text

    async def test_include_deleted_omits_deleted_at_filter(self) -> None:
        """include_deleted=True skips the 'deleted_at IS NULL' condition."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([_make_memory_row()])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            await recall(conn, org_id=org_id, agent_id=agent_id, include_deleted=True)

        call_args = conn.execute.call_args_list[0]
        sql_text = str(call_args[0][0].text)
        assert "deleted_at IS NULL" not in sql_text

    async def test_default_excludes_deleted(self) -> None:
        """Default include_deleted=False includes 'deleted_at IS NULL'."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([_make_memory_row()])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            await recall(conn, org_id=org_id, agent_id=agent_id)

        call_args = conn.execute.call_args_list[0]
        sql_text = str(call_args[0][0].text)
        assert "deleted_at IS NULL" in sql_text


# ---------------------------------------------------------------------------
# recall() — fused relevance scoring
# ---------------------------------------------------------------------------


class TestRecallRelevanceScoring:
    """Test the fused relevance scoring calculation."""

    async def test_no_query_uses_50_50_split(self) -> None:
        """Without a query, relevance = 0.50 * importance + 0.50 * recency."""
        org_id = uuid4()
        agent_id = uuid4()
        # Create a row that was just created (high recency)
        now = datetime.now(tz=UTC)
        row = _make_memory_row(importance_score=0.8, created_at=now)
        conn = _make_mock_conn([row])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(conn, org_id=org_id, agent_id=agent_id)

        assert len(results) == 1
        result = results[0]
        # With very recent memory, recency_score approaches 1.0
        # relevance ~ 0.50 * 0.8 + 0.50 * ~1.0 = ~0.9
        assert result.relevance_score > 0.5

    async def test_with_query_uses_60_25_15_split(self) -> None:
        """With a query, relevance = 0.60 * similarity + 0.25 * importance + 0.15 * recency."""
        org_id = uuid4()
        agent_id = uuid4()
        now = datetime.now(tz=UTC)
        row = _make_memory_row(
            importance_score=0.8,
            created_at=now,
            similarity=0.95,
        )
        conn = _make_mock_conn([row])
        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = [0.1, 0.2, 0.3]

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                query="test query",
                embedding_provider=embed_provider,
            )

        assert len(results) == 1
        result = results[0]
        # 0.60 * 0.95 + 0.25 * 0.8 + 0.15 * ~1.0 ≈ 0.57 + 0.20 + 0.15 = 0.92
        assert result.relevance_score > 0.8

    async def test_recency_score_decreases_with_age(self) -> None:
        """Older memories get lower recency scores."""
        org_id = uuid4()
        agent_id = uuid4()
        # Memory created 30 days ago
        old_time = datetime.now(tz=UTC) - timedelta(days=30)
        row = _make_memory_row(importance_score=0.5, created_at=old_time)
        conn = _make_mock_conn([row])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(conn, org_id=org_id, agent_id=agent_id)

        assert len(results) == 1
        # recency_score = 1/30 ≈ 0.033
        # relevance = 0.50 * 0.5 + 0.50 * 0.033 ≈ 0.267
        assert results[0].relevance_score < 0.4


# ---------------------------------------------------------------------------
# recall() — similarity threshold
# ---------------------------------------------------------------------------


class TestRecallSimilarityThreshold:
    """Test similarity threshold filtering."""

    async def test_below_threshold_filtered_out(self) -> None:
        """Rows with similarity below threshold are excluded."""
        org_id = uuid4()
        agent_id = uuid4()
        row = _make_memory_row(similarity=0.3)
        conn = _make_mock_conn([row])
        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = [0.1, 0.2, 0.3]

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                query="test",
                embedding_provider=embed_provider,
                similarity_threshold=0.5,
            )

        assert len(results) == 0

    async def test_above_threshold_included(self) -> None:
        """Rows with similarity above threshold are included."""
        org_id = uuid4()
        agent_id = uuid4()
        row = _make_memory_row(similarity=0.8)
        conn = _make_mock_conn([row])
        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = [0.1, 0.2, 0.3]

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                query="test",
                embedding_provider=embed_provider,
                similarity_threshold=0.5,
            )

        assert len(results) == 1

    async def test_threshold_not_applied_without_query(self) -> None:
        """Threshold doesn't filter when there's no query (similarity=0.0)."""
        org_id = uuid4()
        agent_id = uuid4()
        row = _make_memory_row(similarity=None)
        conn = _make_mock_conn([row])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                similarity_threshold=0.5,
            )

        # Should include the row since no query means threshold doesn't apply
        assert len(results) == 1


# ---------------------------------------------------------------------------
# recall() — edge cases
# ---------------------------------------------------------------------------


class TestRecallEdgeCases:
    """Test edge cases in recall."""

    async def test_empty_results(self) -> None:
        """recall() returns empty list when no rows match."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(conn, org_id=org_id, agent_id=agent_id)

        # RecallResponse iterates + has __len__ as a back-compat shim.
        assert len(results) == 0
        assert list(results) == []

    async def test_top_k_param_passed(self) -> None:
        """top_k is passed to the SQL query."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            await recall(conn, org_id=org_id, agent_id=agent_id, top_k=5)

        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]
        assert params["top_k"] == 5

    async def test_results_sorted_by_relevance_descending(self) -> None:
        """Results are sorted by relevance_score descending."""
        org_id = uuid4()
        agent_id = uuid4()
        now = datetime.now(tz=UTC)
        # Two rows: one important but old, one less important but recent
        old_row = _make_memory_row(
            importance_score=0.9,
            created_at=now - timedelta(days=10),
        )
        new_row = _make_memory_row(
            importance_score=0.3,
            created_at=now,
        )
        conn = _make_mock_conn([old_row, new_row])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(conn, org_id=org_id, agent_id=agent_id)

        assert len(results) == 2
        assert results[0].relevance_score >= results[1].relevance_score

    async def test_update_recall_count_called(self) -> None:
        """When results exist, recall_count is updated."""
        org_id = uuid4()
        agent_id = uuid4()
        row = _make_memory_row()
        conn = _make_mock_conn([row])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            await recall(conn, org_id=org_id, agent_id=agent_id)

        # Should be called twice: once for SELECT, once for UPDATE
        assert conn.execute.call_count >= 2

    async def test_audit_entry_created(self) -> None:
        """recall() always creates an audit entry."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([])

        with patch(
            "z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock
        ) as mock_audit:
            await recall(conn, org_id=org_id, agent_id=agent_id)

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["operation"] == "recall"
        assert call_kwargs["org_id"] == org_id

    async def test_embedding_provider_empty_embedding_fallback(self) -> None:
        """If embedding_provider returns empty list, use fallback query."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([_make_memory_row()])
        embed_provider = AsyncMock()
        embed_provider.embed_text.return_value = []  # Empty = no embedding

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                query="test",
                embedding_provider=embed_provider,
            )

        # Should still return results via fallback
        assert len(results) == 1

    async def test_invalid_scoring_weights_raises_recall_error(self) -> None:
        """recall() raises RecallError when scoring weights don't sum to ~1.0."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([])

        with pytest.raises(RecallError, match=r"Scoring weights must sum to ~1\.0"):
            await recall(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                similarity_weight=0.5,
                importance_weight=0.5,
                recency_weight=0.5,  # Sum = 1.5, not ~1.0
            )

    async def test_weights_summing_to_one_accepted(self) -> None:
        """recall() accepts weights that sum to exactly 1.0."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([_make_memory_row()])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            # Should not raise
            results = await recall(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                similarity_weight=0.6,
                importance_weight=0.25,
                recency_weight=0.15,
            )
        # Phase C: returns RecallResponse, which iterates like a list.
        assert len(results) == 1

    async def test_weights_within_tolerance_accepted(self) -> None:
        """recall() accepts weights within 0.01 tolerance of 1.0."""
        org_id = uuid4()
        agent_id = uuid4()
        conn = _make_mock_conn([_make_memory_row()])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                similarity_weight=0.6,
                importance_weight=0.25,
                recency_weight=0.155,  # Sum = 1.005, within tolerance
            )
        assert len(results) == 1

    async def test_null_metadata_becomes_empty_dict(self) -> None:
        """Row with None metadata is converted to empty dict."""
        org_id = uuid4()
        agent_id = uuid4()
        row = _make_memory_row(metadata=None)
        # Override the metadata position in the tuple (index 8) to be None
        row_list = list(row)
        row_list[8] = None
        conn = _make_mock_conn([tuple(row_list)])

        with patch("z3rno_core.engine.recall.create_audit_entry", new_callable=AsyncMock):
            results = await recall(conn, org_id=org_id, agent_id=agent_id)

        assert results[0].metadata == {}
