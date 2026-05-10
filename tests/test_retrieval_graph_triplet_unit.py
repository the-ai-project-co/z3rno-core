"""Unit tests for the C.2 GRAPH + TRIPLET strategies (no DB).

Mocks the AGE / LLM dependencies. End-to-end paths against a real
seeded graph live in the integration test suite (deferred to Phase 3
cluster testing for the >=100K-node latency target).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Side-effect: register strategies.
import z3rno_core.retrieval.strategies  # noqa: F401
from z3rno_core.distill.llm_gateway import LLMGatewayError
from z3rno_core.retrieval import get_strategy, registered_strategies
from z3rno_core.retrieval.strategies.graph import GraphStrategy
from z3rno_core.retrieval.strategies.triplet import (
    TripletStrategy,
    _is_safe_edge_label,
    _ParsedTriplet,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_graph_registered(self) -> None:
        assert "GRAPH" in registered_strategies()
        assert get_strategy("GRAPH") is GraphStrategy

    def test_triplet_registered(self) -> None:
        assert "TRIPLET" in registered_strategies()
        assert get_strategy("TRIPLET") is TripletStrategy

    def test_capability_flags(self) -> None:
        assert GraphStrategy.requires_query_embedding is True
        # GRAPH degrades without LLM rather than refusing.
        assert GraphStrategy.requires_llm is False
        # TRIPLET is meaningfully LLM-driven; can't degrade.
        assert TripletStrategy.requires_llm is True


# ---------------------------------------------------------------------------
# GRAPH — short-circuit paths
# ---------------------------------------------------------------------------


class TestGraphShortCircuits:
    async def test_no_query_returns_empty(self) -> None:
        conn = AsyncMock()
        results = await GraphStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="",
            top_k=5,
        )
        assert results == []
        conn.execute.assert_not_called()

    async def test_no_embedding_provider_returns_empty(self) -> None:
        conn = AsyncMock()
        results = await GraphStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="who lives in Paris?",
            top_k=5,
        )
        assert results == []
        conn.execute.assert_not_called()

    async def test_empty_embedding_returns_empty(self) -> None:
        """Provider returns []` (NoOp-style) → GRAPH can't seed → empty."""
        conn = AsyncMock()
        embedding_provider = AsyncMock()
        embedding_provider.embed_text = AsyncMock(return_value=[])

        results = await GraphStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="something",
            top_k=5,
            embedding_provider=embedding_provider,
        )
        assert results == []
        conn.execute.assert_not_called()

    async def test_no_seeds_returns_empty(self) -> None:
        """Vector kNN returns no rows → no seeds to expand → empty."""
        conn = AsyncMock()
        embedding_provider = AsyncMock()
        embedding_provider.embed_text = AsyncMock(return_value=[0.1] * 1536)
        seed_result = MagicMock()
        seed_result.fetchall.return_value = []
        conn.execute = AsyncMock(return_value=seed_result)

        results = await GraphStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="something",
            top_k=5,
            embedding_provider=embedding_provider,
        )
        assert results == []


class TestGraphSeedResultShape:
    async def test_seeds_returned_with_subgraph_empty_when_age_fails(self) -> None:
        """AGE not loaded → run_sync raises → subgraph empty → seeds still returned."""
        from sqlalchemy.exc import DBAPIError  # noqa: PLC0415

        conn = AsyncMock()
        embedding_provider = AsyncMock()
        embedding_provider.embed_text = AsyncMock(return_value=[0.1] * 1536)

        seed_id = uuid4()
        now = datetime.now(tz=UTC)
        seed_row = (
            seed_id,                   # 0 id
            "alice works at acme",     # 1 content
            None,                      # 2 summary
            "semantic",                # 3 memory_type
            0.5,                       # 4 importance_score
            3,                         # 5 recall_count
            now,                       # 6 created_at
            now,                       # 7 valid_from
            {},                        # 8 metadata
            0.85,                      # 9 similarity
        )
        seed_result = MagicMock()
        seed_result.fetchall.return_value = [seed_row]
        conn.execute = AsyncMock(return_value=seed_result)
        # run_sync raises a fake DBAPIError → caught by _expand_subgraphs.
        conn.run_sync = AsyncMock(
            side_effect=DBAPIError("AGE not loaded", None, BaseException("x"))
        )

        results = await GraphStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="alice",
            top_k=5,
            embedding_provider=embedding_provider,
        )

        assert len(results) == 1
        assert results[0].memory_id == seed_id
        assert results[0].content == "alice works at acme"
        # AGE failed → graph_context empty but result still returned.
        assert results[0].graph_context == []
        assert "vector" in results[0].score_components
        # No LLM gateway → no graph_answer key.
        assert "graph_answer" not in results[0].metadata


# ---------------------------------------------------------------------------
# TRIPLET — short-circuit + LLM gating
# ---------------------------------------------------------------------------


class TestTripletShortCircuits:
    async def test_no_llm_gateway_raises(self) -> None:
        """TRIPLET refuses to run without an llm_gateway (per the plan)."""
        conn = AsyncMock()
        with pytest.raises(LLMGatewayError, match="requires an llm_gateway"):
            await TripletStrategy().retrieve(
                conn,
                org_id=uuid4(),
                agent_id=uuid4(),
                query="where does alice work?",
                top_k=5,
            )

    async def test_empty_query_returns_empty(self) -> None:
        conn = AsyncMock()
        llm = MagicMock()
        results = await TripletStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="",
            top_k=5,
            llm_gateway=llm,
        )
        assert results == []

    async def test_whitespace_query_returns_empty(self) -> None:
        conn = AsyncMock()
        llm = MagicMock()
        results = await TripletStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="   ",
            top_k=5,
            llm_gateway=llm,
        )
        assert results == []


class TestParsedTriplet:
    def test_unknown_slot_subject(self) -> None:
        t = _ParsedTriplet(subject="?", predicate="WORKS_AT", object="Acme")
        assert t.unknown_slot == "subject"

    def test_unknown_slot_predicate(self) -> None:
        t = _ParsedTriplet(subject="Alice", predicate="?", object="Acme")
        assert t.unknown_slot == "predicate"

    def test_unknown_slot_object(self) -> None:
        t = _ParsedTriplet(subject="Alice", predicate="WORKS_AT", object="?")
        assert t.unknown_slot == "object"

    def test_zero_unknowns_raises(self) -> None:
        t = _ParsedTriplet(subject="Alice", predicate="WORKS_AT", object="Acme")
        with pytest.raises(ValueError, match="exactly one unknown"):
            _ = t.unknown_slot

    def test_two_unknowns_raises(self) -> None:
        t = _ParsedTriplet(subject="?", predicate="?", object="Acme")
        with pytest.raises(ValueError, match="exactly one unknown"):
            _ = t.unknown_slot


# ---------------------------------------------------------------------------
# Edge-label safety guard
# ---------------------------------------------------------------------------


class TestEdgeLabelGuard:
    def test_accepts_upper_snake(self) -> None:
        assert _is_safe_edge_label("WORKS_AT") is True
        assert _is_safe_edge_label("DERIVED_FROM") is True
        assert _is_safe_edge_label("RELATES_TO") is True

    def test_accepts_alnum(self) -> None:
        assert _is_safe_edge_label("HAS_5_HOPS") is True

    def test_rejects_injection_attempts(self) -> None:
        # Common Cypher-injection shapes mustn't slip through.
        assert _is_safe_edge_label("WORKS_AT'; DROP") is False
        assert _is_safe_edge_label("WORKS_AT $$") is False
        assert _is_safe_edge_label("WORKS-AT") is False
        assert _is_safe_edge_label("WORKS AT") is False
        assert _is_safe_edge_label("") is False

    def test_rejects_leading_digit(self) -> None:
        assert _is_safe_edge_label("3_HOPS") is False
