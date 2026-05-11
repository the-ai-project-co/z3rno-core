"""Unit tests for the C.4 strategies — TEMPORAL, ASK, CYPHER.

No DB. AGE-dependent paths get a mocked AsyncConnection; LLM-driven
paths use StubLLMGateway with structured factories. Where the strategy
genuinely needs the DB (final memo fetch in ASK / CYPHER), we mock
``conn.execute`` to return tabular data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Side-effect: register strategies.
import z3rno_core.retrieval.strategies  # noqa: F401
from z3rno_core.distill.llm_gateway import LLMGatewayError, StubLLMGateway
from z3rno_core.retrieval import get_strategy, registered_strategies
from z3rno_core.retrieval.strategies.ask import (
    AskStrategy,
    _ProposedCypher,
    _validate_cypher,
)
from z3rno_core.retrieval.strategies.cypher import (
    CypherDisabledError,
    CypherStrategy,
    CypherValidationError,
)
from z3rno_core.retrieval.strategies.temporal import (
    TemporalStrategy,
    _ResolvedTimestamp,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistryC4:
    def test_all_three_registered(self) -> None:
        names = registered_strategies()
        for name in ("TEMPORAL", "ASK", "CYPHER"):
            assert name in names

    def test_capability_flags(self) -> None:
        assert TemporalStrategy.requires_query_embedding is True
        assert TemporalStrategy.requires_llm is False
        assert AskStrategy.requires_llm is True
        assert CypherStrategy.requires_llm is False
        assert CypherStrategy.requires_query_embedding is False

    def test_lookup_case_insensitive(self) -> None:
        assert get_strategy("temporal") is TemporalStrategy
        assert get_strategy("Ask") is AskStrategy
        assert get_strategy("CYPHER") is CypherStrategy


# ---------------------------------------------------------------------------
# TEMPORAL — timestamp extraction + delegation
# ---------------------------------------------------------------------------


class TestTemporalTimestamp:
    async def test_no_query_returns_empty(self) -> None:
        conn = AsyncMock()
        results = await TemporalStrategy().retrieve(
            conn, org_id=uuid4(), agent_id=uuid4(), query="", top_k=5
        )
        assert results == []

    async def test_no_embedder_returns_empty(self) -> None:
        """TEMPORAL delegates to VECTOR which needs an embedder."""
        conn = AsyncMock()
        results = await TemporalStrategy().retrieve(
            conn, org_id=uuid4(), agent_id=uuid4(), query="anything", top_k=5
        )
        assert results == []

    async def test_caller_supplied_as_of_skips_llm(self) -> None:
        """Caller's as_of takes precedence — no LLM call attempted."""
        conn = AsyncMock()
        embedder = AsyncMock()
        embedder.embed_text = AsyncMock(return_value=[0.1] * 1536)
        empty = MagicMock()
        empty.fetchall.return_value = []
        conn.execute = AsyncMock(return_value=empty)

        # Stub LLM whose call would fail the test if invoked.
        gateway = StubLLMGateway(
            structured=lambda *_a, **_k: pytest.fail("should not call LLM")
        )

        ts = datetime(2024, 3, 5, tzinfo=UTC)
        results = await TemporalStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="anything",
            top_k=5,
            embedding_provider=embedder,
            as_of=ts,
            llm_gateway=gateway,
        )
        # Empty (no rows) but didn't crash; LLM wasn't called.
        assert results == []

    async def test_llm_resolves_timestamp(self) -> None:
        """LLM returns an ISO-8601 timestamp; strategy delegates with as_of."""
        conn = AsyncMock()
        embedder = AsyncMock()
        embedder.embed_text = AsyncMock(return_value=[0.1] * 1536)
        empty = MagicMock()
        empty.fetchall.return_value = []
        conn.execute = AsyncMock(return_value=empty)

        def _factory(_system: str, _user: str, _model: type) -> Any:
            return _ResolvedTimestamp(
                timestamp="2024-03-05T00:00:00Z", rationale="march 2024"
            )

        gateway = StubLLMGateway(structured=_factory)

        results = await TemporalStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="what did we know on March 5?",
            top_k=5,
            embedding_provider=embedder,
            llm_gateway=gateway,
        )
        assert results == []

    async def test_llm_failure_falls_back_to_current_time(self) -> None:
        """LLM raises → TEMPORAL still runs (no as_of) instead of crashing."""
        conn = AsyncMock()
        embedder = AsyncMock()
        embedder.embed_text = AsyncMock(return_value=[0.1] * 1536)
        empty = MagicMock()
        empty.fetchall.return_value = []
        conn.execute = AsyncMock(return_value=empty)

        def _raise(*_a: Any, **_k: Any) -> Any:
            raise LLMGatewayError("ollama timed out")

        gateway = StubLLMGateway(structured=_raise)
        results = await TemporalStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="what did we know on March 5?",
            top_k=5,
            embedding_provider=embedder,
            llm_gateway=gateway,
        )
        assert results == []

    async def test_empty_timestamp_falls_back(self) -> None:
        """LLM returns empty string → no time hint → current-time recall."""
        conn = AsyncMock()
        embedder = AsyncMock()
        embedder.embed_text = AsyncMock(return_value=[0.1] * 1536)
        empty = MagicMock()
        empty.fetchall.return_value = []
        conn.execute = AsyncMock(return_value=empty)

        def _factory(*_a: Any, **_k: Any) -> Any:
            return _ResolvedTimestamp(timestamp="", rationale="no time hint")

        gateway = StubLLMGateway(structured=_factory)
        results = await TemporalStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="recent thoughts",
            top_k=5,
            embedding_provider=embedder,
            llm_gateway=gateway,
        )
        assert results == []


# ---------------------------------------------------------------------------
# ASK — Cypher validation
# ---------------------------------------------------------------------------


class TestAskCypherValidator:
    def test_accepts_read_only_match(self) -> None:
        assert _validate_cypher("MATCH (m:Memory) RETURN m.id") == ""

    def test_accepts_match_with_where(self) -> None:
        assert _validate_cypher(
            "MATCH (m:Memory) WHERE m.memory_type = 'episodic' RETURN m.id LIMIT 10"
        ) == ""

    def test_accepts_with_clause(self) -> None:
        assert _validate_cypher(
            "MATCH (m:Memory) WITH count(m) AS n RETURN n"
        ) == ""

    @pytest.mark.parametrize(
        "forbidden",
        [
            "CREATE (m:Memory) RETURN m",
            "MATCH (m:Memory) DELETE m",
            "MATCH (m:Memory) DETACH DELETE m",
            "MATCH (m:Memory) SET m.x = 1",
            "MATCH (m:Memory) REMOVE m.x",
            "MERGE (m:Memory) RETURN m",
            "DROP CONSTRAINT foo",
            "CALL apoc.path.expand(...)",
            "LOAD CSV FROM 'x' AS row CREATE (n) RETURN n",
        ],
    )
    def test_rejects_writes_and_dangerous_calls(self, forbidden: str) -> None:
        reason = _validate_cypher(forbidden)
        # Either rejected for a forbidden keyword OR for not starting
        # with MATCH/RETURN/WITH. Both are valid rejections — the test
        # only cares that the validator says "no".
        assert reason != ""

    def test_rejects_empty(self) -> None:
        assert _validate_cypher("") == "empty Cypher"
        assert _validate_cypher("   ") == "empty Cypher"

    def test_rejects_double_dollar_delimiter(self) -> None:
        bad = "MATCH (m) RETURN m $$ extra $$"
        assert "$$" in _validate_cypher(bad)


class TestAskStrategy:
    async def test_no_llm_gateway_raises(self) -> None:
        with pytest.raises(LLMGatewayError):
            await AskStrategy().retrieve(
                AsyncMock(),
                org_id=uuid4(),
                agent_id=uuid4(),
                query="how many",
                top_k=5,
            )

    async def test_empty_query_returns_empty(self) -> None:
        gateway = StubLLMGateway(
            structured=lambda *_a, **_k: pytest.fail("should not call LLM")
        )
        results = await AskStrategy().retrieve(
            AsyncMock(),
            org_id=uuid4(),
            agent_id=uuid4(),
            query="   ",
            top_k=5,
            llm_gateway=gateway,
        )
        assert results == []

    async def test_invalid_cypher_returns_empty(self) -> None:
        """LLM returns a write-flavored cypher → validator rejects → empty."""

        def _factory(_s: str, _u: str, _m: type) -> Any:
            return _ProposedCypher(
                cypher="MATCH (m) CREATE (n) RETURN m", rationale="x"
            )

        gateway = StubLLMGateway(structured=_factory)
        results = await AskStrategy().retrieve(
            AsyncMock(),
            org_id=uuid4(),
            agent_id=uuid4(),
            query="how many",
            top_k=5,
            llm_gateway=gateway,
        )
        assert results == []


# ---------------------------------------------------------------------------
# CYPHER — gate + validation
# ---------------------------------------------------------------------------


class TestCypherGate:
    async def test_disabled_by_default_raises(self) -> None:
        with pytest.raises(CypherDisabledError):
            await CypherStrategy().retrieve(
                AsyncMock(),
                org_id=uuid4(),
                agent_id=uuid4(),
                query="ignored",
                top_k=5,
                raw_cypher="MATCH (m) RETURN m",
            )

    async def test_missing_raw_cypher_raises(self) -> None:
        with pytest.raises(CypherValidationError, match="empty"):
            await CypherStrategy().retrieve(
                AsyncMock(),
                org_id=uuid4(),
                agent_id=uuid4(),
                query="ignored",
                top_k=5,
                allow_cypher_query=True,
            )

    async def test_invalid_cypher_raises(self) -> None:
        with pytest.raises(CypherValidationError, match="rejected"):
            await CypherStrategy().retrieve(
                AsyncMock(),
                org_id=uuid4(),
                agent_id=uuid4(),
                query="ignored",
                top_k=5,
                allow_cypher_query=True,
                raw_cypher="CREATE (m:Memory)",
            )

    async def test_age_failure_returns_empty(self) -> None:
        """When the conn.run_sync raises a DBAPI error (AGE down), return empty."""
        from sqlalchemy.exc import DBAPIError  # noqa: PLC0415

        conn = AsyncMock()
        conn.run_sync = AsyncMock(
            side_effect=DBAPIError("AGE not loaded", None, BaseException("x"))
        )

        results = await CypherStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="ignored",
            top_k=5,
            allow_cypher_query=True,
            raw_cypher="MATCH (m:Memory) RETURN m.id LIMIT 5",
        )
        assert results == []
