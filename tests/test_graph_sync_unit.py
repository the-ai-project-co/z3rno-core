"""Unit tests for z3rno_core.graph.sync — pure logic only (no DB).

Tests Cypher query construction for sync_memory_to_graph,
delete_memory_from_graph, and sync_relationship_to_graph
using mocked connections.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from z3rno_core.graph.sync import (
    GRAPH_NAME,
    _age_preamble,
    _validate_uuid_for_cypher,
    delete_memory_from_graph,
    sync_memory_to_graph,
    sync_relationship_to_graph,
)

# ---------------------------------------------------------------------------
# _validate_uuid_for_cypher
# ---------------------------------------------------------------------------


class TestValidateUuidForCypher:
    """Test UUID validation for Cypher interpolation."""

    def test_valid_uuid_accepted(self) -> None:
        """Standard UUID passes validation."""
        uid = uuid4()
        result = _validate_uuid_for_cypher(uid)
        assert result == str(uid)

    def test_returns_string(self) -> None:
        uid = uuid4()
        assert isinstance(_validate_uuid_for_cypher(uid), str)

    def test_invalid_chars_raise_value_error(self) -> None:
        """UUID with invalid characters raises ValueError."""

        class FakeUUID:
            """Fake UUID that stringifies to a value with invalid chars."""

            def __str__(self) -> str:
                return "12345678-abcd-1234-abcd-12345678GGGG"

        fake = FakeUUID()
        with pytest.raises(ValueError, match="Invalid UUID for Cypher"):
            _validate_uuid_for_cypher(fake)  # type: ignore[arg-type]

    def test_uppercase_hex_raises_value_error(self) -> None:
        """Uppercase hex chars are rejected (only lowercase allowed)."""

        class FakeUUID:
            def __str__(self) -> str:
                return "12345678-ABCD-1234-abcd-123456789abc"

        with pytest.raises(ValueError, match="Invalid UUID for Cypher"):
            _validate_uuid_for_cypher(FakeUUID())  # type: ignore[arg-type]

    def test_injection_attempt_raises_value_error(self) -> None:
        """SQL/Cypher injection attempt is caught."""

        class FakeUUID:
            def __str__(self) -> str:
                return "'; DROP TABLE memories; --"

        with pytest.raises(ValueError, match="Invalid UUID for Cypher"):
            _validate_uuid_for_cypher(FakeUUID())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _age_preamble
# ---------------------------------------------------------------------------


class TestAgePreamble:
    """Test the AGE preamble SQL."""

    def test_returns_string(self) -> None:
        assert isinstance(_age_preamble(), str)

    def test_loads_age_extension(self) -> None:
        assert "LOAD 'age'" in _age_preamble()

    def test_sets_search_path(self) -> None:
        assert "SET search_path" in _age_preamble()
        assert "ag_catalog" in _age_preamble()

    def test_includes_public_schema(self) -> None:
        assert "public" in _age_preamble()


# ---------------------------------------------------------------------------
# GRAPH_NAME constant
# ---------------------------------------------------------------------------


class TestGraphNameConstant:
    """Test the GRAPH_NAME constant."""

    def test_value(self) -> None:
        assert GRAPH_NAME == "memory_graph"


# ---------------------------------------------------------------------------
# sync_memory_to_graph
# ---------------------------------------------------------------------------


class TestSyncMemoryToGraph:
    """Test sync_memory_to_graph Cypher construction."""

    def test_basic_sync(self) -> None:
        """Creates a Memory vertex with required properties."""
        conn = MagicMock()
        mid = uuid4()
        org_id = uuid4()
        agent_id = uuid4()

        sync_memory_to_graph(
            conn,
            memory_id=mid,
            org_id=org_id,
            agent_id=agent_id,
            memory_type="episodic",
        )

        conn.execute.assert_called_once()
        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)

        assert "MERGE (m:Memory" in sql_text
        assert str(mid) in sql_text
        assert str(org_id) in sql_text
        assert str(agent_id) in sql_text
        assert "episodic" in sql_text
        assert "LOAD 'age'" in sql_text

    def test_with_content_preview(self) -> None:
        """Content preview is included and truncated."""
        conn = MagicMock()
        mid = uuid4()
        preview = "This is a test memory content"

        sync_memory_to_graph(
            conn,
            memory_id=mid,
            org_id=uuid4(),
            agent_id=uuid4(),
            memory_type="semantic",
            content_preview=preview,
        )

        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)
        assert "This is a test memory content" in sql_text

    def test_content_preview_truncated_at_200(self) -> None:
        """Long content preview is truncated to 200 characters."""
        conn = MagicMock()
        long_content = "x" * 300

        sync_memory_to_graph(
            conn,
            memory_id=uuid4(),
            org_id=uuid4(),
            agent_id=uuid4(),
            memory_type="working",
            content_preview=long_content,
        )

        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)
        # The preview should be at most 200 x's
        # Count occurrences of 'x' between quotes
        assert "x" * 201 not in sql_text

    def test_none_content_preview(self) -> None:
        """None content preview uses empty string."""
        conn = MagicMock()

        sync_memory_to_graph(
            conn,
            memory_id=uuid4(),
            org_id=uuid4(),
            agent_id=uuid4(),
            memory_type="procedural",
            content_preview=None,
        )

        conn.execute.assert_called_once()

    def test_single_quotes_escaped(self) -> None:
        """Single quotes in preview are escaped."""
        conn = MagicMock()

        sync_memory_to_graph(
            conn,
            memory_id=uuid4(),
            org_id=uuid4(),
            agent_id=uuid4(),
            memory_type="semantic",
            content_preview="it's a test",
        )

        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)
        assert "it\\'s a test" in sql_text

    def test_cypher_uses_merge(self) -> None:
        """Uses MERGE to create or update the vertex."""
        conn = MagicMock()

        sync_memory_to_graph(
            conn,
            memory_id=uuid4(),
            org_id=uuid4(),
            agent_id=uuid4(),
            memory_type="episodic",
        )

        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)
        assert "MERGE" in sql_text


# ---------------------------------------------------------------------------
# delete_memory_from_graph
# ---------------------------------------------------------------------------


class TestDeleteMemoryFromGraph:
    """Test delete_memory_from_graph Cypher construction."""

    def test_basic_delete(self) -> None:
        """Deletes a Memory vertex with DETACH DELETE."""
        conn = MagicMock()
        mid = uuid4()

        delete_memory_from_graph(conn, memory_id=mid)

        conn.execute.assert_called_once()
        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)

        assert "MATCH (m:Memory" in sql_text
        assert "DETACH DELETE m" in sql_text
        assert str(mid) in sql_text
        assert "LOAD 'age'" in sql_text

    def test_uses_graph_name(self) -> None:
        """Uses the GRAPH_NAME constant."""
        conn = MagicMock()
        mid = uuid4()

        delete_memory_from_graph(conn, memory_id=mid)

        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)
        assert GRAPH_NAME in sql_text


# ---------------------------------------------------------------------------
# sync_relationship_to_graph
# ---------------------------------------------------------------------------


class TestSyncRelationshipToGraph:
    """Test sync_relationship_to_graph Cypher construction."""

    def test_basic_relationship(self) -> None:
        """Creates an edge between two Memory vertices."""
        conn = MagicMock()
        source = uuid4()
        target = uuid4()

        sync_relationship_to_graph(
            conn,
            source_id=source,
            target_id=target,
            relationship_type="derived_from",
        )

        conn.execute.assert_called_once()
        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)

        assert str(source) in sql_text
        assert str(target) in sql_text
        assert "DERIVED_FROM" in sql_text  # Uppercased
        assert "MATCH" in sql_text
        assert "CREATE" in sql_text

    def test_relationship_type_uppercased(self) -> None:
        """Relationship type is uppercased for the edge label."""
        conn = MagicMock()

        sync_relationship_to_graph(
            conn,
            source_id=uuid4(),
            target_id=uuid4(),
            relationship_type="related_to",
        )

        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)
        assert "RELATED_TO" in sql_text
        assert "related_to" not in sql_text.split("$$")[1]  # Inside Cypher

    def test_custom_weight(self) -> None:
        """Custom weight is included in the edge properties."""
        conn = MagicMock()

        sync_relationship_to_graph(
            conn,
            source_id=uuid4(),
            target_id=uuid4(),
            relationship_type="supports",
            weight=0.75,
        )

        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)
        assert "0.75" in sql_text

    def test_default_weight_is_one(self) -> None:
        """Default weight is 1.0."""
        conn = MagicMock()

        sync_relationship_to_graph(
            conn,
            source_id=uuid4(),
            target_id=uuid4(),
            relationship_type="contradicts",
        )

        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)
        assert "1.0" in sql_text

    def test_uses_graph_name(self) -> None:
        """Uses the GRAPH_NAME constant."""
        conn = MagicMock()

        sync_relationship_to_graph(
            conn,
            source_id=uuid4(),
            target_id=uuid4(),
            relationship_type="related_to",
        )

        call_args = conn.execute.call_args[0][0]
        sql_text = str(call_args.text)
        assert GRAPH_NAME in sql_text
