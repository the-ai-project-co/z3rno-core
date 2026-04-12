"""Unit tests for graph helper functions.

Tests _age_preamble and other pure helper functions from
z3rno_core.graph.queries and z3rno_core.graph.sync.
No database connection needed.
"""

from __future__ import annotations

from z3rno_core.graph.queries import (
    GRAPH_NAME as QUERIES_GRAPH_NAME,
    _age_preamble as queries_age_preamble,
)
from z3rno_core.graph.sync import GRAPH_NAME as SYNC_GRAPH_NAME, _age_preamble as sync_age_preamble

# ---------------------------------------------------------------------------
# _age_preamble from graph/queries.py
# ---------------------------------------------------------------------------


class TestQueriesAgePreamble:
    """Test _age_preamble from graph/queries.py."""

    def test_returns_string(self) -> None:
        result = queries_age_preamble()
        assert isinstance(result, str)

    def test_loads_age_extension(self) -> None:
        result = queries_age_preamble()
        assert "LOAD 'age'" in result

    def test_sets_search_path(self) -> None:
        result = queries_age_preamble()
        assert "ag_catalog" in result

    def test_includes_public_schema(self) -> None:
        result = queries_age_preamble()
        assert "public" in result

    def test_includes_user_schema(self) -> None:
        result = queries_age_preamble()
        assert '"$user"' in result


# ---------------------------------------------------------------------------
# _age_preamble from graph/sync.py
# ---------------------------------------------------------------------------


class TestSyncAgePreamble:
    """Test _age_preamble from graph/sync.py."""

    def test_returns_string(self) -> None:
        result = sync_age_preamble()
        assert isinstance(result, str)

    def test_loads_age_extension(self) -> None:
        result = sync_age_preamble()
        assert "LOAD 'age'" in result

    def test_sets_search_path(self) -> None:
        result = sync_age_preamble()
        assert "ag_catalog" in result

    def test_matches_queries_preamble(self) -> None:
        """Both modules should produce the same preamble."""
        assert queries_age_preamble() == sync_age_preamble()


# ---------------------------------------------------------------------------
# GRAPH_NAME constants
# ---------------------------------------------------------------------------


class TestGraphNameConstants:
    """Test graph name constants are consistent."""

    def test_queries_graph_name(self) -> None:
        assert QUERIES_GRAPH_NAME == "memory_graph"

    def test_sync_graph_name(self) -> None:
        assert SYNC_GRAPH_NAME == "memory_graph"

    def test_both_match(self) -> None:
        assert QUERIES_GRAPH_NAME == SYNC_GRAPH_NAME


# ---------------------------------------------------------------------------
# Function signatures exist (structural checks)
# ---------------------------------------------------------------------------


class TestGraphQueryFunctionsExist:
    """Verify graph query functions are importable and callable."""

    def test_find_related_memories_exists(self) -> None:
        from z3rno_core.graph.queries import find_related_memories

        assert callable(find_related_memories)

    def test_find_memory_chain_exists(self) -> None:
        from z3rno_core.graph.queries import find_memory_chain

        assert callable(find_memory_chain)

    def test_find_contradictions_exists(self) -> None:
        from z3rno_core.graph.queries import find_contradictions

        assert callable(find_contradictions)

    def test_find_shortest_path_exists(self) -> None:
        from z3rno_core.graph.queries import find_shortest_path

        assert callable(find_shortest_path)


class TestGraphSyncFunctionsExist:
    """Verify graph sync functions are importable and callable."""

    def test_sync_memory_to_graph_exists(self) -> None:
        from z3rno_core.graph.sync import sync_memory_to_graph

        assert callable(sync_memory_to_graph)

    def test_sync_relationship_to_graph_exists(self) -> None:
        from z3rno_core.graph.sync import sync_relationship_to_graph

        assert callable(sync_relationship_to_graph)
