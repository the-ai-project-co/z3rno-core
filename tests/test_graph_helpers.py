"""Unit tests for graph helper functions.

Tests _age_exec and other pure helper functions from
z3rno_core.graph.queries and z3rno_core.graph.sync.
No database connection needed — mocks the SQLAlchemy Connection
to assert the three-statement issue order (LOAD + SET + cypher).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from z3rno_core.graph.queries import (
    GRAPH_NAME as QUERIES_GRAPH_NAME,
    _age_exec as queries_age_exec,
)
from z3rno_core.graph.sync import (
    GRAPH_NAME as SYNC_GRAPH_NAME,
    _age_exec as sync_age_exec,
)


def _statements_issued(mock_conn: MagicMock) -> list[str]:
    """Return the SQL string of every execute() call, in order."""
    return [str(call.args[0]) for call in mock_conn.execute.call_args_list]


# ---------------------------------------------------------------------------
# _age_exec from graph/queries.py
# ---------------------------------------------------------------------------


class TestQueriesAgeExec:
    """v0.21.1 — _age_exec issues LOAD + SET + cypher as three calls.

    Pre-v0.21.1 the preamble was concatenated into one SQL string and
    asyncpg rejected it as a multi-command prepared statement. The
    fix splits the issue order so each execute is a single statement.
    """

    def test_issues_three_statements_in_order(self) -> None:
        conn = MagicMock()
        cypher = "SELECT * FROM cypher('g', $$ MATCH (n) RETURN n $$) AS (n agtype)"
        queries_age_exec(conn, cypher)

        stmts = _statements_issued(conn)
        assert len(stmts) == 3
        assert stmts[0] == "LOAD 'age'"
        assert stmts[1] == 'SET search_path = ag_catalog, "$user", public'
        assert stmts[2] == cypher

    def test_returns_cypher_result(self) -> None:
        """The cypher's CursorResult is returned, not the preamble's."""
        conn = MagicMock()
        cypher_result = MagicMock(name="cypher_result")
        conn.execute.side_effect = [MagicMock(), MagicMock(), cypher_result]
        result = queries_age_exec(conn, "SELECT 1")
        assert result is cypher_result


# ---------------------------------------------------------------------------
# _age_exec from graph/sync.py
# ---------------------------------------------------------------------------


class TestSyncAgeExec:
    """Mirror of TestQueriesAgeExec for the sync.py copy."""

    def test_issues_three_statements_in_order(self) -> None:
        conn = MagicMock()
        cypher = "SELECT * FROM cypher('g', $$ CREATE (n:Memory) $$) AS (v agtype)"
        sync_age_exec(conn, cypher)

        stmts = _statements_issued(conn)
        assert len(stmts) == 3
        assert stmts[0] == "LOAD 'age'"
        assert stmts[1] == 'SET search_path = ag_catalog, "$user", public'
        assert stmts[2] == cypher

    def test_returns_none(self) -> None:
        """sync.py's _age_exec is fire-and-forget — None return."""
        conn = MagicMock()
        result = sync_age_exec(conn, "SELECT 1")
        assert result is None


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
