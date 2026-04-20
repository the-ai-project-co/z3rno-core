"""Cypher traversal queries for the memory graph.

All queries use Apache AGE's cypher() function and return raw agtype
results. The caller is responsible for parsing agtype values.

Query patterns documented in docs/research/apache-age-graph-patterns.md.

NOTE: Apache AGE does **not** support parameterized Cypher queries.
All UUID values are interpolated into query strings after explicit
validation via ``_validate_uuid_for_cypher``.  This is a defense-in-
depth measure; monitor the AGE project for parameterized query support.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import Connection, text
from sqlalchemy.engine import CursorResult

GRAPH_NAME = "memory_graph"


def _age_preamble() -> str:
    """SQL preamble required before every AGE Cypher query."""
    return "LOAD 'age'; SET search_path = ag_catalog, \"$user\", public;"


def _validate_uuid_for_cypher(value: UUID) -> str:
    """Validate and format a UUID for Cypher interpolation.

    AGE does not support parameterized Cypher, so UUIDs are interpolated
    directly into query strings.  This helper ensures only hex digits and
    hyphens appear in the resulting string.
    """
    s = str(value)
    if not all(c in "0123456789abcdef-" for c in s):
        raise ValueError(f"Invalid UUID for Cypher: {s}")
    return s


def find_related_memories(
    conn: Connection,
    memory_id: UUID,
    max_hops: int = 1,
) -> CursorResult[tuple[object, ...]]:
    """Find all memories related to a given memory within N hops.

    Args:
        conn: Active SQLAlchemy connection.
        memory_id: The source memory UUID.
        max_hops: Maximum relationship hops (default 1).

    Returns:
        CursorResult with columns (related_id agtype, rel_type agtype).
    """
    safe_id = _validate_uuid_for_cypher(memory_id)
    cypher = (
        f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
        f"MATCH (m:Memory {{id: '{safe_id}'}})-[r*1..{max_hops}]-(related:Memory) "
        f"RETURN DISTINCT related.id, type(last(r)) "
        f"$$) AS (related_id agtype, rel_type agtype)"
    )
    return conn.execute(text(f"{_age_preamble()} {cypher}"))


def find_memory_chain(
    conn: Connection,
    memory_id: UUID,
    relationship_type: str = "DERIVED_FROM",
    max_depth: int = 10,
) -> CursorResult[tuple[object, ...]]:
    """Find a chain of memories following a specific relationship type.

    E.g. A -[DERIVED_FROM]-> B -[DERIVED_FROM]-> C

    Args:
        conn: Active SQLAlchemy connection.
        memory_id: Starting memory UUID.
        relationship_type: Edge label to follow (default DERIVED_FROM).
        max_depth: Maximum chain length.

    Returns:
        CursorResult with the chain path.
    """
    safe_id = _validate_uuid_for_cypher(memory_id)
    edge_label = relationship_type.upper()
    cypher = (
        f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
        f"MATCH path = (start:Memory {{id: '{safe_id}'}})"
        f"-[:{edge_label}*1..{max_depth}]->(target:Memory) "
        f"RETURN target.id, length(path) "
        f"$$) AS (target_id agtype, depth agtype)"
    )
    return conn.execute(text(f"{_age_preamble()} {cypher}"))


def find_contradictions(
    conn: Connection,
    memory_id: UUID,
) -> CursorResult[tuple[object, ...]]:
    """Find all memories that contradict a given memory.

    Args:
        conn: Active SQLAlchemy connection.
        memory_id: The memory UUID to check for contradictions.

    Returns:
        CursorResult with contradicting memory IDs.
    """
    safe_id = _validate_uuid_for_cypher(memory_id)
    cypher = (
        f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
        f"MATCH (m:Memory {{id: '{safe_id}'}})-[:CONTRADICTS]-(c:Memory) "
        f"RETURN c.id, c.memory_type "
        f"$$) AS (contradicting_id agtype, memory_type agtype)"
    )
    return conn.execute(text(f"{_age_preamble()} {cypher}"))


def find_shortest_path(
    conn: Connection,
    source_id: UUID,
    target_id: UUID,
    max_depth: int = 10,
) -> CursorResult[tuple[object, ...]]:
    """Find the shortest path between two memories in the graph.

    Args:
        conn: Active SQLAlchemy connection.
        source_id: Starting memory UUID.
        target_id: Destination memory UUID.
        max_depth: Maximum path length to search.

    Returns:
        CursorResult with path information.
    """
    safe_source = _validate_uuid_for_cypher(source_id)
    safe_target = _validate_uuid_for_cypher(target_id)
    cypher = (
        f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
        f"MATCH path = shortestPath("
        f"(s:Memory {{id: '{safe_source}'}})-[*..{max_depth}]-(t:Memory {{id: '{safe_target}'}}))"
        f" RETURN length(path) "
        f"$$) AS (path_length agtype)"
    )
    return conn.execute(text(f"{_age_preamble()} {cypher}"))
