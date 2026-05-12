"""Sync relational memory data to the Apache AGE graph.

When a memory is stored or a relationship is created, these functions
mirror the data into the AGE graph as vertices and edges.

All AGE queries require LOAD 'age' and the search_path set to include
ag_catalog. The database-level search_path is set by migration 001.

NOTE: Apache AGE does **not** support parameterized Cypher queries.
All UUID values are interpolated into query strings after explicit
validation via ``_validate_uuid_for_cypher``.  This is a defense-in-
depth measure; monitor the AGE project for parameterized query support.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import Connection, text

GRAPH_NAME = "memory_graph"


def _age_exec(conn: Connection, cypher: str) -> None:
    """Run ``LOAD 'age'`` + ``SET search_path`` + ``cypher`` as three
    separate executes.

    v0.21.1 — pre-fix this issued all three as one concatenated SQL
    string. SQLAlchemy passes it through asyncpg's prepared-statement
    path (true even via ``run_sync`` because the underlying driver is
    still asyncpg under an async engine); asyncpg rejects multi-
    command prepared statements with ``PostgresSyntaxError: cannot
    insert multiple commands into a prepared statement``. Every AGE
    write failed silently (caught by the best-effort savepoint in
    ``graph_writer``), leaving the graph projection empty for any
    distilled corpus. Surfaced during the v0.21 starter-kit smoke
    (Bug G in operator-notes/V0-21-STARTER-KIT-SMOKE-2026-05-12.md).
    """
    conn.execute(text("LOAD 'age'"))
    conn.execute(text('SET search_path = ag_catalog, "$user", public'))
    conn.execute(text(cypher))


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


def sync_memory_to_graph(
    conn: Connection,
    memory_id: UUID,
    org_id: UUID,
    agent_id: UUID,
    memory_type: str,
    content_preview: str | None = None,
) -> None:
    """Create or update a Memory vertex in the AGE graph.

    Args:
        conn: Active SQLAlchemy connection.
        memory_id: The memory's UUID.
        org_id: The tenant's org_id.
        agent_id: The agent's UUID.
        memory_type: One of working/episodic/semantic/procedural.
        content_preview: Optional truncated content for graph display.
    """
    safe_mid = _validate_uuid_for_cypher(memory_id)
    safe_org = _validate_uuid_for_cypher(org_id)
    safe_agent = _validate_uuid_for_cypher(agent_id)
    preview = (content_preview or "")[:200].replace("'", "\\'")
    cypher = (
        f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
        f"MERGE (m:Memory {{id: '{safe_mid}'}}) "
        f"SET m.org_id = '{safe_org}', "
        f"    m.agent_id = '{safe_agent}', "
        f"    m.memory_type = '{memory_type}', "
        f"    m.preview = '{preview}' "
        f"RETURN m "
        f"$$) AS (v agtype)"
    )
    _age_exec(conn, cypher)


def delete_memory_from_graph(
    conn: Connection,
    memory_id: UUID,
) -> None:
    """Delete a Memory vertex and all its edges from the AGE graph.

    DETACH DELETE removes the vertex and any edges connected to it.
    Used by hard_delete (GDPR) to ensure no orphaned graph data remains.

    Args:
        conn: Active SQLAlchemy connection.
        memory_id: The memory's UUID to delete.
    """
    safe_id = _validate_uuid_for_cypher(memory_id)
    cypher = (
        f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
        f"MATCH (m:Memory {{id: '{safe_id}'}}) "
        f"DETACH DELETE m "
        f"$$) AS (v agtype)"
    )
    _age_exec(conn, cypher)


def sync_relationship_to_graph(
    conn: Connection,
    source_id: UUID,
    target_id: UUID,
    relationship_type: str,
    weight: float = 1.0,
) -> None:
    """Create an edge between two Memory vertices in the AGE graph.

    The edge label matches the RelationshipType enum value, uppercased.
    E.g. 'derived_from' -> DERIVED_FROM edge label.

    Args:
        conn: Active SQLAlchemy connection.
        source_id: Source memory UUID.
        target_id: Target memory UUID.
        relationship_type: The relationship type value (e.g. 'derived_from').
        weight: Edge weight (0-1).
    """
    safe_source = _validate_uuid_for_cypher(source_id)
    safe_target = _validate_uuid_for_cypher(target_id)
    edge_label = relationship_type.upper()
    cypher = (
        f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
        f"MATCH (s:Memory {{id: '{safe_source}'}}), (t:Memory {{id: '{safe_target}'}}) "
        f"CREATE (s)-[r:{edge_label} {{weight: {weight}}}]->(t) "
        f"RETURN r "
        f"$$) AS (e agtype)"
    )
    _age_exec(conn, cypher)
