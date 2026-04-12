"""Sync relational memory data to the Apache AGE graph.

When a memory is stored or a relationship is created, these functions
mirror the data into the AGE graph as vertices and edges.

All AGE queries require LOAD 'age' and the search_path set to include
ag_catalog. The database-level search_path is set by migration 001.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import Connection, text

GRAPH_NAME = "memory_graph"


def _age_preamble() -> str:
    """SQL preamble required before every AGE Cypher query."""
    return "LOAD 'age'; SET search_path = ag_catalog, \"$user\", public;"


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
    preview = (content_preview or "")[:200].replace("'", "\\'")
    cypher = (
        f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
        f"MERGE (m:Memory {{id: '{memory_id}'}}) "
        f"SET m.org_id = '{org_id}', "
        f"    m.agent_id = '{agent_id}', "
        f"    m.memory_type = '{memory_type}', "
        f"    m.preview = '{preview}' "
        f"RETURN m "
        f"$$) AS (v agtype)"
    )
    conn.execute(text(f"{_age_preamble()} {cypher}"))


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
    edge_label = relationship_type.upper()
    cypher = (
        f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
        f"MATCH (s:Memory {{id: '{source_id}'}}), (t:Memory {{id: '{target_id}'}}) "
        f"CREATE (s)-[r:{edge_label} {{weight: {weight}}}]->(t) "
        f"RETURN r "
        f"$$) AS (e agtype)"
    )
    conn.execute(text(f"{_age_preamble()} {cypher}"))
