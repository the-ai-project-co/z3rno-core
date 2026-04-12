"""Temporal query functions for SCD Type 2 memory versioning.

All functions operate on a SQLAlchemy Connection and return Row objects.
They are designed to work with RLS (the caller must have set the org
context before calling these functions).

Schema reference: Doc 08 Section 4.3.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Connection, Row, text
from sqlalchemy.engine import CursorResult


def get_memory_at_time(
    conn: Connection,
    memory_id: UUID,
    ts: datetime,
) -> Row[tuple[object, ...]] | None:
    """Return the memory version that was valid at timestamp ``ts``.

    Uses the SCD Type 2 pattern:
      valid_from <= ts AND (valid_to IS NULL OR valid_to > ts)

    Returns None if no version was valid at that time.
    """
    result = conn.execute(
        text("""
            SELECT *
            FROM memories
            WHERE id = :memory_id
              AND valid_from <= :ts
              AND (valid_to IS NULL OR valid_to > :ts)
            LIMIT 1
        """),
        {"memory_id": str(memory_id), "ts": ts},
    )
    return result.fetchone()


def get_memory_history(
    conn: Connection,
    memory_id: UUID,
) -> CursorResult[tuple[object, ...]]:
    """Return all versions of a memory ordered by valid_from (oldest first).

    Includes both current (valid_to IS NULL) and superseded versions.
    """
    return conn.execute(
        text("""
            SELECT *
            FROM memories
            WHERE id = :memory_id
            ORDER BY valid_from ASC
        """),
        {"memory_id": str(memory_id)},
    )


def get_memories_changed_between(
    conn: Connection,
    start: datetime,
    end: datetime,
    org_id: UUID | None = None,
) -> CursorResult[tuple[object, ...]]:
    """Return memories whose valid_from falls within [start, end].

    This finds memories that were created or updated (new version created)
    in the given time range. If org_id is provided, filters to that tenant
    (useful when not relying on RLS, e.g. admin queries).
    """
    if org_id is not None:
        return conn.execute(
            text("""
                SELECT *
                FROM memories
                WHERE valid_from BETWEEN :start AND :end
                  AND org_id = :org_id
                ORDER BY valid_from ASC
            """),
            {"start": start, "end": end, "org_id": str(org_id)},
        )
    return conn.execute(
        text("""
            SELECT *
            FROM memories
            WHERE valid_from BETWEEN :start AND :end
            ORDER BY valid_from ASC
        """),
        {"start": start, "end": end},
    )
