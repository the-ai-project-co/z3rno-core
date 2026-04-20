"""get_memory() - retrieve a single memory by ID.

Returns the current version of a memory (valid_to IS NULL)
for the authenticated tenant (org_id enforced).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass(frozen=True)
class MemoryDetail:
    """Full detail of a single memory."""

    id: UUID
    org_id: UUID
    agent_id: UUID
    user_id: UUID | None
    memory_type: str
    content: str
    summary: str | None
    metadata: dict[str, Any]
    importance_score: float
    recall_count: int
    last_recalled_at: datetime | None
    embedding_model: str | None
    pinned: bool
    created_at: datetime
    valid_from: datetime
    valid_to: datetime | None
    deleted_at: datetime | None


class MemoryNotFoundError(Exception):
    """Raised when a memory is not found."""


async def get_memory(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    memory_id: UUID,
    include_deleted: bool = False,
) -> MemoryDetail:
    """Retrieve a single memory by ID.

    Args:
        conn: Active async connection.
        org_id: Tenant org_id (RLS enforced).
        memory_id: The memory UUID to retrieve.
        include_deleted: If True, also return soft-deleted memories.

    Returns:
        MemoryDetail with full memory data.

    Raises:
        MemoryNotFoundError: If the memory does not exist or is not accessible.
    """
    deleted_filter = "" if include_deleted else "AND deleted_at IS NULL"

    result = await conn.execute(
        text(f"""
            SELECT id, org_id, agent_id, user_id, memory_type, content, summary,
                   metadata, importance_score, recall_count, last_recalled_at,
                   embedding_model, pinned, created_at, valid_from, valid_to, deleted_at
            FROM memories
            WHERE org_id = CAST(:org_id AS uuid)
              AND id = CAST(:memory_id AS uuid)
              AND valid_to IS NULL
              {deleted_filter}
        """),
        {"org_id": str(org_id), "memory_id": str(memory_id)},
    )
    row = result.fetchone()

    if not row:
        raise MemoryNotFoundError(f"Memory {memory_id} not found")

    return MemoryDetail(
        id=row[0],
        org_id=row[1],
        agent_id=row[2],
        user_id=row[3],
        memory_type=row[4],
        content=row[5],
        summary=row[6],
        metadata=row[7] or {},
        importance_score=float(row[8]),
        recall_count=int(row[9]),
        last_recalled_at=row[10],
        embedding_model=row[11],
        pinned=bool(row[12]),
        created_at=row[13],
        valid_from=row[14],
        valid_to=row[15],
        deleted_at=row[16],
    )
