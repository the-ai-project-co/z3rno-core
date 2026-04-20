"""update_memory() - update an existing memory's content, metadata, or importance.

Atomically:
  1. Validates inputs
  2. Executes UPDATE on the memories table (SCD trigger auto-versions)
  3. Creates audit entry with operation='update'
  4. Returns updated memory detail

The entire operation runs in a single transaction.

Schema reference: Doc 08 Section 4.2.1 (memories).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.engine.audit import create_audit_entry
from z3rno_core.engine.get import MemoryDetail, MemoryNotFoundError


class UpdateError(Exception):
    """Raised when update_memory() validation fails."""


@dataclass(frozen=True)
class UpdateResult:
    """Result of a successful update_memory() operation."""

    memory_id: UUID
    content: str
    metadata: dict[str, Any]
    importance_score: float
    updated_at: datetime


async def update_memory(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    memory_id: UUID,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
    importance: float | None = None,
    # Audit context
    agent_id: UUID | None = None,
    api_key_id: UUID | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> MemoryDetail:
    """Update an existing memory's content, metadata, or importance.

    Args:
        conn: An active async connection (must be inside a transaction).
        org_id: The tenant's org_id.
        memory_id: The memory UUID to update.
        content: New content text (optional).
        metadata: New metadata dict (optional, replaces entire metadata).
        importance: New importance score 0-1 (optional).
        agent_id: For audit log context.
        api_key_id: For audit log.
        ip_address: For audit log.
        user_agent: For audit log.
        request_id: For audit log.

    Returns:
        MemoryDetail with the updated memory data.

    Raises:
        UpdateError: If validation fails or no fields provided.
        MemoryNotFoundError: If the memory does not exist.
    """
    # --- Validation ---
    if content is None and metadata is None and importance is None:
        raise UpdateError("At least one of content, metadata, or importance must be provided")

    if content is not None and (not content or not content.strip()):
        raise UpdateError("content must be a non-empty string")

    if importance is not None and not (0.0 <= importance <= 1.0):
        raise UpdateError("importance must be between 0.0 and 1.0")

    # --- Build dynamic SET clause ---
    set_clauses: list[str] = []
    params: dict[str, Any] = {
        "org_id": str(org_id),
        "memory_id": str(memory_id),
    }

    if content is not None:
        set_clauses.append("content = :content")
        params["content"] = content

    if metadata is not None:
        set_clauses.append("metadata = CAST(:metadata AS jsonb)")
        params["metadata"] = json.dumps(metadata, default=str)

    if importance is not None:
        set_clauses.append("importance_score = :importance")
        params["importance"] = importance

    set_clauses.append("updated_at = now()")

    # --- Execute UPDATE (SCD trigger will auto-version) ---
    result = await conn.execute(
        text(f"""
            UPDATE memories
            SET {", ".join(set_clauses)}
            WHERE org_id = CAST(:org_id AS uuid)
              AND id = CAST(:memory_id AS uuid)
              AND valid_to IS NULL
              AND deleted_at IS NULL
            RETURNING id
        """),
        params,
    )
    row = result.fetchone()
    if not row:
        raise MemoryNotFoundError(f"Memory {memory_id} not found")

    # --- Audit log ---
    details: dict[str, Any] = {}
    if content is not None:
        details["content_length"] = len(content)
    if metadata is not None:
        details["metadata_keys"] = list(metadata.keys())
    if importance is not None:
        details["importance_score"] = importance

    # Fetch the agent_id from the memory if not provided
    if agent_id is None:
        agent_result = await conn.execute(
            text("""
                SELECT agent_id FROM memories
                WHERE id = CAST(:memory_id AS uuid)
                  AND org_id = CAST(:org_id AS uuid)
                  AND valid_to IS NULL
                LIMIT 1
            """),
            {"memory_id": str(memory_id), "org_id": str(org_id)},
        )
        agent_row = agent_result.fetchone()
        if agent_row:
            agent_id = agent_row[0]

    await create_audit_entry(
        conn,
        org_id=org_id,
        operation="update",
        agent_id=agent_id,
        memory_id=memory_id,
        details=details,
        api_key_id=api_key_id,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
    )

    # --- Fetch and return updated memory ---
    from z3rno_core.engine.get import get_memory

    return await get_memory(conn, org_id=org_id, memory_id=memory_id)
