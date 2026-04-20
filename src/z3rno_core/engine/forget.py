"""forget() - soft delete, hard delete (GDPR), and cascade forget.

Soft delete: sets deleted_at = now(). Row preserved for audit/recovery.
Hard delete: permanently removes memory, relationships, and graph data.
  Leaves only a GDPR-compliant audit stub with NO content.

Safety: hard_delete requires explicit opt-in. Default is soft delete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.engine.audit import create_audit_entry
from z3rno_core.graph.sync import delete_memory_from_graph


@dataclass(frozen=True)
class ForgetResult:
    """Result of a forget() operation."""

    deleted_count: int
    hard_deleted: bool
    cascade_count: int
    memory_ids: list[UUID]


class ForgetError(Exception):
    """Raised when forget() fails."""


async def forget(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID,
    memory_id: UUID | None = None,
    memory_ids: list[UUID] | None = None,
    hard_delete: bool = False,
    cascade: bool = False,
    cascade_depth: int = 1,
    reason: str | None = None,
    # Audit context
    user_id: UUID | None = None,
    api_key_id: UUID | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> ForgetResult:
    """Forget (delete) one or more memories.

    Args:
        conn: Active async connection (must be in a transaction).
        org_id: Tenant org_id.
        agent_id: Agent performing the forget.
        memory_id: Single memory to forget.
        memory_ids: Multiple memories to forget.
        hard_delete: If True, permanently remove (GDPR). Default: soft delete.
        cascade: If True, also delete related memories up to cascade_depth hops.
        cascade_depth: Maximum number of relationship hops to traverse when
            cascading (default 1 for backward compatibility). Uses BFS.
        reason: Reason for deletion (stored in audit log).
        user_id: For audit log.
        api_key_id: For audit log.
        ip_address: For audit log.
        user_agent: For audit log.
        request_id: For audit log.

    Returns:
        ForgetResult with deletion counts.

    Raises:
        ForgetError: If no memory_id or memory_ids provided.
    """
    # Collect target IDs
    targets: list[UUID] = []
    if memory_id:
        targets.append(memory_id)
    if memory_ids:
        targets.extend(memory_ids)
    if not targets:
        raise ForgetError("must provide memory_id or memory_ids")

    # Cascade: find related memories via BFS up to cascade_depth hops
    cascade_count = 0
    if cascade:
        cascade_targets = await _find_related_memories(conn, org_id, targets, cascade_depth)
        cascade_count = len(cascade_targets)
        targets.extend(cascade_targets)

    # Deduplicate
    targets = list(set(targets))

    if hard_delete:
        deleted = await _hard_delete(conn, org_id, targets)
    else:
        deleted = await _soft_delete(conn, org_id, targets)

    # Audit each deletion
    operation = "gdpr_delete" if hard_delete else "forget"
    for mid in targets:
        details: dict[str, Any] = {
            "reason": reason or ("GDPR right to erasure" if hard_delete else "user requested"),
            "hard_delete": hard_delete,
            "cascade": cascade,
        }
        # GDPR: audit stub must NOT contain content or metadata
        if not hard_delete:
            details["memory_id"] = str(mid)

        await create_audit_entry(
            conn,
            org_id=org_id,
            operation=operation,
            agent_id=agent_id,
            user_id=user_id,
            memory_id=mid,
            details=details,
            api_key_id=api_key_id,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
        )

    return ForgetResult(
        deleted_count=deleted,
        hard_deleted=hard_delete,
        cascade_count=cascade_count,
        memory_ids=targets,
    )


async def _soft_delete(
    conn: AsyncConnection,
    org_id: UUID,
    memory_ids: list[UUID],
) -> int:
    """Set deleted_at = now() on memories. Does NOT touch valid_to."""
    id_list = ",".join(f"'{mid}'" for mid in memory_ids)
    result = await conn.execute(
        text(f"""
            UPDATE memories
            SET deleted_at = now(), updated_at = now()
            WHERE org_id = CAST('{org_id}' AS uuid)
              AND id IN ({id_list})
              AND deleted_at IS NULL
        """)
    )
    return result.rowcount


async def _hard_delete(
    conn: AsyncConnection,
    org_id: UUID,
    memory_ids: list[UUID],
) -> int:
    """Permanently remove memories, relationships, and graph data.

    Deletion order:
      1. AGE graph vertices + edges (DETACH DELETE)
      2. Relational memory_relationships (FK constraint)
      3. Relational memories
    """
    id_list = ",".join(f"'{mid}'" for mid in memory_ids)

    # Delete AGE graph vertices and their edges (GDPR: no orphaned graph data)
    for mid in memory_ids:
        try:
            await conn.run_sync(lambda sync_conn: delete_memory_from_graph(sync_conn, mid))
        except Exception:
            # AGE extension may not be loaded in all environments (e.g., tests)
            pass

    # Delete relationships first (FK constraint)
    await conn.execute(
        text(f"""
            DELETE FROM memory_relationships
            WHERE org_id = CAST('{org_id}' AS uuid)
              AND (source_memory_id IN ({id_list}) OR target_memory_id IN ({id_list}))
        """)
    )

    # Delete memories
    result = await conn.execute(
        text(f"""
            DELETE FROM memories
            WHERE org_id = CAST('{org_id}' AS uuid)
              AND id IN ({id_list})
        """)
    )

    return result.rowcount


async def _find_related_memories(
    conn: AsyncConnection,
    org_id: UUID,
    memory_ids: list[UUID],
    depth: int = 1,
) -> list[UUID]:
    """Find related memories via BFS up to `depth` hops in memory_relationships.

    Args:
        conn: Active async connection.
        org_id: Tenant org_id.
        memory_ids: Starting set of memory IDs.
        depth: Maximum number of relationship hops to traverse (default 1).

    Returns:
        List of discovered related memory IDs (excludes the original memory_ids).
    """
    visited: set[UUID] = set(memory_ids)
    frontier: set[UUID] = set(memory_ids)

    for _ in range(depth):
        if not frontier:
            break
        id_list = ",".join(f"'{mid}'" for mid in frontier)
        result = await conn.execute(
            text(f"""
                SELECT DISTINCT CASE
                    WHEN source_memory_id IN ({id_list}) THEN target_memory_id
                    ELSE source_memory_id
                END AS related_id
                FROM memory_relationships
                WHERE org_id = CAST('{org_id}' AS uuid)
                  AND (source_memory_id IN ({id_list}) OR target_memory_id IN ({id_list}))
            """)
        )
        neighbors = {row[0] for row in result.fetchall()}
        # Next frontier is only newly discovered nodes
        frontier = neighbors - visited
        visited |= frontier

    # Return only the discovered nodes, not the original inputs
    return list(visited - set(memory_ids))
