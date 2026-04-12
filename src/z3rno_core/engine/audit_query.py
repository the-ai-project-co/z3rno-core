"""audit() - paginated, filterable audit log queries.

Provides the audit() query function and analytics helpers for
querying the append-only, hash-chained audit log.

The audit_log table is partitioned monthly on created_at, so
time-range queries are efficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass(frozen=True)
class AuditEntry:
    """A single audit log entry."""

    id: int
    org_id: UUID
    agent_id: UUID | None
    user_id: UUID | None
    operation: str
    memory_id: UUID | None
    memory_type: str | None
    details: dict[str, Any]
    prev_hash: bytes | None
    row_hash: bytes
    ip_address: str | None
    user_agent: str | None
    request_id: str | None
    created_at: datetime


@dataclass(frozen=True)
class AuditPage:
    """Paginated audit log response."""

    entries: list[AuditEntry]
    total: int
    page: int
    page_size: int
    has_next: bool


@dataclass(frozen=True)
class ActivitySummary:
    """Agent activity summary: counts by operation type."""

    agent_id: UUID
    operation_counts: dict[str, int] = field(default_factory=dict)
    total: int = 0


async def audit(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID | None = None,
    user_id: UUID | None = None,
    operation: str | None = None,
    memory_id: UUID | None = None,
    memory_type: str | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    page: int = 1,
    page_size: int = 50,
) -> AuditPage:
    """Query the audit log with optional filters and pagination.

    Args:
        conn: Active async connection.
        org_id: Tenant org_id (required - RLS enforced).
        agent_id: Filter by agent.
        user_id: Filter by user.
        operation: Filter by operation type (e.g. 'store', 'recall').
        memory_id: Filter by memory.
        memory_type: Filter by memory type.
        time_range: Filter by (start, end) created_at range.
        page: Page number (1-indexed).
        page_size: Results per page (max 100).

    Returns:
        AuditPage with entries and pagination metadata.
    """
    page_size = min(page_size, 100)
    offset = (page - 1) * page_size

    conditions: list[str] = ["org_id = CAST(:org_id AS uuid)"]
    params: dict[str, Any] = {"org_id": str(org_id)}

    if agent_id:
        conditions.append("agent_id = CAST(:agent_id AS uuid)")
        params["agent_id"] = str(agent_id)
    if user_id:
        conditions.append("user_id = CAST(:user_id AS uuid)")
        params["user_id"] = str(user_id)
    if operation:
        conditions.append("operation = CAST(:operation AS audit_operation_enum)")
        params["operation"] = operation
    if memory_id:
        conditions.append("memory_id = CAST(:memory_id AS uuid)")
        params["memory_id"] = str(memory_id)
    if memory_type:
        conditions.append("memory_type = CAST(:memory_type AS memory_type_enum)")
        params["memory_type"] = memory_type
    if time_range:
        conditions.append("created_at >= CAST(:time_start AS timestamptz)")
        conditions.append("created_at <= CAST(:time_end AS timestamptz)")
        params["time_start"] = time_range[0].isoformat()
        params["time_end"] = time_range[1].isoformat()

    where_clause = " AND ".join(conditions)

    # Count total
    count_result = await conn.execute(
        text(f"SELECT count(*) FROM audit_log WHERE {where_clause}"),
        params,
    )
    total = count_result.scalar() or 0

    # Fetch page
    params["limit"] = page_size
    params["offset"] = offset
    result = await conn.execute(
        text(f"""
            SELECT id, org_id, agent_id, user_id, operation,
                   memory_id, memory_type, details,
                   prev_hash, row_hash, ip_address, user_agent,
                   request_id, created_at
            FROM audit_log
            WHERE {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )

    entries = [
        AuditEntry(
            id=row[0],
            org_id=row[1],
            agent_id=row[2],
            user_id=row[3],
            operation=row[4],
            memory_id=row[5],
            memory_type=row[6],
            details=row[7] if row[7] else {},
            prev_hash=row[8],
            row_hash=row[9],
            ip_address=row[10],
            user_agent=row[11],
            request_id=row[12],
            created_at=row[13],
        )
        for row in result.fetchall()
    ]

    return AuditPage(
        entries=entries,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


async def get_agent_activity_summary(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID,
    time_range: tuple[datetime, datetime] | None = None,
) -> ActivitySummary:
    """Get operation counts by type for an agent."""
    conditions = [
        "org_id = CAST(:org_id AS uuid)",
        "agent_id = CAST(:agent_id AS uuid)",
    ]
    params: dict[str, Any] = {"org_id": str(org_id), "agent_id": str(agent_id)}

    if time_range:
        conditions.append("created_at >= CAST(:time_start AS timestamptz)")
        conditions.append("created_at <= CAST(:time_end AS timestamptz)")
        params["time_start"] = time_range[0].isoformat()
        params["time_end"] = time_range[1].isoformat()

    where_clause = " AND ".join(conditions)
    result = await conn.execute(
        text(f"""
            SELECT operation, count(*) as cnt
            FROM audit_log
            WHERE {where_clause}
            GROUP BY operation
        """),
        params,
    )

    operation_counts = {row[0]: row[1] for row in result.fetchall()}
    total = sum(operation_counts.values())

    return ActivitySummary(
        agent_id=agent_id,
        operation_counts=operation_counts,
        total=total,
    )


async def get_memory_lifecycle(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    memory_id: UUID,
) -> list[AuditEntry]:
    """Get the full audit history for a single memory."""
    result = await conn.execute(
        text("""
            SELECT id, org_id, agent_id, user_id, operation,
                   memory_id, memory_type, details,
                   prev_hash, row_hash, ip_address, user_agent,
                   request_id, created_at
            FROM audit_log
            WHERE org_id = CAST(:org_id AS uuid)
              AND memory_id = CAST(:memory_id AS uuid)
            ORDER BY created_at ASC, id ASC
        """),
        {"org_id": str(org_id), "memory_id": str(memory_id)},
    )

    return [
        AuditEntry(
            id=row[0],
            org_id=row[1],
            agent_id=row[2],
            user_id=row[3],
            operation=row[4],
            memory_id=row[5],
            memory_type=row[6],
            details=row[7] if row[7] else {},
            prev_hash=row[8],
            row_hash=row[9],
            ip_address=row[10],
            user_agent=row[11],
            request_id=row[12],
            created_at=row[13],
        )
        for row in result.fetchall()
    ]
