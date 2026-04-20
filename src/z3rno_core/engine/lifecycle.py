"""Memory lifecycle management: TTL expiry, retention caps, transitions, and decay.

These functions are designed to be called by Celery workers scheduled
via pg_cron. The pattern is: pg_cron schedules (inside DB), Celery
executes (Python-side with LLM/embedding access).

Authoritative: Doc 08 Section 6.2 (importance + decay), Section 6.3 (forgetting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.engine.audit import create_audit_entry

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepResult:
    """Result of a TTL expiry sweep."""

    expired_count: int
    memory_ids: list[UUID]


@dataclass(frozen=True)
class DecayResult:
    """Result of a decay run."""

    decayed_count: int
    below_threshold_count: int


@dataclass(frozen=True)
class TransitionResult:
    """Result of a memory type transition."""

    memory_id: UUID
    old_type: str
    new_type: str


# ---------------------------------------------------------------------------
# TTL Expiry Sweep
# ---------------------------------------------------------------------------


async def sweep_expired_memories(
    conn: AsyncConnection,
    *,
    org_id: UUID,
) -> SweepResult:
    """Find and soft-delete all memories past their TTL.

    Finds memories where ttl_expires_at < now() AND deleted_at IS NULL,
    sets deleted_at = now(), and creates audit entries.
    """
    # Find expired memories
    result = await conn.execute(
        text("""
            SELECT id FROM memories
            WHERE org_id = CAST(:org_id AS uuid)
              AND ttl_expires_at < now()
              AND deleted_at IS NULL
        """),
        {"org_id": str(org_id)},
    )
    expired_ids = [row[0] for row in result.fetchall()]

    if not expired_ids:
        return SweepResult(expired_count=0, memory_ids=[])

    # Soft-delete
    id_list = ",".join(f"'{mid}'" for mid in expired_ids)
    await conn.execute(
        text(f"""
            UPDATE memories
            SET deleted_at = now(), updated_at = now()
            WHERE id IN ({id_list})
        """)
    )

    # Audit each expiry
    for mid in expired_ids:
        await create_audit_entry(
            conn,
            org_id=org_id,
            operation="forget",
            memory_id=mid,
            details={"reason": "ttl_expiry"},
        )

    return SweepResult(expired_count=len(expired_ids), memory_ids=expired_ids)


# ---------------------------------------------------------------------------
# Retention Cap Enforcement
# ---------------------------------------------------------------------------


async def enforce_retention_cap(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID,
    memory_type: str,
    max_count: int,
) -> SweepResult:
    """Enforce max memory count per agent per type.

    If the count exceeds max_count, soft-deletes the oldest/least-important
    memories to bring it back under the cap.
    """
    # Count current memories
    count_result = await conn.execute(
        text("""
            SELECT count(*) FROM memories
            WHERE org_id = CAST(:org_id AS uuid)
              AND agent_id = CAST(:agent_id AS uuid)
              AND memory_type = CAST(:memory_type AS memory_type_enum)
              AND deleted_at IS NULL
              AND valid_to IS NULL
        """),
        {"org_id": str(org_id), "agent_id": str(agent_id), "memory_type": memory_type},
    )
    current_count = count_result.scalar() or 0

    if current_count <= max_count:
        return SweepResult(expired_count=0, memory_ids=[])

    excess = current_count - max_count

    # Find least important, oldest memories to expire
    result = await conn.execute(
        text("""
            SELECT id FROM memories
            WHERE org_id = CAST(:org_id AS uuid)
              AND agent_id = CAST(:agent_id AS uuid)
              AND memory_type = CAST(:memory_type AS memory_type_enum)
              AND deleted_at IS NULL
              AND valid_to IS NULL
              AND pinned = false
            ORDER BY importance_score ASC, created_at ASC
            LIMIT :excess
        """),
        {
            "org_id": str(org_id),
            "agent_id": str(agent_id),
            "memory_type": memory_type,
            "excess": excess,
        },
    )
    to_expire = [row[0] for row in result.fetchall()]

    if not to_expire:
        return SweepResult(expired_count=0, memory_ids=[])

    id_list = ",".join(f"'{mid}'" for mid in to_expire)
    await conn.execute(
        text(f"""
            UPDATE memories
            SET deleted_at = now(), updated_at = now()
            WHERE id IN ({id_list})
        """)
    )

    for mid in to_expire:
        await create_audit_entry(
            conn,
            org_id=org_id,
            operation="forget",
            agent_id=agent_id,
            memory_id=mid,
            details={"reason": "retention_cap", "max_count": max_count},
        )

    return SweepResult(expired_count=len(to_expire), memory_ids=to_expire)


# ---------------------------------------------------------------------------
# Memory Type Transitions
# ---------------------------------------------------------------------------


async def transition_memory(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    memory_id: UUID,
    new_type: str,
    agent_id: UUID | None = None,
) -> TransitionResult:
    """Transition a memory to a new type (e.g. working -> episodic).

    Creates a new temporal version with the updated memory_type via
    the SCD Type 2 trigger.
    """
    # Get current type
    result = await conn.execute(
        text("""
            SELECT memory_type FROM memories
            WHERE org_id = CAST(:org_id AS uuid)
              AND id = CAST(:memory_id AS uuid)
              AND valid_to IS NULL
              AND deleted_at IS NULL
        """),
        {"org_id": str(org_id), "memory_id": str(memory_id)},
    )
    row = result.fetchone()
    if row is None:
        msg = f"Memory {memory_id} not found or already deleted"
        raise ValueError(msg)

    old_type = row[0]

    # Update memory_type - the SCD trigger will handle versioning
    await conn.execute(
        text("""
            UPDATE memories
            SET memory_type = CAST(:new_type AS memory_type_enum),
                updated_at = now()
            WHERE org_id = CAST(:org_id AS uuid)
              AND id = CAST(:memory_id AS uuid)
              AND valid_to IS NULL
        """),
        {"org_id": str(org_id), "memory_id": str(memory_id), "new_type": new_type},
    )

    # Audit
    await create_audit_entry(
        conn,
        org_id=org_id,
        operation="update",
        agent_id=agent_id,
        memory_id=memory_id,
        memory_type=new_type,
        details={"transition": f"{old_type} -> {new_type}"},
    )

    return TransitionResult(
        memory_id=memory_id,
        old_type=old_type,
        new_type=new_type,
    )


# ---------------------------------------------------------------------------
# Importance Decay
# ---------------------------------------------------------------------------


async def decay_importance(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    decay_rate: float = 0.05,
    min_threshold: float = 0.05,
    decay_floor: float = 0.0,
) -> DecayResult:
    """Apply exponential decay to importance scores based on time since last access.

    Formula: new_score = max(floor, score * e^(-rate * days_since_last_access))

    Memories below min_threshold after decay are candidates for auto-expiry
    (but not auto-expired by this function).

    Args:
        conn: Active async connection.
        org_id: Tenant org_id.
        decay_rate: Exponential decay rate per day.
        min_threshold: Score below which memories are flagged.
        decay_floor: Minimum score (never decays below this).

    Returns:
        DecayResult with counts.
    """
    now = datetime.now().astimezone()

    # Get all active, non-pinned memories with their last access time
    result = await conn.execute(
        text("""
            SELECT id, importance_score, last_recalled_at, created_at
            FROM memories
            WHERE org_id = CAST(:org_id AS uuid)
              AND deleted_at IS NULL
              AND valid_to IS NULL
              AND pinned = false
              AND importance_score > :floor
        """),
        {"org_id": str(org_id), "floor": decay_floor},
    )
    rows = result.fetchall()

    decayed_count = 0
    below_threshold = 0
    updates: list[dict[str, Any]] = []

    for row in rows:
        memory_id = row[0]
        current_score = float(row[1])
        last_access = row[2] or row[3]  # Fall back to created_at

        days_since = max((now - last_access).total_seconds() / 86400, 0)
        new_score = max(decay_floor, current_score * math.exp(-decay_rate * days_since))
        new_score = round(new_score, 6)

        if abs(new_score - current_score) > 0.0001:
            updates.append({"id": str(memory_id), "score": new_score})
            decayed_count += 1
            if new_score < min_threshold:
                below_threshold += 1

    # Batch update using a single UPDATE ... FROM (VALUES ...) statement
    if updates:
        values_list = ", ".join(f"(CAST('{upd['id']}' AS uuid), {upd['score']})" for upd in updates)
        await conn.execute(
            text(f"""
                UPDATE memories AS m
                SET importance_score = v.new_score, updated_at = now()
                FROM (VALUES {values_list}) AS v(id, new_score)
                WHERE m.id = v.id
            """)
        )

    return DecayResult(
        decayed_count=decayed_count,
        below_threshold_count=below_threshold,
    )


# ---------------------------------------------------------------------------
# Audit log partition management
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionResult:
    """Result of an auto-partition run."""

    created_count: int
    partition_names: list[str]


async def ensure_audit_partitions(
    conn: AsyncConnection,
    *,
    months_ahead: int = 3,
) -> PartitionResult:
    """Create audit_log partitions for the next N months if they don't exist.

    Should be called periodically (e.g., daily via Celery) to ensure
    partitions always exist ahead of time. Without pre-created partitions,
    inserts fall into the DEFAULT partition with degraded performance.

    Args:
        conn: Active async connection.
        months_ahead: How many months of future partitions to create.

    Returns:
        PartitionResult with count and names of newly created partitions.
    """
    now = datetime.now(UTC)
    created: list[str] = []

    for offset in range(months_ahead + 1):
        # Calculate the target month
        month = now.month + offset
        year = now.year
        while month > 12:
            month -= 12
            year += 1

        # Next month boundary
        next_month = month + 1
        next_year = year
        if next_month > 12:
            next_month = 1
            next_year += 1

        partition_name = f"audit_log_y{year}m{month:02d}"
        range_start = f"{year}-{month:02d}-01"
        range_end = f"{next_year}-{next_month:02d}-01"

        # Check if partition already exists
        result = await conn.execute(
            text("""
                SELECT 1 FROM pg_class
                WHERE relname = :partition_name
                  AND relkind = 'r'
            """),
            {"partition_name": partition_name},
        )
        if result.fetchone():
            continue

        # Create the partition
        await conn.execute(
            text(f"""
                CREATE TABLE {partition_name} PARTITION OF audit_log
                    FOR VALUES FROM ('{range_start}') TO ('{range_end}')
            """)
        )
        created.append(partition_name)

    return PartitionResult(created_count=len(created), partition_names=created)
