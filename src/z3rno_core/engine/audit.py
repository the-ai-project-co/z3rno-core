"""Audit log helpers - hash chain computation and audit entry creation.

The audit log uses a SHA-256 hash chain for tamper evidence:
  row_hash = SHA-256(prev_hash || canonical_json(row_data))

The chain is per-tenant (per org_id). Each new audit entry fetches the
most recent row_hash for that org and chains from it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


def compute_row_hash(prev_hash: bytes | None, data: dict[str, Any]) -> bytes:
    """Compute the SHA-256 hash for an audit log row.

    Args:
        prev_hash: The row_hash of the previous audit entry (None for first).
        data: The row data to hash (canonical JSON serialization).

    Returns:
        32-byte SHA-256 digest.
    """
    h = hashlib.sha256()
    if prev_hash:
        h.update(prev_hash)
    # Canonical JSON: sorted keys, no whitespace
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    h.update(canonical.encode("utf-8"))
    return h.digest()


async def get_latest_hash(conn: AsyncConnection, org_id: UUID) -> bytes | None:
    """Fetch the most recent row_hash for a tenant's audit chain.

    Returns None if this is the first audit entry for the tenant.
    """
    result = await conn.execute(
        text("""
            SELECT row_hash
            FROM audit_log
            WHERE org_id = :org_id
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """),
        {"org_id": str(org_id)},
    )
    row = result.fetchone()
    if row is None:
        return None
    return row[0]  # type: ignore[no-any-return]


async def create_audit_entry(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    operation: str,
    agent_id: UUID | None = None,
    user_id: UUID | None = None,
    memory_id: UUID | None = None,
    memory_type: str | None = None,
    details: dict[str, Any] | None = None,
    api_key_id: UUID | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> None:
    """Insert a hash-chained audit log entry.

    Fetches the latest hash for the tenant, computes the new row_hash,
    and inserts the entry atomically.
    """
    now = datetime.now().astimezone()
    prev_hash = await get_latest_hash(conn, org_id)

    row_data = {
        "org_id": str(org_id),
        "operation": operation,
        "agent_id": str(agent_id) if agent_id else None,
        "user_id": str(user_id) if user_id else None,
        "memory_id": str(memory_id) if memory_id else None,
        "memory_type": memory_type,
        "details": details or {},
        "created_at": now.isoformat(),
    }
    row_hash = compute_row_hash(prev_hash, row_data)

    await conn.execute(
        text("""
            INSERT INTO audit_log (
                org_id, agent_id, user_id, operation, memory_id,
                memory_type, details, prev_hash, row_hash,
                api_key_id, ip_address, user_agent, request_id,
                created_at, updated_at
            ) VALUES (
                CAST(:org_id AS uuid),
                CAST(:agent_id AS uuid),
                CAST(:user_id AS uuid),
                CAST(:operation AS audit_operation_enum),
                CAST(:memory_id AS uuid),
                CAST(:memory_type AS memory_type_enum),
                CAST(:details AS jsonb),
                :prev_hash,
                :row_hash,
                CAST(:api_key_id AS uuid),
                CAST(:ip_address AS inet),
                :user_agent,
                :request_id,
                :created_at,
                :created_at
            )
        """),
        {
            "org_id": str(org_id),
            "agent_id": str(agent_id) if agent_id else None,
            "user_id": str(user_id) if user_id else None,
            "operation": operation,
            "memory_id": str(memory_id) if memory_id else None,
            "memory_type": memory_type,
            "details": json.dumps(details or {}),
            "prev_hash": prev_hash,
            "row_hash": row_hash,
            "api_key_id": str(api_key_id) if api_key_id else None,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "request_id": request_id,
            "created_at": now.isoformat(),
        },
    )
