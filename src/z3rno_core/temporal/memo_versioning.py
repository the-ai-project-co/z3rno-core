"""Phase F slice 3 — SCD-2 versioning of graph-node properties.

The existing temporal layer (``temporal/queries.py``) walks the
SCD-2 columns on the ``memories`` table itself — versions of the
whole row. This module versions the **graph-projected properties**
of a Memo (memo_type, ontology_uri, distill_provenance, content
snippet) in a parallel ``memo_versions`` table.

Why a parallel table:
  * The AGE graph mirror doesn't natively support SCD-2.
  * Refine's dedupe + reweight changes graph-visible attributes
    independently of the underlying SCD-2 row close — we still want
    "what did this Memo look like at T?" to answer correctly.

Versions are monotonically increasing per (memo_id). Writer closes
the prior open version (``valid_to = now()``) and inserts the new
one in a single transaction. Reader picks the row whose validity
window contains ``as_of``.

RLS enforcement is at the table level (Migration 026); these helpers
trust the caller has set ``app.current_org_id`` before invoking them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass(frozen=True)
class MemoVersion:
    """One row of ``memo_versions``."""

    memo_id: UUID
    version: int
    properties: dict[str, Any]
    valid_from: datetime
    valid_to: datetime | None


async def record_memo_version(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    memo_id: UUID,
    properties: dict[str, Any],
) -> int:
    """Append a new version row for ``memo_id``. Returns the new version number.

    Atomically:
      1. Closes the currently-open version (``valid_to = now()``).
      2. INSERTs a fresh row with ``version = prev + 1``.

    Caller owns the transaction. Idempotency: callers that supply
    identical ``properties`` back-to-back will still create a new
    version — dedup is intentionally not done here (auditing wants
    every state, including no-op writes).
    """
    # Find the current version number.
    row = (
        await conn.execute(
            text("""
                SELECT version FROM public.memo_versions
                WHERE memo_id = CAST(:memo_id AS uuid)
                  AND valid_to IS NULL
                LIMIT 1
            """),
            {"memo_id": str(memo_id)},
        )
    ).fetchone()
    next_version = (row[0] + 1) if row is not None else 1

    # Close any open version (idempotent on the first insert).
    if row is not None:
        await conn.execute(
            text("""
                UPDATE public.memo_versions
                SET valid_to = now()
                WHERE memo_id = CAST(:memo_id AS uuid)
                  AND valid_to IS NULL
            """),
            {"memo_id": str(memo_id)},
        )

    await conn.execute(
        text("""
            INSERT INTO public.memo_versions (
                org_id, memo_id, version, properties, valid_from
            ) VALUES (
                CAST(:org_id AS uuid),
                CAST(:memo_id AS uuid),
                :version,
                CAST(:properties AS jsonb),
                now()
            )
        """),
        {
            "org_id": str(org_id),
            "memo_id": str(memo_id),
            "version": next_version,
            "properties": json.dumps(properties),
        },
    )
    return next_version


async def get_memo_at(
    conn: AsyncConnection,
    *,
    memo_id: UUID,
    as_of: datetime | None = None,
) -> MemoVersion | None:
    """Return the Memo's properties at ``as_of`` (or current state when None).

    Returns ``None`` if no version exists for the Memo. RLS still
    applies — callers in another tenant see ``None``.
    """
    if as_of is None:
        row = (
            await conn.execute(
                text("""
                    SELECT memo_id, version, properties, valid_from, valid_to
                    FROM public.memo_versions
                    WHERE memo_id = CAST(:memo_id AS uuid)
                      AND valid_to IS NULL
                    ORDER BY version DESC
                    LIMIT 1
                """),
                {"memo_id": str(memo_id)},
            )
        ).fetchone()
    else:
        row = (
            await conn.execute(
                text("""
                    SELECT memo_id, version, properties, valid_from, valid_to
                    FROM public.memo_versions
                    WHERE memo_id = CAST(:memo_id AS uuid)
                      AND valid_from <= :as_of
                      AND (valid_to IS NULL OR valid_to > :as_of)
                    ORDER BY version DESC
                    LIMIT 1
                """),
                {"memo_id": str(memo_id), "as_of": as_of},
            )
        ).fetchone()
    if row is None:
        return None
    return MemoVersion(
        memo_id=row[0],
        version=int(row[1]),
        properties=row[2] if row[2] else {},
        valid_from=row[3],
        valid_to=row[4],
    )


async def list_memo_versions(
    conn: AsyncConnection,
    *,
    memo_id: UUID,
    limit: int = 50,
) -> list[MemoVersion]:
    """Return every version of ``memo_id``, newest first."""
    rows = (
        await conn.execute(
            text("""
                SELECT memo_id, version, properties, valid_from, valid_to
                FROM public.memo_versions
                WHERE memo_id = CAST(:memo_id AS uuid)
                ORDER BY version DESC
                LIMIT :limit
            """),
            {"memo_id": str(memo_id), "limit": limit},
        )
    ).fetchall()
    return [
        MemoVersion(
            memo_id=r[0],
            version=int(r[1]),
            properties=r[2] if r[2] else {},
            valid_from=r[3],
            valid_to=r[4],
        )
        for r in rows
    ]
