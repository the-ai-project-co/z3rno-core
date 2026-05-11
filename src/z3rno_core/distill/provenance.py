"""Phase F slice 1 — provenance enforcement.

Two responsibilities:

  1. Build the JSONB blob the graph-writer stamps on every distilled
     Memo (``memories.distill_provenance``).
  2. Validate that a given Memo's provenance chain is intact:
     - the JSONB blob exists,
     - an ``entity_provenance`` row exists for it,
     - an ``audit_log`` (or ``audit_log_pending``) entry exists whose
       ``details->>'correlation_id'`` matches the blob's correlation id.

When ``DISTILL_PROVENANCE_REQUIRED=true`` (set server-side), the
graph-writer raises :class:`ProvenanceRequiredError` rather than
leaving a Memo with a broken chain. With the flag off, missing
provenance is logged and skipped — Phase F is purely additive on the
write path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


class ProvenanceRequiredError(Exception):
    """Raised when DISTILL_PROVENANCE_REQUIRED=true and a Memo's
    provenance chain is incomplete."""


def build_provenance_blob(
    *,
    source_memory_id: UUID,
    model: str,
    prompt_hash: str,
    distill_job_id: UUID,
    chunk_index: int | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    correlation_id: UUID | None = None,
) -> dict[str, Any]:
    """Build the canonical JSONB blob written to ``memories.distill_provenance``.

    ``correlation_id`` is the bridge between this blob and the matching
    audit-chain entry. If not supplied, one is generated.
    """
    return {
        "source_memory_id": str(source_memory_id),
        "model": model,
        "prompt_hash": prompt_hash,
        "distill_job_id": str(distill_job_id),
        "chunk_index": chunk_index,
        "char_start": char_start,
        "char_end": char_end,
        "correlation_id": str(correlation_id or uuid4()),
        "ts": datetime.now(UTC).isoformat(),
    }


async def stamp_provenance(
    conn: AsyncConnection,
    *,
    memo_id: UUID,
    blob: dict[str, Any],
) -> None:
    """Persist the provenance blob on a freshly-written Memo row."""
    import json  # noqa: PLC0415

    await conn.execute(
        text("""
            UPDATE public.memories
            SET distill_provenance = CAST(:blob AS jsonb),
                updated_at = now()
            WHERE id = CAST(:memo_id AS uuid)
        """),
        {"memo_id": str(memo_id), "blob": json.dumps(blob)},
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainVerdict:
    """The result of :func:`verify_chain`."""

    memo_id: UUID
    is_valid: bool
    reason: str = ""

    @property
    def is_broken(self) -> bool:
        return not self.is_valid


async def verify_chain(conn: AsyncConnection, *, memo_id: UUID) -> ChainVerdict:
    """Walk the provenance chain for one Memo. Returns a verdict.

    A chain is valid iff:
      * ``memories.distill_provenance`` is non-NULL.
      * The blob contains a ``correlation_id``.
      * An ``entity_provenance`` row exists for the Memo.
      * An audit entry (either drained ``audit_log`` or pending
        ``audit_log_pending``) carries the same ``correlation_id`` in
        its ``details`` JSONB.

    Caller must have RLS context set; this function only inspects rows
    inside the caller's org.
    """
    row = (
        await conn.execute(
            text("""
                SELECT distill_provenance
                FROM public.memories
                WHERE id = CAST(:memo_id AS uuid)
            """),
            {"memo_id": str(memo_id)},
        )
    ).fetchone()

    if row is None:
        return ChainVerdict(memo_id=memo_id, is_valid=False, reason="memo not found")
    blob = row[0]
    if blob is None:
        return ChainVerdict(
            memo_id=memo_id,
            is_valid=False,
            reason="distill_provenance is NULL (pre-Phase-F Memo or skipped)",
        )

    correlation_id = blob.get("correlation_id") if isinstance(blob, dict) else None
    if not correlation_id:
        return ChainVerdict(
            memo_id=memo_id,
            is_valid=False,
            reason="distill_provenance is missing correlation_id",
        )

    # entity_provenance row
    ep_row = (
        await conn.execute(
            text("""
                SELECT 1 FROM public.entity_provenance
                WHERE memo_id = CAST(:memo_id AS uuid)
                LIMIT 1
            """),
            {"memo_id": str(memo_id)},
        )
    ).fetchone()
    if ep_row is None:
        return ChainVerdict(
            memo_id=memo_id,
            is_valid=False,
            reason="no entity_provenance row for this Memo",
        )

    # audit entry: check chained first, fall back to pending.
    audit_row = (
        await conn.execute(
            text("""
                SELECT 1 FROM public.audit_log
                WHERE memory_id = CAST(:memo_id AS uuid)
                  AND operation = 'distill'
                  AND details->>'correlation_id' = :cid
                LIMIT 1
            """),
            {"memo_id": str(memo_id), "cid": correlation_id},
        )
    ).fetchone()
    if audit_row is None:
        pending_row = (
            await conn.execute(
                text("""
                    SELECT 1 FROM public.audit_log_pending
                    WHERE memory_id = CAST(:memo_id AS uuid)
                      AND operation = 'distill'
                      AND details->>'correlation_id' = :cid
                    LIMIT 1
                """),
                {"memo_id": str(memo_id), "cid": correlation_id},
            )
        ).fetchone()
        if pending_row is None:
            return ChainVerdict(
                memo_id=memo_id,
                is_valid=False,
                reason="no matching distill audit entry (chained or pending)",
            )

    return ChainVerdict(memo_id=memo_id, is_valid=True)
