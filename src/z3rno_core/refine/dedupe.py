"""Dedupe stage of the Refine pipeline (Phase D slice 3).

Merges Memos that point at the same real-world entity. Two Memos are
considered duplicates when *either*:

  1. Both carry the same ``ontology_uri`` (strong signal — the
     ontology resolver canonicalized them, populated by slice 4).
  2. Both share the same ``(memo_type, normalized_name)`` pair, where
     ``normalized_name`` is ``content`` lower-cased + whitespace-
     collapsed (weaker but useful before the resolver is enabled).

Both signals are scoped to ``(org_id, dataset_id)`` — we never merge
across tenants, and dedupe inside one dataset only.

Merge strategy
--------------
Inside a duplicate group:

  * Pick the **primary** = oldest currently-valid row (smallest
    ``valid_from``, ties broken by smallest ``id``). The primary keeps
    living.
  * **Supersede** the losers by setting ``valid_to = now()`` and
    ``deleted_at = now()``. The audit chain stays intact (SCD-2 — no
    rows are hard-deleted).
  * Record the merge in the primary's ``metadata`` JSONB under
    ``refine.merged_from`` so callers can reconstruct lineage.

Why SCD-2 and not a hard merge
------------------------------
A hard merge violates the temporal contract — we'd lose the audit
trail showing the duplicate ever existed. The Phase B.1 launch hit
exactly this with the SCD-2 trigger's recursion guard; staying on the
beaten path here avoids the same trap.

The DELETE of the loser rows is deferred to the existing forget
lifecycle — dedupe only marks them. The lifecycle sweeper already
handles soft-deleted rows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


_WS_RE = re.compile(r"\s+")


def normalize_name(s: str) -> str:
    """Lower-case + whitespace-collapse for fuzzy-equality grouping.

    Pure function so the unit tests can verify grouping without a DB.
    """
    return _WS_RE.sub(" ", s.strip().lower())


@dataclass(frozen=True)
class DedupeGroup:
    """One group of duplicate Memos sharing a key.

    ``primary_id`` is kept; ``loser_ids`` are SCD-2-superseded.
    """

    key: str
    primary_id: UUID
    loser_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class DedupeResult:
    memos_scanned: int
    groups: tuple[DedupeGroup, ...]

    @property
    def memos_deduped(self) -> int:
        return sum(len(g.loser_ids) for g in self.groups)


def _group_rows(rows: list[tuple[UUID, str | None, str | None, str]]) -> list[DedupeGroup]:
    """Group ``(id, memo_type, ontology_uri, content)`` rows into dedupe groups.

    Pure / deterministic. Rows arrive ordered by ``valid_from ASC, id ASC``
    so the first row in each group is the primary.
    """
    by_key: dict[str, list[UUID]] = {}

    for row_id, memo_type, ontology_uri, content in rows:
        if ontology_uri:
            key = f"uri:{ontology_uri}"
        elif memo_type:
            key = f"type:{memo_type}:{normalize_name(content)}"
        else:
            # No ontology_uri, no memo_type → no dedupe signal. Skip.
            continue
        by_key.setdefault(key, []).append(row_id)

    return [
        DedupeGroup(key=key, primary_id=ids[0], loser_ids=tuple(ids[1:]))
        for key, ids in by_key.items()
        if len(ids) > 1
    ]


async def run_dedupe(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    dataset_id: UUID | None = None,
) -> DedupeResult:
    """Scan currently-valid Memos in scope, supersede duplicates.

    ``dataset_id=None`` widens the scan to every dataset in the org.
    """
    where_dataset = (
        "AND dataset_id = CAST(:dataset_id AS uuid)" if dataset_id else "AND dataset_id IS NULL"
    )
    params: dict[str, object] = {"org_id": str(org_id)}
    if dataset_id:
        params["dataset_id"] = str(dataset_id)

    rows = (
        await conn.execute(
            text(f"""
                SELECT id, memo_type, ontology_uri, content
                FROM public.memories
                WHERE org_id = CAST(:org_id AS uuid)
                  AND valid_to IS NULL
                  AND deleted_at IS NULL
                  {where_dataset}
                ORDER BY valid_from ASC, id ASC
            """),  # noqa: S608 — interpolated identifier is constant
            params,
        )
    ).fetchall()

    typed_rows: list[tuple[UUID, str | None, str | None, str]] = [
        (r[0], r[1], r[2], r[3] or "") for r in rows
    ]
    groups = _group_rows(typed_rows)

    for group in groups:
        merged_from = [str(lid) for lid in group.loser_ids]
        # 1. Supersede the losers — SCD-2 close + soft delete.
        await conn.execute(
            text("""
                UPDATE public.memories
                SET valid_to = now(),
                    deleted_at = now(),
                    updated_at = now()
                WHERE id = ANY(CAST(:ids AS uuid[]))
                  AND valid_to IS NULL
                  AND deleted_at IS NULL
            """),
            {"ids": merged_from},
        )
        # 2. Stamp lineage on the primary's metadata.
        await conn.execute(
            text("""
                UPDATE public.memories
                SET metadata = COALESCE(metadata, '{}'::jsonb)
                            || jsonb_build_object(
                                'refine',
                                COALESCE(metadata->'refine', '{}'::jsonb)
                                || jsonb_build_object(
                                    'merged_from',
                                    COALESCE(metadata->'refine'->'merged_from', '[]'::jsonb)
                                        || CAST(:merged AS jsonb)
                                )
                            ),
                    updated_at = now()
                WHERE id = CAST(:primary AS uuid)
            """),
            {"primary": str(group.primary_id), "merged": json.dumps(merged_from)},
        )

    return DedupeResult(memos_scanned=len(typed_rows), groups=tuple(groups))
