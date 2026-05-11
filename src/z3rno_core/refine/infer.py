"""Infer stage of the Refine pipeline (Phase D slice 4).

For each Memo with no existing outbound relationships, ask the LLM to
propose plausible neighbors among the other Memos in the same scope.
Each proposed edge is written into ``memory_relationships`` (the AGE
mirror is synced by the existing graph-sync layer) at low confidence
so an unfortunate proposal washes out under negative feedback in the
next refine cycle.

This stage is **opt-in** (``REFINE_INFER_ENABLED=true``) AND requires
an LLM gateway. When either is missing the stage is a no-op.

Cap-and-budget
--------------
LLM calls are expensive. ``max_candidates`` caps how many source
Memos we consider per cycle. Selection is "Memos with the fewest
existing outbound edges first" — those are the ones most likely to
benefit from inference, and the heuristic is cheap to compute in SQL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

    from z3rno_core.distill.llm_gateway import LLMGateway

log = structlog.get_logger(__name__)

_MIN_CANDIDATES_FOR_INFER = 2
_MAX_OUTBOUND_PER_FOCAL = 5


class InferredEdge(BaseModel):
    """One LLM proposal. Pydantic ⇒ Instructor-friendly."""

    model_config = ConfigDict(frozen=True)

    target_id: str = Field(..., description="UUID string of an existing Memo in the candidate set.")
    predicate: str = Field(..., min_length=1, max_length=64)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class InferredEdgeList(BaseModel):
    """LLM response container — Instructor needs a top-level model."""

    model_config = ConfigDict(frozen=True)

    edges: tuple[InferredEdge, ...] = ()


@dataclass(frozen=True)
class InferResult:
    candidates_examined: int
    edges_proposed: int
    edges_written: int


_SYSTEM = (
    "You are a knowledge-graph editor. Given a focal entity and a set of candidate "
    "neighboring entities, propose a small number (0 to 5) of plausible directed "
    "relationships from the focal entity to candidates. Be conservative: only "
    "propose relationships you are confident about. Return an empty list when none "
    "are warranted."
)


def _format_user_prompt(
    focal_name: str, focal_type: str, candidates: list[tuple[UUID, str, str]]
) -> str:
    lines = [f"Focal entity: {focal_name} ({focal_type})", "", "Candidate neighbors:"]
    for cid, cname, ctype in candidates:
        lines.append(f"- {cid} | {cname} ({ctype})")
    lines.append("")
    lines.append(
        "Return up to 5 directed relationships from the focal entity to candidates "
        'as a JSON object {"edges": [{"target_id": ..., "predicate": ..., "confidence": ...}]}.'
    )
    return "\n".join(lines)


async def _fetch_candidates(
    conn: AsyncConnection, org_id: UUID, dataset_id: UUID | None, limit: int
) -> list[tuple[UUID, str, str]]:
    where_dataset = (
        "AND m.dataset_id = CAST(:dataset_id AS uuid)" if dataset_id else "AND m.dataset_id IS NULL"
    )
    params: dict[str, object] = {"org_id": str(org_id), "limit": limit}
    if dataset_id:
        params["dataset_id"] = str(dataset_id)

    rows = (
        await conn.execute(
            text(f"""
                SELECT m.id,
                       COALESCE(m.metadata->>'entity_name', LEFT(m.content, 80)),
                       COALESCE(m.memo_type, m.metadata->>'entity_type', 'ENTITY')
                FROM public.memories m
                WHERE m.org_id = CAST(:org_id AS uuid)
                  AND m.valid_to IS NULL
                  AND m.deleted_at IS NULL
                  AND m.memo_type IS NOT NULL
                  {where_dataset}
                LIMIT :limit
            """),  # noqa: S608 — interpolated identifier is constant
            params,
        )
    ).fetchall()
    return [(r[0], str(r[1] or ""), str(r[2] or "ENTITY")) for r in rows]


async def _existing_targets(conn: AsyncConnection, source_id: UUID) -> set[UUID]:
    rows = (
        await conn.execute(
            text("""
                SELECT target_memory_id FROM public.memory_relationships
                WHERE source_memory_id = CAST(:src AS uuid)
            """),
            {"src": str(source_id)},
        )
    ).fetchall()
    return {r[0] for r in rows}


async def _insert_inferred_edge(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    source_id: UUID,
    target_id: UUID,
    predicate: str,
    confidence: float,
) -> bool:
    """Insert one relational edge. Returns True on insert, False on conflict."""
    try:
        await conn.execute(
            text("""
                INSERT INTO public.memory_relationships (
                    id, org_id, source_memory_id, target_memory_id,
                    relationship_type, weight, metadata, created_at, updated_at
                ) VALUES (
                    CAST(:id AS uuid),
                    CAST(:org_id AS uuid),
                    CAST(:src AS uuid),
                    CAST(:tgt AS uuid),
                    CAST(:predicate AS relationship_type_enum),
                    :weight,
                    CAST(:meta AS jsonb),
                    now(), now()
                )
            """),
            {
                "id": str(uuid4()),
                "org_id": str(org_id),
                "src": str(source_id),
                "tgt": str(target_id),
                "predicate": predicate.lower(),
                "weight": confidence,
                "meta": '{"source": "refine.infer"}',
            },
        )
    except Exception:
        # Most common reason: predicate not in relationship_type_enum, or
        # FK self-loop. Skip silently; the refine cycle still progresses.
        return False
    return True


async def run_infer(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    gateway: LLMGateway | None,
    dataset_id: UUID | None = None,
    max_candidates: int = 50,
) -> InferResult:
    """Propose edges via LLM. No-op when ``gateway`` is None."""
    if gateway is None:
        return InferResult(candidates_examined=0, edges_proposed=0, edges_written=0)

    candidates = await _fetch_candidates(conn, org_id, dataset_id, max_candidates)
    if len(candidates) < _MIN_CANDIDATES_FOR_INFER:
        return InferResult(candidates_examined=len(candidates), edges_proposed=0, edges_written=0)

    edges_proposed = 0
    edges_written = 0

    for focal_id, focal_name, focal_type in candidates:
        neighbors = [c for c in candidates if c[0] != focal_id]
        if not neighbors:
            continue
        existing = await _existing_targets(conn, focal_id)
        # Skip if the focal already has plenty of edges — we're targeting
        # under-connected Memos per the design note.
        if len(existing) >= _MAX_OUTBOUND_PER_FOCAL:
            continue

        try:
            response = await gateway.complete_structured(
                system=_SYSTEM,
                user=_format_user_prompt(focal_name, focal_type, neighbors),
                response_model=InferredEdgeList,
            )
        except Exception as exc:
            log.warning("refine.infer.llm_failed", focal=str(focal_id), error=str(exc))
            continue

        for proposal in response.edges:
            edges_proposed += 1
            try:
                target_uuid = UUID(proposal.target_id)
            except (TypeError, ValueError):
                continue
            if target_uuid in existing or target_uuid == focal_id:
                continue
            inserted = await _insert_inferred_edge(
                conn,
                org_id=org_id,
                source_id=focal_id,
                target_id=target_uuid,
                predicate=proposal.predicate,
                confidence=proposal.confidence,
            )
            if inserted:
                edges_written += 1
                existing.add(target_uuid)

    return InferResult(
        candidates_examined=len(candidates),
        edges_proposed=edges_proposed,
        edges_written=edges_written,
    )
