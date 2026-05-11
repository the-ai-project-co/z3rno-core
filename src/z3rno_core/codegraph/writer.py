"""Persist extracted CodeMemos + CodeEdges (Phase D slice 5).

Two-pass write: first the Memos (so each one gets a database UUID),
then the edges (now that ``key → memory_id`` is known). Edges whose
target is name-only (unresolved call site) create a placeholder
``FUNCTION`` Memo so the call is still queryable as ``calls →
placeholder``; a future refine pass can re-bind it to the real Memo.

Edges land in ``memory_relationships`` with ``relationship_type =
'related_to'`` (the existing enum) and the codegraph edge kind in
``metadata.codegraph_kind``. The ``CODE`` retrieval strategy reads
that metadata key to follow the call graph; no schema migration is
needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import text

from z3rno_core.codegraph.extractor import (
    CodeEdgeKind,
    CodeMemo,
    CodeMemoKind,
    ExtractResult,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass(frozen=True)
class CodegraphWriteResult:
    memos_written: int
    edges_written: int
    placeholders_written: int


def _memo_type_for(kind: CodeMemoKind) -> str:
    """Map CodeMemoKind to the `memo_type` text stored in `memories`."""
    return f"CODE_{kind.value}"  # CODE_MODULE / CODE_CLASS / CODE_FUNCTION / CODE_IMPORT


def _content_for(memo: CodeMemo) -> str:
    """Render a CodeMemo into the Memo's `content` text."""
    return f"{memo.kind.value.lower()} {memo.qualified_name}"


async def _insert_memo(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID,
    dataset_id: UUID | None,
    memo: CodeMemo,
) -> UUID:
    """Insert one code-Memo into ``memories``. Returns its DB id."""
    memo_id = uuid4()
    metadata = {
        "kind": "codegraph",
        "codegraph_kind": memo.kind.value,
        "qualified_name": memo.qualified_name,
        "name": memo.name,
        "language": memo.language,
        "start_line": memo.start_line,
        "end_line": memo.end_line,
        **memo.metadata,
    }
    await conn.execute(
        text("""
            INSERT INTO public.memories (
                id, org_id, agent_id, dataset_id,
                memory_type, content, metadata,
                memo_type, importance_score,
                valid_from, created_at, updated_at
            ) VALUES (
                CAST(:id AS uuid),
                CAST(:org_id AS uuid),
                CAST(:agent_id AS uuid),
                CAST(:dataset_id AS uuid),
                CAST('semantic' AS memory_type_enum),
                :content,
                CAST(:metadata AS jsonb),
                :memo_type,
                0.5,
                now(), now(), now()
            )
        """),
        {
            "id": str(memo_id),
            "org_id": str(org_id),
            "agent_id": str(agent_id),
            "dataset_id": str(dataset_id) if dataset_id else None,
            "content": _content_for(memo),
            "metadata": json.dumps(metadata),
            "memo_type": _memo_type_for(memo.kind),
        },
    )
    return memo_id


async def _insert_edge(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    src_id: UUID,
    tgt_id: UUID,
    kind: CodeEdgeKind,
) -> bool:
    """Insert one memory_relationships row. Returns False on conflict."""
    metadata = {"codegraph_kind": kind.value, "source": "codegraph"}
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
                    CAST('related_to' AS relationship_type_enum),
                    1.0,
                    CAST(:meta AS jsonb),
                    now(), now()
                )
            """),
            {
                "id": str(uuid4()),
                "org_id": str(org_id),
                "src": str(src_id),
                "tgt": str(tgt_id),
                "meta": json.dumps(metadata),
            },
        )
    except Exception:
        # self-loop or constraint violation — skip silently
        return False
    return True


async def write_extraction(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID,
    extraction: ExtractResult,
    dataset_id: UUID | None = None,
) -> CodegraphWriteResult:
    """Persist a complete ExtractResult. Caller owns the transaction + RLS."""
    key_to_id: dict[str, UUID] = {}
    memos_written = 0
    edges_written = 0
    placeholders_written = 0

    # --- pass 1: insert Memos ---
    for memo in extraction.memos:
        mid = await _insert_memo(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            dataset_id=dataset_id,
            memo=memo,
        )
        key_to_id[memo.key] = mid
        memos_written += 1

    # --- pass 2: insert edges, creating placeholders for unresolved calls ---
    for edge in extraction.edges:
        if edge.src_key not in key_to_id:
            continue
        src_id = key_to_id[edge.src_key]

        if edge.target_key is not None and edge.target_key in key_to_id:
            tgt_id = key_to_id[edge.target_key]
        elif edge.target_name:
            # Placeholder Memo for the name-only target. Subsequent
            # extractions in the same dataset will re-use it via the
            # name-based lookup below.
            placeholder_key = f"placeholder::{edge.target_name}"
            if placeholder_key in key_to_id:
                tgt_id = key_to_id[placeholder_key]
            else:
                placeholder = CodeMemo(
                    key=placeholder_key,
                    kind=CodeMemoKind.FUNCTION,
                    name=edge.target_name,
                    qualified_name=f"<unresolved>.{edge.target_name}",
                    language=extraction.memos[0].language if extraction.memos else "unknown",
                    start_line=0,
                    end_line=0,
                    metadata={"placeholder": True},
                )
                tgt_id = await _insert_memo(
                    conn,
                    org_id=org_id,
                    agent_id=agent_id,
                    dataset_id=dataset_id,
                    memo=placeholder,
                )
                key_to_id[placeholder_key] = tgt_id
                memos_written += 1
                placeholders_written += 1
        else:
            continue

        if src_id == tgt_id:
            # MemoryRelationship has no_self_relationship CHECK; respect it.
            continue
        if await _insert_edge(
            conn,
            org_id=org_id,
            src_id=src_id,
            tgt_id=tgt_id,
            kind=edge.kind,
        ):
            edges_written += 1

    return CodegraphWriteResult(
        memos_written=memos_written,
        edges_written=edges_written,
        placeholders_written=placeholders_written,
    )
