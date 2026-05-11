"""Graph writer — the *retain* stage of the Forge pipeline (Phase A).

Persists a :class:`DistillResult` atomically into the four stores Z3rno
already manages:

  1. **Postgres ``memories``** — every extracted Entity becomes a SEMANTIC
     Memo via the existing :func:`z3rno_core.engine.store.store` (which
     in turn writes the audit-log entry and hash-links the chain).
  2. **Apache AGE** — Memo nodes are mirrored as ``Memory`` vertices and
     each Relationship becomes an edge whose label is the LLM's
     predicate (e.g. ``WORKS_FOR``, ``COMPETES_WITH``). AGE writes are
     best-effort: a missing extension or graph schema is logged and
     does not abort the transaction.
  3. **``entity_provenance``** — one row per Memo links it back to the
     source memory_id, model, prompt hash, and chunk char-span.
  4. **``audit_log``** — handled transitively by ``store()``.

Idempotency is the orchestrator's responsibility (it checks
``entity_provenance`` before invoking us). The writer assumes its caller
already opened a transaction and called ``set_org_context`` for RLS.

Phase F will tighten this: ``DISTILL_PROVENANCE_REQUIRED`` will hard-fail
writes that lack a prompt_hash, audit_chain_id, or provenance row. Phase A
records all of these but does not yet enforce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import structlog
from sqlalchemy import text

from z3rno_core.engine.store import store
from z3rno_core.graph.sync import sync_memory_to_graph, sync_relationship_to_graph
from z3rno_core.models.enums import MemoryType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

    from z3rno_core.distill.schemas import DistillResult, Entity
    from z3rno_core.engine.embedding import EmbeddingProvider

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WriteResult:
    """Counters returned to the orchestrator after a successful retain pass."""

    memos_written: int
    edges_written: int
    provenance_written: int
    summary_memo_id: UUID | None = None


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def write_distill_result(
    conn: AsyncConnection,
    *,
    result: DistillResult,
    org_id: UUID,
    agent_id: UUID,
    source_memory_id: UUID,
    distill_job_id: UUID,
    prompt_hash: str = "",
    embedding_provider: EmbeddingProvider | None = None,
    api_key_id: UUID | None = None,
    request_id: str | None = None,
    ontology_resolver: Any = None,
) -> WriteResult:
    """Persist a :class:`DistillResult` into Postgres + AGE + provenance + audit.

    The caller MUST hold an open transaction with RLS context set
    (``set_org_context(conn, org_id)``). On any exception the surrounding
    transaction should roll back; this function does not commit.
    """
    name_to_id: dict[tuple[str, str], UUID] = {}
    memos_written = 0
    edges_written = 0
    provenance_written = 0

    # ----------------------------------------------------------------- entities
    for entity in result.entities:
        memo_id = await _write_entity(
            conn,
            entity=entity,
            org_id=org_id,
            agent_id=agent_id,
            source_memory_id=source_memory_id,
            distill_job_id=distill_job_id,
            embedding_provider=embedding_provider,
            api_key_id=api_key_id,
            request_id=request_id,
        )
        memos_written += 1
        name_to_id[(entity.name.lower(), entity.type.lower())] = memo_id

        # Phase D slice 4 — ontology grounding. Resolver is optional;
        # when None we leave memo_type / ontology_uri NULL and the Memo
        # behaves exactly as it did pre-Phase-D.
        if ontology_resolver is not None:
            await _apply_ontology_grounding(
                conn,
                memo_id=memo_id,
                entity_name=entity.name,
                entity_type=entity.type,
                resolver=ontology_resolver,
            )

        await _insert_provenance(
            conn,
            org_id=org_id,
            memo_id=memo_id,
            source_memory_id=source_memory_id,
            distill_job_id=distill_job_id,
            model=result.model,
            prompt_hash=prompt_hash,
            chunk_index=result.chunk_index,
            char_start=result.char_start,
            char_end=result.char_end,
        )
        provenance_written += 1

        await _safe_age_node(conn, memo_id, org_id, agent_id, entity.name)

    # ---------------------------------------------------------------- relationships
    for rel in result.relationships:
        src_id = _resolve_endpoint(name_to_id, rel.source)
        tgt_id = _resolve_endpoint(name_to_id, rel.target)
        if src_id is None or tgt_id is None:
            log.debug(
                "distill.graph_writer.unresolved_rel",
                source=rel.source,
                target=rel.target,
                predicate=rel.predicate,
            )
            continue
        wrote = await _safe_age_edge(
            conn,
            src_id,
            tgt_id,
            rel.predicate,
            rel.confidence,
        )
        if wrote:
            edges_written += 1

    # ---------------------------------------------------------------- summary
    summary_memo_id: UUID | None = None
    if result.summary and result.summary.strip():
        summary_memo_id = await _write_summary(
            conn,
            summary=result.summary,
            org_id=org_id,
            agent_id=agent_id,
            source_memory_id=source_memory_id,
            distill_job_id=distill_job_id,
            embedding_provider=embedding_provider,
            api_key_id=api_key_id,
            request_id=request_id,
        )
        memos_written += 1
        await _insert_provenance(
            conn,
            org_id=org_id,
            memo_id=summary_memo_id,
            source_memory_id=source_memory_id,
            distill_job_id=distill_job_id,
            model=result.model,
            prompt_hash=prompt_hash,
            chunk_index=result.chunk_index,
            char_start=result.char_start,
            char_end=result.char_end,
        )
        provenance_written += 1
        await _safe_age_node(conn, summary_memo_id, org_id, agent_id, "summary")

    return WriteResult(
        memos_written=memos_written,
        edges_written=edges_written,
        provenance_written=provenance_written,
        summary_memo_id=summary_memo_id,
    )


async def _apply_ontology_grounding(
    conn: AsyncConnection,
    *,
    memo_id: UUID,
    entity_name: str,
    entity_type: str,
    resolver: Any,
) -> None:
    """Resolve ``entity_name`` against the ontology and stamp memo_type + ontology_uri.

    Failures are logged and swallowed — grounding is best-effort, the
    Memo row already exists and the distill flow continues.
    """
    try:
        match = resolver.resolve(entity_name, type_hint=entity_type)
    except Exception as exc:
        log.warning(
            "distill.graph_writer.ontology_resolve_failed",
            memo_id=str(memo_id),
            entity=entity_name,
            error=str(exc),
        )
        return

    # Always set memo_type (even when resolver returned None) — it's the
    # cheap signal that lets Phase D dedupe group by (memo_type, name).
    memo_type = entity_type.upper() if entity_type else None
    ontology_uri = match.uri if match else None
    if memo_type is None and ontology_uri is None:
        return

    await conn.execute(
        text("""
            UPDATE public.memories
            SET memo_type = COALESCE(:memo_type, memo_type),
                ontology_uri = COALESCE(:ontology_uri, ontology_uri),
                updated_at = now()
            WHERE id = CAST(:memo_id AS uuid)
        """),
        {
            "memo_type": memo_type,
            "ontology_uri": ontology_uri,
            "memo_id": str(memo_id),
        },
    )


# ---------------------------------------------------------------------------
# Entity / summary Memo writers
# ---------------------------------------------------------------------------


async def _write_entity(
    conn: AsyncConnection,
    *,
    entity: Entity,
    org_id: UUID,
    agent_id: UUID,
    source_memory_id: UUID,
    distill_job_id: UUID,
    embedding_provider: EmbeddingProvider | None,
    api_key_id: UUID | None,
    request_id: str | None,
) -> UUID:
    content = _format_entity_content(entity)
    metadata = {
        "kind": "entity",
        "entity_name": entity.name,
        "entity_type": entity.type,
        "aliases": list(entity.aliases),
        "confidence": entity.confidence,
        "distill_job_id": str(distill_job_id),
        "source_memory_id": str(source_memory_id),
    }
    res = await store(
        conn,
        org_id=org_id,
        agent_id=agent_id,
        content=content,
        memory_type=MemoryType.SEMANTIC,
        embedding_provider=embedding_provider,
        metadata=metadata,
        importance=min(1.0, max(0.0, entity.confidence)),
        api_key_id=api_key_id,
        request_id=request_id,
    )
    return res.memory_id


async def _write_summary(
    conn: AsyncConnection,
    *,
    summary: str,
    org_id: UUID,
    agent_id: UUID,
    source_memory_id: UUID,
    distill_job_id: UUID,
    embedding_provider: EmbeddingProvider | None,
    api_key_id: UUID | None,
    request_id: str | None,
) -> UUID:
    metadata = {
        "kind": "summary",
        "distill_job_id": str(distill_job_id),
        "source_memory_id": str(source_memory_id),
    }
    res = await store(
        conn,
        org_id=org_id,
        agent_id=agent_id,
        content=summary.strip(),
        memory_type=MemoryType.SEMANTIC,
        embedding_provider=embedding_provider,
        metadata=metadata,
        api_key_id=api_key_id,
        request_id=request_id,
    )
    return res.memory_id


def _format_entity_content(entity: Entity) -> str:
    """Render an Entity into the text we store as the Memo's content field.

    Format chosen for two reasons:
      * downstream embeddings discriminate names from descriptions
      * recall surfaces the canonical name first
    """
    if entity.description:
        return f"{entity.name} ({entity.type}) — {entity.description}"
    return f"{entity.name} ({entity.type})"


# ---------------------------------------------------------------------------
# Provenance row
# ---------------------------------------------------------------------------


async def _insert_provenance(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    memo_id: UUID,
    source_memory_id: UUID,
    distill_job_id: UUID,
    model: str,
    prompt_hash: str,
    chunk_index: int | None,
    char_start: int | None,
    char_end: int | None,
) -> None:
    await conn.execute(
        text("""
            INSERT INTO entity_provenance (
                id, org_id, memo_id, source_memory_id, distill_job_id,
                model, prompt_hash, chunk_index, char_start, char_end,
                audit_chain_id, created_at
            ) VALUES (
                CAST(:id AS uuid),
                CAST(:org_id AS uuid),
                CAST(:memo_id AS uuid),
                CAST(:source_memory_id AS uuid),
                CAST(:distill_job_id AS uuid),
                :model, :prompt_hash, :chunk_index, :char_start, :char_end,
                NULL, now()
            )
        """),
        {
            "id": str(uuid4()),
            "org_id": str(org_id),
            "memo_id": str(memo_id),
            "source_memory_id": str(source_memory_id),
            "distill_job_id": str(distill_job_id),
            "model": model or "",
            "prompt_hash": prompt_hash or "",
            "chunk_index": chunk_index,
            "char_start": char_start,
            "char_end": char_end,
        },
    )


# ---------------------------------------------------------------------------
# AGE wrappers — best effort
# ---------------------------------------------------------------------------


async def _safe_age_node(
    conn: AsyncConnection,
    memo_id: UUID,
    org_id: UUID,
    agent_id: UUID,
    name: str,
) -> bool:
    """Mirror a Memo into the AGE graph. Returns False if AGE is unavailable.

    Wrapped in a savepoint so a failure (e.g. AGE extension not loaded
    on the testcontainer) doesn't abort the surrounding distill transaction.
    """
    try:
        async with conn.begin_nested():
            await conn.run_sync(
                lambda sync_conn: sync_memory_to_graph(
                    sync_conn,
                    memory_id=memo_id,
                    org_id=org_id,
                    agent_id=agent_id,
                    memory_type="semantic",
                    content_preview=name,
                )
            )
    except Exception as exc:
        log.warning(
            "distill.graph_writer.age_node_failed",
            memo_id=str(memo_id),
            error=str(exc),
        )
        return False
    return True


async def _safe_age_edge(
    conn: AsyncConnection,
    src_id: UUID,
    tgt_id: UUID,
    predicate: str,
    weight: float,
) -> bool:
    """Create an AGE edge labelled with the predicate. Returns False on failure.

    Wrapped in a savepoint so a failure (e.g. AGE extension not loaded)
    doesn't abort the surrounding distill transaction.
    """
    edge_label = _normalize_predicate(predicate)
    try:
        async with conn.begin_nested():
            await conn.run_sync(
                lambda sync_conn: sync_relationship_to_graph(
                    sync_conn,
                    source_id=src_id,
                    target_id=tgt_id,
                    relationship_type=edge_label,
                    weight=weight,
                )
            )
    except Exception as exc:
        log.warning(
            "distill.graph_writer.age_edge_failed",
            src=str(src_id),
            tgt=str(tgt_id),
            predicate=predicate,
            error=str(exc),
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_endpoint(
    name_to_id: dict[tuple[str, str], UUID],
    name: str,
) -> UUID | None:
    """Resolve an entity name to a Memo id created in this batch.

    First tries an exact ``(name, type)`` match across known types, then
    a name-only match. Returns ``None`` if the name was never written.
    """
    lname = name.lower()
    for (n, _t), mid in name_to_id.items():
        if n == lname:
            return mid
    return None


_PREDICATE_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")


def _normalize_predicate(predicate: str) -> str:
    """Reduce an LLM predicate to a safe AGE edge label.

    Lowercases, replaces spaces with underscores, and strips characters
    that aren't ``[a-z0-9_]``. AGE labels are unquoted Cypher identifiers
    so unsafe characters would break the query. Empty input falls back
    to ``related_to``.
    """
    cleaned = "".join(c if c in _PREDICATE_SAFE_CHARS else "_" for c in predicate.lower().strip())
    cleaned = cleaned.strip("_")
    return cleaned or "related_to"


# ---------------------------------------------------------------------------
# Distill-job state helpers
# ---------------------------------------------------------------------------


async def insert_distill_job(
    conn: AsyncConnection,
    *,
    job_id: UUID,
    org_id: UUID,
    agent_id: UUID,
    memory_ids: list[UUID],
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    max_concurrency: int,
) -> None:
    """Insert a fresh ``distill_jobs`` row in ``queued`` status."""
    # asyncpg expects a Python list for ARRAY columns; passing a
    # Postgres-literal-formatted string works only with psycopg.
    await conn.execute(
        text("""
            INSERT INTO distill_jobs (
                id, org_id, agent_id, memory_ids, status, model,
                chunk_size, chunk_overlap, max_concurrency,
                created_at, updated_at
            ) VALUES (
                CAST(:id AS uuid),
                CAST(:org_id AS uuid),
                CAST(:agent_id AS uuid),
                :memory_ids,
                'queued',
                :model, :chunk_size, :chunk_overlap, :max_concurrency,
                now(), now()
            )
        """),
        {
            "id": str(job_id),
            "org_id": str(org_id),
            "agent_id": str(agent_id),
            "memory_ids": [str(m) for m in memory_ids],
            "model": model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_concurrency": max_concurrency,
        },
    )


async def update_distill_job(
    conn: AsyncConnection,
    *,
    job_id: UUID,
    status: str | None = None,
    chunks_total: int | None = None,
    chunks_failed: int | None = None,
    entities_extracted: int | None = None,
    relationships_extracted: int | None = None,
    memos_written: int | None = None,
    error: str | None = None,
    started_at_now: bool = False,
    completed_at_now: bool = False,
) -> None:
    """Patch a ``distill_jobs`` row. Only non-None columns are updated."""
    sets: list[str] = ["updated_at = now()"]
    params: dict[str, object] = {"id": str(job_id)}

    def _add(col: str, val: object | None) -> None:
        if val is not None:
            sets.append(f"{col} = :{col}")
            params[col] = val

    _add("status", status)
    _add("chunks_total", chunks_total)
    _add("chunks_failed", chunks_failed)
    _add("entities_extracted", entities_extracted)
    _add("relationships_extracted", relationships_extracted)
    _add("memos_written", memos_written)
    _add("error", error)
    if started_at_now:
        sets.append("started_at = now()")
    if completed_at_now:
        sets.append("completed_at = now()")

    # Column names appended to `sets` are hardcoded literals from the _add()
    # calls above; only the values flow through bind parameters. Safe.
    sql = "UPDATE distill_jobs SET " + ", ".join(sets) + " WHERE id = CAST(:id AS uuid)"  # noqa: S608
    await conn.execute(text(sql), params)


async def already_distilled(
    conn: AsyncConnection,
    *,
    distill_job_id: UUID,
    source_memory_id: UUID,
) -> bool:
    """Return True if ``(distill_job_id, source_memory_id)`` already has provenance rows."""
    res = await conn.execute(
        text("""
            SELECT 1 FROM entity_provenance
            WHERE distill_job_id = CAST(:job AS uuid)
              AND source_memory_id = CAST(:src AS uuid)
            LIMIT 1
        """),
        {"job": str(distill_job_id), "src": str(source_memory_id)},
    )
    return res.fetchone() is not None
