"""ForgePipeline — the Phase A orchestrator.

Wires the Phase A modules into the canonical **parse → distill → retain**
flow over a list of source ``memory_ids``:

  parse    — load source content, run :func:`chunk_by_tokens`
  distill  — run :func:`extract_from_chunks` (LLM), optional summary
  retain   — :func:`write_distill_result` into Postgres + AGE + audit chain

The orchestrator owns transaction boundaries. Each source memory is
processed in its own transaction so partial progress survives crashes
and ``distill_jobs`` row counters update incrementally. RLS is set at
the start of every transaction.

Idempotency
-----------
:func:`already_distilled` is consulted before each per-memory pass. A
re-run of the same job over the same memory is a no-op — safe under
Celery retries and worker restarts.

Concurrency
-----------
The chunk-level fan-out inside one memory is bounded by
``options.max_concurrency`` (passed through to :func:`extract_from_chunks`).
Per-memory iteration is sequential — keeps audit-log ordering tidy and
avoids surprising lock contention against ``distill_jobs``.

Phase A scope
-------------
Multimodal loaders and dataset-level orchestration land in Phase B.
Refinement / dedupe / feedback land in Phase D. Distributed sharding
across worker pools lands in Phase F. The shape of this pipeline does
not change in those phases — they extend, not rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import structlog
from sqlalchemy import text

from z3rno_core.chunking import chunk_by_tokens
from z3rno_core.distill.extract import build_extraction_prompts, extract_from_chunks
from z3rno_core.distill.graph_writer import (
    already_distilled,
    insert_distill_job,
    update_distill_job,
    write_distill_result,
)
from z3rno_core.distill.summarize import rolling_summarize
from z3rno_core.security.rls import set_org_context

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

    from z3rno_core.distill.llm_gateway import LLMGateway
    from z3rno_core.distill.summarize import SummaryStyle
    from z3rno_core.engine.embedding import EmbeddingProvider

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Options + run summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgeOptions:
    """Tuning knobs for a single :meth:`ForgePipeline.run` invocation.

    Defaults match the Phase A settings in ``z3rno_server.config.Settings``
    so the API endpoint can pass them through unchanged.
    """

    chunk_size: int = 1024
    chunk_overlap: int = 128
    max_concurrency: int = 4
    summary_style: SummaryStyle = "concise"
    include_summary: bool = True
    temperature: float = 0.0


@dataclass
class ForgeRunSummary:
    """Outcome of one :meth:`ForgePipeline.run` invocation."""

    job_id: UUID
    status: str  # "completed" | "failed"
    memories_processed: int = 0
    memories_skipped: int = 0
    chunks_total: int = 0
    chunks_failed: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    memos_written: int = 0
    error: str | None = None
    skipped_memory_ids: list[UUID] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ForgePipeline:
    """End-to-end Forge orchestrator.

    Construction is cheap (no I/O). The engine, gateway, and embedding
    provider are injected so the pipeline is testable with stubs.
    """

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        embedding_provider: EmbeddingProvider | None = None,
        options: ForgeOptions | None = None,
    ) -> None:
        self._gateway = gateway
        self._embedding_provider = embedding_provider
        self._options = options or ForgeOptions()

    @property
    def options(self) -> ForgeOptions:
        return self._options

    # --- main entry --------------------------------------------------------

    async def run(
        self,
        engine: AsyncEngine,
        *,
        org_id: UUID,
        agent_id: UUID,
        memory_ids: list[UUID],
        job_id: UUID | None = None,
        api_key_id: UUID | None = None,
        request_id: str | None = None,
    ) -> ForgeRunSummary:
        """Execute the Forge over ``memory_ids`` and return a summary.

        If ``job_id`` is provided the caller has already inserted the
        ``distill_jobs`` row (typical for the API path: the request
        handler inserts the row then enqueues a Celery task that calls
        run() with the existing job_id). When ``job_id`` is ``None`` the
        pipeline creates the row itself (useful for direct invocation
        from tests or scripts).
        """
        job_id = job_id or uuid4()
        summary = ForgeRunSummary(job_id=job_id, status="failed")

        # ---- ensure job row exists, mark running ------------------------
        try:
            async with engine.begin() as conn:
                await self._set_rls(conn, org_id)
                if not await self._job_exists(conn, job_id):
                    await insert_distill_job(
                        conn,
                        job_id=job_id,
                        org_id=org_id,
                        agent_id=agent_id,
                        memory_ids=memory_ids,
                        model=self._gateway.model_name,
                        chunk_size=self._options.chunk_size,
                        chunk_overlap=self._options.chunk_overlap,
                        max_concurrency=self._options.max_concurrency,
                    )
                await update_distill_job(
                    conn,
                    job_id=job_id,
                    status="running",
                    started_at_now=True,
                )
            summary.started_at = datetime.now().astimezone()
        except Exception as exc:
            log.exception("forge.run.bootstrap_failed", job_id=str(job_id))
            await self._mark_failed(engine, org_id, job_id, str(exc))
            summary.error = str(exc)
            summary.completed_at = datetime.now().astimezone()
            return summary

        # ---- per-memory loop -------------------------------------------
        try:
            for mid in memory_ids:
                outcome = await self._process_memory(
                    engine,
                    org_id=org_id,
                    agent_id=agent_id,
                    memory_id=mid,
                    job_id=job_id,
                    api_key_id=api_key_id,
                    request_id=request_id,
                )
                if outcome.skipped:
                    summary.memories_skipped += 1
                    summary.skipped_memory_ids.append(mid)
                    continue
                summary.memories_processed += 1
                summary.chunks_total += outcome.chunks_total
                summary.chunks_failed += outcome.chunks_failed
                summary.entities_extracted += outcome.entities_extracted
                summary.relationships_extracted += outcome.relationships_extracted
                summary.memos_written += outcome.memos_written
        except Exception as exc:
            log.exception("forge.run.failed", job_id=str(job_id))
            summary.error = str(exc)
            summary.completed_at = datetime.now().astimezone()
            await self._mark_failed(engine, org_id, job_id, str(exc), counters=summary)
            return summary

        # ---- mark completed ---------------------------------------------
        summary.completed_at = datetime.now().astimezone()
        summary.status = "completed"
        try:
            async with engine.begin() as conn:
                await self._set_rls(conn, org_id)
                await update_distill_job(
                    conn,
                    job_id=job_id,
                    status="completed",
                    chunks_total=summary.chunks_total,
                    chunks_failed=summary.chunks_failed,
                    entities_extracted=summary.entities_extracted,
                    relationships_extracted=summary.relationships_extracted,
                    memos_written=summary.memos_written,
                    completed_at_now=True,
                )
        except Exception as exc:  # pragma: no cover — final-state update is best-effort
            log.exception("forge.run.final_update_failed", job_id=str(job_id))
            summary.error = str(exc)
            summary.status = "failed"
        return summary

    # --- per-memory --------------------------------------------------------

    @dataclass
    class _MemoryOutcome:
        skipped: bool = False
        chunks_total: int = 0
        chunks_failed: int = 0
        entities_extracted: int = 0
        relationships_extracted: int = 0
        memos_written: int = 0

    async def _process_memory(
        self,
        engine: AsyncEngine,
        *,
        org_id: UUID,
        agent_id: UUID,
        memory_id: UUID,
        job_id: UUID,
        api_key_id: UUID | None,
        request_id: str | None,
    ) -> _MemoryOutcome:
        async with engine.begin() as conn:
            await self._set_rls(conn, org_id)

            if await already_distilled(conn, distill_job_id=job_id, source_memory_id=memory_id):
                log.info(
                    "forge.process_memory.skipped_already_distilled",
                    memory_id=str(memory_id),
                    job_id=str(job_id),
                )
                return self._MemoryOutcome(skipped=True)

            content = await self._load_memory_content(conn, memory_id)
            if content is None or not content.strip():
                log.info(
                    "forge.process_memory.empty_or_missing",
                    memory_id=str(memory_id),
                )
                return self._MemoryOutcome(skipped=True)

            # ---- parse ----
            chunks = chunk_by_tokens(
                content,
                chunk_size=self._options.chunk_size,
                overlap=self._options.chunk_overlap,
            )
            if not chunks:
                return self._MemoryOutcome(skipped=True)

            # ---- distill: extract ----
            result = await extract_from_chunks(
                chunks,
                gateway=self._gateway,
                source_memory_id=memory_id,
                max_concurrency=self._options.max_concurrency,
                temperature=self._options.temperature,
            )

            # ---- distill: optional summary ----
            # include_summary=False is a hard skip — even if the LLM
            # already produced a summary as part of extraction we drop
            # it so no summary Memo is written downstream.
            if not self._options.include_summary:
                if result.summary:
                    result = result.model_copy(update={"summary": ""})
            elif not result.summary:
                try:
                    s = await rolling_summarize(
                        chunks,
                        gateway=self._gateway,
                        style=self._options.summary_style,
                        temperature=self._options.temperature,
                    )
                except Exception:
                    log.warning("forge.process_memory.summary_failed", memory_id=str(memory_id))
                    s = ""
                if s:
                    result = result.model_copy(update={"summary": s})

            # ---- retain ----
            prompt_hash = _hash_prompts(*build_extraction_prompts("X"))
            write = await write_distill_result(
                conn,
                result=result,
                org_id=org_id,
                agent_id=agent_id,
                source_memory_id=memory_id,
                distill_job_id=job_id,
                prompt_hash=prompt_hash,
                embedding_provider=self._embedding_provider,
                api_key_id=api_key_id,
                request_id=request_id,
            )

            return self._MemoryOutcome(
                skipped=False,
                chunks_total=len(chunks),
                chunks_failed=0,  # extract_from_chunks absorbs failures silently in v1
                entities_extracted=len(result.entities),
                relationships_extracted=len(result.relationships),
                memos_written=write.memos_written,
            )

    # --- helpers -----------------------------------------------------------

    @staticmethod
    async def _set_rls(conn: AsyncConnection, org_id: UUID) -> None:
        await conn.run_sync(lambda sync_conn: set_org_context(sync_conn, org_id))

    @staticmethod
    async def _job_exists(conn: AsyncConnection, job_id: UUID) -> bool:
        res = await conn.execute(
            text("SELECT 1 FROM distill_jobs WHERE id = CAST(:id AS uuid) LIMIT 1"),
            {"id": str(job_id)},
        )
        return res.fetchone() is not None

    @staticmethod
    async def _load_memory_content(conn: AsyncConnection, memory_id: UUID) -> str | None:
        res = await conn.execute(
            text("""
                SELECT content
                FROM memories
                WHERE id = CAST(:mid AS uuid)
                  AND deleted_at IS NULL
                  AND valid_to IS NULL
            """),
            {"mid": str(memory_id)},
        )
        row = res.fetchone()
        return row[0] if row else None

    async def _mark_failed(
        self,
        engine: AsyncEngine,
        org_id: UUID,
        job_id: UUID,
        error: str,
        counters: ForgeRunSummary | None = None,
    ) -> None:
        try:
            async with engine.begin() as conn:
                await self._set_rls(conn, org_id)
                await update_distill_job(
                    conn,
                    job_id=job_id,
                    status="failed",
                    error=error[:2000],  # cap; full error stays in logs
                    chunks_total=counters.chunks_total if counters else None,
                    chunks_failed=counters.chunks_failed if counters else None,
                    entities_extracted=counters.entities_extracted if counters else None,
                    relationships_extracted=(
                        counters.relationships_extracted if counters else None
                    ),
                    memos_written=counters.memos_written if counters else None,
                    completed_at_now=True,
                )
        except Exception:  # pragma: no cover — last-resort, just don't crash
            log.exception("forge.mark_failed.update_error", job_id=str(job_id))


# ---------------------------------------------------------------------------
# Helpers exposed for tests / audits
# ---------------------------------------------------------------------------


def _hash_prompts(system: str, user: str) -> str:
    """SHA-256 over the canonical extraction prompts.

    Stamped into ``entity_provenance.prompt_hash`` for every Memo the
    Forge writes. Phase F will require this hash to validate audit
    chains.
    """
    h = sha256()
    h.update(b"system\x00")
    h.update(system.encode("utf-8"))
    h.update(b"\x00user\x00")
    h.update(user.encode("utf-8"))
    return h.hexdigest()
