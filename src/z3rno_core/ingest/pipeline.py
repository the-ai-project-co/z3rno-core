"""IngestPipeline — the Phase B.1 orchestrator.

Bridges the new :mod:`z3rno_core.loaders` and :mod:`z3rno_core.storage`
packages with the existing Phase A engine. The flow per job:

  1. **materialize** — turn the input into a tuple of
     ``(content_bytes, content_type, filename, source_uri)``. For
     ``url`` inputs we fetch via :func:`fetch_url`. For ``file`` inputs
     we persist the raw bytes through the configured
     :class:`StorageBackend` and return the canonical ``file://`` URI.
     For ``text`` inputs we encode to UTF-8 bytes; ``source_uri`` is
     left ``None`` (text re-ingestion is intentionally not idempotent).
  2. **dedupe** — when ``source_uri`` is meaningful (URL kind) and the
     ``(org_id, dataset_id, source_uri)`` triple already maps to a
     Memo, return that id and skip; never duplicate.
  3. **load** — dispatch the right :class:`Loader` for the bytes;
     produce a :class:`LoaderResult` (text + structural metadata).
  4. **store** — call the existing :func:`z3rno_core.engine.store.store`
     to create one Memo whose ``content`` is the loader's text and
     whose ``metadata`` carries the loader output, source URI, and
     dataset_id.
  5. **post_ingest** *(optional)* — if the caller passes a
     ``post_ingest`` callable (the server-side worker hooks
     :func:`forge_distill.delay` here) we invoke it with the run
     summary; the callable returns a ``distill_job_id`` we record on
     the ingest_jobs row.

The orchestrator owns transaction boundaries: bootstrap, the per-input
write, and the final-state update each get their own transaction so
partial progress survives crashes. RLS is set at the start of every
transaction via :func:`set_org_context`.

Phase B.1 scope
---------------
* One Memo per ingest. Multi-document expansion (e.g. one CSV → N rows
  → N Memos) is intentionally out of scope; chunking happens later in
  the Forge.
* Auto-distill is "fire and trust" — we record the distill_job_id but
  don't poll it.
* Multimodal (image/audio) lands in Phase B.2.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import structlog
from sqlalchemy import text as sa_text

from z3rno_core.engine.store import store
from z3rno_core.ingest.schemas import (
    IngestInput,
    IngestOptions,
    IngestRunSummary,
)
from z3rno_core.ingest.state import (
    find_memory_by_source_uri,
    insert_ingest_job,
    update_ingest_job,
)
from z3rno_core.loaders.url import fetch_url
from z3rno_core.models.enums import MemoryType
from z3rno_core.security.rls import set_org_context

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

    from z3rno_core.engine.embedding import EmbeddingProvider
    from z3rno_core.loaders.registry import LoaderRegistry
    from z3rno_core.storage.base import StorageBackend

log = structlog.get_logger(__name__)


PostIngestCallback = Callable[["IngestRunSummary"], Awaitable[UUID | None]]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestPipeline:
    """End-to-end ingest orchestrator. Construction does no I/O."""

    def __init__(
        self,
        *,
        registry: LoaderRegistry,
        storage: StorageBackend,
        embedding_provider: EmbeddingProvider | None = None,
        url_fetch_max_bytes: int = 50 * 1024 * 1024,
        url_fetch_timeout_seconds: float = 15.0,
        url_allowed_schemes: tuple[str, ...] = ("http", "https"),
        url_playwright_enabled: bool = False,
        url_playwright_min_chars: int = 200,
        url_playwright_timeout_seconds: float = 30.0,
    ) -> None:
        self._registry = registry
        self._storage = storage
        self._embedding_provider = embedding_provider
        self._url_fetch_max_bytes = url_fetch_max_bytes
        self._url_fetch_timeout_seconds = url_fetch_timeout_seconds
        self._url_allowed_schemes = url_allowed_schemes
        self._url_playwright_enabled = url_playwright_enabled
        self._url_playwright_min_chars = url_playwright_min_chars
        self._url_playwright_timeout_seconds = url_playwright_timeout_seconds

    # ---- public entry ----------------------------------------------------

    async def run(  # noqa: PLR0915 — orchestrator stages are linear by design
        self,
        engine: AsyncEngine,
        *,
        org_id: UUID,
        agent_id: UUID,
        ingest_input: IngestInput,
        dataset_id: UUID | None = None,
        job_id: UUID | None = None,
        options: IngestOptions | None = None,
        post_ingest: PostIngestCallback | None = None,
        api_key_id: UUID | None = None,
        request_id: str | None = None,
    ) -> IngestRunSummary:
        """Execute one ingest end-to-end and return the summary."""
        job_id = job_id or uuid4()
        opts = options or IngestOptions()
        summary = IngestRunSummary(job_id=job_id, status="failed")

        _validate_input(ingest_input)

        # 1. Bootstrap row + mark running.
        try:
            async with engine.begin() as conn:
                await self._set_rls(conn, org_id)
                if not await self._job_exists(conn, job_id):
                    await insert_ingest_job(
                        conn,
                        job_id=job_id,
                        org_id=org_id,
                        agent_id=agent_id,
                        kind=ingest_input.kind,
                        dataset_id=dataset_id,
                        source_uri=None,  # populated after materialize
                        content_type=ingest_input.content_type,
                        filename=ingest_input.filename,
                    )
                await update_ingest_job(conn, job_id=job_id, status="running", started_at_now=True)
            summary.started_at = datetime.now().astimezone()
        except Exception as exc:
            log.exception("ingest.run.bootstrap_failed", job_id=str(job_id))
            summary.error = str(exc)
            summary.completed_at = datetime.now().astimezone()
            await self._mark_failed(engine, org_id, job_id, str(exc))
            return summary

        # 2-5. Materialize → dedupe → load → store → optional post_ingest.
        try:
            materialized = await self._materialize(ingest_input, org_id=org_id)
            summary.source_uri = materialized.source_uri
            summary.content_type = materialized.content_type
            summary.filename = materialized.filename
            summary.file_size = len(materialized.content)

            # Dedupe — only when source_uri is stable across runs.
            if materialized.source_uri is not None and ingest_input.kind == "url":
                async with engine.begin() as conn:
                    await self._set_rls(conn, org_id)
                    existing = await find_memory_by_source_uri(
                        conn,
                        org_id=org_id,
                        source_uri=materialized.source_uri,
                        dataset_id=dataset_id,
                    )
                if existing is not None:
                    summary.memory_ids = [existing]
                    summary.skipped_existing = [existing]
                    summary.status = "completed"
                    summary.completed_at = datetime.now().astimezone()
                    await self._mark_completed(engine, org_id, summary)
                    return summary

            # Load → text + metadata.
            loader = self._registry.get_loader(
                materialized.content,
                mime_type=materialized.content_type,
                filename=materialized.filename,
            )
            loader_result = await loader.load(
                materialized.content,
                filename=materialized.filename,
                mime_type=materialized.content_type,
            )

            # Surface non-fatal loader anomalies as structured warnings so
            # the operator polling /v1/ingest/{job_id} sees them. Today:
            # CSV row-cap truncation. New loaders can add more codes
            # without changing this list — anything the loader emits as
            # a recognised "warn"-shaped metadata key gets propagated.
            if loader_result.metadata.get("truncated"):
                summary.warnings.append(
                    {
                        "code": "csv_truncated",
                        "detail": (
                            f"row cap hit at {loader_result.metadata.get('row_count', '?')} "
                            "rows; raise INGEST_MAX_CSV_ROWS to ingest more"
                        ),
                    }
                )

            # Store — one Memo per ingest in B.1.
            metadata = {
                "ingest_kind": ingest_input.kind,
                "source_uri": materialized.source_uri,
                "dataset_id": str(dataset_id) if dataset_id else None,
                "ingest_job_id": str(job_id),
                "loader": loader.name,
                **loader_result.metadata,
            }
            async with engine.begin() as conn:
                await self._set_rls(conn, org_id)
                store_res = await store(
                    conn,
                    org_id=org_id,
                    agent_id=agent_id,
                    content=loader_result.text or "(empty)",
                    memory_type=MemoryType.EPISODIC,
                    embedding_provider=self._embedding_provider,
                    metadata=metadata,
                    dataset_id=dataset_id,
                    api_key_id=api_key_id,
                    request_id=request_id,
                )
            summary.memory_ids = [store_res.memory_id]

            # Phase D slice 5 — optional codegraph extraction. Runs after
            # the text Memo is stored so the call-graph Memos share the
            # ingest_job lineage. Failures are non-fatal — the text Memo
            # is already persisted.
            if opts.codegraph_enabled:
                lang = loader_result.metadata.get("language")
                if isinstance(lang, str) and lang.lower() in ("python", "typescript"):
                    try:
                        from z3rno_core.codegraph import (  # noqa: PLC0415
                            extract,
                            parse_source,
                            write_extraction,
                        )

                        parsed = parse_source(loader_result.text, language=lang.lower())
                        module_name = (
                            materialized.filename or materialized.source_uri or f"ingest-{job_id}"
                        )
                        extraction = extract(parsed, module_name=str(module_name))
                        async with engine.begin() as conn:
                            await self._set_rls(conn, org_id)
                            cg_result = await write_extraction(
                                conn,
                                org_id=org_id,
                                agent_id=agent_id,
                                extraction=extraction,
                                dataset_id=dataset_id,
                            )
                        summary.codegraph_memos_written = cg_result.memos_written
                        summary.codegraph_edges_written = cg_result.edges_written
                    except Exception as cg_exc:
                        log.warning(
                            "ingest.run.codegraph_failed",
                            job_id=str(job_id),
                            error=str(cg_exc),
                        )
                        summary.warnings.append({"code": "codegraph_failed", "detail": str(cg_exc)})
        except Exception as exc:
            log.exception("ingest.run.failed", job_id=str(job_id))
            summary.error = str(exc)
            summary.completed_at = datetime.now().astimezone()
            await self._mark_failed(engine, org_id, job_id, str(exc), summary=summary)
            return summary

        # 6. Optional post-ingest hook (e.g. enqueue forge_distill).
        if opts.auto_distill and post_ingest is not None:
            try:
                summary.distill_job_id = await post_ingest(summary)
            except Exception:
                log.exception("ingest.run.post_ingest_failed", job_id=str(job_id))
                # Non-fatal: the ingest itself succeeded, distill can be retried.

        # 7. Mark completed.
        summary.status = "completed"
        summary.completed_at = datetime.now().astimezone()
        await self._mark_completed(engine, org_id, summary)
        return summary

    # ---- materialize -----------------------------------------------------

    @dataclass(frozen=True)
    class _Materialized:
        content: bytes
        content_type: str
        filename: str | None
        source_uri: str | None

    async def _materialize(
        self,
        inp: IngestInput,
        *,
        org_id: UUID,
    ) -> _Materialized:
        """Resolve the input into bytes + content_type + source_uri."""
        if inp.kind == "text":
            assert inp.text is not None
            return self._Materialized(
                content=inp.text.encode("utf-8"),
                content_type=inp.content_type or "text/plain",
                filename=inp.filename,
                source_uri=None,  # text is not idempotent
            )

        if inp.kind == "url":
            assert inp.url is not None
            fetched = await fetch_url(
                inp.url,
                allowed_schemes=self._url_allowed_schemes,
                timeout_seconds=self._url_fetch_timeout_seconds,
                max_bytes=self._url_fetch_max_bytes,
                playwright_enabled=self._url_playwright_enabled,
                playwright_min_chars=self._url_playwright_min_chars,
                playwright_timeout_seconds=self._url_playwright_timeout_seconds,
            )
            return self._Materialized(
                content=fetched.content,
                content_type=fetched.content_type,
                filename=inp.filename,
                # Use the canonical post-redirect URL as the dedupe key.
                source_uri=fetched.url,
            )

        if inp.kind == "s3_uri":
            # Phase B.2.1 direct-to-S3 flow: the artifact is already at
            # ``source_uri`` (the client uploaded out-of-band). Read
            # bytes via the storage backend and route them through the
            # loader registry like a normal file.
            assert inp.source_uri is not None
            content = await self._storage.read_artifact(inp.source_uri)
            return self._Materialized(
                content=content,
                content_type=inp.content_type or "application/octet-stream",
                filename=inp.filename,
                source_uri=inp.source_uri,
            )

        # kind == "file" — persist via storage backend; the storage URI
        # becomes the source_uri.
        assert inp.content is not None
        content_type = inp.content_type or "application/octet-stream"
        source_uri = await self._storage.store_artifact(
            org_id=org_id,
            content=inp.content,
            content_type=content_type,
            filename=inp.filename,
        )
        return self._Materialized(
            content=inp.content,
            content_type=content_type,
            filename=inp.filename,
            source_uri=source_uri,
        )

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    async def _set_rls(conn: AsyncConnection, org_id: UUID) -> None:
        await conn.run_sync(lambda sync_conn: set_org_context(sync_conn, org_id))

    @staticmethod
    async def _job_exists(conn: AsyncConnection, job_id: UUID) -> bool:
        res = await conn.execute(
            sa_text("SELECT 1 FROM ingest_jobs WHERE id = CAST(:id AS uuid) LIMIT 1"),
            {"id": str(job_id)},
        )
        return res.fetchone() is not None

    async def _mark_failed(
        self,
        engine: AsyncEngine,
        org_id: UUID,
        job_id: UUID,
        error: str,
        summary: IngestRunSummary | None = None,
    ) -> None:
        try:
            async with engine.begin() as conn:
                await self._set_rls(conn, org_id)
                await update_ingest_job(
                    conn,
                    job_id=job_id,
                    status="failed",
                    error=error[:2000],
                    source_uri=summary.source_uri if summary else None,
                    content_type=summary.content_type if summary else None,
                    filename=summary.filename if summary else None,
                    file_size=summary.file_size if summary else None,
                    memory_ids=summary.memory_ids if summary else None,
                    completed_at_now=True,
                )
        except Exception:  # pragma: no cover — last-resort, just don't crash
            log.exception("ingest.mark_failed.update_error", job_id=str(job_id))

    async def _mark_completed(
        self,
        engine: AsyncEngine,
        org_id: UUID,
        summary: IngestRunSummary,
    ) -> None:
        async with engine.begin() as conn:
            await self._set_rls(conn, org_id)
            await update_ingest_job(
                conn,
                job_id=summary.job_id,
                status="completed",
                source_uri=summary.source_uri,
                content_type=summary.content_type,
                filename=summary.filename,
                file_size=summary.file_size,
                memory_ids=summary.memory_ids,
                memos_written=len(summary.memory_ids),
                distill_job_id=summary.distill_job_id,
                warnings=summary.warnings if summary.warnings else None,
                completed_at_now=True,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_REQUIRED_FIELD_BY_KIND: dict[str, str] = {
    "text": "text",
    "url": "url",
    "file": "content",
    "s3_uri": "source_uri",
}


def _validate_input(inp: IngestInput) -> None:
    """Reject malformed :class:`IngestInput` early with a clear error."""
    required = _REQUIRED_FIELD_BY_KIND.get(inp.kind)
    if required is None:
        raise ValueError(f"unknown ingest kind: {inp.kind!r}")
    value = getattr(inp, required)
    if value is None or (isinstance(value, str) and not value):
        raise ValueError(f"ingest_input.kind={inp.kind!r} requires {required}=...")
    forbidden = [f for f in ("text", "url", "content", "source_uri") if f != required]
    extras = [f for f in forbidden if getattr(inp, f) is not None]
    if extras:
        raise ValueError(f"ingest_input.kind={inp.kind!r} must not set {', '.join(extras)}")
