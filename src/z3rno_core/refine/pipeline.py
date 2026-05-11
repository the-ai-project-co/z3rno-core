"""``RefinePipeline`` — Phase D slice 3 orchestrator.

Stages, in order:

  1. **dedupe** — merge Memos sharing ``ontology_uri`` or
     ``(memo_type, normalized_name)``. SCD-2 supersede losers.
  2. **reweight** — drain ``feedback``, EMA-update
     ``memory_relationships.weight``.
  3. **prune** — drop relationships with weight below threshold.

Slice 4 will add ``infer`` (LLM proposes missing edges) and
``summarize`` (per-subgraph summaries) stages between dedupe and
reweight.

Lifecycle
---------
The pipeline owns the ``refine_jobs`` row: marks it running on entry,
updates counters as each stage completes, transitions to completed /
failed on exit. The caller is responsible for setting the RLS context
on the connection before calling :meth:`run`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import structlog

from z3rno_core.refine.dedupe import DedupeResult, run_dedupe
from z3rno_core.refine.infer import InferResult, run_infer
from z3rno_core.refine.prune import PruneResult, run_prune
from z3rno_core.refine.reweight import ReweightResult, run_reweight
from z3rno_core.refine.state import insert_refine_job, update_refine_job
from z3rno_core.refine.summarize import SummarizeResult, run_summarize

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

    from z3rno_core.distill.llm_gateway import LLMGateway

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RefineOptions:
    """Tuning for one refine run."""

    feedback_weight_decay: float = 0.95
    prune_threshold: float = 0.05
    trigger: str = "api"  # 'api' | 'beat'
    # Phase D slice 4 — LLM-driven stages. Each requires a gateway AND
    # the corresponding flag. False ⇒ stage is a no-op.
    infer_enabled: bool = False
    summarize_enabled: bool = False
    infer_max_candidates: int = 50


@dataclass
class RefineRunSummary:
    """What one ``RefinePipeline.run()`` accomplished."""

    job_id: UUID
    status: str = "queued"
    memos_scanned: int = 0
    memos_deduped: int = 0
    edges_reweighted: int = 0
    edges_pruned: int = 0
    feedback_drained: int = 0
    error: str | None = None
    dedupe: DedupeResult | None = field(default=None, repr=False)
    reweight: ReweightResult | None = field(default=None, repr=False)
    prune: PruneResult | None = field(default=None, repr=False)
    infer: InferResult | None = field(default=None, repr=False)
    summarize: SummarizeResult | None = field(default=None, repr=False)


class RefinePipeline:
    """Orchestrate one end-to-end refine pass."""

    def __init__(
        self,
        options: RefineOptions | None = None,
        *,
        gateway: LLMGateway | None = None,
    ) -> None:
        self.options = options or RefineOptions()
        # Phase D slice 4 — optional LLM gateway shared by infer +
        # summarize stages. None ⇒ both stages no-op even if their
        # flags are set.
        self._gateway = gateway

    async def run(
        self,
        conn: AsyncConnection,
        *,
        org_id: UUID,
        dataset_id: UUID | None = None,
        job_id: UUID | None = None,
    ) -> RefineRunSummary:
        """Execute dedupe → reweight → prune. Returns the summary.

        Creates a ``refine_jobs`` row on entry (unless ``job_id`` is
        supplied, in which case the row is expected to already exist —
        the Celery task path pre-inserts so the row is visible to
        ``GET /v1/refine/{job_id}`` immediately).
        """
        rj_id = job_id or uuid4()
        summary = RefineRunSummary(job_id=rj_id)

        if job_id is None:
            await insert_refine_job(
                conn,
                job_id=rj_id,
                org_id=org_id,
                dataset_id=dataset_id,
                trigger=self.options.trigger,
                status="running",
            )
        await update_refine_job(conn, job_id=rj_id, status="running", started_at_now=True)

        try:
            # --- 1. dedupe ---
            dedupe_result = await run_dedupe(conn, org_id=org_id, dataset_id=dataset_id)
            summary.dedupe = dedupe_result
            summary.memos_scanned = dedupe_result.memos_scanned
            summary.memos_deduped = dedupe_result.memos_deduped
            await update_refine_job(
                conn,
                job_id=rj_id,
                memos_scanned=summary.memos_scanned,
                memos_deduped=summary.memos_deduped,
            )

            # --- 1b. infer (opt-in, LLM-required) ---
            stage_meta: dict[str, object] = {}
            if self.options.infer_enabled and self._gateway is not None:
                infer_result = await run_infer(
                    conn,
                    org_id=org_id,
                    gateway=self._gateway,
                    dataset_id=dataset_id,
                    max_candidates=self.options.infer_max_candidates,
                )
                summary.infer = infer_result
                stage_meta["infer"] = {
                    "candidates_examined": infer_result.candidates_examined,
                    "edges_proposed": infer_result.edges_proposed,
                    "edges_written": infer_result.edges_written,
                }

            # --- 1c. summarize (opt-in, LLM-required) ---
            if self.options.summarize_enabled and self._gateway is not None:
                summarize_result = await run_summarize(
                    conn,
                    org_id=org_id,
                    gateway=self._gateway,
                    dataset_id=dataset_id,
                )
                summary.summarize = summarize_result
                stage_meta["summarize"] = {
                    "clusters_examined": summarize_result.clusters_examined,
                    "summaries_written": summarize_result.summaries_written,
                    "summaries_skipped_cached": summarize_result.summaries_skipped_cached,
                }

            if stage_meta:
                await update_refine_job(conn, job_id=rj_id, job_metadata=stage_meta)

            # --- 2. reweight ---
            reweight_result = await run_reweight(
                conn, org_id=org_id, decay=self.options.feedback_weight_decay
            )
            summary.reweight = reweight_result
            summary.edges_reweighted = reweight_result.edges_reweighted
            summary.feedback_drained = reweight_result.feedback_drained
            await update_refine_job(
                conn,
                job_id=rj_id,
                edges_reweighted=summary.edges_reweighted,
                feedback_drained=summary.feedback_drained,
            )

            # --- 3. prune ---
            prune_result = await run_prune(
                conn, org_id=org_id, threshold=self.options.prune_threshold
            )
            summary.prune = prune_result
            summary.edges_pruned = prune_result.edges_pruned
            await update_refine_job(conn, job_id=rj_id, edges_pruned=summary.edges_pruned)

            summary.status = "completed"
            await update_refine_job(conn, job_id=rj_id, status="completed", completed_at_now=True)
        except Exception as exc:
            log.exception("refine.pipeline.failed", job_id=str(rj_id))
            summary.status = "failed"
            summary.error = str(exc)
            await update_refine_job(
                conn,
                job_id=rj_id,
                status="failed",
                error=str(exc),
                completed_at_now=True,
            )
            raise

        return summary
