"""z3rno_core.refine — Phase D graph-improvement pipeline.

Slice 2 shipped feedback ingestion. Slice 3 added the orchestrator and
the dedupe → reweight → prune stages. Slice 4 plugs in optional
LLM-driven infer + summarize stages between dedupe and reweight.
"""

from __future__ import annotations

from z3rno_core.refine.dedupe import DedupeGroup, DedupeResult, run_dedupe
from z3rno_core.refine.feedback import record_feedback
from z3rno_core.refine.infer import InferResult, run_infer
from z3rno_core.refine.pipeline import RefineOptions, RefinePipeline, RefineRunSummary
from z3rno_core.refine.prune import PruneResult, run_prune
from z3rno_core.refine.reweight import ReweightResult, run_reweight
from z3rno_core.refine.summarize import SummarizeResult, run_summarize

__all__ = [
    "DedupeGroup",
    "DedupeResult",
    "InferResult",
    "PruneResult",
    "RefineOptions",
    "RefinePipeline",
    "RefineRunSummary",
    "ReweightResult",
    "SummarizeResult",
    "record_feedback",
    "run_dedupe",
    "run_infer",
    "run_prune",
    "run_reweight",
    "run_summarize",
]
