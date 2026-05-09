"""z3rno_core.forge — the Forge pipeline orchestrator (Phase A).

The **Forge** is Z3rno's end-to-end pipeline that turns raw input into a
structured, provenance-stamped knowledge graph. It runs three stages:

  1. **parse**    — normalize input (Phase A: text only; Phase B adds loaders)
  2. **distill**  — extract entities, relationships, summaries via LLM
  3. **retain**   — persist Memos + edges to Postgres / pgvector / Apache AGE
                    and stamp provenance into the audit chain

Modules
-------

- ``pipeline``    — ``ForgePipeline`` orchestrator with stage dispatch
- ``state``       — ``DistillJob`` lifecycle helpers (queued / running / completed / failed)
- ``idempotency`` — skip-if-already-distilled guard via ``entity_provenance`` lookup

The orchestrator is async, bounded by ``DISTILL_MAX_CONCURRENCY``, idempotent
per ``memory_id``, and fully RLS-aware (operates only within the caller's
``app.current_org_id`` context).
"""

from __future__ import annotations

from z3rno_core.forge.pipeline import ForgeOptions, ForgePipeline, ForgeRunSummary

__all__ = ["ForgeOptions", "ForgePipeline", "ForgeRunSummary"]
