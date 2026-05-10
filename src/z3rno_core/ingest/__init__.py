"""z3rno_core.ingest — the ingest pipeline (Phase B.1).

Bridges the new :mod:`z3rno_core.loaders` and :mod:`z3rno_core.storage`
packages with the existing Phase A engine and (optionally) the Forge
distillation pipeline.

The ``ingest()`` flow is:

  1. Persist the raw artifact via the configured ``StorageBackend`` and
     compute a stable ``source_uri``.
  2. Dispatch the right ``Loader`` for the input's MIME type.
  3. Convert the loaded text into one or more :class:`Memory` rows via
     the existing :func:`z3rno_core.engine.store.store`.
  4. If ``INGEST_AUTO_DISTILL`` is on, enqueue ``z3rno.forge_distill`` for
     the new memory IDs so the resulting graph appears without a second
     round-trip.

Modules
-------

- ``pipeline`` — :class:`IngestPipeline` orchestrator
- ``state``    — ``ingest_jobs`` row helpers (insert / update / lookup)
- ``schemas``  — :class:`IngestRunSummary` and related Pydantic types

The entire surface is dormant unless ``INGEST_ENABLED=true``.
"""

from __future__ import annotations

from z3rno_core.ingest.pipeline import IngestPipeline, PostIngestCallback
from z3rno_core.ingest.schemas import (
    IngestInput,
    IngestKind,
    IngestOptions,
    IngestRunSummary,
)
from z3rno_core.ingest.state import (
    find_memory_by_source_uri,
    insert_ingest_job,
    update_ingest_job,
)

__all__ = [
    "IngestInput",
    "IngestKind",
    "IngestOptions",
    "IngestPipeline",
    "IngestRunSummary",
    "PostIngestCallback",
    "find_memory_by_source_uri",
    "insert_ingest_job",
    "update_ingest_job",
]
