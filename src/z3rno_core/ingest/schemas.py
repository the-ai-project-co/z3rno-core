"""Schemas for the ingest pipeline (Phase B.1).

Three input shapes feed the Forge:

  * **text** — a raw text body (already a string)
  * **url**  — an HTTP(S) URL we'll fetch, normalize, and load
  * **file** — bytes uploaded by the client, with a content_type + filename hint

These are encoded as :class:`IngestInput` so downstream worker / API code
flow them around as a single typed value. :class:`IngestOptions` carries
per-job tuning, :class:`IngestRunSummary` is the orchestrator's return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID

IngestKind = Literal["text", "url", "file"]


@dataclass(frozen=True)
class IngestInput:
    """One of the three input shapes the IngestPipeline accepts.

    Exactly one of ``text`` / ``url`` / ``content`` must be populated,
    matching the ``kind`` discriminator. The :class:`IngestPipeline`
    validates this on entry.
    """

    kind: IngestKind
    text: str | None = None
    url: str | None = None
    content: bytes | None = None
    filename: str | None = None
    content_type: str | None = None


@dataclass(frozen=True)
class IngestOptions:
    """Per-job tuning."""

    auto_distill: bool = True
    chunk_size: int = 1024
    chunk_overlap: int = 128
    summary_style: Literal["concise", "bullet", "abstractive"] = "concise"


@dataclass
class IngestRunSummary:
    """Outcome of one :meth:`IngestPipeline.run`.

    Mirrors the shape of ``ingest_jobs`` so the server-side API can
    return this directly (or fetch it from the row) without an
    extra translation layer.
    """

    job_id: UUID
    status: str  # "completed" | "failed"
    memory_ids: list[UUID] = field(default_factory=list)
    # If the dedupe check found a matching prior Memo, its id is also
    # listed here for caller visibility.
    skipped_existing: list[UUID] = field(default_factory=list)
    source_uri: str | None = None
    content_type: str | None = None
    filename: str | None = None
    file_size: int | None = None
    distill_job_id: UUID | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
