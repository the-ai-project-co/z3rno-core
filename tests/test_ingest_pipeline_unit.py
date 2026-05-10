"""Unit tests for z3rno_core.ingest pure helpers + schemas (Phase B.1).

End-to-end DB-backed coverage lives in the integration suite (Task 37).
These tests exercise the parts that don't need a connection so the fast
unit suite still proves the boundary logic of the orchestrator.
"""

from __future__ import annotations

import tempfile
from uuid import uuid4

import pytest

from z3rno_core.ingest import (
    IngestInput,
    IngestOptions,
    IngestPipeline,
    IngestRunSummary,
)
from z3rno_core.ingest.pipeline import _validate_input
from z3rno_core.loaders import get_default_registry
from z3rno_core.storage import LocalStorageBackend

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TestIngestOptions:
    def test_defaults(self) -> None:
        o = IngestOptions()
        assert o.auto_distill is True
        assert o.chunk_size == 1024
        assert o.chunk_overlap == 128
        assert o.summary_style == "concise"


class TestIngestRunSummary:
    def test_list_default_factory_isolates_instances(self) -> None:
        s1 = IngestRunSummary(job_id=uuid4(), status="completed")
        s2 = IngestRunSummary(job_id=uuid4(), status="completed")
        s1.memory_ids.append(uuid4())
        assert len(s1.memory_ids) == 1
        assert len(s2.memory_ids) == 0

    def test_default_state(self) -> None:
        s = IngestRunSummary(job_id=uuid4(), status="failed")
        assert s.memory_ids == []
        assert s.skipped_existing == []
        assert s.source_uri is None
        assert s.distill_job_id is None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_valid_text(self) -> None:
        _validate_input(IngestInput(kind="text", text="hello"))

    def test_valid_url(self) -> None:
        _validate_input(IngestInput(kind="url", url="https://example.com"))

    def test_valid_file(self) -> None:
        _validate_input(IngestInput(kind="file", content=b"bytes"))

    def test_text_missing_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires text"):
            _validate_input(IngestInput(kind="text"))

    def test_text_with_extra_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not set"):
            _validate_input(IngestInput(kind="text", text="x", url="http://x"))

    def test_url_missing_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires url"):
            _validate_input(IngestInput(kind="url"))

    def test_url_with_extra_content_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not set"):
            _validate_input(IngestInput(kind="url", url="http://x", content=b"x"))

    def test_file_missing_content_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires content"):
            _validate_input(IngestInput(kind="file"))

    def test_file_with_extra_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not set"):
            _validate_input(IngestInput(kind="file", content=b"x", text="y"))

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown ingest kind"):
            _validate_input(IngestInput(kind="bogus"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


class TestIngestPipelineConstruction:
    def test_construction_is_io_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = IngestPipeline(
                registry=get_default_registry(),
                storage=LocalStorageBackend(tmp),
            )
            assert pipeline is not None

    def test_url_settings_passed_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = IngestPipeline(
                registry=get_default_registry(),
                storage=LocalStorageBackend(tmp),
                url_fetch_max_bytes=1024,
                url_fetch_timeout_seconds=2.5,
                url_allowed_schemes=("https",),
            )
            assert pipeline._url_fetch_max_bytes == 1024
            assert pipeline._url_fetch_timeout_seconds == 2.5
            assert pipeline._url_allowed_schemes == ("https",)
