"""Unit tests for Phase F slice 1 — provenance enforcement.

Covers the pure helpers (build_provenance_blob, ProvenanceRequiredError)
and the verify_chain validator with a mocked DB. Live integration
against a real postgres lives in the integration suite.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from z3rno_core.distill.provenance import (
    ChainVerdict,
    ProvenanceRequiredError,
    build_provenance_blob,
    verify_chain,
)

# ---------------------------------------------------------------------------
# build_provenance_blob
# ---------------------------------------------------------------------------


def test_build_blob_has_all_canonical_fields() -> None:
    src = uuid4()
    job = uuid4()
    blob = build_provenance_blob(
        source_memory_id=src,
        model="openai/gpt-4o-mini",
        prompt_hash="deadbeef",
        distill_job_id=job,
        chunk_index=2,
        char_start=10,
        char_end=120,
    )
    assert blob["source_memory_id"] == str(src)
    assert blob["distill_job_id"] == str(job)
    assert blob["model"] == "openai/gpt-4o-mini"
    assert blob["prompt_hash"] == "deadbeef"
    assert blob["chunk_index"] == 2
    assert blob["char_start"] == 10
    assert blob["char_end"] == 120
    # correlation_id auto-generated when not supplied
    assert "correlation_id" in blob
    UUID(blob["correlation_id"])  # parses
    assert "ts" in blob


def test_build_blob_respects_supplied_correlation_id() -> None:
    fixed = uuid4()
    blob = build_provenance_blob(
        source_memory_id=uuid4(),
        model="x",
        prompt_hash="y",
        distill_job_id=uuid4(),
        correlation_id=fixed,
    )
    assert blob["correlation_id"] == str(fixed)


# ---------------------------------------------------------------------------
# ProvenanceRequiredError
# ---------------------------------------------------------------------------


def test_provenance_error_message_carries_memo_id() -> None:
    mid = uuid4()
    err = ProvenanceRequiredError(f"failed to stamp provenance for memo {mid}")
    assert str(mid) in str(err)


# ---------------------------------------------------------------------------
# verify_chain
# ---------------------------------------------------------------------------


class _ChainTestFixture:
    """Helper to build a MagicMock AsyncConnection scripted with a
    sequence of fetchone() return values."""

    def __init__(self, *return_values: object) -> None:
        self._values = list(return_values)
        self.conn = MagicMock()

        async def _execute(*args: object, **kwargs: object) -> MagicMock:
            v = self._values.pop(0) if self._values else None
            result = MagicMock()
            result.fetchone = lambda: v
            return result

        self.conn.execute = AsyncMock(side_effect=_execute)


@pytest.mark.asyncio
async def test_verify_chain_returns_invalid_when_memo_missing() -> None:
    fx = _ChainTestFixture(None)
    verdict = await verify_chain(fx.conn, memo_id=uuid4())
    assert verdict.is_broken
    assert "not found" in verdict.reason


@pytest.mark.asyncio
async def test_verify_chain_returns_invalid_when_provenance_null() -> None:
    # Memo row exists but distill_provenance is NULL.
    fx = _ChainTestFixture((None,))
    verdict = await verify_chain(fx.conn, memo_id=uuid4())
    assert verdict.is_broken
    assert "NULL" in verdict.reason


@pytest.mark.asyncio
async def test_verify_chain_returns_invalid_when_correlation_id_missing() -> None:
    blob_without_corr = {"source_memory_id": "x", "model": "m"}
    fx = _ChainTestFixture((blob_without_corr,))
    verdict = await verify_chain(fx.conn, memo_id=uuid4())
    assert verdict.is_broken
    assert "correlation_id" in verdict.reason


@pytest.mark.asyncio
async def test_verify_chain_invalid_when_entity_provenance_missing() -> None:
    blob = {"correlation_id": "abc"}
    # Memo row → blob; entity_provenance lookup → None.
    fx = _ChainTestFixture((blob,), None)
    verdict = await verify_chain(fx.conn, memo_id=uuid4())
    assert verdict.is_broken
    assert "entity_provenance" in verdict.reason


@pytest.mark.asyncio
async def test_verify_chain_valid_when_audit_row_chained() -> None:
    blob = {"correlation_id": "abc"}
    # Memo blob → ep row → audit_log row.
    fx = _ChainTestFixture((blob,), (1,), (1,))
    verdict = await verify_chain(fx.conn, memo_id=uuid4())
    assert verdict.is_valid


@pytest.mark.asyncio
async def test_verify_chain_valid_when_audit_row_only_pending() -> None:
    blob = {"correlation_id": "abc"}
    # Memo blob → ep row → audit_log MISSING → audit_log_pending row.
    fx = _ChainTestFixture((blob,), (1,), None, (1,))
    verdict = await verify_chain(fx.conn, memo_id=uuid4())
    assert verdict.is_valid


@pytest.mark.asyncio
async def test_verify_chain_invalid_when_no_audit_anywhere() -> None:
    blob = {"correlation_id": "abc"}
    fx = _ChainTestFixture((blob,), (1,), None, None)
    verdict = await verify_chain(fx.conn, memo_id=uuid4())
    assert verdict.is_broken
    assert "audit" in verdict.reason


# ---------------------------------------------------------------------------
# ChainVerdict ergonomics
# ---------------------------------------------------------------------------


def test_chain_verdict_is_broken_when_invalid() -> None:
    v = ChainVerdict(memo_id=uuid4(), is_valid=False, reason="x")
    assert v.is_broken is True


def test_chain_verdict_is_not_broken_when_valid() -> None:
    v = ChainVerdict(memo_id=uuid4(), is_valid=True)
    assert v.is_broken is False
