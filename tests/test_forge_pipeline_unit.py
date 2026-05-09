"""Unit tests for pure helpers and construction of z3rno_core.forge.pipeline.

End-to-end run() coverage is in the integration suite (Task 15) where a
real Postgres + AGE testcontainer is available.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from z3rno_core.distill.llm_gateway import StubLLMGateway
from z3rno_core.forge import ForgeOptions, ForgePipeline, ForgeRunSummary
from z3rno_core.forge.pipeline import _hash_prompts


class TestForgeOptions:
    def test_defaults(self) -> None:
        o = ForgeOptions()
        assert o.chunk_size == 1024
        assert o.chunk_overlap == 128
        assert o.max_concurrency == 4
        assert o.summary_style == "concise"
        assert o.include_summary is True

    def test_frozen(self) -> None:
        o = ForgeOptions()
        with pytest.raises(Exception):  # noqa: B017, PT011
            o.chunk_size = 99  # type: ignore[misc]


class TestForgeRunSummary:
    def test_list_default_factory_isolates_instances(self) -> None:
        s1 = ForgeRunSummary(job_id=uuid4(), status="completed")
        s2 = ForgeRunSummary(job_id=uuid4(), status="completed")
        s1.skipped_memory_ids.append(uuid4())
        assert len(s1.skipped_memory_ids) == 1
        assert len(s2.skipped_memory_ids) == 0

    def test_default_counters_zero(self) -> None:
        s = ForgeRunSummary(job_id=uuid4(), status="completed")
        assert s.memories_processed == 0
        assert s.memos_written == 0
        assert s.error is None
        assert s.started_at is None


class TestForgePipelineConstruction:
    def test_construct_with_default_options(self) -> None:
        gw = StubLLMGateway(model="m")
        p = ForgePipeline(gateway=gw)
        assert p.options.chunk_size == 1024
        assert p.options.include_summary is True

    def test_construct_with_custom_options(self) -> None:
        gw = StubLLMGateway(model="m")
        opts = ForgeOptions(chunk_size=256, max_concurrency=2, include_summary=False)
        p = ForgePipeline(gateway=gw, options=opts)
        assert p.options.chunk_size == 256
        assert p.options.include_summary is False


class TestHashPrompts:
    def test_deterministic(self) -> None:
        assert _hash_prompts("a", "b") == _hash_prompts("a", "b")

    def test_sensitive_to_changes(self) -> None:
        assert _hash_prompts("a", "b") != _hash_prompts("a ", "b")
        assert _hash_prompts("a", "b") != _hash_prompts("a", "b ")

    def test_64_char_hex(self) -> None:
        h = _hash_prompts("sys", "user")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
