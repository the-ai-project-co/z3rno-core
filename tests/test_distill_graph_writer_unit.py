"""Unit tests for the pure helpers of z3rno_core.distill.graph_writer.

Full DB-backed integration coverage lives in the integration suite —
these tests exercise the helpers that don't need a connection so the
fast unit suite still proves the boundary logic of the writer.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from z3rno_core.distill.graph_writer import (
    WriteResult,
    _format_entity_content,
    _normalize_predicate,
    _resolve_endpoint,
)
from z3rno_core.distill.schemas import Entity


class TestFormatEntityContent:
    def test_without_description(self) -> None:
        e = Entity(name="Z3rno", type="product")
        assert _format_entity_content(e) == "Z3rno (product)"

    def test_with_description(self) -> None:
        e = Entity(name="Z3rno", type="product", description="smart memory")
        assert _format_entity_content(e) == "Z3rno (product) — smart memory"


class TestNormalizePredicate:
    def test_simple_lowercase(self) -> None:
        assert _normalize_predicate("OWNS") == "owns"

    def test_spaces_become_underscores(self) -> None:
        assert _normalize_predicate("Works For") == "works_for"

    def test_unsafe_chars_stripped(self) -> None:
        assert _normalize_predicate("competes-with!") == "competes_with"

    def test_empty_falls_back_to_related_to(self) -> None:
        assert _normalize_predicate("") == "related_to"

    def test_only_unsafe_falls_back_to_related_to(self) -> None:
        assert _normalize_predicate("!!!") == "related_to"
        assert _normalize_predicate("   ___   ") == "related_to"

    def test_unicode_collapsed_to_underscores(self) -> None:
        assert _normalize_predicate("foo→bar") == "foo_bar"


class TestResolveEndpoint:
    def test_exact_lowercase_match(self) -> None:
        m = uuid4()
        table = {("z3rno", "product"): m}
        assert _resolve_endpoint(table, "Z3rno") == m

    def test_case_insensitive_match(self) -> None:
        m = uuid4()
        table = {("acme corp", "org"): m}
        assert _resolve_endpoint(table, "ACME CORP") == m

    def test_unknown_returns_none(self) -> None:
        m = uuid4()
        table = {("z3rno", "product"): m}
        assert _resolve_endpoint(table, "cognee") is None

    def test_empty_table(self) -> None:
        assert _resolve_endpoint({}, "anything") is None


class TestWriteResult:
    def test_default_summary_memo_id_is_none(self) -> None:
        r = WriteResult(memos_written=1, edges_written=0, provenance_written=1)
        assert r.summary_memo_id is None

    def test_frozen(self) -> None:
        r = WriteResult(memos_written=1, edges_written=0, provenance_written=1)
        with pytest.raises(Exception):  # noqa: B017, PT011
            r.memos_written = 99  # type: ignore[misc]
