"""Unit tests for the dedupe stage (Phase D slice 3).

Pure-function tests for ``normalize_name`` and ``_group_rows``; live
DB behavior is covered by the integration test.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from z3rno_core.refine.dedupe import _group_rows, normalize_name


def test_normalize_name_lowercases() -> None:
    assert normalize_name("Ada Lovelace") == "ada lovelace"


def test_normalize_name_collapses_whitespace() -> None:
    assert normalize_name("Ada\t Lovelace\n") == "ada lovelace"


def test_normalize_name_strips_edges() -> None:
    assert normalize_name("   Ada   ") == "ada"


def _row(id_: UUID, type_: str | None, uri: str | None, content: str):
    return (id_, type_, uri, content)


def test_group_rows_groups_by_ontology_uri() -> None:
    a, b, c = uuid4(), uuid4(), uuid4()
    rows = [
        _row(a, "PERSON", "uri:Ada", "Ada Lovelace"),
        _row(b, "PERSON", "uri:Ada", "Ada"),
        _row(c, "PERSON", "uri:Grace", "Grace Hopper"),
    ]
    groups = _group_rows(rows)
    assert len(groups) == 1
    assert groups[0].primary_id == a
    assert groups[0].loser_ids == (b,)


def test_group_rows_groups_by_type_and_normalized_name() -> None:
    a, b, c = uuid4(), uuid4(), uuid4()
    rows = [
        _row(a, "PERSON", None, "Ada Lovelace"),
        _row(b, "PERSON", None, "ada  lovelace"),  # whitespace + case differ
        _row(c, "ORG", None, "Ada Lovelace"),  # different type → not a dup
    ]
    groups = _group_rows(rows)
    assert len(groups) == 1
    assert groups[0].primary_id == a
    assert groups[0].loser_ids == (b,)


def test_group_rows_ignores_singletons() -> None:
    rows = [_row(uuid4(), "PERSON", None, "Ada"), _row(uuid4(), "PERSON", None, "Grace")]
    assert _group_rows(rows) == []


def test_group_rows_skips_rows_with_no_signal() -> None:
    a, b = uuid4(), uuid4()
    rows = [_row(a, None, None, "anonymous"), _row(b, None, None, "anonymous")]
    assert _group_rows(rows) == []


def test_group_rows_uri_beats_type_signal() -> None:
    """When ontology_uri exists, we group on it even if memo_type differs."""
    a, b = uuid4(), uuid4()
    rows = [
        _row(a, "PERSON", "uri:Ada", "Ada"),
        _row(b, "ENTITY", "uri:Ada", "lovelace, ada"),
    ]
    groups = _group_rows(rows)
    assert len(groups) == 1
    assert {a, b} == {groups[0].primary_id, *groups[0].loser_ids}


def test_group_rows_primary_is_first_in_input_order() -> None:
    """Caller passes rows ordered by valid_from ASC, id ASC; primary = first."""
    a, b, c = uuid4(), uuid4(), uuid4()
    rows = [
        _row(a, "PERSON", "uri:X", "x"),
        _row(b, "PERSON", "uri:X", "x"),
        _row(c, "PERSON", "uri:X", "x"),
    ]
    groups = _group_rows(rows)
    assert groups[0].primary_id == a
    assert groups[0].loser_ids == (b, c)


def test_group_rows_meets_50pct_dedup_bar_on_synthetic_fixture() -> None:
    """Acceptance criterion #3: ≥50% duplicate reduction on a synthetic set.

    Build 100 Memos where 60 are dup pairs (30 groups of 2) and 40 are
    singletons. Expected: 30 losers superseded → 30/60 = 50% reduction
    of the dup population (50/100 = 50% of total rows touched).
    """
    rows = []
    for i in range(30):
        rows.append(_row(uuid4(), "PERSON", f"uri:E{i}", f"e{i}"))
        rows.append(_row(uuid4(), "PERSON", f"uri:E{i}", f"e{i}"))
    rows.extend(_row(uuid4(), "PERSON", f"uri:U{i}", f"u{i}") for i in range(40))

    groups = _group_rows(rows)
    losers = sum(len(g.loser_ids) for g in groups)
    assert losers == 30
    # 30 superseded out of 60 duplicates = 50%; out of 100 total = 30%.
    # The acceptance bar is "≥50% reduction of dup count" — verified.
    assert losers / 60 >= 0.5
