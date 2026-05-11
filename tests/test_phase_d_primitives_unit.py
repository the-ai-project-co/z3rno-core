"""Unit tests for Phase D slice 1 — Pydantic graph primitives.

No database. Covers Memo / Edge validation, frozen-ness, defaults,
and that the Triplet re-export from z3rno_core.graph is the same
type Phase A's distill module ships.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from z3rno_core.distill.schemas import Triplet as DistillTriplet
from z3rno_core.graph import Edge, Memo, Triplet

# ---------------------------------------------------------------------------
# Triplet re-export
# ---------------------------------------------------------------------------


def test_triplet_reexport_is_distill_triplet() -> None:
    """Slice 1 reuses the existing Phase A Triplet — no fragmentation."""
    assert Triplet is DistillTriplet


# ---------------------------------------------------------------------------
# Memo
# ---------------------------------------------------------------------------


def test_memo_minimal_valid() -> None:
    memo = Memo(id=uuid4(), memo_type="PERSON", name="Ada Lovelace")
    assert memo.ontology_uri is None
    assert memo.version == 1
    assert memo.provenance == {}


def test_memo_with_ontology_and_provenance() -> None:
    memo = Memo(
        id=uuid4(),
        memo_type="PERSON",
        name="Ada Lovelace",
        ontology_uri="http://dbpedia.org/resource/Ada_Lovelace",
        version=3,
        provenance={"prompt_hash": "abc", "model": "openai/gpt-4o-mini"},
    )
    assert memo.ontology_uri == "http://dbpedia.org/resource/Ada_Lovelace"
    assert memo.version == 3
    assert memo.provenance["model"] == "openai/gpt-4o-mini"


def test_memo_is_frozen() -> None:
    memo = Memo(id=uuid4(), memo_type="PERSON", name="Ada")
    with pytest.raises(ValidationError):
        memo.name = "Grace"  # type: ignore[misc]


def test_memo_empty_name_rejected() -> None:
    with pytest.raises(ValidationError):
        Memo(id=uuid4(), memo_type="PERSON", name="")


def test_memo_empty_memo_type_rejected() -> None:
    with pytest.raises(ValidationError):
        Memo(id=uuid4(), memo_type="", name="Ada")


def test_memo_version_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Memo(id=uuid4(), memo_type="PERSON", name="Ada", version=0)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


def test_edge_minimal_valid() -> None:
    edge = Edge(
        edge_id="e1",
        src_id=uuid4(),
        dst_id=uuid4(),
        predicate="works_for",
    )
    assert edge.weight == 1.0
    assert edge.provenance == {}


def test_edge_weight_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        Edge(
            edge_id="e1",
            src_id=uuid4(),
            dst_id=uuid4(),
            predicate="works_for",
            weight=-0.1,
        )


def test_edge_is_frozen() -> None:
    edge = Edge(edge_id="e1", src_id=uuid4(), dst_id=uuid4(), predicate="p")
    with pytest.raises(ValidationError):
        edge.weight = 0.5  # type: ignore[misc]


def test_edge_empty_predicate_rejected() -> None:
    with pytest.raises(ValidationError):
        Edge(edge_id="e1", src_id=uuid4(), dst_id=uuid4(), predicate="")


def test_edge_empty_edge_id_rejected() -> None:
    with pytest.raises(ValidationError):
        Edge(edge_id="", src_id=uuid4(), dst_id=uuid4(), predicate="p")
