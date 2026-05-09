"""Unit tests for z3rno_core.distill.schemas.

Validates the Pydantic schemas the LLM Gateway hands back: Entity,
Relationship, Triplet, DistillResult — including the merge() dedupe.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from z3rno_core.distill.schemas import (
    DistillResult,
    Entity,
    Relationship,
    Triplet,
)


class TestEntity:
    def test_minimal(self) -> None:
        e = Entity(name="Z3rno")
        assert e.name == "Z3rno"
        assert e.type == "thing"  # default
        assert e.confidence == 1.0
        assert e.aliases == ()

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Entity(name="")

    def test_confidence_clamped(self) -> None:
        with pytest.raises(ValidationError):
            Entity(name="x", confidence=1.5)
        with pytest.raises(ValidationError):
            Entity(name="x", confidence=-0.1)

    def test_aliases_tuple(self) -> None:
        e = Entity(name="x", aliases=("a", "b"))
        assert e.aliases == ("a", "b")

    def test_frozen(self) -> None:
        e = Entity(name="x")
        with pytest.raises(ValidationError):
            e.name = "y"  # type: ignore[misc]


class TestRelationship:
    def test_minimal(self) -> None:
        r = Relationship(source="a", target="b", predicate="connects")
        assert r.confidence == 1.0
        assert r.description == ""

    def test_empty_endpoints_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Relationship(source="", target="b", predicate="x")
        with pytest.raises(ValidationError):
            Relationship(source="a", target="", predicate="x")
        with pytest.raises(ValidationError):
            Relationship(source="a", target="b", predicate="")


class TestTriplet:
    def test_minimal(self) -> None:
        t = Triplet(subject="s", predicate="p", obj="o")
        assert t.confidence == 1.0


class TestDistillResult:
    def test_empty_default(self) -> None:
        r = DistillResult()
        assert r.is_empty is True
        assert r.entities == ()
        assert r.relationships == ()
        assert r.triplets == ()

    def test_non_empty(self) -> None:
        r = DistillResult(
            entities=(Entity(name="x"),),
            summary="s",
        )
        assert r.is_empty is False

    def test_provenance_fields(self) -> None:
        mid = uuid4()
        r = DistillResult(
            entities=(Entity(name="x"),),
            source_memory_id=mid,
            chunk_index=2,
            char_start=10,
            char_end=20,
            model="m",
        )
        assert r.source_memory_id == mid
        assert r.chunk_index == 2
        assert r.char_start == 10
        assert r.model == "m"


class TestMerge:
    def test_merge_dedupes_entities_case_insensitively(self) -> None:
        a = DistillResult(entities=(Entity(name="Z3rno", type="Product"),))
        b = DistillResult(entities=(Entity(name="z3rno", type="product"),))
        m = a.merge(b)
        assert len(m.entities) == 1

    def test_merge_keeps_higher_confidence_entity(self) -> None:
        low = Entity(name="x", confidence=0.4)
        high = Entity(name="x", confidence=0.95)
        a = DistillResult(entities=(low,))
        b = DistillResult(entities=(high,))
        m = a.merge(b)
        assert m.entities[0].confidence == 0.95

    def test_merge_dedupes_relationships(self) -> None:
        r = Relationship(source="a", target="b", predicate="rel")
        a = DistillResult(relationships=(r,))
        b = DistillResult(relationships=(r,))
        m = a.merge(b)
        assert len(m.relationships) == 1

    def test_merge_dedupes_triplets_case_insensitively(self) -> None:
        a = DistillResult(triplets=(Triplet(subject="A", predicate="P", obj="O"),))
        b = DistillResult(triplets=(Triplet(subject="a", predicate="p", obj="o"),))
        m = a.merge(b)
        assert len(m.triplets) == 1

    def test_merge_preserves_summary_when_self_has_one(self) -> None:
        a = DistillResult(summary="A's summary")
        b = DistillResult(summary="B's summary")
        m = a.merge(b)
        assert m.summary == "A's summary"

    def test_merge_falls_back_to_other_summary(self) -> None:
        a = DistillResult()
        b = DistillResult(summary="B's summary")
        m = a.merge(b)
        assert m.summary == "B's summary"

    def test_merge_preserves_self_provenance(self) -> None:
        mid_a = uuid4()
        mid_b = uuid4()
        a = DistillResult(source_memory_id=mid_a, chunk_index=1, char_start=0, char_end=10)
        b = DistillResult(source_memory_id=mid_b, chunk_index=2, char_start=20, char_end=30)
        m = a.merge(b)
        assert m.source_memory_id == mid_a
        assert m.chunk_index == 1
        assert m.char_start == 0
        assert m.char_end == 10
