"""Unit tests for z3rno_core.engine.store — pure logic only (no DB).

Tests validation logic, helper functions, and dataclass construction.
"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

import pytest

from z3rno_core.engine.store import (
    DuplicateMemoryError,
    RelationshipInput,
    StoreError,
    StoreResult,
    _format_vector,
    _json_dumps,
)

# ---------------------------------------------------------------------------
# StoreError validation
# ---------------------------------------------------------------------------


class TestStoreError:
    """Test StoreError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(StoreError, Exception)

    def test_message(self) -> None:
        err = StoreError("content must be a non-empty string")
        assert str(err) == "content must be a non-empty string"

    def test_raises(self) -> None:
        with pytest.raises(StoreError, match="non-empty"):
            raise StoreError("content must be a non-empty string")


# ---------------------------------------------------------------------------
# _json_dumps helper
# ---------------------------------------------------------------------------


class TestJsonDumps:
    """Test the _json_dumps helper."""

    def test_empty_dict(self) -> None:
        assert _json_dumps({}) == "{}"

    def test_simple_dict(self) -> None:
        result = _json_dumps({"key": "value"})
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_nested_dict(self) -> None:
        data = {"outer": {"inner": [1, 2, 3]}}
        result = _json_dumps(data)
        parsed = json.loads(result)
        assert parsed == data

    def test_uuid_serialization(self) -> None:
        uid = uuid4()
        result = _json_dumps({"id": uid})
        parsed = json.loads(result)
        assert parsed["id"] == str(uid)

    def test_datetime_serialization(self) -> None:
        now = datetime.now()
        result = _json_dumps({"ts": now})
        parsed = json.loads(result)
        assert parsed["ts"] == str(now)

    def test_returns_string(self) -> None:
        assert isinstance(_json_dumps({"a": 1}), str)


# ---------------------------------------------------------------------------
# _format_vector helper
# ---------------------------------------------------------------------------


class TestFormatVector:
    """Test the _format_vector pgvector formatter."""

    def test_none_returns_none(self) -> None:
        assert _format_vector(None) is None

    def test_single_element(self) -> None:
        result = _format_vector([1.0])
        assert result == "[1.0]"

    def test_multiple_elements(self) -> None:
        result = _format_vector([0.1, 0.2, 0.3])
        assert result == "[0.1,0.2,0.3]"

    def test_empty_list(self) -> None:
        result = _format_vector([])
        assert result == "[]"

    def test_integers_become_strings(self) -> None:
        result = _format_vector([1, 2, 3])
        assert result == "[1,2,3]"

    def test_negative_values(self) -> None:
        result = _format_vector([-0.5, 0.5])
        assert result == "[-0.5,0.5]"


# ---------------------------------------------------------------------------
# StoreResult dataclass
# ---------------------------------------------------------------------------


class TestStoreResult:
    """Test StoreResult frozen dataclass."""

    def test_construction(self) -> None:
        mid = uuid4()
        now = datetime.now()
        result = StoreResult(
            memory_id=mid,
            embedding_model="text-embedding-3-small",
            importance_score=0.75,
            created_at=now,
        )
        assert result.memory_id == mid
        assert result.embedding_model == "text-embedding-3-small"
        assert result.importance_score == 0.75
        assert result.created_at == now

    def test_none_embedding_model(self) -> None:
        result = StoreResult(
            memory_id=uuid4(),
            embedding_model=None,
            importance_score=0.5,
            created_at=datetime.now(),
        )
        assert result.embedding_model is None

    def test_frozen(self) -> None:
        result = StoreResult(
            memory_id=uuid4(),
            embedding_model=None,
            importance_score=0.5,
            created_at=datetime.now(),
        )
        with pytest.raises(AttributeError):
            result.importance_score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RelationshipInput dataclass
# ---------------------------------------------------------------------------


class TestRelationshipInput:
    """Test RelationshipInput frozen dataclass."""

    def test_construction(self) -> None:
        target = uuid4()
        rel = RelationshipInput(
            target_memory_id=target,
            relationship_type="derived_from",
        )
        assert rel.target_memory_id == target
        assert rel.relationship_type == "derived_from"
        assert rel.weight == 1.0
        assert rel.metadata == {}

    def test_custom_weight(self) -> None:
        rel = RelationshipInput(
            target_memory_id=uuid4(),
            relationship_type="supports",
            weight=0.8,
        )
        assert rel.weight == 0.8

    def test_custom_metadata(self) -> None:
        meta = {"source": "extraction"}
        rel = RelationshipInput(
            target_memory_id=uuid4(),
            relationship_type="related_to",
            metadata=meta,
        )
        assert rel.metadata == meta

    def test_frozen(self) -> None:
        rel = RelationshipInput(
            target_memory_id=uuid4(),
            relationship_type="supports",
        )
        with pytest.raises(AttributeError):
            rel.weight = 0.5  # type: ignore[misc]

    def test_default_metadata_not_shared(self) -> None:
        """Each instance should get its own default dict."""
        r1 = RelationshipInput(target_memory_id=uuid4(), relationship_type="a")
        r2 = RelationshipInput(target_memory_id=uuid4(), relationship_type="b")
        assert r1.metadata is not r2.metadata or r1.metadata == {}


# ---------------------------------------------------------------------------
# DuplicateMemoryError
# ---------------------------------------------------------------------------


class TestDuplicateMemoryError:
    """Test DuplicateMemoryError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(DuplicateMemoryError, Exception)

    def test_message_contains_memory_id(self) -> None:
        mid = uuid4()
        err = DuplicateMemoryError(existing_memory_id=mid, similarity=0.98)
        assert str(mid) in str(err)

    def test_message_contains_similarity(self) -> None:
        err = DuplicateMemoryError(existing_memory_id=uuid4(), similarity=0.9612)
        assert "0.9612" in str(err)

    def test_attributes(self) -> None:
        mid = uuid4()
        err = DuplicateMemoryError(existing_memory_id=mid, similarity=0.97)
        assert err.existing_memory_id == mid
        assert err.similarity == 0.97

    def test_raises(self) -> None:
        mid = uuid4()
        with pytest.raises(DuplicateMemoryError, match="Near-duplicate"):
            raise DuplicateMemoryError(existing_memory_id=mid, similarity=0.99)
