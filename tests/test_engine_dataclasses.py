"""Unit tests for all engine dataclass/result types and error classes.

Tests construction, field defaults, and frozen behaviour for:
- RecallResult, RecallError (recall.py)
- ForgetResult, ForgetError (forget.py)
- AuditEntry, AuditPage, ActivitySummary (audit_query.py)
- SweepResult, DecayResult, TransitionResult (lifecycle.py)
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from z3rno_core.engine.audit_query import ActivitySummary, AuditEntry, AuditPage
from z3rno_core.engine.forget import ForgetError, ForgetResult
from z3rno_core.engine.lifecycle import DecayResult, SweepResult, TransitionResult
from z3rno_core.engine.recall import RecallError, RecallResult, _fallback_query

# ---------------------------------------------------------------------------
# RecallResult
# ---------------------------------------------------------------------------


class TestRecallResult:
    """Test RecallResult frozen dataclass."""

    def _make(self, **overrides: object) -> RecallResult:
        defaults = {
            "memory_id": uuid4(),
            "content": "test content",
            "summary": None,
            "memory_type": "episodic",
            "similarity_score": 0.85,
            "importance_score": 0.6,
            "relevance_score": 0.75,
            "recall_count": 3,
            "created_at": datetime.now(tz=UTC),
            "valid_from": datetime.now(tz=UTC),
            "metadata": {},
        }
        defaults.update(overrides)
        return RecallResult(**defaults)  # type: ignore[arg-type]

    def test_construction(self) -> None:
        r = self._make()
        assert r.content == "test content"
        assert r.graph_context == []

    def test_graph_context_default(self) -> None:
        r = self._make()
        assert r.graph_context == []

    def test_graph_context_custom(self) -> None:
        ctx = [{"rel": "derived_from", "id": "abc"}]
        r = self._make(graph_context=ctx)
        assert r.graph_context == ctx

    def test_frozen(self) -> None:
        r = self._make()
        with pytest.raises(AttributeError):
            r.content = "modified"  # type: ignore[misc]

    def test_summary_nullable(self) -> None:
        r = self._make(summary="a summary")
        assert r.summary == "a summary"

    def test_metadata_dict(self) -> None:
        r = self._make(metadata={"tag": "important"})
        assert r.metadata == {"tag": "important"}


class TestRecallError:
    def test_is_exception(self) -> None:
        assert issubclass(RecallError, Exception)

    def test_message(self) -> None:
        err = RecallError("recall failed")
        assert str(err) == "recall failed"


class TestFallbackQuery:
    """Test _fallback_query SQL generation."""

    def test_returns_string(self) -> None:
        result = _fallback_query("org_id = CAST(:org_id AS uuid)")
        assert isinstance(result, str)

    def test_contains_where_clause(self) -> None:
        clause = "org_id = CAST(:org_id AS uuid) AND agent_id = CAST(:agent_id AS uuid)"
        result = _fallback_query(clause)
        assert clause in result

    def test_orders_by_importance_then_recency(self) -> None:
        result = _fallback_query("1=1")
        assert "ORDER BY importance_score DESC, created_at DESC" in result

    def test_has_limit_placeholder(self) -> None:
        result = _fallback_query("1=1")
        assert ":top_k" in result

    def test_selects_null_similarity(self) -> None:
        result = _fallback_query("1=1")
        assert "NULL AS similarity" in result

    def test_selects_from_memories(self) -> None:
        result = _fallback_query("1=1")
        assert "FROM memories" in result


# ---------------------------------------------------------------------------
# ForgetResult / ForgetError
# ---------------------------------------------------------------------------


class TestForgetResult:
    """Test ForgetResult frozen dataclass."""

    def test_construction(self) -> None:
        ids = [uuid4(), uuid4()]
        r = ForgetResult(
            deleted_count=2,
            hard_deleted=False,
            cascade_count=0,
            memory_ids=ids,
        )
        assert r.deleted_count == 2
        assert r.hard_deleted is False
        assert r.cascade_count == 0
        assert r.memory_ids == ids

    def test_hard_delete_flag(self) -> None:
        r = ForgetResult(
            deleted_count=1,
            hard_deleted=True,
            cascade_count=3,
            memory_ids=[uuid4()],
        )
        assert r.hard_deleted is True
        assert r.cascade_count == 3

    def test_frozen(self) -> None:
        r = ForgetResult(
            deleted_count=1,
            hard_deleted=False,
            cascade_count=0,
            memory_ids=[uuid4()],
        )
        with pytest.raises(AttributeError):
            r.deleted_count = 5  # type: ignore[misc]


class TestForgetError:
    def test_is_exception(self) -> None:
        assert issubclass(ForgetError, Exception)

    def test_no_memory_id_message(self) -> None:
        err = ForgetError("must provide memory_id or memory_ids")
        assert "memory_id" in str(err)

    def test_raises(self) -> None:
        with pytest.raises(ForgetError, match="must provide"):
            raise ForgetError("must provide memory_id or memory_ids")


# ---------------------------------------------------------------------------
# AuditEntry / AuditPage / ActivitySummary
# ---------------------------------------------------------------------------


class TestAuditEntry:
    """Test AuditEntry frozen dataclass."""

    def _make(self, **overrides: object) -> AuditEntry:
        defaults = {
            "id": 1,
            "org_id": uuid4(),
            "agent_id": uuid4(),
            "user_id": None,
            "operation": "store",
            "memory_id": uuid4(),
            "memory_type": "episodic",
            "details": {},
            "prev_hash": None,
            "row_hash": b"\x00" * 32,
            "ip_address": None,
            "user_agent": None,
            "request_id": None,
            "created_at": datetime.now(tz=UTC),
        }
        defaults.update(overrides)
        return AuditEntry(**defaults)  # type: ignore[arg-type]

    def test_construction(self) -> None:
        entry = self._make()
        assert entry.operation == "store"
        assert entry.prev_hash is None

    def test_all_fields_populated(self) -> None:
        entry = self._make(
            user_id=uuid4(),
            prev_hash=b"\xab" * 32,
            ip_address="192.168.1.1",
            user_agent="test-agent/1.0",
            request_id="req-123",
        )
        assert entry.user_id is not None
        assert entry.ip_address == "192.168.1.1"
        assert entry.user_agent == "test-agent/1.0"
        assert entry.request_id == "req-123"

    def test_frozen(self) -> None:
        entry = self._make()
        with pytest.raises(AttributeError):
            entry.operation = "recall"  # type: ignore[misc]


class TestAuditPage:
    """Test AuditPage frozen dataclass."""

    def test_construction(self) -> None:
        page = AuditPage(
            entries=[],
            total=0,
            page=1,
            page_size=50,
            has_next=False,
        )
        assert page.entries == []
        assert page.total == 0
        assert page.page == 1
        assert page.page_size == 50
        assert page.has_next is False

    def test_has_next_true(self) -> None:
        page = AuditPage(
            entries=[],
            total=100,
            page=1,
            page_size=50,
            has_next=True,
        )
        assert page.has_next is True

    def test_frozen(self) -> None:
        page = AuditPage(
            entries=[],
            total=0,
            page=1,
            page_size=50,
            has_next=False,
        )
        with pytest.raises(AttributeError):
            page.total = 99  # type: ignore[misc]


class TestActivitySummary:
    """Test ActivitySummary frozen dataclass."""

    def test_construction_defaults(self) -> None:
        agent = uuid4()
        summary = ActivitySummary(agent_id=agent)
        assert summary.agent_id == agent
        assert summary.operation_counts == {}
        assert summary.total == 0

    def test_construction_with_counts(self) -> None:
        agent = uuid4()
        counts = {"store": 10, "recall": 5, "forget": 2}
        summary = ActivitySummary(
            agent_id=agent,
            operation_counts=counts,
            total=17,
        )
        assert summary.operation_counts == counts
        assert summary.total == 17

    def test_frozen(self) -> None:
        summary = ActivitySummary(agent_id=uuid4())
        with pytest.raises(AttributeError):
            summary.total = 42  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SweepResult / DecayResult / TransitionResult
# ---------------------------------------------------------------------------


class TestSweepResult:
    """Test SweepResult frozen dataclass."""

    def test_construction(self) -> None:
        ids = [uuid4(), uuid4()]
        r = SweepResult(expired_count=2, memory_ids=ids)
        assert r.expired_count == 2
        assert r.memory_ids == ids

    def test_empty_sweep(self) -> None:
        r = SweepResult(expired_count=0, memory_ids=[])
        assert r.expired_count == 0
        assert r.memory_ids == []

    def test_frozen(self) -> None:
        r = SweepResult(expired_count=1, memory_ids=[uuid4()])
        with pytest.raises(AttributeError):
            r.expired_count = 0  # type: ignore[misc]


class TestDecayResult:
    """Test DecayResult frozen dataclass."""

    def test_construction(self) -> None:
        r = DecayResult(decayed_count=15, below_threshold_count=3)
        assert r.decayed_count == 15
        assert r.below_threshold_count == 3

    def test_zero_counts(self) -> None:
        r = DecayResult(decayed_count=0, below_threshold_count=0)
        assert r.decayed_count == 0
        assert r.below_threshold_count == 0

    def test_frozen(self) -> None:
        r = DecayResult(decayed_count=1, below_threshold_count=0)
        with pytest.raises(AttributeError):
            r.decayed_count = 99  # type: ignore[misc]


class TestTransitionResult:
    """Test TransitionResult frozen dataclass."""

    def test_construction(self) -> None:
        mid = uuid4()
        r = TransitionResult(memory_id=mid, old_type="working", new_type="episodic")
        assert r.memory_id == mid
        assert r.old_type == "working"
        assert r.new_type == "episodic"

    def test_frozen(self) -> None:
        r = TransitionResult(memory_id=uuid4(), old_type="working", new_type="semantic")
        with pytest.raises(AttributeError):
            r.new_type = "procedural"  # type: ignore[misc]
