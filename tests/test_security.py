"""Unit tests for z3rno_core.security.rls.

Tests that the RLS functions exist and have correct signatures.
No database connection needed for structural checks.
"""

from __future__ import annotations

import inspect

from z3rno_core.security.rls import clear_org_context, set_org_context


class TestSetOrgContext:
    """Test set_org_context function signature and structure."""

    def test_is_callable(self) -> None:
        assert callable(set_org_context)

    def test_accepts_conn_and_org_id(self) -> None:
        sig = inspect.signature(set_org_context)
        params = list(sig.parameters.keys())
        assert "conn" in params
        assert "org_id" in params

    def test_has_two_parameters(self) -> None:
        sig = inspect.signature(set_org_context)
        assert len(sig.parameters) == 2

    def test_return_annotation_is_none(self) -> None:
        sig = inspect.signature(set_org_context)
        assert sig.return_annotation in (None, "None")


class TestClearOrgContext:
    """Test clear_org_context function signature and structure."""

    def test_is_callable(self) -> None:
        assert callable(clear_org_context)

    def test_accepts_conn(self) -> None:
        sig = inspect.signature(clear_org_context)
        params = list(sig.parameters.keys())
        assert "conn" in params

    def test_has_one_parameter(self) -> None:
        sig = inspect.signature(clear_org_context)
        assert len(sig.parameters) == 1

    def test_return_annotation_is_none(self) -> None:
        sig = inspect.signature(clear_org_context)
        assert sig.return_annotation in (None, "None")


class TestTemporalQueriesFunctionsExist:
    """Verify temporal query functions are importable and callable."""

    def test_get_memory_at_time_exists(self) -> None:
        from z3rno_core.temporal.queries import get_memory_at_time

        assert callable(get_memory_at_time)

    def test_get_memory_history_exists(self) -> None:
        from z3rno_core.temporal.queries import get_memory_history

        assert callable(get_memory_history)

    def test_get_memories_changed_between_exists(self) -> None:
        from z3rno_core.temporal.queries import get_memories_changed_between

        assert callable(get_memories_changed_between)
