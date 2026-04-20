"""Unit tests for z3rno_core.security.rls — set_org_context / clear_org_context."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from z3rno_core.security.rls import clear_org_context, set_org_context


class TestSetOrgContext:
    """Test set_org_context issues the correct SET LOCAL."""

    def test_sets_session_variable(self) -> None:
        conn = MagicMock()
        oid = uuid4()
        set_org_context(conn, oid)
        assert conn.execute.called
        sql = str(conn.execute.call_args[0][0].text)
        assert "SET LOCAL app.current_org_id" in sql
        assert str(oid) in sql

    def test_accepts_string_org_id(self) -> None:
        conn = MagicMock()
        set_org_context(conn, "my-org-id")
        sql = str(conn.execute.call_args[0][0].text)
        assert "my-org-id" in sql


class TestClearOrgContext:
    """Test clear_org_context issues RESET."""

    def test_resets_session_variable(self) -> None:
        conn = MagicMock()
        clear_org_context(conn)
        assert conn.execute.called
        sql = str(conn.execute.call_args[0][0].text)
        assert "RESET app.current_org_id" in sql
