"""Phase G slice 2 — conversation primitives unit tests.

Mocks the AsyncConnection so each helper's SQL + parameter dict can
be asserted without a live DB. Integration tests against the
testcontainer cover the actual table behaviour.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from z3rno_core.conversations import (
    Conversation,
    ConversationNotFoundError,
    add_turn,
    create_conversation,
    get_conversation,
    list_turns,
    mark_summary_emitted,
    needs_summary,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic Conversation rows for the cadence logic
# ---------------------------------------------------------------------------


def _conv(turn_count: int, last_summary_turn: int, cadence: int = 10) -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=UUID(int=1),
        org_id=UUID(int=2),
        agent_id=UUID(int=3),
        user_id=None,
        title=None,
        summary_cadence=cadence,
        turn_count=turn_count,
        last_summary_turn=last_summary_turn,
        metadata={},
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _mock_conn() -> MagicMock:
    conn = MagicMock()
    conn.execute = AsyncMock()
    return conn


def _fake_row(*values: object) -> MagicMock:
    r = MagicMock()
    r.__getitem__ = lambda self, i: values[i]  # type: ignore[misc]
    return r


def _fake_result(row: object | None = None, rows: list[object] | None = None) -> MagicMock:
    r = MagicMock()
    r.fetchone = MagicMock(return_value=row)
    r.fetchall = MagicMock(return_value=rows or [])
    return r


# ---------------------------------------------------------------------------
# needs_summary
# ---------------------------------------------------------------------------


def test_needs_summary_false_at_start() -> None:
    assert needs_summary(_conv(turn_count=0, last_summary_turn=0)) is False


def test_needs_summary_false_below_threshold() -> None:
    assert needs_summary(_conv(turn_count=5, last_summary_turn=0, cadence=10)) is False


def test_needs_summary_true_at_threshold() -> None:
    assert needs_summary(_conv(turn_count=10, last_summary_turn=0, cadence=10)) is True


def test_needs_summary_resets_after_summary() -> None:
    # 10 turns covered → False until 10 more arrive
    assert needs_summary(_conv(turn_count=15, last_summary_turn=10, cadence=10)) is False
    assert needs_summary(_conv(turn_count=20, last_summary_turn=10, cadence=10)) is True


def test_needs_summary_respects_per_conversation_cadence() -> None:
    """A chatty agent might lower cadence to 3."""
    assert needs_summary(_conv(turn_count=3, last_summary_turn=0, cadence=3)) is True
    assert needs_summary(_conv(turn_count=2, last_summary_turn=0, cadence=3)) is False


# ---------------------------------------------------------------------------
# create_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_conversation_rejects_zero_cadence() -> None:
    conn = _mock_conn()
    with pytest.raises(ValueError, match="summary_cadence"):
        await create_conversation(conn, org_id=uuid4(), agent_id=uuid4(), summary_cadence=0)


@pytest.mark.asyncio
async def test_create_conversation_inserts_then_fetches() -> None:
    conn = _mock_conn()
    org = uuid4()
    agent = uuid4()
    # Second execute (the SELECT inside get_conversation) returns the row.
    now = datetime.now(UTC)

    def _exec_side_effect(*args: object, **_kwargs: object) -> MagicMock:
        # First call (INSERT) — no row to return; second call (SELECT) returns shape.
        if conn.execute.await_count == 1:
            return _fake_result()
        return _fake_result(
            row=_fake_row(
                UUID(int=42),  # id
                org,
                agent,
                None,  # user_id
                "demo",
                7,  # summary_cadence
                0,
                0,
                {},
                now,
                now,
                None,
            )
        )

    conn.execute.side_effect = _exec_side_effect

    conv = await create_conversation(
        conn, org_id=org, agent_id=agent, summary_cadence=7, title="demo"
    )
    assert conv.summary_cadence == 7
    assert conv.title == "demo"
    # Two executes: INSERT then SELECT.
    assert conn.execute.await_count == 2


# ---------------------------------------------------------------------------
# get_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_conversation_404_on_missing() -> None:
    conn = _mock_conn()
    conn.execute.return_value = _fake_result(row=None)
    with pytest.raises(ConversationNotFoundError):
        await get_conversation(conn, org_id=uuid4(), conversation_id=uuid4())


# ---------------------------------------------------------------------------
# add_turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_turn_rejects_unknown_role() -> None:
    conn = _mock_conn()
    with pytest.raises(ValueError, match="turn_role"):
        await add_turn(
            conn,
            org_id=uuid4(),
            conversation_id=uuid4(),
            memory_id=uuid4(),
            turn_role="bogus",
        )


@pytest.mark.asyncio
async def test_add_turn_assigns_sequential_index() -> None:
    conn = _mock_conn()
    # First call (UPDATE … RETURNING turn_count) gives turn_count=3
    # Second call (UPDATE memories) returns nothing meaningful.
    conn.execute.side_effect = [
        _fake_result(row=_fake_row(3)),
        _fake_result(),
    ]
    idx = await add_turn(
        conn,
        org_id=uuid4(),
        conversation_id=uuid4(),
        memory_id=uuid4(),
        turn_role="assistant",
    )
    assert idx == 3
    assert conn.execute.await_count == 2


@pytest.mark.asyncio
async def test_add_turn_404_when_conversation_missing() -> None:
    conn = _mock_conn()
    conn.execute.return_value = _fake_result(row=None)
    with pytest.raises(ConversationNotFoundError):
        await add_turn(
            conn,
            org_id=uuid4(),
            conversation_id=uuid4(),
            memory_id=uuid4(),
            turn_role="user",
        )


# ---------------------------------------------------------------------------
# list_turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_turns_builds_sql_with_filters() -> None:
    conn = _mock_conn()
    now = datetime.now(UTC)
    conn.execute.return_value = _fake_result(
        rows=[
            _fake_row(uuid4(), 1, "user", "hi", now),
            _fake_row(uuid4(), 2, "assistant", "hello", now),
        ]
    )
    turns = await list_turns(
        conn,
        org_id=uuid4(),
        conversation_id=uuid4(),
        after_turn=0,
        limit=10,
    )
    assert [t.turn_index for t in turns] == [1, 2]
    assert [t.turn_role for t in turns] == ["user", "assistant"]
    # Verify SQL string + params on the executed call.
    args, _ = conn.execute.call_args
    sql_text = args[0]
    sql_str = sql_text.text if hasattr(sql_text, "text") else str(sql_text)
    assert "ORDER BY turn_index ASC" in sql_str
    assert "turn_index > :after" in sql_str
    assert "LIMIT :lim" in sql_str


# ---------------------------------------------------------------------------
# mark_summary_emitted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_summary_emitted_uses_greatest() -> None:
    """``last_summary_turn`` only advances — passing a lower value is a no-op."""
    conn = _mock_conn()
    conn.execute.return_value = _fake_result()
    await mark_summary_emitted(conn, org_id=uuid4(), conversation_id=uuid4(), through_turn=20)
    args, _ = conn.execute.call_args
    sql_text = args[0]
    sql_str = sql_text.text if hasattr(sql_text, "text") else str(sql_text)
    assert "GREATEST(last_summary_turn, :through)" in sql_str


# ---------------------------------------------------------------------------
# build_where_clause integration with conversation_id
# ---------------------------------------------------------------------------


def test_where_clause_includes_conversation_id() -> None:
    from z3rno_core.retrieval._filters import build_where_clause

    org = uuid4()
    agent = uuid4()
    conv = uuid4()
    where, params = build_where_clause(org_id=org, agent_id=agent, conversation_id=conv)
    assert "conversation_id = CAST(:conversation_id AS uuid)" in where
    assert params["conversation_id"] == str(conv)


def test_where_clause_omits_conversation_id_when_unset() -> None:
    from z3rno_core.retrieval._filters import build_where_clause

    where, params = build_where_clause(org_id=uuid4(), agent_id=uuid4())
    assert "conversation_id" not in where
    assert "conversation_id" not in params
