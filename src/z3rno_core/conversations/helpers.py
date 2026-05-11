"""Conversation storage helpers.

All functions take an active ``AsyncConnection`` inside a transaction
— same contract as ``engine.store``, ``engine.recall``, etc. The
helpers never start their own transactions so they compose with the
existing engine flows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


class ConversationNotFoundError(Exception):
    """Raised when ``get_conversation`` finds no row for ``(org_id, id)``."""


@dataclass(frozen=True)
class Conversation:
    id: UUID
    org_id: UUID
    agent_id: UUID
    user_id: UUID | None
    title: str | None
    summary_cadence: int
    turn_count: int
    last_summary_turn: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None


@dataclass(frozen=True)
class Turn:
    """One turn within a conversation."""

    memory_id: UUID
    turn_index: int
    turn_role: str
    content: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Create + fetch
# ---------------------------------------------------------------------------


async def create_conversation(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID,
    user_id: UUID | None = None,
    title: str | None = None,
    summary_cadence: int = 10,
    metadata: dict[str, Any] | None = None,
) -> Conversation:
    """INSERT a new conversation row and return its frozen view.

    ``summary_cadence`` must be ≥ 1 (enforced by the DB CHECK).
    """
    if summary_cadence < 1:
        raise ValueError("summary_cadence must be ≥ 1")
    conv_id = uuid4()
    await conn.execute(
        text("""
            INSERT INTO conversations (
                id, org_id, agent_id, user_id,
                title, summary_cadence,
                turn_count, last_summary_turn,
                metadata,
                created_at, updated_at
            ) VALUES (
                CAST(:id AS uuid),
                CAST(:org_id AS uuid),
                CAST(:agent_id AS uuid),
                CAST(:user_id AS uuid),
                :title, :cadence,
                0, 0,
                CAST(:metadata AS jsonb),
                now(), now()
            )
        """),
        {
            "id": str(conv_id),
            "org_id": str(org_id),
            "agent_id": str(agent_id),
            "user_id": str(user_id) if user_id else None,
            "title": title,
            "cadence": summary_cadence,
            "metadata": json.dumps(metadata or {}),
        },
    )
    return await get_conversation(conn, org_id=org_id, conversation_id=conv_id)


async def get_conversation(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    conversation_id: UUID,
) -> Conversation:
    """RLS-isolated fetch. Raises ``ConversationNotFoundError`` on
    cross-tenant lookups (RLS already drops the row; the explicit
    NotFound makes the 404 path obvious to callers)."""
    result = await conn.execute(
        text("""
            SELECT id, org_id, agent_id, user_id, title,
                   summary_cadence, turn_count, last_summary_turn,
                   metadata, created_at, updated_at, deleted_at
            FROM conversations
            WHERE org_id = CAST(:org_id AS uuid)
              AND id = CAST(:id AS uuid)
              AND deleted_at IS NULL
        """),
        {"org_id": str(org_id), "id": str(conversation_id)},
    )
    row = result.fetchone()
    if row is None:
        raise ConversationNotFoundError(str(conversation_id))
    return Conversation(
        id=row[0],
        org_id=row[1],
        agent_id=row[2],
        user_id=row[3],
        title=row[4],
        summary_cadence=int(row[5]),
        turn_count=int(row[6]),
        last_summary_turn=int(row[7]),
        metadata=dict(row[8] or {}),
        created_at=row[9],
        updated_at=row[10],
        deleted_at=row[11],
    )


# ---------------------------------------------------------------------------
# Turn management
# ---------------------------------------------------------------------------


_VALID_ROLES = {"user", "assistant", "system", "tool", "summary"}


async def add_turn(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    conversation_id: UUID,
    memory_id: UUID,
    turn_role: str,
) -> int:
    """Stamp a freshly-stored Memo as the next turn of ``conversation_id``.

    Atomic: bumps ``turn_count`` on the conversation row, fetches the
    new value, then updates the Memo's ``turn_index`` + ``turn_role``
    + ``conversation_id`` so they match. Returns the assigned
    ``turn_index``.

    Caller responsibility: pass a Memo that was created in the same
    transaction (no commits in between). The Memo's
    ``conversation_id`` column gets set here so the engine.store path
    doesn't need to know about conversations.
    """
    if turn_role not in _VALID_ROLES:
        raise ValueError(f"turn_role must be one of {_VALID_ROLES}, got {turn_role!r}")

    # 1. Bump the conversation counter and capture the new turn index
    #    in a single UPDATE ... RETURNING so we don't race two
    #    concurrent adders.
    result = await conn.execute(
        text("""
            UPDATE conversations
            SET turn_count = turn_count + 1,
                updated_at = now()
            WHERE org_id = CAST(:org_id AS uuid)
              AND id = CAST(:id AS uuid)
              AND deleted_at IS NULL
            RETURNING turn_count
        """),
        {"org_id": str(org_id), "id": str(conversation_id)},
    )
    row = result.fetchone()
    if row is None:
        raise ConversationNotFoundError(str(conversation_id))
    turn_index = int(row[0])

    # 2. Stamp the Memo. Direct UPDATE (not via SCD-2 trigger path)
    #    because conversation linkage is metadata about retrieval,
    #    not content history — the Memo's content itself is what
    #    SCD-2 tracks.
    await conn.execute(
        text("""
            UPDATE memories
            SET conversation_id = CAST(:conv_id AS uuid),
                turn_index = :turn_index,
                turn_role = :turn_role,
                updated_at = now()
            WHERE org_id = CAST(:org_id AS uuid)
              AND id = CAST(:memory_id AS uuid)
        """),
        {
            "conv_id": str(conversation_id),
            "turn_index": turn_index,
            "turn_role": turn_role,
            "org_id": str(org_id),
            "memory_id": str(memory_id),
        },
    )
    return turn_index


async def list_turns(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    conversation_id: UUID,
    limit: int | None = None,
    after_turn: int | None = None,
) -> list[Turn]:
    """Return the conversation's turns ordered by ``turn_index``.

    ``after_turn`` lets callers paginate forward (``after_turn=5``
    yields turns 6, 7, 8, …). ``limit`` caps the returned slice.
    Defaults: full history, no cap.
    """
    params: dict[str, Any] = {
        "org_id": str(org_id),
        "conv_id": str(conversation_id),
    }
    where = (
        "WHERE org_id = CAST(:org_id AS uuid) "
        "AND conversation_id = CAST(:conv_id AS uuid) "
        "AND deleted_at IS NULL"
    )
    if after_turn is not None:
        where += " AND turn_index > :after"
        params["after"] = after_turn
    limit_clause = ""
    if limit is not None:
        limit_clause = " LIMIT :lim"
        params["lim"] = limit
    base = "SELECT id, turn_index, turn_role, content, created_at FROM memories "
    sql = base + where + " ORDER BY turn_index ASC" + limit_clause
    result = await conn.execute(text(sql), params)
    return [
        Turn(
            memory_id=row[0],
            turn_index=int(row[1]),
            turn_role=row[2] or "",
            content=row[3],
            created_at=row[4],
        )
        for row in result.fetchall()
    ]


# ---------------------------------------------------------------------------
# Summary cadence
# ---------------------------------------------------------------------------


def needs_summary(conv: Conversation) -> bool:
    """True when at least ``summary_cadence`` turns have passed since
    the last summary (or since conversation start, if none yet).

    The engine's auto-summary path checks this *after* an ``add_turn``
    and enqueues a Forge summarization job when it flips True. The
    summary itself is recorded as a Memo with ``turn_role='summary'``
    via ``add_turn``, then ``mark_summary_emitted`` advances the
    high-water mark.
    """
    return (conv.turn_count - conv.last_summary_turn) >= conv.summary_cadence


async def mark_summary_emitted(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    conversation_id: UUID,
    through_turn: int,
) -> None:
    """Advance ``last_summary_turn`` to ``through_turn``.

    Idempotent: a lower or equal ``through_turn`` is a no-op. Callers
    should pass the highest turn index covered by the just-emitted
    summary.
    """
    await conn.execute(
        text("""
            UPDATE conversations
            SET last_summary_turn = GREATEST(last_summary_turn, :through),
                updated_at = now()
            WHERE org_id = CAST(:org_id AS uuid)
              AND id = CAST(:id AS uuid)
              AND deleted_at IS NULL
        """),
        {
            "through": int(through_turn),
            "org_id": str(org_id),
            "id": str(conversation_id),
        },
    )
