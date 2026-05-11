"""Phase G slice 2 — conversation memory primitives.

Public surface:

  * ``Conversation`` — frozen row shape.
  * ``Turn`` — frozen turn record (Memo + role + index).
  * ``create_conversation`` — INSERT a new conversation row.
  * ``add_turn`` — append a turn (atomic: bumps ``turn_count``,
    stamps the Memo's conversation_id/turn_index/turn_role).
  * ``get_conversation`` — RLS-isolated fetch.
  * ``list_turns`` — ordered turn list, optional pagination.
  * ``needs_summary`` — checks whether the cadence threshold has
    been crossed since the last summary.
  * ``mark_summary_emitted`` — bumps ``last_summary_turn`` after a
    summarization task lands.

The summarization *task itself* is left to the operator's chosen
Forge gateway — this module deals with the storage primitives; the
caller wires the LLM call.
"""

from __future__ import annotations

from z3rno_core.conversations.helpers import (
    Conversation,
    ConversationNotFoundError,
    Turn,
    add_turn,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_turns,
    mark_summary_emitted,
    needs_summary,
)

__all__ = [
    "Conversation",
    "ConversationNotFoundError",
    "Turn",
    "add_turn",
    "create_conversation",
    "delete_conversation",
    "get_conversation",
    "list_turns",
    "mark_summary_emitted",
    "needs_summary",
]
