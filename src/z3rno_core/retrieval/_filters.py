"""Shared SQL-filter helpers used by every strategy.

The agent / org / memory_type / metadata / temporal / soft-delete
filters are identical across strategies — keeping them in one place
means a Phase-D change (e.g. dataset filter) lands once instead of
once per strategy.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID


def build_where_clause(
    *,
    org_id: UUID,
    agent_id: UUID,
    # v0.21.1 — scope by end-user id (``memories.user_id`` column).
    # Multi-user agents need this for per-user recall isolation. Indexed
    # by migration 034 (partial index over (org_id, agent_id, user_id)).
    # Optional — most agents are single-user-per-agent and don't pass it.
    user_id: UUID | None = None,
    memory_type: str | None = None,
    # v0.21.2 — renamed from ``filters``. Semantic is
    # ``metadata @> :metadata_filter`` (JSONB containment); the old
    # name read like a general where-clause builder and caused silent
    # zero-hit drops when callers passed keys that don't exist in
    # stored metadata. ``filters`` stays as a deprecated alias one
    # layer up in ``engine.recall``; the private helper takes the
    # clean name only.
    metadata_filter: dict[str, Any] | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    as_of: datetime | None = None,
    include_deleted: bool = False,
    # Phase G slice 2 — scope to one conversation.
    conversation_id: UUID | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(where_clause_sql, params_dict)`` for the standard filters.

    org_id is intentionally the FIRST condition so Postgres can pre-
    filter via the B-tree index before any expensive index scan
    (HNSW vector, GIN tsvector). Don't reorder.
    """
    conditions: list[str] = ["org_id = CAST(:org_id AS uuid)"]
    params: dict[str, Any] = {"org_id": str(org_id)}

    conditions.append("agent_id = CAST(:agent_id AS uuid)")
    params["agent_id"] = str(agent_id)

    if user_id is not None:
        conditions.append("user_id = CAST(:user_id AS uuid)")
        params["user_id"] = str(user_id)

    if as_of:
        conditions.append("valid_from <= CAST(:as_of AS timestamptz)")
        conditions.append("(valid_to IS NULL OR valid_to > CAST(:as_of AS timestamptz))")
        params["as_of"] = as_of
    else:
        conditions.append("valid_to IS NULL")

    if not include_deleted:
        conditions.append("deleted_at IS NULL")

    if memory_type:
        conditions.append("memory_type = CAST(:memory_type AS memory_type_enum)")
        params["memory_type"] = memory_type

    if time_range:
        conditions.append("created_at >= CAST(:time_start AS timestamptz)")
        conditions.append("created_at <= CAST(:time_end AS timestamptz)")
        params["time_start"] = time_range[0]
        params["time_end"] = time_range[1]

    if metadata_filter:
        conditions.append("metadata @> CAST(:meta_filter AS jsonb)")
        params["meta_filter"] = json.dumps(metadata_filter)

    if conversation_id is not None:
        conditions.append("conversation_id = CAST(:conversation_id AS uuid)")
        params["conversation_id"] = str(conversation_id)

    return " AND ".join(conditions), params
