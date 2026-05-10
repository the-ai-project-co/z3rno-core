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
    memory_type: str | None = None,
    filters: dict[str, Any] | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    as_of: datetime | None = None,
    include_deleted: bool = False,
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

    if filters:
        conditions.append("metadata @> CAST(:meta_filter AS jsonb)")
        params["meta_filter"] = json.dumps(filters)

    return " AND ".join(conditions), params
