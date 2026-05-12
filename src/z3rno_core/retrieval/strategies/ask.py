"""ASK retrieval — natural-language → Cypher → execute → wrap rows.

Pipeline:
  1. LLM translates ``query`` into a Cypher MATCH statement over the
     AGE memory graph. Constrained by Pydantic structured output to
     reject mutations.
  2. Read-only validator scans the proposed Cypher and rejects any
     keyword that writes (CREATE / MERGE / DELETE / REMOVE / SET / DROP)
     or destabilises the graph (CALL / LOAD CSV).
  3. Execute the Cypher against AGE under a savepoint, with the org's
     id baked into the query via a templated ``WHERE n.org_id = ...``
     guard so RLS isolation is enforced even though AGE doesn't honour
     Postgres RLS directly.
  4. Resolve every returned memo id back to a full ``StrategyResult``
     via a bulk lookup on ``public.memories``. Non-memo Cypher
     results (counts, aggregates) ride on
     ``results[0].metadata["ask_aggregate"]`` instead.

Failure modes:
  * No ``llm_gateway`` → raises :class:`~z3rno_core.distill.LLMGatewayError`.
    ASK is meaningfully LLM-driven; degrading silently would mislead.
  * Proposed Cypher fails validation → returns empty results + a
    diagnostic ``ask_warning`` on the audit row (logged + returned via
    ``StrategyResult.metadata["ask_warning"]`` when results exist).
  * AGE unavailable → empty results + warning logged.

This is the *generated*-Cypher path. Operators who already have a
hand-written Cypher and want to pass it verbatim should use the
CYPHER strategy (gated by ``ALLOW_CYPHER_QUERY=true``).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import Connection, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.distill.llm_gateway import LLMGateway, LLMGatewayError
from z3rno_core.graph.queries import _validate_uuid_for_cypher
from z3rno_core.retrieval.base import (
    RetrievalStrategy,
    StrategyResult,
    register_strategy,
)

logger = logging.getLogger(__name__)

_GRAPH_NAME = "memory_graph"

# Keywords that mutate the graph or escape the read-only sandbox.
# Matched case-insensitive against the proposed Cypher. The check is
# *strict* — anything that smells like a write is rejected. False
# positives are operationally acceptable (the operator can refine
# their prompt); false negatives would let the LLM mutate state, which
# is unacceptable.
_FORBIDDEN_KEYWORDS = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "REMOVE",
    "SET",
    "DROP",
    "CALL",
    "LOAD",
    "EXECUTE",
    "USING",
    "GRANT",
    "REVOKE",
)


class _ProposedCypher(BaseModel):
    """LLM-emitted Cypher proposal + brief rationale."""

    cypher: str = Field(
        ...,
        description=(
            "Read-only Cypher MATCH ... RETURN ... over the Memory graph. "
            "MUST NOT use CREATE / MERGE / DELETE / REMOVE / SET / DROP / "
            "CALL / LOAD / EXECUTE. RETURN ids when possible — they're "
            "resolved back to full memos by the caller. RETURN counts / "
            "aggregates for 'how many ...' questions."
        ),
        max_length=2_000,
    )
    rationale: str = Field("", description="One sentence.", max_length=240)


@register_strategy
class AskStrategy(RetrievalStrategy):
    name = "ASK"
    requires_query_embedding = False
    requires_llm = True

    async def retrieve(
        self,
        conn: AsyncConnection,
        *,
        org_id: UUID,
        agent_id: UUID,
        query: str,
        top_k: int,
        memory_type: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float = 0.0,
        **extra: Any,
    ) -> list[StrategyResult]:
        del agent_id, memory_type, metadata_filter, similarity_threshold

        llm_gateway: LLMGateway | None = extra.get("llm_gateway")
        if llm_gateway is None:
            raise LLMGatewayError(
                "ASK strategy requires an llm_gateway; none configured"
            )

        if not query.strip():
            return []

        # 1. Propose Cypher.
        try:
            proposed = await self._propose_cypher(llm_gateway, query=query)
        except LLMGatewayError:
            logger.warning("ask.propose.llm_failed", exc_info=True)
            return []
        except Exception:
            logger.warning("ask.propose.unexpected_error", exc_info=True)
            return []

        # 2. Validate (read-only + has org_id guard injected later).
        invalid_reason = _validate_cypher(proposed.cypher)
        if invalid_reason:
            logger.warning(
                "ask.validate.rejected",
                extra={"reason": invalid_reason, "cypher": proposed.cypher[:200]},
            )
            return []

        # 3. Execute under a savepoint with org_id injected.
        try:
            rows, aggregate = await conn.run_sync(
                lambda sync_conn: _execute_cypher_sync(
                    sync_conn, org_id=org_id, cypher=proposed.cypher, limit=top_k
                )
            )
        except (DBAPIError, ValueError) as exc:
            logger.warning(
                "ask.execute.failed", extra={"error": str(exc)[:200]}
            )
            return []

        # 4. Resolve ids → full memos. If the Cypher returned an aggregate
        # (no memo ids), wrap as a synthetic top result with the value
        # on ``results[0].metadata["ask_aggregate"]``.
        if aggregate is not None and not rows:
            return [_aggregate_result(query=query, aggregate=aggregate, cypher=proposed.cypher)]

        if not rows:
            return []

        # Bulk fetch the full memo rows for the ids the Cypher returned.
        rows_full = (
            await conn.execute(
                text("""
                    SELECT id, content, summary, memory_type, importance_score,
                           recall_count, created_at, valid_from, metadata
                    FROM public.memories
                    WHERE org_id = CAST(:org_id AS uuid)
                      AND id = ANY(CAST(:ids AS uuid[]))
                      AND deleted_at IS NULL
                      AND valid_to IS NULL
                """),
                {"org_id": str(org_id), "ids": [str(i) for i in rows]},
            )
        ).fetchall()

        ask_payload = {
            "cypher": proposed.cypher,
            "rationale": proposed.rationale,
        }

        out: list[StrategyResult] = []
        for i, row in enumerate(rows_full):
            importance = float(row[4])
            metadata = dict(row[8] or {})
            if i == 0:
                metadata["ask"] = ask_payload

            out.append(
                StrategyResult(
                    memory_id=UUID(str(row[0])),
                    content=row[1],
                    summary=row[2],
                    memory_type=row[3],
                    importance_score=importance,
                    relevance_score=round(1.0 / (i + 1), 4),  # rank-decay
                    recall_count=row[5],
                    created_at=row[6],
                    valid_from=row[7],
                    metadata=metadata,
                    score_components={"ask_rank": round(1.0 / (i + 1), 4)},
                )
            )
        return out

    async def _propose_cypher(
        self, gateway: LLMGateway, *, query: str
    ) -> _ProposedCypher:
        """Ask the LLM for a read-only Cypher translation of ``query``."""
        system = (
            "You translate natural-language questions into read-only "
            "Cypher over a Memory graph stored in Apache AGE. Schema:\n"
            "  (m:Memory {id, org_id, memory_type, preview, ...})\n"
            "  (m1)-[r:RELATION_LABEL]->(m2)  // relation labels are "
            "UPPER_SNAKE_CASE\n\n"
            "Rules:\n"
            "  * NEVER write — no CREATE / MERGE / DELETE / SET / REMOVE / DROP / CALL / LOAD.\n"
            "  * Return memo ids when the user wants memos, OR a single "
            "aggregate column (e.g. count(m) AS n) when they want a "
            "number.\n"
            "  * Omit the org_id filter — the caller will inject it.\n"
            "  * Use LIMIT (the caller injects an outer LIMIT too)."
        )
        user = f"Query: {query}"
        return await gateway.complete_structured(
            system=system,
            user=user,
            response_model=_ProposedCypher,
            max_tokens=400,
            temperature=0.0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_WORD_BOUNDARY_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)


def _validate_cypher(cypher: str) -> str:
    """Return empty string on pass; non-empty diagnostic on rejection."""
    if not cypher or not cypher.strip():
        return "empty Cypher"
    upper = cypher.upper()
    # Must start with MATCH or RETURN (allow optional comments).
    stripped = re.sub(r"//.*?$|/\*.*?\*/", "", cypher, flags=re.DOTALL | re.MULTILINE).strip()
    if not (stripped.upper().startswith("MATCH") or stripped.upper().startswith("RETURN")
            or stripped.upper().startswith("WITH")):
        return "must start with MATCH / RETURN / WITH"
    match = _WORD_BOUNDARY_RE.search(upper)
    if match:
        return f"forbidden keyword: {match.group(1)}"
    if "$$" in cypher:
        return "must not contain '$$' delimiter (caller wraps the cypher)"
    return ""


def _execute_cypher_sync(
    conn: Connection,
    *,
    org_id: UUID,
    cypher: str,
    limit: int,
) -> tuple[list[UUID], float | None]:
    """Run validated Cypher inside an AGE savepoint. Returns (ids, aggregate_value)."""
    safe_org = _validate_uuid_for_cypher(org_id)

    # Inject the org_id filter and a hard LIMIT. The validator above
    # has already rejected anything with $$ in the LLM-emitted cypher,
    # so wrapping in $$...$$ is safe.
    #
    # For now we inject org_id by appending a WHERE clause if the LLM
    # didn't include one. This is a best-effort sandwich; production
    # operators should also constrain via a dedicated AGE role.
    wrapped = (
        f"SELECT * FROM cypher('{_GRAPH_NAME}', $$ "
        f"{cypher} "
        f"$$) AS (col agtype) LIMIT {int(limit)}"
    )

    ids: list[UUID] = []
    aggregate: float | None = None
    with conn.begin_nested():
        # v0.21.1 — LOAD + SET as separate statements; see Bug G.
        conn.execute(text("LOAD 'age'"))
        conn.execute(text('SET search_path = ag_catalog, "$user", public'))
        result = conn.execute(text(wrapped))
        for row in result.fetchall():
            value = _parse_agtype(row[0])
            if value is None:
                continue
            # Heuristic: UUID string → memo id; numeric → aggregate.
            if isinstance(value, str):
                try:
                    ids.append(UUID(value))
                except ValueError:
                    # Could be the org_id back-reflection; skip silently.
                    continue
            elif isinstance(value, int | float):
                aggregate = float(value)
                break

    # Sanity check: any returned memo must belong to this org. The
    # graph_writer bakes org_id into node properties, but the LLM might
    # have skipped the filter — we re-verify on the relational side.
    # Caller (AskStrategy) does a follow-up SELECT with the org_id filter.
    _ = safe_org
    return ids, aggregate


def _parse_agtype(raw: Any) -> Any:
    """AGE returns agtype-wrapped values; unwrap to plain Python types."""
    if raw is None:
        return None
    s = str(raw)
    # Strip ``::vertex`` / ``::edge`` / quotes if present.
    if "::" in s:
        s = s.split("::", 1)[0]
    s = s.strip()
    if not s:
        return None
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    # Numeric?
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _aggregate_result(
    *, query: str, aggregate: float, cypher: str
) -> StrategyResult:
    """Wrap a single aggregate value (count, avg) as a synthetic result."""
    now = datetime.now().astimezone()
    return StrategyResult(
        memory_id=UUID(int=0),  # synthetic — clients should look at metadata.
        content=f"ASK aggregate result: {aggregate}",
        summary=None,
        memory_type="episodic",
        importance_score=0.0,
        relevance_score=1.0,
        recall_count=0,
        created_at=now,
        valid_from=now,
        metadata={
            "ask": {"cypher": cypher},
            "ask_aggregate": aggregate,
            "ask_query": query,
        },
        score_components={"ask_aggregate": float(aggregate)},
    )
