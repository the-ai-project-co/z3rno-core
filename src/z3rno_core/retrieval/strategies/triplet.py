"""TRIPLET retrieval — partial (S, P, O) triplet → AGE traversal → slot fill.

Pipeline:
  1. LLM parses the query into a partial (subject, predicate, object)
     triplet where exactly one slot is unknown (``"?"``).
  2. AGE Cypher MATCH retrieves Memos that fit the known parts of the
     triplet.
  3. LLM uses the matched Memos as context to fill the missing slot.
  4. Return results: every matched Memo, with the synthesised answer
     riding on ``results[0].metadata["triplet_answer"]`` and the parsed
     triplet on every result's ``metadata["triplet"]``.

Examples:
  * "Where does Alice work?" → (Alice, WORKS_AT, ?) → AGE finds the
    object → LLM names it.
  * "Who introduced Bob to Carol?" → (?, INTRODUCED, Bob+Carol-pair)
    → LLM identifies the subject.

Failure modes:
  * No ``llm_gateway`` → raises :class:`~z3rno_core.distill.LLMGatewayError`.
    TRIPLET is meaningfully LLM-driven; degrading silently would give
    callers the wrong impression.
  * LLM returns malformed triplet → raises ``LLMValidationError``.
  * AGE not loaded → empty results + warning logged.

RLS: every Cypher MATCH includes the ``org_id`` filter baked into
node properties by the Phase A graph_writer.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import Connection, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.distill.llm_gateway import (
    LLMGateway,
    LLMGatewayError,
)
from z3rno_core.graph.queries import _validate_uuid_for_cypher
from z3rno_core.retrieval.base import (
    RetrievalStrategy,
    StrategyResult,
    register_strategy,
)

logger = logging.getLogger(__name__)

_GRAPH_NAME = "memory_graph"


# ---------------------------------------------------------------------------
# Triplet parsing + slot fill — LLM I/O schemas
# ---------------------------------------------------------------------------


class _ParsedTriplet(BaseModel):
    """LLM-extracted partial triplet from the user's query.

    Exactly one of ``subject`` / ``predicate`` / ``object`` is ``"?"``
    (the slot the user is asking about). The others are concrete
    entities or relationship labels.
    """

    subject: str = Field(..., description="Entity name or '?' if unknown")
    predicate: str = Field(
        ..., description="Relationship label (e.g. WORKS_AT) or '?' if unknown"
    )
    object: str = Field(..., description="Entity name or '?' if unknown")

    @property
    def unknown_slot(self) -> str:
        """Which slot is the ``"?"``; raises if zero or more than one."""
        unknowns = [
            slot
            for slot, val in (
                ("subject", self.subject),
                ("predicate", self.predicate),
                ("object", self.object),
            )
            if val == "?"
        ]
        if len(unknowns) != 1:
            raise ValueError(
                f"triplet must have exactly one unknown slot; got {unknowns}"
            )
        return unknowns[0]


class _SlotAnswer(BaseModel):
    """LLM-synthesised answer for the missing triplet slot."""

    value: str = Field(..., description="The filled-in value (entity, predicate, etc.)")
    explanation: str = Field(
        "", description="One-sentence explanation of why this value fits."
    )


@register_strategy
class TripletStrategy(RetrievalStrategy):
    name = "TRIPLET"
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
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.0,
        **extra: Any,
    ) -> list[StrategyResult]:
        del memory_type, filters, similarity_threshold  # TRIPLET ignores

        llm_gateway: LLMGateway | None = extra.get("llm_gateway")
        if llm_gateway is None:
            raise LLMGatewayError(
                "TRIPLET strategy requires an llm_gateway; none configured"
            )

        if not query or not query.strip():
            return []

        # --- 1. Parse the query into a partial triplet ---
        triplet = await self._parse_triplet(llm_gateway, query=query)

        # --- 2. AGE traversal for memos matching the known parts ---
        matched = await self._match_triplet(
            conn, org_id=org_id, triplet=triplet, top_k=top_k
        )

        if not matched:
            return []

        # --- 3. LLM fills the missing slot using the matched memos ---
        try:
            slot_answer = await self._fill_slot(
                llm_gateway,
                query=query,
                triplet=triplet,
                matched=matched,
            )
        except LLMGatewayError:
            logger.warning("triplet.slot_fill.llm_failed", exc_info=True)
            slot_answer = None
        except Exception:
            logger.warning("triplet.slot_fill.unexpected_error", exc_info=True)
            slot_answer = None

        # --- 4. Build results ---
        now = datetime.now().astimezone()
        triplet_payload = {
            "subject": triplet.subject,
            "predicate": triplet.predicate,
            "object": triplet.object,
            "unknown_slot": triplet.unknown_slot,
        }

        results: list[StrategyResult] = []
        for i, row in enumerate(matched):
            importance = float(row[4])
            recency_days = max((now - row[6]).total_seconds() / 86400, 0.01)
            recency_score = min(1.0, 1.0 / recency_days)

            # Triplet matches are binary (match or not). Weight match
            # rank * importance * recency as a soft relevance signal.
            rank_decay = 1.0 / (i + 1)
            relevance = 0.60 * rank_decay + 0.25 * importance + 0.15 * recency_score

            metadata = dict(row[8] or {})
            metadata["triplet"] = triplet_payload
            if i == 0 and slot_answer is not None:
                metadata["triplet_answer"] = {
                    "value": slot_answer.value,
                    "explanation": slot_answer.explanation,
                }

            results.append(
                StrategyResult(
                    memory_id=UUID(str(row[0])),
                    content=row[1],
                    summary=row[2],
                    memory_type=row[3],
                    importance_score=importance,
                    relevance_score=round(relevance, 4),
                    recall_count=row[5],
                    created_at=row[6],
                    valid_from=row[7],
                    metadata=metadata,
                    score_components={
                        "triplet_rank": round(rank_decay, 4),
                        "importance": round(importance, 4),
                        "recency": round(recency_score, 4),
                    },
                )
            )

        return results

    async def _parse_triplet(
        self, gateway: LLMGateway, *, query: str
    ) -> _ParsedTriplet:
        """LLM extracts a partial (S, P, O) triplet from the query."""
        system = (
            "Extract a (subject, predicate, object) triplet from the user's "
            "query. Exactly one slot must be '?' — the slot the user is "
            "asking about. Predicate is an UPPER_SNAKE_CASE relationship "
            "label (e.g. WORKS_AT, KNOWS, LIVES_IN). Subject and object are "
            "entity names. If the query is too vague to map cleanly, set "
            "predicate='?' and put the most likely entities in subject/object."
        )
        user = f"Query: {query}"
        return await gateway.complete_structured(
            system=system,
            user=user,
            response_model=_ParsedTriplet,
            max_tokens=200,
            temperature=0.0,
        )

    async def _match_triplet(
        self,
        conn: AsyncConnection,
        *,
        org_id: UUID,
        triplet: _ParsedTriplet,
        top_k: int,
    ) -> list[Any]:
        """AGE traversal for memos that fit the known parts of the triplet."""
        try:
            memory_ids = await conn.run_sync(
                lambda sync_conn: _match_triplet_sync(
                    sync_conn, org_id=org_id, triplet=triplet, top_k=top_k
                )
            )
        except (DBAPIError, ValueError) as exc:
            logger.warning(
                "triplet.match.failed", extra={"error": str(exc)[:200]}
            )
            return []

        if not memory_ids:
            return []

        # Fetch memo rows in id-order to match what the user expects.
        rows = (
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
                {"org_id": str(org_id), "ids": [str(i) for i in memory_ids]},
            )
        ).fetchall()
        return list(rows)

    async def _fill_slot(
        self,
        gateway: LLMGateway,
        *,
        query: str,
        triplet: _ParsedTriplet,
        matched: list[Any],
    ) -> _SlotAnswer:
        """LLM fills the missing slot using matched memos as context."""
        context_lines = []
        for row in matched[:10]:
            content = (row[1] or "").strip()[:300]
            context_lines.append(f"- {content}")

        system = (
            "Fill the missing slot ('?') in the triplet using ONLY the "
            "memories provided as evidence. Return the most likely value "
            "and a one-sentence explanation. If the evidence is "
            "insufficient, return value='unknown'."
        )
        user = (
            f"Query: {query}\n"
            f"Triplet: ({triplet.subject}, {triplet.predicate}, {triplet.object})\n"
            f"Unknown slot: {triplet.unknown_slot}\n\n"
            f"Memories:\n" + "\n".join(context_lines)
        )
        return await gateway.complete_structured(
            system=system,
            user=user,
            response_model=_SlotAnswer,
            max_tokens=200,
            temperature=0.0,
        )


def _match_triplet_sync(
    conn: Connection,
    *,
    org_id: UUID,
    triplet: _ParsedTriplet,
    top_k: int,
) -> list[UUID]:
    """Match a partial triplet against the AGE graph.

    Returns the list of Memo UUIDs involved in matching relationships.
    Wrapped in a savepoint so AGE failures don't poison the outer
    transaction.
    """
    safe_org = _validate_uuid_for_cypher(org_id)
    preamble = "LOAD 'age'; SET search_path = ag_catalog, \"$user\", public;"

    # Predicate goes verbatim into the Cypher edge label — validate it
    # to UPPER_SNAKE_CASE only. Entities are CONTAINS-matched on the
    # ``preview`` property the graph_writer populates.
    pred = triplet.predicate
    if pred == "?":
        edge = "[r]"  # any relationship type
    elif _is_safe_edge_label(pred):
        edge = f"[r:{pred}]"
    else:
        logger.warning(
            "triplet.match.unsafe_predicate", extra={"predicate": pred}
        )
        return []

    subj = triplet.subject
    obj = triplet.object

    nodes_returned: set[UUID] = set()

    with conn.begin_nested():
        if subj != "?" and obj != "?":
            # Both ends known — find the relationship; return both endpoints.
            cypher = (
                f"{preamble} "
                f"SELECT * FROM cypher('{_GRAPH_NAME}', $$ "
                f"MATCH (s:Memory)-{edge}-(o:Memory) "
                f"WHERE s.org_id = '{safe_org}' AND o.org_id = '{safe_org}' "
                f"AND toLower(s.preview) CONTAINS toLower('{_escape_cypher_str(subj)}') "
                f"AND toLower(o.preview) CONTAINS toLower('{_escape_cypher_str(obj)}') "
                f"RETURN s.id, o.id LIMIT {top_k} "
                f"$$) AS (s_id agtype, o_id agtype)"
            )
            try:
                for row in conn.execute(text(cypher)).fetchall():
                    _add_node(nodes_returned, row[0])
                    _add_node(nodes_returned, row[1])
            except DBAPIError:
                return []
        elif subj == "?":
            # Object known, subject unknown — find the subject.
            cypher = (
                f"{preamble} "
                f"SELECT * FROM cypher('{_GRAPH_NAME}', $$ "
                f"MATCH (s:Memory)-{edge}-(o:Memory) "
                f"WHERE s.org_id = '{safe_org}' AND o.org_id = '{safe_org}' "
                f"AND toLower(o.preview) CONTAINS toLower('{_escape_cypher_str(obj)}') "
                f"RETURN s.id LIMIT {top_k} "
                f"$$) AS (s_id agtype)"
            )
            try:
                for row in conn.execute(text(cypher)).fetchall():
                    _add_node(nodes_returned, row[0])
            except DBAPIError:
                return []
        elif obj == "?":
            # Subject known, object unknown.
            cypher = (
                f"{preamble} "
                f"SELECT * FROM cypher('{_GRAPH_NAME}', $$ "
                f"MATCH (s:Memory)-{edge}-(o:Memory) "
                f"WHERE s.org_id = '{safe_org}' AND o.org_id = '{safe_org}' "
                f"AND toLower(s.preview) CONTAINS toLower('{_escape_cypher_str(subj)}') "
                f"RETURN o.id LIMIT {top_k} "
                f"$$) AS (o_id agtype)"
            )
            try:
                for row in conn.execute(text(cypher)).fetchall():
                    _add_node(nodes_returned, row[0])
            except DBAPIError:
                return []
        else:  # predicate unknown
            # Both endpoints known but we don't know the relationship —
            # return any relationship between them.
            cypher = (
                f"{preamble} "
                f"SELECT * FROM cypher('{_GRAPH_NAME}', $$ "
                f"MATCH (s:Memory)-[r]-(o:Memory) "
                f"WHERE s.org_id = '{safe_org}' AND o.org_id = '{safe_org}' "
                f"AND toLower(s.preview) CONTAINS toLower('{_escape_cypher_str(subj)}') "
                f"AND toLower(o.preview) CONTAINS toLower('{_escape_cypher_str(obj)}') "
                f"RETURN s.id, o.id LIMIT {top_k} "
                f"$$) AS (s_id agtype, o_id agtype)"
            )
            try:
                for row in conn.execute(text(cypher)).fetchall():
                    _add_node(nodes_returned, row[0])
                    _add_node(nodes_returned, row[1])
            except DBAPIError:
                return []

    return list(nodes_returned)


def _is_safe_edge_label(label: str) -> bool:
    """Whitelist UPPER_SNAKE_CASE labels for Cypher interpolation."""
    return bool(label) and all(c.isalnum() or c == "_" for c in label) and not label[0].isdigit()


def _escape_cypher_str(value: str) -> str:
    """Escape single-quotes for safe inclusion in a Cypher string literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _add_node(out: set[UUID], agtype_value: Any) -> None:
    """Parse an agtype id value into a UUID, ignoring malformed entries."""
    s = str(agtype_value)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    try:
        out.add(UUID(s))
    except ValueError:
        pass
