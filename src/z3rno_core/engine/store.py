"""store() - the core memory creation operation.

Atomically:
  1. Validates inputs
  2. Generates embedding (async, via pluggable provider)
  3. INSERTs into memories table
  4. INSERTs relationships (if any) into memory_relationships + AGE graph
  5. INSERTs audit log entry with hash chain
  6. Returns the created memory ID

The entire operation runs in a single transaction. If any step fails,
everything rolls back.

Schema reference: Doc 08 Section 1.3 (request flow), Section 4.2.1 (memories).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.engine.audit import create_audit_entry
from z3rno_core.engine.embedding import EmbeddingProvider
from z3rno_core.models.enums import MemoryType


@dataclass(frozen=True)
class RelationshipInput:
    """Input for creating a memory relationship alongside a store()."""

    target_memory_id: UUID
    relationship_type: str  # One of RelationshipType values
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoreResult:
    """Result of a successful store() operation."""

    memory_id: UUID
    embedding_model: str | None
    importance_score: float
    created_at: datetime


class StoreError(Exception):
    """Raised when store() validation fails."""


class DuplicateMemoryError(Exception):
    """Raised when a near-duplicate memory is detected during store().

    Attributes:
        existing_memory_id: The UUID of the existing memory that matches.
        similarity: The cosine similarity score between the two embeddings.
    """

    def __init__(self, existing_memory_id: UUID, similarity: float) -> None:
        self.existing_memory_id = existing_memory_id
        self.similarity = similarity
        super().__init__(
            f"Near-duplicate memory detected: existing_memory_id={existing_memory_id}, "
            f"similarity={similarity:.4f}"
        )


async def store(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID,
    content: str,
    memory_type: MemoryType,
    embedding_provider: EmbeddingProvider | None = None,
    user_id: UUID | None = None,
    metadata: dict[str, Any] | None = None,
    relationships: list[RelationshipInput] | None = None,
    ttl_seconds: int | None = None,
    importance: float | None = None,
    check_duplicates: bool = False,
    duplicate_threshold: float = 0.95,
    # Phase B.1 — dataset scoping
    dataset_id: UUID | None = None,
    # Audit context
    api_key_id: UUID | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> StoreResult:
    """Store a new memory atomically.

    Args:
        conn: An active async connection (must be inside a transaction).
        org_id: The tenant's org_id.
        agent_id: The agent creating this memory.
        content: The memory text content (required, non-empty).
        memory_type: One of working/episodic/semantic/procedural.
        embedding_provider: Provider to generate the embedding vector.
            If None, embedding is deferred (NULL in DB, worker fills later).
        user_id: Optional end-user ID the agent is serving.
        metadata: Optional JSONB metadata dict.
        relationships: Optional list of relationships to create.
        ttl_seconds: Optional TTL in seconds (sets ttl_expires_at).
        importance: Optional importance score (0-1). Default 0.5.
        check_duplicates: If True, check for near-duplicate memories before
            storing. Requires an embedding_provider that returns a non-empty
            embedding. Default False.
        duplicate_threshold: Cosine similarity threshold (0-1) above which
            a memory is considered a near-duplicate. Default 0.95.
        api_key_id: For audit log.
        ip_address: For audit log.
        user_agent: For audit log.
        request_id: For audit log.

    Returns:
        StoreResult with the created memory ID and metadata.

    Raises:
        StoreError: If validation fails.
        DuplicateMemoryError: If check_duplicates is True and a near-duplicate
            memory exists (similarity > duplicate_threshold).
    """
    # --- Validation ---
    if not content or not content.strip():
        raise StoreError("content must be a non-empty string")

    importance_score = importance if importance is not None else 0.5
    if not 0.0 <= importance_score <= 1.0:
        raise StoreError("importance must be between 0.0 and 1.0")

    # --- Generate embedding ---
    embedding: list[float] | None = None
    embedding_model: str | None = None
    if embedding_provider is not None:
        embedding = await embedding_provider.embed_text(content)
        embedding_model = embedding_provider.model_name
        if not embedding:  # NoOp provider returns empty list
            embedding = None

    # --- Near-duplicate detection ---
    if check_duplicates and embedding is not None:
        vector_str = _format_vector(embedding)
        dup_result = await conn.execute(
            text("""
                SELECT id,
                       1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                FROM memories
                WHERE org_id = CAST(:org_id AS uuid)
                  AND agent_id = CAST(:agent_id AS uuid)
                  AND embedding IS NOT NULL
                  AND deleted_at IS NULL
                  AND valid_to IS NULL
                ORDER BY embedding <=> CAST(:query_vec AS vector)
                LIMIT 1
            """),
            {
                "query_vec": vector_str,
                "org_id": str(org_id),
                "agent_id": str(agent_id),
            },
        )
        dup_row = dup_result.fetchone()
        if dup_row is not None:
            existing_id, similarity = dup_row[0], float(dup_row[1])
            if similarity > duplicate_threshold:
                raise DuplicateMemoryError(
                    existing_memory_id=existing_id,
                    similarity=similarity,
                )

    # --- Compute TTL ---
    now = datetime.now().astimezone()
    ttl_expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None

    # --- INSERT memory ---
    memory_id = uuid4()
    await conn.execute(
        text("""
            INSERT INTO memories (
                id, org_id, agent_id, user_id, memory_type,
                content, summary, metadata, embedding, embedding_model,
                importance_score, recall_count, last_recalled_at,
                valid_from, valid_to, pinned, ttl_expires_at,
                deleted_at, quarantined, anomaly_score,
                dataset_id,
                created_at, updated_at
            ) VALUES (
                CAST(:id AS uuid),
                CAST(:org_id AS uuid),
                CAST(:agent_id AS uuid),
                CAST(:user_id AS uuid),
                CAST(:memory_type AS memory_type_enum),
                :content, NULL,
                CAST(:metadata AS jsonb),
                :embedding,
                :embedding_model,
                :importance_score, 0, NULL,
                :now, NULL, false, :ttl_expires_at,
                NULL, false, 0.0,
                CAST(:dataset_id AS uuid),
                :now, :now
            )
        """),
        {
            "id": str(memory_id),
            "org_id": str(org_id),
            "agent_id": str(agent_id),
            "user_id": str(user_id) if user_id else None,
            "memory_type": memory_type.value,
            "content": content,
            "metadata": _json_dumps(metadata or {}),
            "embedding": _format_vector(embedding),
            "embedding_model": embedding_model,
            "importance_score": importance_score,
            "now": now,
            "ttl_expires_at": ttl_expires_at,
            "dataset_id": str(dataset_id) if dataset_id else None,
        },
    )

    # --- INSERT relationships ---
    if relationships:
        for rel in relationships:
            rel_id = uuid4()
            await conn.execute(
                text("""
                    INSERT INTO memory_relationships (
                        id, org_id, source_memory_id, target_memory_id,
                        relationship_type, weight, metadata,
                        created_at, updated_at
                    ) VALUES (
                        CAST(:id AS uuid),
                        CAST(:org_id AS uuid),
                        CAST(:source AS uuid),
                        CAST(:target AS uuid),
                        CAST(:rel_type AS relationship_type_enum),
                        :weight,
                        CAST(:metadata AS jsonb),
                        :now, :now
                    )
                """),
                {
                    "id": str(rel_id),
                    "org_id": str(org_id),
                    "source": str(memory_id),
                    "target": str(rel.target_memory_id),
                    "rel_type": rel.relationship_type,
                    "weight": rel.weight,
                    "metadata": _json_dumps(rel.metadata),
                    "now": now,
                },
            )

    # --- Audit log ---
    await create_audit_entry(
        conn,
        org_id=org_id,
        operation="store",
        agent_id=agent_id,
        user_id=user_id,
        memory_id=memory_id,
        memory_type=memory_type.value,
        details={
            "content_length": len(content),
            "has_embedding": embedding is not None,
            "relationship_count": len(relationships) if relationships else 0,
            "ttl_seconds": ttl_seconds,
            "importance_score": importance_score,
        },
        api_key_id=api_key_id,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
    )

    return StoreResult(
        memory_id=memory_id,
        embedding_model=embedding_model,
        importance_score=importance_score,
        created_at=now,
    )


def _json_dumps(data: dict[str, Any]) -> str:
    """Serialize dict to JSON string for SQL parameter."""
    return json.dumps(data, default=str)


def _format_vector(embedding: list[float] | None) -> str | None:
    """Format embedding as pgvector literal string."""
    if embedding is None:
        return None
    return "[" + ",".join(str(x) for x in embedding) + "]"
