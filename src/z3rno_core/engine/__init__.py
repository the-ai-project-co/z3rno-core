"""Memory engine - core store / recall / forget / audit functions.

The engine layer provides the transactional business logic for memory
operations. Each function takes an AsyncConnection and performs all
database writes atomically within a single transaction.
"""

from __future__ import annotations

from z3rno_core.engine.audit import compute_row_hash, create_audit_entry
from z3rno_core.engine.audit_query import (
    ActivitySummary,
    AuditEntry,
    AuditPage,
    audit,
    get_agent_activity_summary,
    get_memory_lifecycle,
)
from z3rno_core.engine.embedding import (
    EmbeddingProvider,
    LiteLLMEmbeddingProvider,
    NoOpEmbeddingProvider,
    get_embedding_provider,
)
from z3rno_core.engine.forget import ForgetError, ForgetResult, forget
from z3rno_core.engine.lifecycle import (
    DecayResult,
    SweepResult,
    TransitionResult,
    decay_importance,
    enforce_retention_cap,
    sweep_expired_memories,
    transition_memory,
)
from z3rno_core.engine.recall import RecallError, RecallResult, recall
from z3rno_core.engine.store import RelationshipInput, StoreError, StoreResult, store

__all__ = [
    "ActivitySummary",
    "AuditEntry",
    "AuditPage",
    "DecayResult",
    "EmbeddingProvider",
    "ForgetError",
    "ForgetResult",
    "LiteLLMEmbeddingProvider",
    "NoOpEmbeddingProvider",
    "RecallError",
    "RecallResult",
    "RelationshipInput",
    "StoreError",
    "StoreResult",
    "SweepResult",
    "TransitionResult",
    "audit",
    "compute_row_hash",
    "create_audit_entry",
    "decay_importance",
    "enforce_retention_cap",
    "forget",
    "get_agent_activity_summary",
    "get_embedding_provider",
    "get_memory_lifecycle",
    "recall",
    "store",
    "sweep_expired_memories",
    "transition_memory",
]
