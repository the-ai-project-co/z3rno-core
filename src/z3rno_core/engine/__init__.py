"""Memory engine - core store / recall / forget / audit functions.

The engine layer provides the transactional business logic for memory
operations. Each function takes an AsyncConnection and performs all
database writes atomically within a single transaction.
"""

from __future__ import annotations

from z3rno_core.engine.audit import compute_row_hash, create_audit_entry
from z3rno_core.engine.embedding import (
    EmbeddingProvider,
    LiteLLMEmbeddingProvider,
    NoOpEmbeddingProvider,
    get_embedding_provider,
)
from z3rno_core.engine.forget import ForgetError, ForgetResult, forget
from z3rno_core.engine.recall import RecallError, RecallResult, recall
from z3rno_core.engine.store import RelationshipInput, StoreError, StoreResult, store

__all__ = [
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
    "compute_row_hash",
    "create_audit_entry",
    "forget",
    "get_embedding_provider",
    "recall",
    "store",
]
