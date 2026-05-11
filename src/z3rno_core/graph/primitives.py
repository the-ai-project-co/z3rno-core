"""Typed graph primitives for Phase D.

These Pydantic models are the *graph-level* projection of the
relational state. A ``Memo`` is the in-memory shape passed between
the refine stages (dedupe, infer, summarize, reweight) and the AGE
graph writer; it is intentionally distinct from the SA ``Memory``
row, which carries lifecycle / temporal / vector concerns the graph
layer doesn't need.

Why two types
-------------
The SA ``Memory`` is a database row — many fields, SCD-2 versioned,
soft-deletable. A ``Memo`` is what flows through the refine pipeline:
identity + type + grounding + provenance, nothing else. Keeping them
separate avoids dragging SA session lifecycles into pure compute paths
and lets refine stages be tested without a database.

``Triplet`` is re-exported from ``z3rno_core.distill.schemas`` — Phase
A already shipped a frozen, validated triplet; defining a parallel one
here would fragment the type.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from z3rno_core.distill.schemas import Triplet

__all__ = ["Edge", "Memo", "Triplet"]


class Memo(BaseModel):
    """A typed graph node projected from one or more ``Memory`` rows.

    ``ontology_uri`` is optional — the resolver may not have run, or
    the entity may sit outside the loaded ontology. ``provenance``
    carries the same shape Forge already emits (prompt_hash, model,
    chunk_index, char_start/end) so refine stages can audit-trail
    their mutations.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(..., description="Stable Memo identifier (matches a memories.id row).")
    memo_type: str = Field(
        ...,
        min_length=1,
        description="Graph-node subtype label. Free-form; ontology-driven values preferred.",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Canonical surface form — used for dedupe in refine.",
    )
    ontology_uri: str | None = Field(
        default=None,
        description="Canonical entity URI when the Memo is grounded in an OWL concept.",
    )
    version: int = Field(default=1, ge=1, description="Refine-managed version counter.")
    provenance: dict[str, str] = Field(
        default_factory=dict,
        description="Audit trail entries; refine stages append their own keys.",
    )


class Edge(BaseModel):
    """A typed directed edge between two Memos.

    ``weight`` is the value slice 3's reweight stage manipulates from
    ``feedback`` signals. Defaults to 1.0 so newly-inferred edges from
    slice 4's infer stage start neutral.

    ``edge_id`` is a stable string identifier matching the value
    written to AGE and referenced by ``feedback.edge_id``.
    """

    model_config = ConfigDict(frozen=True)

    edge_id: str = Field(..., min_length=1)
    src_id: UUID = Field(...)
    dst_id: UUID = Field(...)
    predicate: str = Field(..., min_length=1)
    weight: float = Field(default=1.0, ge=0.0)
    provenance: dict[str, str] = Field(default_factory=dict)
