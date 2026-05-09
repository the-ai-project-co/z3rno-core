"""Pydantic schemas for the *distill* stage of the Forge pipeline (Phase A).

These models serve two roles:

  1. **Instructor response targets.** ``LLMGateway.complete_structured(...)``
     uses them as the schema the LLM must conform to.
  2. **Internal data transport.** Downstream stages (graph_writer, audit
     stamping) accept these types directly — no looser dicts allowed.

Design rules
------------
* Every result class is a ``ConfigDict(frozen=True)`` Pydantic model so it
  hashes / compares structurally and can flow through async tasks safely.
* String fields are ``str`` (not ``str | None``) with sensible empty
  defaults. The Forge prefers "missing slot, empty string" over nullable
  noise.
* Confidence floats are constrained to ``[0.0, 1.0]``.
* Aliases / mention spans are optional; the LLM is allowed to omit them.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    """A canonical entity extracted from text.

    A future Phase D will attach an ``ontology_uri``; for Phase A the
    entity is identified solely by ``name`` + ``type``.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, description="Canonical surface form of the entity.")
    type: str = Field(
        default="thing",
        description="Coarse category (e.g. 'person', 'org', 'product', 'event').",
    )
    description: str = Field(default="", description="Free-text description, may be empty.")
    aliases: tuple[str, ...] = Field(
        default=(), description="Other surface forms seen for the same entity in the text."
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Relationship(BaseModel):
    """A directed relationship between two entities."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(
        ..., min_length=1, description="Source entity name (must match an Entity.name)."
    )
    target: str = Field(
        ..., min_length=1, description="Target entity name (must match an Entity.name)."
    )
    predicate: str = Field(
        ..., min_length=1, description="Relationship label, e.g. 'works_for', 'launched', 'owns'."
    )
    description: str = Field(default="")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Triplet(BaseModel):
    """A subject-predicate-object triplet.

    Used both as an LLM target (when extraction is triplet-first) and as
    a normalized form derivable from Entity + Relationship.
    """

    model_config = ConfigDict(frozen=True)

    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    obj: str = Field(..., min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class DistillResult(BaseModel):
    """The output of distilling one chunk (or a merged list of chunks).

    Carries enough provenance to write Memos with audit-chain entries
    pointing back to the source memory_id and char span.
    """

    model_config = ConfigDict(frozen=True)

    entities: tuple[Entity, ...] = ()
    relationships: tuple[Relationship, ...] = ()
    triplets: tuple[Triplet, ...] = ()
    summary: str = ""

    # Provenance — populated by the orchestrator, not the LLM.
    source_memory_id: UUID | None = None
    chunk_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    model: str = ""

    @property
    def is_empty(self) -> bool:
        return (
            not self.entities and not self.relationships and not self.triplets and not self.summary
        )

    def merge(self, other: DistillResult) -> DistillResult:
        """Combine two results into one.

        Entities are deduped by ``(name.lower(), type.lower())``; the
        higher-confidence copy wins. Relationships and triplets are
        deduped by their full natural key. Provenance fields from
        ``self`` are preserved (the merge target is canonical).
        """
        ent_index: dict[tuple[str, str], Entity] = {
            (e.name.lower(), e.type.lower()): e for e in self.entities
        }
        for e in other.entities:
            key = (e.name.lower(), e.type.lower())
            existing = ent_index.get(key)
            if existing is None or e.confidence > existing.confidence:
                ent_index[key] = e

        rel_index: dict[tuple[str, str, str], Relationship] = {
            (r.source.lower(), r.predicate.lower(), r.target.lower()): r for r in self.relationships
        }
        for r in other.relationships:
            key2 = (r.source.lower(), r.predicate.lower(), r.target.lower())
            existing_rel = rel_index.get(key2)
            if existing_rel is None or r.confidence > existing_rel.confidence:
                rel_index[key2] = r

        trip_index: dict[tuple[str, str, str], Triplet] = {
            (t.subject.lower(), t.predicate.lower(), t.obj.lower()): t for t in self.triplets
        }
        for t in other.triplets:
            key3 = (t.subject.lower(), t.predicate.lower(), t.obj.lower())
            existing_t = trip_index.get(key3)
            if existing_t is None or t.confidence > existing_t.confidence:
                trip_index[key3] = t

        return DistillResult(
            entities=tuple(ent_index.values()),
            relationships=tuple(rel_index.values()),
            triplets=tuple(trip_index.values()),
            summary=self.summary or other.summary,
            source_memory_id=self.source_memory_id,
            chunk_index=self.chunk_index,
            char_start=self.char_start,
            char_end=self.char_end,
            model=self.model or other.model,
        )
