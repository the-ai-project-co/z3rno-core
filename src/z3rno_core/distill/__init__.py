"""z3rno_core.distill — LLM-driven knowledge extraction (Phase A).

This package implements the *distill* stage of the **Forge** pipeline. It
takes raw text and produces typed graph artifacts (entities, relationships,
triplets, summaries) using a provider-agnostic LLM Gateway.

Modules
-------

- ``llm_gateway`` — provider-agnostic LLM completion + structured output
- ``schemas``     — Pydantic models for distillation results
- ``extract``     — entity / relationship extraction from text chunks
- ``summarize``   — per-document and rolling summarization
- ``graph_writer`` — persists distillation results to Postgres + AGE + audit chain

Phase A surface is **opt-in**: every public verb is a no-op unless
``DISTILL_ENABLED=true`` in z3rno-server settings.

See ``z3rno-process-docs/improvements/plans/03-phase-a-extraction-layer.md`` for
the full Phase A design and acceptance criteria.
"""

from __future__ import annotations

from z3rno_core.distill.extract import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_extraction_prompts,
    extract_from_chunk,
    extract_from_chunks,
)
from z3rno_core.distill.graph_writer import (
    WriteResult,
    already_distilled,
    insert_distill_job,
    update_distill_job,
    write_distill_result,
)
from z3rno_core.distill.llm_gateway import (
    LiteLLMGateway,
    LLMGateway,
    LLMGatewayError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMValidationError,
    StubLLMGateway,
    get_llm_gateway,
)
from z3rno_core.distill.schemas import (
    DistillResult,
    Entity,
    Relationship,
    Triplet,
)
from z3rno_core.distill.provenance import (
    ChainVerdict,
    ProvenanceRequiredError,
    build_provenance_blob,
    stamp_provenance,
    verify_chain,
)
from z3rno_core.distill.summarize import (
    SummaryStyle,
    rolling_summarize,
    summarize_text,
)

__all__ = [
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "ChainVerdict",
    "DistillResult",
    "Entity",
    "LLMGateway",
    "LLMGatewayError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMValidationError",
    "LiteLLMGateway",
    "ProvenanceRequiredError",
    "Relationship",
    "StubLLMGateway",
    "SummaryStyle",
    "Triplet",
    "WriteResult",
    "already_distilled",
    "build_extraction_prompts",
    "build_provenance_blob",
    "extract_from_chunk",
    "extract_from_chunks",
    "get_llm_gateway",
    "insert_distill_job",
    "rolling_summarize",
    "stamp_provenance",
    "summarize_text",
    "update_distill_job",
    "verify_chain",
    "write_distill_result",
]
