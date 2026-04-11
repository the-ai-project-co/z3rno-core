# Architecture Decision Records (ADRs)

This directory holds the canonical record of every significant technical decision made on `z3rno-core` and the broader Z3rno stack. ADRs are how we keep institutional memory alive — six months from now, a new contributor (or one of the founders) will read these to understand *why* a decision was made, not just *what* the current state is.

## Format

We follow the [Michael Nygard ADR template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions). Every ADR has these sections:

```markdown
# {NNN}: {Short title in present tense}

**Status:** Proposed | Accepted | Deprecated | Superseded by ADR-XYZ
**Date:** YYYY-MM-DD
**Deciders:** F1, F2, F3 (or specific owners)
**Tags:** {comma-separated topical tags}

## Context

What is the problem? What constraints exist? What forces are pushing in
different directions? Why are we even having this conversation?

## Decision

The decision in one or two sentences, present tense, active voice.
"We will use X." not "We considered using X."

## Consequences

What becomes easier or harder as a result of this decision?
List positive, negative, and neutral consequences. Be honest about trade-offs.

## Alternatives Considered

Each alternative gets its own subsection with:
- Brief description
- Why we rejected it (or why we'd reconsider it later)

## References

Links to relevant docs, PRs, benchmarks, prior art.
```

## Lifecycle

ADRs are **immutable once Accepted**. If a decision changes, you do NOT edit the existing ADR — you write a **new ADR** that supersedes it, and update the old one's Status to `Superseded by ADR-NNN`. This preserves the historical reasoning so future readers can see "we used to do X for these reasons, but then Y changed and we switched to Z."

The only edits allowed to an Accepted ADR are:
- Status changes (Accepted → Deprecated, Accepted → Superseded)
- Typo fixes
- Adding cross-references to newer ADRs

## Numbering

ADRs are numbered sequentially in three-digit zero-padded format (`001`, `002`, ..., `099`, `100`, ...). Numbers are never reused, even if an ADR is superseded.

## When to write an ADR

Write an ADR when:

- You picked one technology over alternatives that someone might reasonably choose differently
- You committed to a specific schema, protocol, or interface that's hard to change later
- You decided **not** to do something that seems obvious (e.g. "we will not support multi-region writes in v1")
- You established a convention that contributors need to follow
- Future-you will ask "why did we do it this way?"

Don't write an ADR for:

- Obvious choices (e.g. "we will use git for version control")
- Implementation details that can be changed without external impact
- Style preferences that have a linter rule
- Things already documented in `docs/SCHEMA.md`, `docs/MULTI_TENANCY.md`, etc.

## Index

| # | Title | Status | Date |
|---|---|---|---|
| [001](001-embedding-model.md) | Embedding model: OpenAI text-embedding-3-small (1536 dims) | Accepted | 2026-04-11 |

When you write a new ADR, add it to this table.
