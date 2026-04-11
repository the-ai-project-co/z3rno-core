# Research Documents

This directory holds research artifacts — investigations, design explorations, and pattern catalogs that inform the Z3rno implementation but are not themselves specifications.

## Research vs ADRs vs specs

| Type | Purpose | Lifecycle | Lives in |
|---|---|---|---|
| **Research** | Explore the design space, document patterns, compare alternatives, surface gotchas | Mutable; updated as we learn more | `docs/research/` |
| **ADR** | Lock in a single decision with rationale | Immutable once Accepted; superseded via new ADR | `docs/adr/` |
| **Spec** | Document the implementation contract (schema, API, protocol) | Mutable, version-controlled with the code | `docs/SCHEMA.md`, `docs/MULTI_TENANCY.md`, etc. |

A research doc may inform an ADR (e.g. `apache-age-graph-patterns.md` informs the Week 1 Friday graph schema decision). An ADR may then inform a spec (e.g. ADR-001 informs the `embedding vector(1536)` line in the schema).

## When to write a research doc

- You're exploring how to solve a problem and don't yet know the right answer
- You want to document patterns and gotchas so future-you doesn't re-discover them
- You need a place to put benchmark results, prototype findings, comparison tables
- You want to gather options before writing an ADR

## When NOT to write a research doc

- You already know the answer — write an ADR
- It's a one-line decision — put it in a code comment
- It's a runbook or how-to — that's docs, not research

## Index

| File | Topic | Last Updated |
|---|---|---|
| [apache-age-graph-patterns.md](apache-age-graph-patterns.md) | openCypher patterns Z3rno will use for memory relationships, AGE gotchas, performance trade-offs | 2026-04-11 |

When you add a new research doc, add it to this table.
