# z3rno-core

> The core memory engine for Z3rno. PostgreSQL schema, SQLAlchemy models, Alembic migrations, and the `store` / `recall` / `forget` / `audit` library functions that power the `z3rno-server` REST API.

**License:** Apache 2.0
**Status:** Early development — not yet on PyPI
**Part of:** [Z3rno](https://github.com/the-ai-project-co) — the database for AI agent memory

## What this is

`z3rno-core` is a Python library that contains the authoritative PostgreSQL schema for the Z3rno memory database, the SQLAlchemy 2.0 models that back it, the Alembic migrations that evolve it, and the pure-Python engine functions that implement the four core operations: `store`, `recall`, `forget`, `audit`.

It is designed to be imported as a library by `z3rno-server` (the FastAPI REST API), not used directly by end users. End users talk to the server via `z3rno-sdk-python` or `z3rno-sdk-typescript`.

## What this is not

- Not a web server (that is `z3rno-server`).
- Not an SDK (those are `z3rno-sdk-python` and `z3rno-sdk-typescript`).
- Not a database driver — it uses `asyncpg` + `SQLAlchemy` under the hood.

## Why

Every AI agent needs persistent memory. Existing solutions are fragmented (Mem0 is paywalled, Letta is framework-locked, Zep is immature). Z3rno is the purpose-built, PostgreSQL-native, Apache-2.0 answer — combining vector similarity (pgvector), graph relationships (Apache AGE), and temporal versioning (SCD Type 2) in one unified engine.

See the architecture doc for the full rationale.
