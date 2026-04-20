"""z3rno-core — the memory engine for Z3rno.

This package contains the PostgreSQL schema, SQLAlchemy 2.0 models, Alembic
migrations, and the pure-Python `store / recall / forget / audit` engine
functions that power the Z3rno memory database.

It is designed to be imported as a library by `z3rno-server` (the FastAPI
REST server), not used directly by end users. End users talk to the server
via `z3rno-sdk-python` or `@z3rno/sdk`.

Subpackages
-----------

- ``models``    SQLAlchemy 2.0 declarative models for every table.
- ``engine``    Memory engine functions: store, recall, forget, audit.
- ``graph``     Apache AGE Cypher query builder and agtype decoder.
- ``temporal``  SCD Type 2 helpers for point-in-time queries.
- ``security``  RLS context helpers and API-key hashing.

See Also
--------

- Architecture: ``z3rno-process-docs/08-Architecture-Document.md``
- Tech stack:   ``z3rno-process-docs/09-Tech-Stack-Document.md``
- ADR-001:      ``docs/adr/001-embedding-model.md``
"""

from z3rno_core._version import __version__
from z3rno_core.config import ConnectionConfig, create_async_engine_from_config

__all__ = ["ConnectionConfig", "__version__", "create_async_engine_from_config"]
