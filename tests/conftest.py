"""Shared pytest fixtures for the z3rno-core test suite.

Provides a session-scoped PostgreSQL container via testcontainers for
integration tests.  Unit tests continue to work without any database.

The container uses ``pgvector/pgvector:pg17`` which ships with pgvector
and pgcrypto but **not** Apache AGE, vectorscale, pg_cron, or pgaudit.
Migrations that depend on those extensions (001 extensions, 010 graph
schema) are handled specially: we create only the available extensions
and stamp the AGE migration without executing it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module-level state for the container managed by pytest hooks
# ---------------------------------------------------------------------------
_container: Any = None
_container_url: str | None = None


def _testcontainers_available() -> bool:
    """Check whether the testcontainers package is installed."""
    try:
        import testcontainers.postgres  # noqa: F401
    except ImportError:
        return False
    else:
        return True


def _docker_available() -> bool:
    """Check whether Docker is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"],  # noqa: S607
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    else:
        return result.returncode == 0


def _should_start_container() -> bool:
    """Determine if we need to spin up a testcontainers PostgreSQL instance."""
    # Already have a database URL -- no need for a container
    if os.environ.get("DATABASE_URL"):
        return False
    if not _testcontainers_available():
        return False
    return _docker_available()


def _run_migrations(db_url: str) -> None:
    """Apply Alembic migrations against the test database.

    Strategy (to work around unavailable extensions in the pgvector image):
      1. Create pgcrypto + vector extensions manually.
      2. Stamp migration 001 (extension creation) so Alembic skips it.
      3. Run migrations 002 -> 009 normally.
      4. Stamp migration 010 (AGE graph schema) -- AGE is not available.
      5. Run migrations 011 -> head normally.
    """
    from sqlalchemy import create_engine, text

    # Create extensions available in pgvector/pgvector:pg17 ----------------
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    engine.dispose()

    # Run Alembic -----------------------------------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alembic_cmd = [sys.executable, "-m", "alembic"]
    env = {**os.environ, "DATABASE_URL": db_url}

    def _alembic(*args: str) -> None:
        result = subprocess.run(  # noqa: S603
            [*alembic_cmd, *args],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            msg = (
                f"Alembic {' '.join(args)} failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            raise RuntimeError(msg)

    _alembic("stamp", "001")  # skip extension creation
    _alembic("upgrade", "009")  # 002 -> 009
    _alembic("stamp", "010")  # skip AGE graph schema
    _alembic("upgrade", "head")  # 011 -> head


# ---------------------------------------------------------------------------
# pytest hooks -- start/stop the container around the whole session
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Start a PostgreSQL testcontainer before test collection.

    By running here (before collection), the ``DATABASE_URL`` env-var is
    set in time for the ``skipif(DATABASE_URL is None)`` guards in the
    integration test modules to evaluate correctly.
    """
    global _container, _container_url  # noqa: PLW0603

    if not _should_start_container():
        return

    from testcontainers.postgres import PostgresContainer

    pg = PostgresContainer(
        image="pgvector/pgvector:pg17",
        username="z3rno_test",
        password="z3rno_test",
        dbname="z3rno_test",
        driver="psycopg",
    ).with_bind_ports(5432, 0)

    pg.start()
    _container = pg

    db_url = pg.get_connection_url()
    _container_url = db_url

    # Run migrations
    _run_migrations(db_url)

    # Export for downstream tests
    os.environ["DATABASE_URL"] = db_url


def pytest_unconfigure(config: pytest.Config) -> None:
    """Stop the PostgreSQL testcontainer after all tests have finished."""
    global _container, _container_url  # noqa: PLW0603

    if _container is not None:
        _container.stop()
        _container = None
        _container_url = None
        os.environ.pop("DATABASE_URL", None)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sentinel() -> str:
    """Trivial fixture proving conftest.py is loaded."""
    return "z3rno-core test suite"
