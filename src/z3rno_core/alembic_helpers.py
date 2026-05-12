"""v0.20.2 — run the bundled Alembic migration tree from inside a deployed wheel.

The migration tree (``migrations/``) and its ``alembic.ini`` ship inside
the wheel under ``z3rno_core/_alembic/`` via the hatchling
``force-include`` directive in ``pyproject.toml``. That sidesteps the
pre-v0.20.2 trap where downstream containers (z3rno-server image,
helm-managed deploys) shelled out to ``alembic upgrade head`` but the
container didn't ship the migrations directory — the entrypoint silently
swallowed the error and the database ended up on an older head than the
installed engine code expected.

Caller contract:

    from z3rno_core.alembic_helpers import upgrade_to_head

    # Sync DSN required — Alembic uses psycopg, not asyncpg.
    upgrade_to_head("postgresql+psycopg://user:pass@host:5432/z3rno")

Designed to fail loud, not silently — a missing migration directory,
an unreachable DB, or a failed upgrade all surface as exceptions for
the caller (typically the container entrypoint) to log and abort on.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Iterator


def _alembic_ini_path() -> Path:
    """Resolve ``z3rno_core/_alembic/alembic.ini`` from the installed wheel."""
    resource = files("z3rno_core").joinpath("_alembic", "alembic.ini")
    # ``files`` returns a Traversable; alembic.Config wants a real path.
    return Path(str(resource))


@contextmanager
def _env_override(database_url: str | None) -> Iterator[None]:
    if database_url is None:
        yield
        return
    prior = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prior


def upgrade_to_head(database_url: str | None = None) -> None:
    """Run ``alembic upgrade head`` against the bundled migration tree.

    Parameters
    ----------
    database_url
        Optional. When set, temporarily exported as ``DATABASE_URL`` so
        the bundled ``env.py`` picks it up. Must use the sync psycopg
        driver — convert any ``postgresql+asyncpg://`` URLs first
        (``url.replace("+asyncpg", "+psycopg")``).
    """
    # Lazy import — alembic is a relatively heavy module to load and
    # callers that never run migrations shouldn't pay for it.
    from alembic import command
    from alembic.config import Config

    ini_path = _alembic_ini_path()
    if not ini_path.is_file():
        raise RuntimeError(
            f"z3rno_core alembic.ini not found at {ini_path}. "
            "Did the wheel ship without the bundled migration tree? "
            "This is a packaging bug — please file an issue."
        )

    with _env_override(database_url):
        cfg = Config(str(ini_path))
        command.upgrade(cfg, "head")


def current_revision(database_url: str | None = None) -> str | None:
    """Return the current head revision applied to the database, or
    ``None`` if the ``alembic_version`` table is empty / missing."""
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine

    ini_path = _alembic_ini_path()
    if not ini_path.is_file():
        raise RuntimeError(f"alembic.ini not found at {ini_path}")

    with _env_override(database_url):
        # DATABASE_URL wins — alembic.ini ships with a localhost default
        # that's almost certainly wrong in production.
        url = os.environ.get("DATABASE_URL") or Config(str(ini_path)).get_main_option(
            "sqlalchemy.url"
        )
        if not url:
            raise RuntimeError(
                "DATABASE_URL not set and alembic.ini has no sqlalchemy.url"
            )
        engine = create_engine(url)
        try:
            with engine.begin() as conn:
                ctx = MigrationContext.configure(conn)
                return ctx.get_current_revision()
        finally:
            engine.dispose()
        # Reference unused import for type checkers — alembic.command kept
        # available for callers chaining `current_revision` → `upgrade_to_head`
        # without re-importing.
        _ = command  # noqa: F841
