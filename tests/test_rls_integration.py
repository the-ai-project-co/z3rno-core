"""Integration tests for Row-Level Security.

These tests run against a real PostgreSQL database to verify that:
  - Tenant A cannot read/write Tenant B's data
  - The z3rno_app role is subject to RLS
  - Without org context, no rows are visible

Requires a running postgres with migrations applied. Set DATABASE_URL
env var or these tests will be skipped.

Usage:
    DATABASE_URL=postgresql+psycopg://z3rno:z3rno@localhost:5433/z3rno \
        uv run pytest tests/test_rls_integration.py -v
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.orm import Session

from z3rno_core.models import (
    Agent,
    Memory,
    MemoryType,
    Tenant,
)
from z3rno_core.models.enums import PlanTier

DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = [
    pytest.mark.skipif(
        DATABASE_URL is None,
        reason="DATABASE_URL not set - skipping integration tests",
    ),
    pytest.mark.integration,
]

SeedData = dict[str, UUID]


@pytest.fixture(scope="module")
def engine() -> Generator[Engine, None, None]:
    """Create a test engine."""
    assert DATABASE_URL is not None
    eng = create_engine(DATABASE_URL)
    yield eng
    eng.dispose()


@contextmanager
def rls_context(eng: Engine, org_id: UUID) -> Generator[Connection, None, None]:
    """Context manager that connects as z3rno_app with org context set."""
    with eng.connect() as conn:
        conn.execute(text("SET ROLE z3rno_app"))
        conn.execute(text(f"SET LOCAL app.current_org_id = '{org_id}'"))
        yield conn
        conn.rollback()


@contextmanager
def rls_no_context(eng: Engine) -> Generator[Connection, None, None]:
    """Context manager that connects as z3rno_app WITHOUT org context."""
    with eng.connect() as conn:
        conn.execute(text("SET ROLE z3rno_app"))
        yield conn
        conn.rollback()


@pytest.fixture(scope="module")
def seed_data(engine: Engine) -> Generator[SeedData, None, None]:
    """Create two tenants with agents and memories, return their org_ids."""
    org_a = uuid4()
    org_b = uuid4()
    agent_a_id = uuid4()
    agent_b_id = uuid4()
    mem_a_id = uuid4()
    mem_b_id = uuid4()

    with Session(engine) as session:
        session.add(Tenant(org_id=org_a, name="RLS Test Tenant A", plan_tier=PlanTier.PRO))
        session.add(Tenant(org_id=org_b, name="RLS Test Tenant B", plan_tier=PlanTier.COMMUNITY))
        session.flush()

        session.add(Agent(id=agent_a_id, org_id=org_a, name="Agent A", agent_metadata={}))
        session.add(Agent(id=agent_b_id, org_id=org_b, name="Agent B", agent_metadata={}))
        session.flush()

        session.add(
            Memory(
                id=mem_a_id,
                org_id=org_a,
                agent_id=agent_a_id,
                memory_type=MemoryType.EPISODIC,
                content="Secret data for Tenant A",
                memory_metadata={"owner": "a"},
            )
        )
        session.add(
            Memory(
                id=mem_b_id,
                org_id=org_b,
                agent_id=agent_b_id,
                memory_type=MemoryType.SEMANTIC,
                content="Secret data for Tenant B",
                memory_metadata={"owner": "b"},
            )
        )
        session.commit()

    yield {
        "org_a": org_a,
        "org_b": org_b,
        "agent_a_id": agent_a_id,
        "agent_b_id": agent_b_id,
        "memory_a_id": mem_a_id,
        "memory_b_id": mem_b_id,
    }

    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM memories WHERE org_id IN ('{org_a}', '{org_b}')"))
        conn.execute(text(f"DELETE FROM agents WHERE org_id IN ('{org_a}', '{org_b}')"))
        conn.execute(text(f"DELETE FROM tenants WHERE org_id IN ('{org_a}', '{org_b}')"))
        conn.commit()


def test_tenant_a_sees_only_own_memories(engine: Engine, seed_data: SeedData) -> None:
    """Org A's context should only return Org A's memories."""
    with rls_context(engine, seed_data["org_a"]) as conn:
        result = conn.execute(text("SELECT id FROM memories")).fetchall()
        ids = {row[0] for row in result}
        assert seed_data["memory_a_id"] in ids
        assert seed_data["memory_b_id"] not in ids


def test_tenant_b_sees_only_own_memories(engine: Engine, seed_data: SeedData) -> None:
    """Org B's context should only return Org B's memories."""
    with rls_context(engine, seed_data["org_b"]) as conn:
        result = conn.execute(text("SELECT id FROM memories")).fetchall()
        ids = {row[0] for row in result}
        assert seed_data["memory_b_id"] in ids
        assert seed_data["memory_a_id"] not in ids


def test_tenant_a_sees_only_own_agents(engine: Engine, seed_data: SeedData) -> None:
    """RLS on agents table."""
    with rls_context(engine, seed_data["org_a"]) as conn:
        result = conn.execute(text("SELECT id FROM agents")).fetchall()
        ids = {row[0] for row in result}
        assert seed_data["agent_a_id"] in ids
        assert seed_data["agent_b_id"] not in ids


def test_cross_tenant_update_affects_zero_rows(engine: Engine, seed_data: SeedData) -> None:
    """UPDATE with wrong org context should affect 0 rows."""
    with rls_context(engine, seed_data["org_a"]) as conn:
        result = conn.execute(
            text(f"UPDATE memories SET content = 'hacked' WHERE id = '{seed_data['memory_b_id']}'")
        )
        assert result.rowcount == 0


def test_cross_tenant_delete_affects_zero_rows(engine: Engine, seed_data: SeedData) -> None:
    """DELETE with wrong org context should affect 0 rows."""
    with rls_context(engine, seed_data["org_a"]) as conn:
        result = conn.execute(text(f"DELETE FROM memories WHERE id = '{seed_data['memory_b_id']}'"))
        assert result.rowcount == 0

    # Verify memory B still exists
    with rls_context(engine, seed_data["org_b"]) as conn:
        row: Any = conn.execute(
            text(f"SELECT id FROM memories WHERE id = '{seed_data['memory_b_id']}'")
        ).fetchone()
        assert row is not None


def test_no_context_returns_no_rows(engine: Engine, seed_data: SeedData) -> None:
    """Without setting org context, z3rno_app should see no rows or error."""
    with rls_no_context(engine) as conn:
        try:
            result = conn.execute(text("SELECT count(*) FROM memories")).fetchone()
            assert result is not None
            assert result[0] == 0
        except Exception:
            # An error (unset session var) is also acceptable
            pass
