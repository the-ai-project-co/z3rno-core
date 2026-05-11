"""Integration tests for Phase D slice 1 schema additions.

Covers Migration 023:
  * ``memories.memo_type`` + ``memories.ontology_uri`` accept values
    and round-trip through the SA model.
  * ``ix_memories_org_ontology_uri`` partial index exists.
  * ``feedback`` table CHECK constraints (signal range, exactly-one-of).
  * ``feedback`` RLS isolation across two orgs.

Skipped without DATABASE_URL.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from z3rno_core.models import (
    Agent,
    Feedback,
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


@pytest.fixture(scope="module")
def engine() -> Generator[Engine, None, None]:
    assert DATABASE_URL is not None
    eng = create_engine(DATABASE_URL)
    yield eng
    eng.dispose()


@contextmanager
def rls_context(eng: Engine, org_id: UUID) -> Generator[Connection, None, None]:
    with eng.connect() as conn:
        conn.execute(text("SET ROLE z3rno_app"))
        conn.execute(text(f"SET LOCAL app.current_org_id = '{org_id}'"))
        yield conn
        conn.rollback()


@pytest.fixture(scope="module")
def seed(engine: Engine) -> Generator[dict[str, UUID], None, None]:
    org_a, org_b = uuid4(), uuid4()
    agent_a, agent_b = uuid4(), uuid4()
    mem_a = uuid4()

    with Session(engine) as session:
        session.add(Tenant(org_id=org_a, name="Phase D Tenant A", plan_tier=PlanTier.PRO))
        session.add(Tenant(org_id=org_b, name="Phase D Tenant B", plan_tier=PlanTier.COMMUNITY))
        session.flush()
        session.add(Agent(id=agent_a, org_id=org_a, name="A", agent_metadata={}))
        session.add(Agent(id=agent_b, org_id=org_b, name="B", agent_metadata={}))
        session.flush()
        session.add(
            Memory(
                id=mem_a,
                org_id=org_a,
                agent_id=agent_a,
                memory_type=MemoryType.EPISODIC,
                content="Ada Lovelace was a mathematician.",
                memory_metadata={},
                memo_type="PERSON",
                ontology_uri="http://dbpedia.org/resource/Ada_Lovelace",
            )
        )
        session.commit()

    yield {
        "org_a": org_a,
        "org_b": org_b,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "memory_a": mem_a,
    }

    with engine.connect() as conn:
        for tbl in ("feedback", "memories", "agents", "tenants"):
            conn.execute(text(f"DELETE FROM {tbl} WHERE org_id IN ('{org_a}', '{org_b}')"))
        conn.commit()


# ---------------------------------------------------------------------------
# memories.memo_type + memories.ontology_uri
# ---------------------------------------------------------------------------


def test_memory_round_trips_memo_type_and_ontology_uri(
    engine: Engine, seed: dict[str, UUID]
) -> None:
    with Session(engine) as session:
        memory = session.get(Memory, seed["memory_a"])
        assert memory is not None
        assert memory.memo_type == "PERSON"
        assert memory.ontology_uri == "http://dbpedia.org/resource/Ada_Lovelace"


def test_memory_memo_type_defaults_null_for_pre_phase_d_rows(
    engine: Engine, seed: dict[str, UUID]
) -> None:
    """A Memory inserted without memo_type / ontology_uri stays NULL."""
    org = seed["org_a"]
    agent = seed["agent_a"]
    mid = uuid4()
    with Session(engine) as session:
        session.add(
            Memory(
                id=mid,
                org_id=org,
                agent_id=agent,
                memory_type=MemoryType.EPISODIC,
                content="no memo type",
                memory_metadata={},
            )
        )
        session.commit()
        m = session.get(Memory, mid)
        assert m is not None
        assert m.memo_type is None
        assert m.ontology_uri is None
        session.delete(m)
        session.commit()


def test_partial_index_on_ontology_uri_exists(engine: Engine) -> None:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE schemaname = 'public' "
                "AND indexname = 'ix_memories_org_ontology_uri'"
            )
        ).fetchone()
    assert row is not None
    assert "ontology_uri IS NOT NULL" in row[0]


# ---------------------------------------------------------------------------
# feedback CHECK constraints
# ---------------------------------------------------------------------------


def _insert_feedback(session: Session, **kwargs: object) -> None:
    """Flush a Feedback row in one statement so pytest.raises blocks stay tight."""
    session.add(Feedback(**kwargs))  # type: ignore[arg-type]
    session.flush()


def test_feedback_signal_out_of_range_rejected(engine: Engine, seed: dict[str, UUID]) -> None:
    with Session(engine) as session, pytest.raises(IntegrityError):
        _insert_feedback(
            session,
            org_id=seed["org_a"],
            agent_id=seed["agent_a"],
            memory_id=seed["memory_a"],
            signal=2,
        )


def test_feedback_requires_exactly_one_target(engine: Engine, seed: dict[str, UUID]) -> None:
    """Neither memory_id nor edge_id → rejected."""
    with Session(engine) as session, pytest.raises(IntegrityError):
        _insert_feedback(
            session,
            org_id=seed["org_a"],
            agent_id=seed["agent_a"],
            signal=1,
        )


def test_feedback_both_targets_rejected(engine: Engine, seed: dict[str, UUID]) -> None:
    """Both memory_id AND edge_id → rejected."""
    with Session(engine) as session, pytest.raises(IntegrityError):
        _insert_feedback(
            session,
            org_id=seed["org_a"],
            agent_id=seed["agent_a"],
            memory_id=seed["memory_a"],
            edge_id="e:1",
            signal=1,
        )


def test_feedback_round_trip(engine: Engine, seed: dict[str, UUID]) -> None:
    fid = uuid4()
    with Session(engine) as session:
        session.add(
            Feedback(
                id=fid,
                org_id=seed["org_a"],
                agent_id=seed["agent_a"],
                memory_id=seed["memory_a"],
                signal=1,
                reason="helpful answer",
            )
        )
        session.commit()
        fb = session.get(Feedback, fid)
        assert fb is not None
        assert fb.signal == 1
        assert fb.reason == "helpful answer"
        session.delete(fb)
        session.commit()


# ---------------------------------------------------------------------------
# feedback RLS isolation
# ---------------------------------------------------------------------------


def test_feedback_rls_isolation(engine: Engine, seed: dict[str, UUID]) -> None:
    """Org B cannot see feedback inserted under Org A."""
    fid = uuid4()
    with Session(engine) as session:
        session.add(
            Feedback(
                id=fid,
                org_id=seed["org_a"],
                agent_id=seed["agent_a"],
                memory_id=seed["memory_a"],
                signal=-1,
                reason="bad",
            )
        )
        session.commit()

    with rls_context(engine, seed["org_a"]) as conn:
        ids_a = {row[0] for row in conn.execute(text("SELECT id FROM feedback")).fetchall()}
        assert fid in ids_a

    with rls_context(engine, seed["org_b"]) as conn:
        ids_b = {row[0] for row in conn.execute(text("SELECT id FROM feedback")).fetchall()}
        assert fid not in ids_b

    with Session(engine) as session:
        fb = session.get(Feedback, fid)
        if fb is not None:
            session.delete(fb)
            session.commit()
