"""Development seed data for z3rno-core.

Populates a fresh database with realistic-looking test data:
  - 2 tenants (Acme Corp on pro, Globex on community)
  - 3 agents per tenant (6 total)
  - 500 memories (working: 50, episodic: 200, semantic: 200, procedural: 50)
  - 100 memory relationships
  - 1,000 audit log entries

Embeddings are random 1536-dim unit vectors (fast, no API call needed).
For realistic embeddings, see seeds/realistic_seed.py.

Usage:
    DATABASE_URL=postgresql+psycopg://z3rno:z3rno@localhost:5432/z3rno \
        uv run python -m seeds.dev_seed
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# Ensure src/ is on the path for z3rno_core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from z3rno_core.models import (
    Agent,
    ApiKey,
    AuditOperation,
    LifecyclePolicy,
    Memory,
    MemoryRelationship,
    MemoryType,
    RelationshipType,
    Tenant,
)
from z3rno_core.models.enums import DecayCurve, PlanTier, RetentionPolicy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://z3rno:z3rno@localhost:5432/z3rno",
)

RNG = random.Random(42)
NP_RNG = np.random.default_rng(42)

MEMORY_TYPE_DISTRIBUTION = {
    MemoryType.WORKING: 50,
    MemoryType.EPISODIC: 200,
    MemoryType.SEMANTIC: 200,
    MemoryType.PROCEDURAL: 50,
}

SAMPLE_CONTENTS = {
    MemoryType.WORKING: [
        "User is currently asking about quarterly revenue figures.",
        "Active conversation about deployment pipeline issues.",
        "User mentioned they prefer dark mode UI.",
        "Currently debugging a null pointer in the auth module.",
        "Discussing API rate limiting strategy for the free tier.",
    ],
    MemoryType.EPISODIC: [
        "User reported a login failure on 2026-03-15. Root cause was expired JWT.",
        "Team standup: decided to migrate from REST to gRPC for internal services.",
        "User asked about data export compliance for GDPR. Provided DPA template.",
        "Deployment rollback occurred due to memory leak in v2.3.1.",
        "Customer escalation: response time exceeded SLA by 200ms during peak.",
        "User onboarding session completed. Configured SSO with Okta.",
        "Sprint retro: team agreed to add automated regression tests.",
        "Incident post-mortem: database failover took 45s instead of expected 10s.",
    ],
    MemoryType.SEMANTIC: [
        "PostgreSQL HNSW indexes support approximate nearest neighbor search.",
        "GDPR Article 17 grants the right to erasure (right to be forgotten).",
        "The company uses a microservices architecture with 12 core services.",
        "Authentication flow: API key -> BCrypt verify -> JWT issue -> Redis session.",
        "Memory decay follows an exponential curve: score *= e^(-rate * days).",
        "Tenant isolation is enforced via Row-Level Security on org_id.",
        "Vector similarity search uses cosine distance for semantic matching.",
        "The four memory types map to cognitive science: working, episodic, semantic, procedural.",
    ],
    MemoryType.PROCEDURAL: [
        "To deploy: run `make deploy ENV=staging`, wait for healthcheck, then promote.",
        "Database backup procedure: pg_dump with --format=custom, upload to S3.",
        "When importance_score < 0.1, auto-forget the memory after summarisation.",
        "API key rotation: generate new key, update client, revoke old key within 24h.",
        "To run migrations: `make migrate` (wraps alembic upgrade head).",
    ],
}


def random_unit_vector(dim: int = 1536) -> list[float]:
    """Generate a random unit vector (L2 norm = 1)."""
    vec = NP_RNG.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()  # type: ignore[no-any-return]


def random_datetime(start: datetime, end: datetime) -> datetime:
    """Random datetime between start and end."""
    delta = end - start
    random_seconds = RNG.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def compute_row_hash(prev_hash: bytes | None, data: str) -> bytes:
    """Simple SHA-256 hash for seed data audit chain."""
    h = hashlib.sha256()
    if prev_hash:
        h.update(prev_hash)
    h.update(data.encode())
    return h.digest()


def seed(session: Session) -> None:
    """Populate the database with dev seed data."""
    now = datetime.now(UTC)
    month_ago = now - timedelta(days=30)

    # ---- Tenants ----
    tenants = [
        Tenant(
            org_id=uuid4(),
            name="Acme Corp",
            plan_tier=PlanTier.PRO,
            settings={"embedding_model": "text-embedding-3-small", "embedding_dimension": 1536},
        ),
        Tenant(
            org_id=uuid4(),
            name="Globex Inc",
            plan_tier=PlanTier.COMMUNITY,
            settings={"embedding_model": "text-embedding-3-small"},
        ),
    ]
    session.add_all(tenants)
    session.flush()
    print(f"  Created {len(tenants)} tenants")

    # ---- Agents (3 per tenant) ----
    agent_names = ["Customer Support Bot", "Code Assistant", "Analytics Agent"]
    agents: list[Agent] = []
    for tenant in tenants:
        for i, name in enumerate(agent_names):
            agent = Agent(
                id=uuid4(),
                org_id=tenant.org_id,
                external_id=f"agent-{i + 1}",
                name=name,
                agent_metadata={"version": "1.0", "capabilities": ["chat", "search"]},
            )
            agents.append(agent)
    session.add_all(agents)
    session.flush()
    print(f"  Created {len(agents)} agents")

    # ---- API Keys (1 per tenant) ----
    api_keys = []
    for tenant in tenants:
        api_key = ApiKey(
            id=uuid4(),
            org_id=tenant.org_id,
            prefix="z3r_" + RNG.randbytes(4).hex(),
            key_hash=hashlib.sha256(RNG.randbytes(32)).digest(),  # Not real BCrypt, just seed data
            name="Development Key",
        )
        api_keys.append(api_key)
    session.add_all(api_keys)
    session.flush()
    print(f"  Created {len(api_keys)} API keys")

    # ---- Lifecycle Policies (1 per memory type per tenant) ----
    policies = []
    for tenant in tenants:
        for mem_type in MemoryType:
            policy = LifecyclePolicy(
                id=uuid4(),
                org_id=tenant.org_id,
                memory_type=mem_type,
                max_ttl_days=7 if mem_type == MemoryType.WORKING else None,
                max_count=100 if mem_type == MemoryType.WORKING else None,
                min_importance=0.1,
                decay_enabled=True,
                decay_rate=0.05 if mem_type == MemoryType.WORKING else 0.01,
                decay_curve=DecayCurve.EXPONENTIAL,
                decay_floor=0.1,
                retention_policy=RetentionPolicy.STANDARD,
                summarize_before_delete=mem_type != MemoryType.WORKING,
            )
            policies.append(policy)
    session.add_all(policies)
    session.flush()
    print(f"  Created {len(policies)} lifecycle policies")

    # ---- Memories ----
    memories: list[Memory] = []
    for mem_type, count in MEMORY_TYPE_DISTRIBUTION.items():
        contents = SAMPLE_CONTENTS[mem_type]
        for _ in range(count):
            tenant = RNG.choice(tenants)
            tenant_agents = [a for a in agents if a.org_id == tenant.org_id]
            agent = RNG.choice(tenant_agents)
            created = random_datetime(month_ago, now)

            memory = Memory(
                id=uuid4(),
                org_id=tenant.org_id,
                agent_id=agent.id,
                user_id=uuid4() if RNG.random() > 0.3 else None,
                memory_type=mem_type,
                content=RNG.choice(contents),
                summary=None,
                memory_metadata={"source": "seed", "tags": [mem_type.value]},
                embedding=random_unit_vector(),
                embedding_model="text-embedding-3-small",
                importance_score=round(RNG.uniform(0.1, 1.0), 3),
                recall_count=RNG.randint(0, 20),
                last_recalled_at=created + timedelta(hours=RNG.randint(1, 48))
                if RNG.random() > 0.5
                else None,
                valid_from=created,
                valid_to=None,  # All current
                pinned=RNG.random() > 0.95,
                ttl_expires_at=created + timedelta(days=7)
                if mem_type == MemoryType.WORKING
                else None,
                deleted_at=None,
                quarantined=False,
                anomaly_score=0.0,
            )
            memories.append(memory)
    session.add_all(memories)
    session.flush()
    print(f"  Created {len(memories)} memories")

    # ---- Memory Relationships (100) ----
    relationships = []
    for _ in range(100):
        # Pick two different memories from the same tenant
        tenant = RNG.choice(tenants)
        tenant_memories = [m for m in memories if m.org_id == tenant.org_id]
        if len(tenant_memories) < 2:
            continue
        src, tgt = RNG.sample(tenant_memories, 2)
        rel = MemoryRelationship(
            id=uuid4(),
            org_id=tenant.org_id,
            source_memory_id=src.id,
            target_memory_id=tgt.id,
            relationship_type=RNG.choice(list(RelationshipType)),
            weight=round(RNG.uniform(0.3, 1.0), 3),
            relationship_metadata={"source": "seed"},
        )
        relationships.append(rel)
    session.add_all(relationships)
    session.flush()
    print(f"  Created {len(relationships)} memory relationships")

    # ---- Audit Log (1000 entries) ----
    operations = list(AuditOperation)
    # Weight towards store/recall which are the most common
    op_weights = [0.3, 0.3, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02, 0.05]
    prev_hash: bytes | None = None

    # Use raw SQL for audit_log since the partitioned table needs special handling
    audit_entries = []
    for i in range(1000):
        tenant = RNG.choice(tenants)
        tenant_agents_ids = [a.id for a in agents if a.org_id == tenant.org_id]
        tenant_memory_ids = [m.id for m in memories if m.org_id == tenant.org_id]
        op = RNG.choices(operations, weights=op_weights, k=1)[0]
        created = random_datetime(month_ago, now)

        row_data = f"{tenant.org_id}:{op.value}:{i}"
        row_hash = compute_row_hash(prev_hash, row_data)

        audit_entries.append(
            {
                "org_id": str(tenant.org_id),
                "agent_id": str(RNG.choice(tenant_agents_ids)),
                "user_id": str(uuid4()) if RNG.random() > 0.3 else None,
                "operation": op.value,
                "memory_id": str(RNG.choice(tenant_memory_ids)) if tenant_memory_ids else None,
                "memory_type": RNG.choice(list(MemoryType)).value if RNG.random() > 0.2 else None,
                "details": f'{{"seed_index": {i}}}',
                "prev_hash": prev_hash,
                "row_hash": row_hash,
                "created_at": created.isoformat(),
                "updated_at": created.isoformat(),
            }
        )
        prev_hash = row_hash

    # Batch insert via raw SQL for partitioned table
    for entry in audit_entries:
        session.execute(
            text("""
                INSERT INTO audit_log (org_id, agent_id, user_id, operation, memory_id,
                    memory_type, details, prev_hash, row_hash, created_at, updated_at)
                VALUES (
                    CAST(:org_id AS uuid),
                    CAST(:agent_id AS uuid),
                    CAST(:user_id AS uuid),
                    CAST(:operation AS audit_operation_enum),
                    CAST(:memory_id AS uuid),
                    CAST(:memory_type AS memory_type_enum),
                    CAST(:details AS jsonb),
                    :prev_hash,
                    :row_hash,
                    CAST(:created_at AS timestamptz),
                    CAST(:updated_at AS timestamptz)
                )
            """),
            entry,
        )
    session.flush()
    print(f"  Created {len(audit_entries)} audit log entries")

    session.commit()
    print("\nSeed complete!")


def main() -> None:
    print(f"Connecting to: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        seed(session)


if __name__ == "__main__":
    main()
