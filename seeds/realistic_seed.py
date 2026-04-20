"""Realistic seed data with actual embeddings from OpenAI.

Generates embeddings by calling text-embedding-3-small via LiteLLM.
Produces a smaller dataset (50 memories) but with real semantic vectors,
enabling meaningful recall quality testing.

Requires:
    OPENAI_API_KEY environment variable set.

Usage:
    OPENAI_API_KEY=sk-... \
    DATABASE_URL=postgresql+psycopg://z3rno:z3rno@localhost:5432/z3rno \
        uv run python -m seeds.realistic_seed
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# Ensure src/ is on the path for z3rno_core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from z3rno_core.models import (
    Agent,
    Memory,
    MemoryRelationship,
    MemoryType,
    RelationshipType,
    Tenant,
)
from z3rno_core.models.enums import PlanTier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://z3rno:z3rno@localhost:5432/z3rno",
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# Realistic memory content — designed so recall queries return meaningful
# results with semantic similarity. Grouped by topic clusters.
# ---------------------------------------------------------------------------

MEMORIES: list[dict[str, str]] = [
    # --- Cluster 1: Authentication & Security ---
    {
        "type": "semantic",
        "content": "JWT tokens expire after 24 hours. Refresh tokens are valid for 30 days and stored in HttpOnly cookies.",
    },
    {
        "type": "semantic",
        "content": "API key authentication uses BCrypt hashing with a cost factor of 12. Keys are stored as salted hashes, never plaintext.",
    },
    {
        "type": "procedural",
        "content": "To rotate an API key: generate a new key via POST /v1/api-keys, update the client configuration, verify the new key works, then revoke the old key within 24 hours.",
    },
    {
        "type": "episodic",
        "content": "On 2026-03-15, a customer reported unauthorized access to their account. Investigation revealed the JWT secret had been committed to a public GitHub repository. Immediate remediation: rotated all secrets, invalidated all sessions, enabled mandatory 2FA.",
    },
    {
        "type": "working",
        "content": "User is asking about implementing OAuth2 PKCE flow for their mobile application integration.",
    },
    # --- Cluster 2: Database & Performance ---
    {
        "type": "semantic",
        "content": "PostgreSQL HNSW indexes use a hierarchical navigable small world graph for approximate nearest neighbor search. Parameters m=16 and ef_construction=200 provide a good balance of recall and build speed.",
    },
    {
        "type": "semantic",
        "content": "Row-Level Security in PostgreSQL enforces tenant isolation at the database level. Each query is filtered by org_id using current_setting('app.current_org_id').",
    },
    {
        "type": "procedural",
        "content": "To run database migrations: execute 'make migrate' which wraps 'alembic upgrade head'. Always run migrations before deploying a new server version.",
    },
    {
        "type": "episodic",
        "content": "During the March load test, the database connection pool was exhausted at 500 concurrent connections. We increased pool_size from 20 to 50 and added PgBouncer as a connection multiplexer.",
    },
    {
        "type": "episodic",
        "content": "The weekly database vacuum job failed on 2026-03-22 because the disk was 95% full. We cleared old WAL files and increased the disk from 100GB to 500GB.",
    },
    # --- Cluster 3: Machine Learning & Embeddings ---
    {
        "type": "semantic",
        "content": "Text embeddings convert natural language into dense vector representations. OpenAI text-embedding-3-small produces 1536-dimensional vectors optimized for cosine similarity search.",
    },
    {
        "type": "semantic",
        "content": "Memory importance scoring uses a weighted combination of recency (30%), access frequency (30%), and explicit user rating (40%). The decay function follows an exponential curve.",
    },
    {
        "type": "semantic",
        "content": "Retrieval-augmented generation (RAG) combines vector similarity search with a language model to generate contextually relevant responses. The memory engine provides the retrieval component.",
    },
    {
        "type": "procedural",
        "content": "To benchmark embedding quality: generate embeddings for a test corpus, compute recall@10 against ground truth, compare across models. Target recall@10 > 0.95 for production use.",
    },
    {
        "type": "episodic",
        "content": "We evaluated three embedding models in March 2026: OpenAI text-embedding-3-small (1536d, $0.02/1M tokens), Cohere embed-v3 (1024d, $0.01/1M tokens), and nomic-embed-text (768d, free). OpenAI won on quality; Cohere was close but different dimension.",
    },
    # --- Cluster 4: Customer Support Scenarios ---
    {
        "type": "episodic",
        "content": "Customer Acme Corp reported that their agent's memory was returning irrelevant results. Root cause: embedding model mismatch between store and recall operations. Fixed by standardizing on text-embedding-3-small.",
    },
    {
        "type": "episodic",
        "content": "New customer onboarding for Globex Inc completed on 2026-03-10. They are using the Python SDK with LangChain integration. Primary use case: customer support chatbot with memory.",
    },
    {
        "type": "semantic",
        "content": "The four memory types in Z3rno map to cognitive science: working memory (active context, short-lived), episodic memory (event records), semantic memory (factual knowledge), and procedural memory (how-to instructions).",
    },
    {
        "type": "working",
        "content": "User is troubleshooting why their recall queries return zero results. Likely causes: empty embedding column, wrong memory_type filter, or RLS context not set.",
    },
    {
        "type": "working",
        "content": "Customer is requesting a bulk import API to migrate 50,000 memories from their legacy system to Z3rno.",
    },
    # --- Cluster 5: Infrastructure & DevOps ---
    {
        "type": "semantic",
        "content": "The Z3rno deployment architecture uses Kubernetes with CloudNativePG for PostgreSQL operator management. Helm charts are available for self-hosting.",
    },
    {
        "type": "procedural",
        "content": "To deploy to staging: run 'make deploy ENV=staging', wait for the health check endpoint to return 200, verify key metrics in Grafana, then promote to production with 'make deploy ENV=prod'.",
    },
    {
        "type": "episodic",
        "content": "Production incident on 2026-03-18: memory leak in the Celery worker caused OOM kills every 6 hours. Root cause was unbounded batch size in the decay job. Fixed by chunking to 1000 memories per batch.",
    },
    {
        "type": "episodic",
        "content": "CI pipeline migration from GitHub Actions to self-hosted runners completed on 2026-03-20. Build times reduced from 4 minutes to 90 seconds.",
    },
    {
        "type": "procedural",
        "content": "Disaster recovery procedure: restore from the latest automated pg_basebackup, apply WAL files from S3, verify data integrity with checksums, switch DNS to the recovery instance.",
    },
    # --- Cluster 6: GDPR & Compliance ---
    {
        "type": "semantic",
        "content": "GDPR Article 17 grants individuals the right to erasure. In Z3rno, this is implemented via forget(hard_delete=True), which permanently removes the memory, relationships, graph vertices, and embedding vectors.",
    },
    {
        "type": "semantic",
        "content": "The audit log is append-only with hash chain integrity. Each entry contains prev_hash and row_hash fields that form a tamper-evident linked list. The immutability trigger prevents UPDATE and DELETE operations.",
    },
    {
        "type": "procedural",
        "content": "To process a GDPR deletion request: verify the requester's identity, call forget() with hard_delete=True and the subject's user_id, verify the audit log contains a gdpr_delete entry, issue a deletion certificate.",
    },
    {
        "type": "episodic",
        "content": "First GDPR deletion request received on 2026-03-25 from a Globex user. Successfully processed in 3 minutes. Audit log confirms complete erasure with no residual content.",
    },
    # --- Cluster 7: Agent Behavior & Memory Patterns ---
    {
        "type": "semantic",
        "content": "Agent memory consolidation follows a sleep-wake cycle: working memories from active sessions are promoted to episodic when the session ends, then periodically consolidated into semantic memories via summarization.",
    },
    {
        "type": "semantic",
        "content": "Memory poisoning detection identifies anomalous patterns in stored memories: sudden importance spikes, content that contradicts established semantic knowledge, or embedding vectors that are statistical outliers.",
    },
    {
        "type": "episodic",
        "content": "The customer support agent for Acme Corp processed 1,247 conversations in March 2026, storing 8,340 episodic memories and consolidating 312 into semantic knowledge.",
    },
    {
        "type": "working",
        "content": "Agent is currently processing a complex multi-turn conversation about migrating from Mem0 to Z3rno. User has 100K existing memories to import.",
    },
    {
        "type": "working",
        "content": "Active debugging session: agent recall latency spiked to 500ms. Investigating whether the HNSW index needs rebuilding after a large batch insert.",
    },
    # --- Cluster 8: API & Integration ---
    {
        "type": "semantic",
        "content": "The Z3rno REST API follows OpenAPI 3.1 conventions. All endpoints require an X-API-Key header. Rate limits are 100 requests per second on the free tier and 1000 on Pro.",
    },
    {
        "type": "procedural",
        "content": "To integrate Z3rno with LangChain: install z3rno-sdk-python, initialize Z3rnoMemory with your API key, and pass it as the memory parameter to your LangChain agent.",
    },
    {
        "type": "episodic",
        "content": "LangChain integration v0.1.0 released on 2026-03-28. Supports store, recall, and forget operations as a LangChain BaseMemory subclass. CrewAI integration scheduled for next sprint.",
    },
    {
        "type": "semantic",
        "content": "Webhook events are fired for memory lifecycle changes: memory.created, memory.updated, memory.deleted, memory.decayed, memory.transitioned. Webhooks use HMAC-SHA256 signatures for verification.",
    },
    # --- Additional diverse memories ---
    {
        "type": "episodic",
        "content": "Quarterly business review: Z3rno processed 2.1 million memory operations in Q1 2026. Top use cases: customer support (45%), code assistance (30%), analytics (25%).",
    },
    {
        "type": "semantic",
        "content": "Apache AGE enables Cypher graph queries inside PostgreSQL. Memory relationships like DERIVED_FROM, CONTRADICTS, and SUPPORTS form a knowledge graph that enriches recall with multi-hop reasoning.",
    },
    {
        "type": "procedural",
        "content": "To create a memory relationship: use the relationships parameter in store(), passing a list of RelationshipInput objects with target_memory_id, relationship_type, and optional weight.",
    },
    {
        "type": "episodic",
        "content": "Graph traversal performance test: 3-hop Cypher query across 10,000 vertices completed in 12ms. Acceptable for real-time recall augmentation.",
    },
    {
        "type": "semantic",
        "content": "Temporal versioning (SCD Type 2) in Z3rno means every memory mutation creates a new version with valid_from/valid_to timestamps. Point-in-time queries use get_memory_at_time(memory_id, timestamp).",
    },
    {
        "type": "working",
        "content": "User is asking about the difference between soft delete and hard delete in Z3rno. Soft delete sets deleted_at timestamp; hard delete permanently removes all data including embeddings.",
    },
    {
        "type": "semantic",
        "content": "Connection pooling is critical for multi-tenant databases. PgBouncer in transaction mode allows thousands of application connections to share a smaller pool of PostgreSQL connections.",
    },
    {
        "type": "episodic",
        "content": "First enterprise design partner signed: TechCorp agreed to a 90-day pilot with Z3rno for their internal knowledge management agent. 500K memory budget, Pro tier.",
    },
    {
        "type": "procedural",
        "content": "To monitor Z3rno health: check /v1/health for API status, review Grafana dashboards for latency percentiles, and set up PagerDuty alerts for p99 > 200ms or error rate > 1%.",
    },
    {
        "type": "semantic",
        "content": "The Z3rno pricing model: Community (free, 10K memories, 1 agent), Pro ($29/mo, 100K memories, 10 agents), Team ($199/mo, 1M memories, unlimited agents), Enterprise (custom).",
    },
]


def embed_batch(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Generate embeddings via LiteLLM (calls OpenAI by default)."""
    import litellm  # noqa: PLC0415

    response = litellm.embedding(model=model, input=texts)
    return [item["embedding"] for item in response.data]


def compute_row_hash(prev_hash: bytes | None, data: str) -> bytes:
    h = hashlib.sha256()
    if prev_hash:
        h.update(prev_hash)
    h.update(data.encode())
    return h.digest()


def seed(session: Session) -> None:
    """Populate the database with realistic memories and real embeddings."""
    now = datetime.now(UTC)
    month_ago = now - timedelta(days=30)

    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  Memories to create: {len(MEMORIES)}")

    # ---- Tenant ----
    org_id = uuid4()
    tenant = Tenant(
        org_id=org_id,
        name="Realistic Seed Tenant",
        plan_tier=PlanTier.PRO,
        settings={"embedding_model": EMBEDDING_MODEL, "embedding_dimension": 1536},
    )
    session.add(tenant)
    session.flush()
    print(f"  Created tenant: {tenant.name} ({org_id})")

    # ---- Agent ----
    agent_id = uuid4()
    agent = Agent(
        id=agent_id,
        org_id=org_id,
        external_id="realistic-seed-agent",
        name="Realistic Seed Agent",
    )
    session.add(agent)
    session.flush()
    print(f"  Created agent: {agent.name} ({agent_id})")

    # ---- Generate embeddings in batches of 20 ----
    print("  Generating embeddings via LiteLLM...")
    all_texts = [m["content"] for m in MEMORIES]
    all_embeddings: list[list[float]] = []
    batch_size = 20
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        t0 = time.time()
        embeddings = embed_batch(batch)
        elapsed = time.time() - t0
        all_embeddings.extend(embeddings)
        print(
            f"    Batch {i // batch_size + 1}/{(len(all_texts) + batch_size - 1) // batch_size}: {len(batch)} texts, {elapsed:.2f}s"
        )

    # ---- Create memories ----
    type_map = {
        "working": MemoryType.WORKING,
        "episodic": MemoryType.EPISODIC,
        "semantic": MemoryType.SEMANTIC,
        "procedural": MemoryType.PROCEDURAL,
    }
    memory_ids: list[uuid4] = []

    for i, (mem_data, embedding) in enumerate(zip(MEMORIES, all_embeddings, strict=True)):
        mid = uuid4()
        memory_ids.append(mid)
        created_at = month_ago + timedelta(hours=i * 12)

        memory = Memory(
            id=mid,
            org_id=org_id,
            agent_id=agent_id,
            memory_type=type_map[mem_data["type"]],
            content=mem_data["content"],
            embedding=embedding,
            embedding_model=EMBEDDING_MODEL,
            importance_score=0.5 + (i % 5) * 0.1,  # 0.5 to 0.9
            recall_count=i % 10,
            valid_from=created_at,
            created_at=created_at,
            updated_at=created_at,
        )
        session.add(memory)

    session.flush()
    print(f"  Created {len(MEMORIES)} memories with real embeddings")

    # ---- Create relationships between related memories ----
    # Link memories within the same cluster
    relationship_pairs = [
        # Auth cluster
        (0, 1, RelationshipType.RELATED_TO),
        (0, 2, RelationshipType.RELATED_TO),
        (1, 3, RelationshipType.CAUSED_BY),
        # DB cluster
        (5, 6, RelationshipType.RELATED_TO),
        (5, 7, RelationshipType.RELATED_TO),
        (8, 9, RelationshipType.CAUSED_BY),
        # ML cluster
        (10, 11, RelationshipType.RELATED_TO),
        (10, 12, RelationshipType.SUPPORTS),
        (13, 14, RelationshipType.DERIVED_FROM),
        # Cross-cluster: ML <-> Customer Support
        (10, 15, RelationshipType.RELATED_TO),
        # GDPR cluster
        (25, 26, RelationshipType.SUPPORTS),
        (25, 27, RelationshipType.RELATED_TO),
        (26, 28, RelationshipType.DERIVED_FROM),
        # Agent behavior
        (29, 30, RelationshipType.RELATED_TO),
        # API cluster
        (34, 35, RelationshipType.RELATED_TO),
        (35, 36, RelationshipType.DERIVED_FROM),
    ]

    for src_idx, tgt_idx, rel_type in relationship_pairs:
        rel = MemoryRelationship(
            id=uuid4(),
            org_id=org_id,
            source_memory_id=memory_ids[src_idx],
            target_memory_id=memory_ids[tgt_idx],
            relationship_type=rel_type,
            weight=0.8,
        )
        session.add(rel)

    session.flush()
    print(f"  Created {len(relationship_pairs)} memory relationships")

    # ---- Audit log entries ----
    prev_hash: bytes | None = None
    for i, mid in enumerate(memory_ids):
        data = f"store:{mid}:{now.isoformat()}"
        row_hash = compute_row_hash(prev_hash, data)
        session.execute(
            text(
                "INSERT INTO audit_log (org_id, agent_id, operation, memory_id, "
                "memory_type, details, prev_hash, row_hash, created_at) VALUES "
                "(CAST(:org AS uuid), CAST(:aid AS uuid), CAST(:op AS audit_operation_enum), "
                "CAST(:mid AS uuid), CAST(:mt AS memory_type_enum), CAST(:details AS jsonb), "
                ":prev_hash, :row_hash, :created_at)"
            ),
            {
                "org": str(org_id),
                "aid": str(agent_id),
                "op": "store",
                "mid": str(mid),
                "mt": MEMORIES[i]["type"],
                "details": "{}",
                "prev_hash": prev_hash,
                "row_hash": row_hash,
                "created_at": now - timedelta(hours=len(MEMORIES) - i),
            },
        )
        prev_hash = row_hash

    session.flush()
    print(f"  Created {len(MEMORIES)} audit log entries")

    session.commit()
    print(f"\n  Seed complete. Tenant org_id: {org_id}")
    print(f"  Agent ID: {agent_id}")
    print(
        "  Try: recall() with queries like 'GDPR deletion', 'database performance', 'authentication security'"
    )


def main() -> None:
    print("Z3rno Realistic Seed — generating embeddings via LiteLLM")
    print(f"Database: {DATABASE_URL}")

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not set. This script requires an OpenAI API key.")
        print("Usage: OPENAI_API_KEY=sk-... uv run python -m seeds.realistic_seed")
        sys.exit(1)

    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        seed(session)


if __name__ == "__main__":
    main()
