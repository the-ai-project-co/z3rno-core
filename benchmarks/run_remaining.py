"""Run remaining benchmarks: IVFFlat comparison + 1M audit log."""

from __future__ import annotations

import statistics
import time
from uuid import uuid4

import numpy as np
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg://z3rno:z3rno_bench@localhost:5433/z3rno_bench"
engine = create_engine(DB_URL)


def random_unit_vector(dim: int = 1536) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


def format_vector(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def main() -> None:
    with engine.connect() as conn:
        org_id = conn.execute(text("SELECT org_id FROM tenants LIMIT 1")).scalar()
        agent_id = conn.execute(text("SELECT id FROM agents LIMIT 1")).scalar()
        print(f"Using org_id={org_id}, agent_id={agent_id}")

        # --- IVFFlat build + queries ---
        print("\n=== IVFFlat Build ===")
        conn.execute(text("DROP INDEX IF EXISTS bench_idx"))
        conn.execute(text("DROP INDEX IF EXISTS idx_memories_embedding_hnsw"))
        conn.commit()
        conn.execute(text("SET maintenance_work_mem = '256MB'"))
        t0 = time.perf_counter()
        conn.execute(
            text(
                "CREATE INDEX bench_idx ON memories "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
        )
        conn.commit()
        build_time = time.perf_counter() - t0
        print(f"IVFFlat build: {build_time:.2f}s")

        latencies = []
        for _ in range(50):
            qvec = format_vector(random_unit_vector())
            t0 = time.perf_counter()
            conn.execute(
                text(
                    "SELECT id FROM memories WHERE embedding IS NOT NULL "
                    "ORDER BY embedding <=> CAST(:qvec AS vector) LIMIT 10"
                ),
                {"qvec": qvec},
            ).fetchall()
            latencies.append((time.perf_counter() - t0) * 1000)
        print(
            f"IVFFlat latency: avg={statistics.mean(latencies):.1f}ms, "
            f"p50={statistics.median(latencies):.1f}ms, "
            f"p95={sorted(latencies)[47]:.1f}ms"
        )

        # Restore HNSW
        print("\nRestoring HNSW index...")
        conn.execute(text("DROP INDEX IF EXISTS bench_idx"))
        conn.commit()
        conn.execute(text("SET maintenance_work_mem = '256MB'"))
        t0 = time.perf_counter()
        conn.execute(
            text(
                "CREATE INDEX idx_memories_embedding_hnsw ON memories "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 200)"
            )
        )
        conn.commit()
        print(f"HNSW restore: {time.perf_counter() - t0:.2f}s")

        # --- Audit 1M ---
        print("\n=== Audit Log 1M Insert ===")
        conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_update"))
        conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete"))
        conn.execute(text("DELETE FROM audit_log"))
        conn.commit()

        operations = ["store", "recall", "forget", "update"]
        memory_types = ["episodic", "semantic", "working", "procedural"]
        target = 1_000_000
        batch_size = 5000

        t0 = time.perf_counter()
        for batch_start in range(0, target, batch_size):
            batch_end = min(batch_start + batch_size, target)
            values = []
            params: dict[str, str] = {}
            for i in range(batch_start, batch_end):
                params[f"org_{i}"] = str(org_id)
                params[f"aid_{i}"] = str(agent_id)
                params[f"op_{i}"] = operations[i % 4]
                params[f"mid_{i}"] = str(uuid4())
                params[f"mt_{i}"] = memory_types[i % 4]
                values.append(
                    f"(CAST(:org_{i} AS uuid), CAST(:aid_{i} AS uuid), "
                    f"CAST(:op_{i} AS audit_operation_enum), "
                    f"CAST(:mid_{i} AS uuid), CAST(:mt_{i} AS memory_type_enum), "
                    f"now())"
                )
            conn.execute(
                text(
                    "INSERT INTO audit_log (org_id, agent_id, operation, "
                    "memory_id, memory_type, created_at) VALUES " + ", ".join(values)
                ),
                params,
            )
            conn.commit()
            if (batch_start + batch_size) % 100_000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {batch_start + batch_size:,} / {target:,} ({elapsed:.1f}s)")

        insert_time = time.perf_counter() - t0
        print(f"Total insert: {insert_time:.2f}s ({target / insert_time:,.0f} rows/sec)")

        # Query benchmarks
        queries = {
            "count_by_org": "SELECT count(*) FROM audit_log WHERE org_id = CAST(:org AS uuid)",
            "recent_50": (
                "SELECT * FROM audit_log WHERE org_id = CAST(:org AS uuid) "
                "ORDER BY created_at DESC LIMIT 50"
            ),
            "filter_op": (
                "SELECT * FROM audit_log WHERE org_id = CAST(:org AS uuid) "
                "AND operation = 'store' ORDER BY created_at DESC LIMIT 50"
            ),
        }
        print("\nQuery benchmarks:")
        for name, sql in queries.items():
            lats = []
            for _ in range(10):
                t0 = time.perf_counter()
                conn.execute(text(sql), {"org": str(org_id)}).fetchall()
                lats.append((time.perf_counter() - t0) * 1000)
            avg = statistics.mean(lats)
            status = "PASS" if avg < 50 else "SLOW"
            print(f"  {name}: avg={avg:.1f}ms [{status}]")

        conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_update"))
        conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_delete"))
        conn.commit()
        print("\nDone!")


if __name__ == "__main__":
    main()
