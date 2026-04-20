"""Z3rno-core benchmark suite.

Runs three benchmark sets against a live PostgreSQL instance:
  1. HNSW vector index — build time and query latency at 10K/50K/100K vectors
  2. IVFFlat vs HNSW — recall@10 and latency comparison
  3. Audit log partitioning — insert 1M rows, verify query performance

Usage:
    DATABASE_URL="postgresql+psycopg://z3rno:z3rno_bench@localhost:5433/z3rno_bench" \
        uv run python benchmarks/run_benchmarks.py

Results are printed to stdout and written to benchmarks/results/.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from sqlalchemy import create_engine, text

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://z3rno:z3rno_bench@localhost:5433/z3rno_bench",
)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

engine = create_engine(DB_URL)


def random_unit_vector(dim: int = 1536) -> list[float]:
    """Generate a random unit vector of given dimension."""
    v = np.random.randn(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


def format_vector(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# ─── Setup ────────────────────────────────────────────────────────────────────


def setup_bench_tenant(conn):  # noqa: ANN001, ANN201
    """Create a benchmark tenant and agent."""
    org_id = uuid4()
    agent_id = uuid4()
    conn.execute(
        text(
            "INSERT INTO tenants (org_id, name, plan_tier) "
            "VALUES (CAST(:org AS uuid), 'Benchmark Tenant', 'pro')"
        ),
        {"org": str(org_id)},
    )
    conn.execute(
        text(
            "INSERT INTO agents (id, org_id, name) "
            "VALUES (CAST(:aid AS uuid), CAST(:org AS uuid), 'Bench Agent')"
        ),
        {"aid": str(agent_id), "org": str(org_id)},
    )
    conn.commit()
    return org_id, agent_id


# ─── Benchmark 1: HNSW Vector Index ──────────────────────────────────────────


def bench_hnsw(conn, org_id, agent_id, sizes=(10_000, 50_000, 100_000)):  # noqa: ANN001, ANN201
    """Benchmark HNSW index build time and query latency."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: HNSW Vector Index — Build Time & Query Latency")
    print("=" * 70)

    results = {}
    total_inserted = 0

    for size in sizes:
        to_insert = size - total_inserted
        print(f"\n--- {size:,} vectors ---")

        # Insert vectors in batches of 1000
        t0 = time.perf_counter()
        batch_size = 200
        for batch_start in range(0, to_insert, batch_size):
            batch_end = min(batch_start + batch_size, to_insert)
            values = []
            params = {}
            for i in range(batch_start, batch_end):
                idx = total_inserted + i
                mid = uuid4()
                vec = random_unit_vector()
                params[f"id_{idx}"] = str(mid)
                params[f"org_{idx}"] = str(org_id)
                params[f"aid_{idx}"] = str(agent_id)
                params[f"content_{idx}"] = f"Benchmark memory {idx}"
                params[f"emb_{idx}"] = format_vector(vec)
                values.append(
                    f"(CAST(:id_{idx} AS uuid), CAST(:org_{idx} AS uuid), "
                    f"CAST(:aid_{idx} AS uuid), 'semantic', :content_{idx}, "
                    f"CAST(:emb_{idx} AS vector), 0.5, now(), now())"
                )
            conn.execute(
                text(
                    "INSERT INTO memories (id, org_id, agent_id, memory_type, content, "
                    "embedding, importance_score, valid_from, created_at) VALUES "
                    + ", ".join(values)
                ),
                params,
            )
            conn.commit()
            if (batch_start + batch_size) % 10000 == 0:
                print(f"  Inserted {total_inserted + batch_start + batch_size:,} / {size:,}")

        total_inserted = size
        insert_time = time.perf_counter() - t0
        print(f"  Insert time: {insert_time:.2f}s ({to_insert:,} rows)")

        # Reindex HNSW to measure build time
        print("  Rebuilding HNSW index...")
        conn.execute(text("DROP INDEX IF EXISTS idx_memories_embedding_hnsw"))
        conn.commit()
        t0 = time.perf_counter()
        conn.execute(
            text(
                "CREATE INDEX idx_memories_embedding_hnsw ON memories "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 200)"
            )
        )
        conn.commit()
        build_time = time.perf_counter() - t0
        print(f"  HNSW build time: {build_time:.2f}s")

        # Query latency (50 random queries)
        latencies = []
        for _ in range(50):
            qvec = format_vector(random_unit_vector())
            t0 = time.perf_counter()
            conn.execute(
                text(
                    "SELECT id, 1 - (embedding <=> CAST(:qvec AS vector)) AS similarity "
                    "FROM memories WHERE embedding IS NOT NULL "
                    "ORDER BY embedding <=> CAST(:qvec AS vector) LIMIT 10"
                ),
                {"qvec": qvec},
            ).fetchall()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

        avg_lat = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        print(f"  Query latency (top-10, 50 queries): avg={avg_lat:.1f}ms, p50={p50:.1f}ms, p95={p95:.1f}ms, p99={p99:.1f}ms")

        results[size] = {
            "insert_time_s": round(insert_time, 2),
            "hnsw_build_time_s": round(build_time, 2),
            "query_latency_ms": {
                "avg": round(avg_lat, 1),
                "p50": round(p50, 1),
                "p95": round(p95, 1),
                "p99": round(p99, 1),
            },
            "rows_inserted": to_insert,
        }

    return results


# ─── Benchmark 2: IVFFlat vs HNSW ────────────────────────────────────────────


def bench_ivfflat_vs_hnsw(conn, num_queries=50):  # noqa: ANN001, ANN201
    """Compare IVFFlat and HNSW recall@10 and latency using existing data."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: IVFFlat vs HNSW — Recall@10 & Latency Comparison")
    print("=" * 70)

    # Get exact (sequential scan) results for ground truth
    queries = [random_unit_vector() for _ in range(num_queries)]

    # Sequential scan (ground truth)
    print("\n  Computing ground truth (sequential scan)...")
    conn.execute(text("SET enable_indexscan = off"))
    ground_truths = []
    for qvec in queries:
        rows = conn.execute(
            text(
                "SELECT id FROM memories WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> CAST(:qvec AS vector) LIMIT 10"
            ),
            {"qvec": format_vector(qvec)},
        ).fetchall()
        ground_truths.append({r[0] for r in rows})
    conn.execute(text("SET enable_indexscan = on"))

    results = {}

    for index_type, create_sql in [
        ("hnsw", "CREATE INDEX bench_idx ON memories USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)"),
        ("ivfflat", "CREATE INDEX bench_idx ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"),
    ]:
        print(f"\n  --- {index_type.upper()} ---")
        conn.execute(text("DROP INDEX IF EXISTS idx_memories_embedding_hnsw"))
        conn.execute(text("DROP INDEX IF EXISTS bench_idx"))
        conn.commit()

        t0 = time.perf_counter()
        conn.execute(text(create_sql))
        conn.commit()
        build_time = time.perf_counter() - t0
        print(f"  Build time: {build_time:.2f}s")

        latencies = []
        recall_scores = []
        for i, qvec in enumerate(queries):
            t0 = time.perf_counter()
            rows = conn.execute(
                text(
                    "SELECT id FROM memories WHERE embedding IS NOT NULL "
                    "ORDER BY embedding <=> CAST(:qvec AS vector) LIMIT 10"
                ),
                {"qvec": format_vector(qvec)},
            ).fetchall()
            latencies.append((time.perf_counter() - t0) * 1000)
            result_ids = {r[0] for r in rows}
            recall = len(result_ids & ground_truths[i]) / 10.0
            recall_scores.append(recall)

        avg_recall = statistics.mean(recall_scores)
        avg_lat = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"  Recall@10: {avg_recall:.2%}")
        print(f"  Latency: avg={avg_lat:.1f}ms, p50={p50:.1f}ms, p95={p95:.1f}ms")

        results[index_type] = {
            "build_time_s": round(build_time, 2),
            "recall_at_10": round(avg_recall, 4),
            "latency_ms": {
                "avg": round(avg_lat, 1),
                "p50": round(p50, 1),
                "p95": round(p95, 1),
            },
        }

    # Restore HNSW as the production index
    conn.execute(text("DROP INDEX IF EXISTS bench_idx"))
    conn.execute(
        text(
            "CREATE INDEX idx_memories_embedding_hnsw ON memories "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 200)"
        )
    )
    conn.commit()

    return results


# ─── Benchmark 3: Audit Log 1M Rows ──────────────────────────────────────────


def bench_audit_log(conn, org_id, agent_id, target=1_000_000):  # noqa: ANN001, ANN201
    """Insert 1M audit log rows and benchmark query performance."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Audit Log — 1M Row Insert & Query Performance")
    print("=" * 70)

    # Disable immutability trigger for bulk insert performance
    conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_update"))
    conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete"))
    conn.commit()

    operations = ["store", "recall", "forget", "update"]
    memory_types = ["episodic", "semantic", "working", "procedural"]
    batch_size = 5000

    print(f"\n  Inserting {target:,} audit log rows...")
    t0 = time.perf_counter()

    for batch_start in range(0, target, batch_size):
        batch_end = min(batch_start + batch_size, target)
        values = []
        params = {}
        for i in range(batch_start, batch_end):
            op = operations[i % 4]
            mt = memory_types[i % 4]
            params[f"org_{i}"] = str(org_id)
            params[f"aid_{i}"] = str(agent_id)
            params[f"op_{i}"] = op
            params[f"mid_{i}"] = str(uuid4())
            params[f"mt_{i}"] = mt
            values.append(
                f"(CAST(:org_{i} AS uuid), CAST(:aid_{i} AS uuid), "
                f"CAST(:op_{i} AS audit_operation_enum), "
                f"CAST(:mid_{i} AS uuid), CAST(:mt_{i} AS memory_type_enum), "
                f"now())"
            )
        conn.execute(
            text(
                "INSERT INTO audit_log (org_id, agent_id, operation, memory_id, memory_type, created_at) VALUES "
                + ", ".join(values)
            ),
            params,
        )
        conn.commit()
        if (batch_start + batch_size) % 100_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {batch_start + batch_size:,} / {target:,} ({elapsed:.1f}s)")

    insert_time = time.perf_counter() - t0
    print(f"  Total insert time: {insert_time:.2f}s ({target / insert_time:,.0f} rows/sec)")

    # Verify row count
    count = conn.execute(text("SELECT count(*) FROM audit_log")).scalar()
    print(f"  Total rows in audit_log: {count:,}")

    # Query benchmarks
    queries = {
        "count_by_org": "SELECT count(*) FROM audit_log WHERE org_id = CAST(:org AS uuid)",
        "count_by_operation": "SELECT operation, count(*) FROM audit_log WHERE org_id = CAST(:org AS uuid) GROUP BY operation",
        "recent_50": "SELECT * FROM audit_log WHERE org_id = CAST(:org AS uuid) ORDER BY created_at DESC LIMIT 50",
        "filter_by_operation": "SELECT * FROM audit_log WHERE org_id = CAST(:org AS uuid) AND operation = 'store' ORDER BY created_at DESC LIMIT 50",
        "time_range_scan": f"SELECT count(*) FROM audit_log WHERE org_id = CAST(:org AS uuid) AND created_at >= '2026-04-20' AND created_at < '2026-04-21'",
    }

    results = {"insert_time_s": round(insert_time, 2), "rows": target, "rows_per_sec": round(target / insert_time), "queries": {}}

    print("\n  Query benchmarks (10 runs each):")
    for name, sql in queries.items():
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            conn.execute(text(sql), {"org": str(org_id)}).fetchall()
            latencies.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        passed = avg < 50
        status = "PASS" if passed else "SLOW"
        print(f"    {name}: avg={avg:.1f}ms, p50={p50:.1f}ms [{status}]")
        results["queries"][name] = {
            "avg_ms": round(avg, 1),
            "p50_ms": round(p50, 1),
            "under_50ms": passed,
        }

    # Re-enable triggers
    conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_update"))
    conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_delete"))
    conn.commit()

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Z3rno-core Benchmark Suite")
    print(f"Database: {DB_URL}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    with engine.connect() as conn:
        org_id, agent_id = setup_bench_tenant(conn)

        hnsw_results = bench_hnsw(conn, org_id, agent_id)
        ivf_results = bench_ivfflat_vs_hnsw(conn)
        audit_results = bench_audit_log(conn, org_id, agent_id)

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": DB_URL.split("@")[1] if "@" in DB_URL else DB_URL,
        "hnsw_benchmark": hnsw_results,
        "ivfflat_vs_hnsw": ivf_results,
        "audit_log_benchmark": audit_results,
    }

    results_file = RESULTS_DIR / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
