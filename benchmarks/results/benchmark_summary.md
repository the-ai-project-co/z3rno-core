# Z3rno Vector Index Benchmark Summary

**Date:** 2026-04-20 | **Prepared for:** Stakeholder Review

---

## Key Findings

- **Query latency stays under 15ms at P95 for 100K vectors** — well within the <50ms SLA target, confirming HNSW scales sub-linearly for recall operations.
- **HNSW chosen as production default over IVFFlat** — despite 13x slower builds on native hardware, HNSW provides superior recall accuracy and supports online workloads without periodic re-indexing.
- **Audit log queries meet performance targets** — paginated queries return in <3ms at 1M rows; only full `count(*)` (67ms) slightly exceeds the 50ms target but is not in the hot path.

---

## Performance Table

### HNSW Query Latency (1536-dim, cosine, top-10)

| Vectors | Query Avg | Query P50 | Query P95 | Query P99 |
|---------|-----------|-----------|-----------|-----------|
| 10,000  | 2.1ms     | 2.0ms     | 2.5ms     | 9.0ms     |
| 50,000  | 7.1ms     | 5.8ms     | 14.7ms    | 18.4ms    |
| 100,000 | 7.6ms     | 6.0ms     | 13.8ms    | 17.2ms    |

### Index Build Time (estimated native amd64)

| Vectors | HNSW Build | IVFFlat Build |
|---------|-----------|---------------|
| 10,000  | ~2.5s     | —             |
| 50,000  | ~28s      | —             |
| 100,000 | ~90s      | 7.4s          |

### Audit Log (1M rows)

| Query Type | Latency | Target | Status |
|------------|---------|--------|--------|
| Recent-50 paginated | 2.2ms | <50ms | PASS |
| Filtered by operation | 1.2ms | <50ms | PASS |
| count(*) by org_id | 67.0ms | <50ms | SLOW (not hot path) |

---

## Recommendations

1. **Ship HNSW as default** — query performance meets all SLA targets at projected 100K vector scale.
2. **Offer IVFFlat as opt-in** — for tenants with frequent bulk re-imports who can tolerate slightly lower recall accuracy.
3. **Implement monthly partitioning** for audit logs before exceeding 1M rows/month to keep count queries manageable.
4. **Add covering index** on `(org_id, created_at)` if analytics dashboards require frequent count queries.

---

## Environment

| Component | Version/Config |
|-----------|---------------|
| PostgreSQL | 17 |
| pgvector | 0.8.2 |
| pgvectorscale | 0.9.0 |
| Embedding dimension | 1536 (text-embedding-3-small) |
| HNSW params | m=16, ef_construction=200 |
| IVFFlat params | lists=100 |
| Platform | Docker (linux/amd64) |
