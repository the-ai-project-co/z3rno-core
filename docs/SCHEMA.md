# Z3rno Database Schema Reference

Authoritative schema for the Z3rno memory engine. All column names, types, and
conventions are locked per **Doc 08 (Architecture Document) Section 4.2**.

> **Single source of truth:** The SQLAlchemy models in `src/z3rno_core/models/`
> are the canonical schema definition. Alembic auto-generates migrations from
> them. This document is a human-readable reference, not a substitute.

---

## Tables (dependency order)

| # | Table                  | PK type   | Tenant-scoped | Notes                                     |
|---|------------------------|-----------|---------------|--------------------------------------------|
| 1 | `tenants`              | UUID      | No (root)     | One row per organisation                   |
| 2 | `agents`               | UUID      | Yes           | FK -> tenants.org_id                       |
| 3 | `api_keys`             | UUID      | Yes           | FK -> tenants.org_id, BCrypt key_hash      |
| 4 | `lifecycle_policies`   | UUID      | Yes           | UNIQUE(org_id, memory_type)                |
| 5 | `memories`             | UUID      | Yes           | Core table, Vector(1536), SCD Type 2       |
| 6 | `memory_relationships` | UUID      | Yes           | Two FKs to memories.id                     |
| 7 | `audit_log`            | BIGSERIAL | Yes           | Append-only, hash-chained, monthly partitioned |

---

## Naming Conventions

All constraint/index names follow the SQLAlchemy naming convention configured
in `Base.metadata`:

```
pk  -> pk_%(table_name)s
fk  -> fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s
ix  -> ix_%(column_0_label)s
uq  -> uq_%(table_name)s_%(column_0_name)s
ck  -> ck_%(table_name)s_%(constraint_name)s
```

---

## Locked Decisions

These decisions are **not negotiable** without a new ADR:

| Decision | Rationale |
|----------|-----------|
| Tenant key is `org_id`, not `tenant_id` | Consistency with multi-org SaaS pattern |
| `valid_to` is NULLable (NULL = current) | SCD Type 2; no 9999-12-31 sentinel |
| No `is_current` or `is_deleted` booleans | Derived from `valid_to IS NULL` and `deleted_at IS NOT NULL` |
| `recall_count` / `last_recalled_at` | Not `access_count` / `last_accessed_at` |
| Audit column is `operation`, not `action` | Matches enum name `AuditOperation` |
| `audit_log.memory_id` is NOT a FK | Audit rows must survive memory deletion (GDPR) |
| `audit_log.id` is BIGSERIAL, not UUID | Millions of rows per tenant; sequential for partitioning |
| Embedding dimension is 1536 | ADR-001: OpenAI text-embedding-3-small, hardcoded for MVP |

---

## Enums

| Enum             | Values |
|------------------|--------|
| `MemoryType`     | `working`, `episodic`, `semantic`, `procedural` |
| `PlanTier`       | `community`, `pro`, `team`, `enterprise` |
| `AuditOperation` | `store`, `recall`, `forget`, `update`, `quarantine`, `pin`, `unpin`, `export`, `gdpr_delete` |
| `RelationshipType` | `derived_from`, `contradicts`, `supports`, `supersedes`, `related_to`, `caused_by` |
| `DecayCurve`     | `exponential`, `logarithmic`, `step` |
| `RetentionPolicy` | `standard`, `gdpr_strict`, `hipaa`, `financial` |

---

## Indexing Strategy

### `memories` table

| Index | Type | Columns / Expression | Purpose |
|-------|------|----------------------|---------|
| `ix_memories_org_agent_valid` | B-tree (composite) | `org_id, agent_id, valid_to` | Hot-path: all tenant+agent recall queries with "currently valid" filter |
| `ix_memories_valid_range` | B-tree (composite) | `valid_from, valid_to` | Temporal point-in-time queries |
| `ix_memories_active` | B-tree (partial) | `org_id, agent_id, memory_type WHERE deleted_at IS NULL` | Skip soft-deleted rows on the hot path |
| `ix_memories_metadata` | GIN | `metadata` | JSONB containment/key-exists queries |
| `ix_memories_embedding_hnsw` | HNSW | `embedding vector_cosine_ops (m=16, ef_construction=200)` | Similarity search via pgvector |
| `ix_agent_id` | B-tree | `agent_id` | Single-column lookup (auto from `index=True`) |
| `ix_user_id` | B-tree | `user_id` | Single-column lookup (auto from `index=True`) |
| `ix_org_id` | B-tree | `org_id` | Inherited from `OrgScopedMixin` |

### `audit_log` table

| Index | Type | Columns | Purpose |
|-------|------|---------|---------|
| `ix_audit_log_org_created` | B-tree (composite) | `org_id, created_at` | Time-range scans within a tenant (dashboard, export, compliance) |
| `ix_audit_log_org_memory` | B-tree (composite) | `org_id, memory_id` | Full audit trail for a specific memory |
| `ix_agent_id` | B-tree | `agent_id` | Auto from `index=True` |
| `ix_user_id` | B-tree | `user_id` | Auto from `index=True` |
| `ix_operation` | B-tree | `operation` | Auto from `index=True` |
| `ix_memory_id` | B-tree | `memory_id` | Auto from `index=True` |
| `ix_api_key_id` | B-tree | `api_key_id` | Auto from `index=True` |

### Other tables

All tenant-scoped tables inherit `org_id` (B-tree indexed) from `OrgScopedMixin`.

| Table | Additional Indexes |
|-------|--------------------|
| `agents` | `external_id` (B-tree) |
| `api_keys` | `prefix` (B-tree) |
| `memory_relationships` | `source_memory_id`, `target_memory_id`, `relationship_type` (B-tree each) |
| `lifecycle_policies` | `UNIQUE(org_id, memory_type)` |

---

## Partitioning

**`audit_log`** is partitioned monthly on `created_at` using PostgreSQL native
declarative partitioning. Partition management is a Celery task that
pre-creates partitions 3 months ahead.

> Partitioning DDL is created in the Alembic migration, not in the SQLAlchemy
> model (SQLAlchemy's `__table_args__` doesn't natively support `PARTITION BY`).

---

## Row-Level Security (RLS)

Every tenant-scoped table has RLS policies that filter on:

```sql
org_id = current_setting('app.current_org_id')::uuid
```

RLS is created in Alembic migration (Week 1 Thursday). The application sets
the session variable before every query.

---

## Constraints

### `memories`

| Constraint | Type | Expression |
|------------|------|------------|
| `importance_score_range` | CHECK | `importance_score >= 0 AND importance_score <= 1` |
| `anomaly_score_range` | CHECK | `anomaly_score >= 0 AND anomaly_score <= 1` |
| `recall_count_non_negative` | CHECK | `recall_count >= 0` |

### `lifecycle_policies`

| Constraint | Type | Expression |
|------------|------|------------|
| `uq_lifecycle_policies_org_id_memory_type` | UNIQUE | `(org_id, memory_type)` |
| `min_importance_range` | CHECK | `min_importance >= 0 AND min_importance <= 1` |
| `decay_floor_range` | CHECK | `decay_floor >= 0 AND decay_floor <= 1` |
| `decay_rate_non_negative` | CHECK | `decay_rate >= 0` |

### `memory_relationships`

| Constraint | Type | Expression |
|------------|------|------------|
| `weight_range` | CHECK | `weight >= 0 AND weight <= 1` |
| `no_self_relationship` | CHECK | `source_memory_id != target_memory_id` |

---

## ER Diagram (text)

```
tenants (org_id PK)
  |
  +--< agents (org_id FK)
  |
  +--< api_keys (org_id FK)
  |
  +--< lifecycle_policies (org_id FK)
  |
  +--< memories (org_id FK)
  |       |
  |       +--< memory_relationships (source_memory_id FK, target_memory_id FK)
  |
  +--< audit_log (org_id FK, memory_id NOT a FK)
```

> Note: `memories.agent_id` is a UUID column but NOT a FK to `agents.id`.
> This allows memories to survive agent deletion.
