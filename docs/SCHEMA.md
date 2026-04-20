# Z3rno Database Schema

> Auto-generated from SQLAlchemy models on 2026-04-19 20:40 UTC.
> Do not edit manually. Run `python scripts/generate_schema.py` to regenerate.

## Tables

- [agents](#agents)
- [api_keys](#api_keys)
- [audit_log](#audit_log)
- [lifecycle_policies](#lifecycle_policies)
- [memories](#memories)
- [memory_relationships](#memory_relationships)
- [tenants](#tenants)

---

## agents

### Columns

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | `UUID` | No | gen_random_uuid() |
| `external_id` | `VARCHAR(255)` | Yes |  |
| `name` | `VARCHAR(255)` | No |  |
| `metadata` | `JSONB` | No | {} |
| `org_id` | `UUID` | No |  |
| `created_at` | `DATETIME` | No | now() |
| `updated_at` | `DATETIME` | No | now() |

**Primary Key:** `id`

### Foreign Keys

- `org_id` -> `tenants.org_id`

### Indexes

- `ix_agents_external_id`: `external_id`
- `ix_agents_org_id`: `org_id`

---

## api_keys

### Columns

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | `UUID` | No | gen_random_uuid() |
| `name` | `VARCHAR(255)` | No |  |
| `prefix` | `VARCHAR(32)` | No |  |
| `key_hash` | `BLOB` | No |  |
| `last_used_at` | `DATETIME` | Yes |  |
| `expires_at` | `DATETIME` | Yes |  |
| `revoked_at` | `DATETIME` | Yes |  |
| `org_id` | `UUID` | No |  |
| `created_at` | `DATETIME` | No | now() |
| `updated_at` | `DATETIME` | No | now() |

**Primary Key:** `id`

### Foreign Keys

- `org_id` -> `tenants.org_id`

### Indexes

- `ix_api_keys_org_id`: `org_id`
- `ix_api_keys_prefix`: `prefix`

---

## audit_log

### Columns

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | `BIGINT` | No |  |
| `agent_id` | `UUID` | Yes |  |
| `user_id` | `UUID` | Yes |  |
| `operation` | `VARCHAR(11)` | No |  |
| `memory_id` | `UUID` | Yes |  |
| `memory_type` | `VARCHAR(10)` | Yes |  |
| `details` | `JSONB` | No | {} |
| `prev_hash` | `BLOB` | Yes |  |
| `row_hash` | `BLOB` | No |  |
| `api_key_id` | `UUID` | Yes |  |
| `ip_address` | `INET` | Yes |  |
| `user_agent` | `TEXT` | Yes |  |
| `request_id` | `VARCHAR(64)` | Yes |  |
| `org_id` | `UUID` | No |  |
| `created_at` | `DATETIME` | No | now() |
| `updated_at` | `DATETIME` | No | now() |

**Primary Key:** `id`

### Foreign Keys

- `org_id` -> `tenants.org_id`

### Indexes

- `ix_audit_log_agent_id`: `agent_id`
- `ix_audit_log_api_key_id`: `api_key_id`
- `ix_audit_log_memory_id`: `memory_id`
- `ix_audit_log_operation`: `operation`
- `ix_audit_log_org_created`: `org_id`, `created_at`
- `ix_audit_log_org_id`: `org_id`
- `ix_audit_log_org_memory`: `org_id`, `memory_id`
- `ix_audit_log_user_id`: `user_id`

---

## lifecycle_policies

### Columns

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | `UUID` | No | gen_random_uuid() |
| `memory_type` | `VARCHAR(10)` | No |  |
| `max_ttl_days` | `INTEGER` | Yes |  |
| `max_count` | `INTEGER` | Yes |  |
| `min_importance` | `FLOAT` | No | 0.0 |
| `decay_enabled` | `BOOLEAN` | No | true |
| `decay_rate` | `FLOAT` | No | 0.01 |
| `decay_curve` | `VARCHAR(11)` | No | exponential |
| `decay_floor` | `FLOAT` | No | 0.1 |
| `retention_policy` | `VARCHAR(11)` | No | standard |
| `gdpr_auto_delete_days` | `INTEGER` | Yes |  |
| `summarize_before_delete` | `BOOLEAN` | No | true |
| `org_id` | `UUID` | No |  |
| `created_at` | `DATETIME` | No | now() |
| `updated_at` | `DATETIME` | No | now() |

**Primary Key:** `id`

### Foreign Keys

- `org_id` -> `tenants.org_id`

### Indexes

- `ix_lifecycle_policies_org_id`: `org_id`

### Unique Constraints

- `uq_lifecycle_policies_org_id_memory_type`: `org_id`, `memory_type`

---

## memories

### Columns

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | `UUID` | No | gen_random_uuid() |
| `agent_id` | `UUID` | No |  |
| `user_id` | `UUID` | Yes |  |
| `memory_type` | `VARCHAR(10)` | No | episodic |
| `content` | `TEXT` | No |  |
| `summary` | `TEXT` | Yes |  |
| `metadata` | `JSONB` | No | {} |
| `embedding` | `VECTOR(1536)` | Yes |  |
| `embedding_model` | `VARCHAR(100)` | Yes |  |
| `importance_score` | `FLOAT` | No | 0.5 |
| `recall_count` | `INTEGER` | No | 0 |
| `last_recalled_at` | `DATETIME` | Yes |  |
| `valid_from` | `DATETIME` | No | now() |
| `valid_to` | `DATETIME` | Yes |  |
| `pinned` | `BOOLEAN` | No | false |
| `ttl_expires_at` | `DATETIME` | Yes |  |
| `deleted_at` | `DATETIME` | Yes |  |
| `quarantined` | `BOOLEAN` | No | false |
| `anomaly_score` | `FLOAT` | No | 0.0 |
| `org_id` | `UUID` | No |  |
| `created_at` | `DATETIME` | No | now() |
| `updated_at` | `DATETIME` | No | now() |

**Primary Key:** `id`

### Foreign Keys

- `org_id` -> `tenants.org_id`

### Indexes

- `ix_memories_active`: `org_id`, `agent_id`, `memory_type`
- `ix_memories_agent_id`: `agent_id`
- `ix_memories_embedding_hnsw`: `embedding`
- `ix_memories_metadata`: `metadata`
- `ix_memories_org_agent_valid`: `org_id`, `agent_id`, `valid_to`
- `ix_memories_org_id`: `org_id`
- `ix_memories_user_id`: `user_id`
- `ix_memories_valid_range`: `valid_from`, `valid_to`

---

## memory_relationships

### Columns

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | `UUID` | No | gen_random_uuid() |
| `source_memory_id` | `UUID` | No |  |
| `target_memory_id` | `UUID` | No |  |
| `relationship_type` | `VARCHAR(12)` | No |  |
| `weight` | `FLOAT` | No | 1.0 |
| `metadata` | `JSONB` | No | {} |
| `org_id` | `UUID` | No |  |
| `created_at` | `DATETIME` | No | now() |
| `updated_at` | `DATETIME` | No | now() |

**Primary Key:** `id`

### Foreign Keys

- `source_memory_id` -> `memories.id`
- `target_memory_id` -> `memories.id`
- `org_id` -> `tenants.org_id`

### Indexes

- `ix_memory_relationships_org_id`: `org_id`
- `ix_memory_relationships_relationship_type`: `relationship_type`
- `ix_memory_relationships_source_memory_id`: `source_memory_id`
- `ix_memory_relationships_target_memory_id`: `target_memory_id`

---

## tenants

### Columns

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `org_id` | `UUID` | No | gen_random_uuid() |
| `name` | `VARCHAR(255)` | No |  |
| `plan_tier` | `VARCHAR(10)` | No | community |
| `settings` | `JSONB` | No | {} |
| `suspended_at` | `DATETIME` | Yes |  |
| `created_at` | `DATETIME` | No | now() |
| `updated_at` | `DATETIME` | No | now() |

**Primary Key:** `org_id`

