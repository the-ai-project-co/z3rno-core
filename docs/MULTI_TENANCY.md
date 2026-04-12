# Multi-Tenancy & Row-Level Security

Z3rno uses PostgreSQL Row-Level Security (RLS) as the primary defence-in-depth
mechanism for tenant isolation. Even if the application layer has a bug, the
database itself blocks cross-tenant access.

## Architecture

```
API Request
    |
    v
API Key Lookup -> org_id
    |
    v
SET LOCAL app.current_org_id = '<org_id>'
    |
    v
All SQL queries filtered by RLS policy
```

## Session Variable

The session variable is **`app.current_org_id`** (locked across all docs).

```sql
SET LOCAL app.current_org_id = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
```

`SET LOCAL` scopes the value to the current transaction. It automatically
clears on COMMIT or ROLLBACK.

## RLS Policies

Every tenant-scoped table has an identical policy:

```sql
CREATE POLICY tenant_isolation ON <table>
    FOR ALL
    USING (org_id = current_setting('app.current_org_id')::uuid)
    WITH CHECK (org_id = current_setting('app.current_org_id')::uuid);
```

Tables with RLS enabled:
- `agents`
- `api_keys`
- `lifecycle_policies`
- `memories`
- `memory_relationships`
- `audit_log`

The `tenants` table itself does NOT have RLS (it is the root table).

## Roles

| Role | RLS | Purpose |
|------|-----|---------|
| `z3rno` (database owner) | Superuser (bypasses) | Database setup, migrations |
| `z3rno_admin` | BYPASSRLS | Admin operations, data export |
| `z3rno_app` | Subject to RLS | All application queries |

The z3rno-server application connects as `z3rno` and uses `SET ROLE z3rno_app`
before executing tenant queries. This ensures all application SQL is filtered
by RLS.

## Application Integration

```python
from z3rno_core.security import set_org_context

# In z3rno-server middleware (after API key -> org_id lookup):
async with async_session() as session:
    conn = await session.connection()
    await conn.run_sync(set_org_context, org_id)
    # All queries in this session are now filtered to org_id
```

## pgvector HNSW and RLS

PostgreSQL RLS is applied **after** the index scan. This means:

1. The HNSW index itself is not tenant-aware - it indexes all vectors.
2. RLS filters the results after the index returns candidates.
3. This is correct for isolation but means the index may scan more
   vectors than necessary.

**Production optimization (post-MVP):** Use a partial HNSW index per
tenant, or prepend `org_id` filtering before the vector search to
leverage the composite index `ix_memories_org_agent_valid`.

For MVP, the current approach is correct and secure. The performance
impact is negligible at the expected data volumes.

## Testing

Integration tests in `tests/test_rls_integration.py` verify:
- Tenant A cannot SELECT Tenant B's memories
- Tenant B cannot SELECT Tenant A's memories
- Tenant A cannot SELECT Tenant B's agents
- Cross-tenant UPDATE affects 0 rows
- Cross-tenant DELETE affects 0 rows
- No org context returns 0 rows

Run: `DATABASE_URL=... uv run pytest tests/test_rls_integration.py -v`
