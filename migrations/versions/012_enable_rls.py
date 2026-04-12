"""012 - Enable Row-Level Security on all tenant-scoped tables.

Creates two database roles:
  - z3rno_admin: superuser-like, bypasses RLS (for migrations, admin ops)
  - z3rno_app: application role, subject to RLS

RLS policy shape (same on all 6 tables):
  FOR ALL USING (org_id = current_setting('app.current_org_id')::uuid)
  WITH CHECK (org_id = current_setting('app.current_org_id')::uuid)

The application must SET LOCAL app.current_org_id = '<uuid>' at the start
of every transaction. See z3rno_core.security.set_org_context().

Revision ID: 012
Revises: 011
Create Date: 2026-04-12
"""

from alembic import op

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None

# All tenant-scoped tables that need RLS
RLS_TABLES = [
    "agents",
    "api_keys",
    "lifecycle_policies",
    "memories",
    "memory_relationships",
    "audit_log",
]

POLICY_EXPR = "org_id = current_setting('app.current_org_id')::uuid"


def upgrade() -> None:
    # --- Create roles ---
    # IF NOT EXISTS prevents errors on re-run or if roles already exist
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_admin') THEN
                CREATE ROLE z3rno_admin NOLOGIN BYPASSRLS;
            END IF;
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_app') THEN
                CREATE ROLE z3rno_app NOLOGIN;
            END IF;
        END
        $$
    """)

    # --- Enable RLS + create policies ---
    for table in RLS_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        # Force RLS even for table owners (defense in depth)
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        # Single policy: FOR ALL covers SELECT, INSERT, UPDATE, DELETE
        op.execute(f"""
            CREATE POLICY tenant_isolation ON {table}
                FOR ALL
                USING ({POLICY_EXPR})
                WITH CHECK ({POLICY_EXPR})
        """)

    # --- Grant permissions to application role ---
    for table in RLS_TABLES:
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO z3rno_app")
    # tenants table: app role can read but not modify
    op.execute("GRANT SELECT ON tenants TO z3rno_app")
    # audit_log sequence (for BIGSERIAL inserts)
    op.execute("GRANT USAGE, SELECT ON SEQUENCE audit_log_id_seq TO z3rno_app")
    # alembic_version: only admin needs this
    op.execute("GRANT ALL ON alembic_version TO z3rno_admin")
    # Allow the database owner to SET ROLE to z3rno_app (needed for testing)
    op.execute("""
        DO $$
        BEGIN
            EXECUTE format('GRANT z3rno_app TO %I', current_user);
            EXECUTE format('GRANT z3rno_admin TO %I', current_user);
        END
        $$
    """)


def downgrade() -> None:
    for table in reversed(RLS_TABLES):
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"REVOKE SELECT, INSERT, UPDATE, DELETE ON {table} FROM z3rno_app")
    op.execute("REVOKE SELECT ON tenants FROM z3rno_app")

    # Drop roles
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_app') THEN
                DROP ROLE z3rno_app;
            END IF;
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_admin') THEN
                DROP ROLE z3rno_admin;
            END IF;
        END
        $$
    """)
