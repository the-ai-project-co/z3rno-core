"""027 - Phase F slice 5: forget_certificates table.

Cryptographically-signed proof that a forget() actually happened.

Each row is **append-only by convention** (no UPDATE / DELETE path
in the engine code) — the cert is the contract handed to the user
who exercised right-to-erasure, plus the auditor who later asks
"prove you actually deleted this".

Schema:
  * ``cert_id`` — uuid; the public handle on ``GET /v1/forget/{cert_id}``
  * ``memory_ids`` — uuid[]; the set of Memos covered by this proof
  * ``merkle_root`` — bytea; root of a tree built over leaf hashes
    of ``(memory_id, content_hash, audit_entry_hash)`` for each Memo
  * ``signature`` — bytea; ed25519 signature over a canonical JSON
    blob `{cert_id, org_id, memory_ids[], merkle_root, signed_at,
    audit_seq_range, signer_key_id}`
  * ``signer_key_id`` — text; operator-chosen identifier of the
    public key (rotation-friendly: the CLI looks up which pubkey to
    verify against by this id)
  * ``audit_seq_start`` / ``audit_seq_end`` — bigint; the audit_log
    sequence range that contains the matching forget operations.
    Used by the verifier CLI to anchor the proof to the audit chain.
  * ``signed_at`` — timestamptz
  * ``hard_delete`` — boolean; cert applies to a hard- vs soft-forget

RLS-scoped by ``org_id`` so certs never leak across tenants.

Revision ID: 027
Revises: 026
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forget_certificates",
        sa.Column(
            "cert_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "memory_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
        ),
        sa.Column("merkle_root", postgresql.BYTEA(), nullable=False),
        sa.Column("signature", postgresql.BYTEA(), nullable=False),
        sa.Column("signer_key_id", sa.Text(), nullable=False),
        sa.Column("audit_seq_start", sa.BigInteger(), nullable=True),
        sa.Column("audit_seq_end", sa.BigInteger(), nullable=True),
        sa.Column("hard_delete", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "signed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["tenants.org_id"],
            name="fk_forget_certificates_org",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "cardinality(memory_ids) >= 1",
            name="ck_forget_certificates_nonempty",
        ),
    )

    op.create_index(
        "ix_forget_certificates_org",
        "forget_certificates",
        ["org_id", "signed_at"],
    )
    op.execute(
        "CREATE INDEX ix_forget_certificates_memory_ids "
        "ON public.forget_certificates USING GIN (memory_ids)"
    )

    # RLS — mirror the tenant-scoped pattern.
    policy_expr = "org_id = current_setting('app.current_org_id')::uuid"
    op.execute("ALTER TABLE public.forget_certificates ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE public.forget_certificates FORCE ROW LEVEL SECURITY")
    op.execute(f"""
        CREATE POLICY tenant_isolation ON forget_certificates
            FOR ALL
            USING ({policy_expr})
            WITH CHECK ({policy_expr})
    """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_app') THEN
                EXECUTE 'GRANT SELECT, INSERT ON forget_certificates TO z3rno_app';
            END IF;
        END
        $$
    """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON forget_certificates")
    op.execute("ALTER TABLE IF EXISTS public.forget_certificates DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE IF EXISTS public.forget_certificates NO FORCE ROW LEVEL SECURITY")
    op.execute("DROP INDEX IF EXISTS public.ix_forget_certificates_memory_ids")
    op.drop_index("ix_forget_certificates_org", table_name="forget_certificates")
    op.drop_table("forget_certificates")
