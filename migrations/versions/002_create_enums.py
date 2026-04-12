"""002 - Create all PostgreSQL enum types.

Creating enums in a dedicated migration avoids ordering issues when
multiple tables reference the same enum (e.g. memory_type_enum is used
by both lifecycle_policies and memories).

Revision ID: 002
Revises: 001
Create Date: 2026-04-12
"""

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None

ENUMS = [
    ("plan_tier_enum", ["community", "pro", "team", "enterprise"]),
    ("memory_type_enum", ["working", "episodic", "semantic", "procedural"]),
    (
        "audit_operation_enum",
        [
            "store",
            "recall",
            "forget",
            "update",
            "quarantine",
            "pin",
            "unpin",
            "export",
            "gdpr_delete",
        ],
    ),
    (
        "relationship_type_enum",
        [
            "derived_from",
            "contradicts",
            "supports",
            "supersedes",
            "related_to",
            "caused_by",
        ],
    ),
    ("decay_curve_enum", ["exponential", "logarithmic", "step"]),
    ("retention_policy_enum", ["standard", "gdpr_strict", "hipaa", "financial"]),
]


def upgrade() -> None:
    for name, values in ENUMS:
        value_list = ", ".join(f"'{v}'" for v in values)
        op.execute(f"CREATE TYPE {name} AS ENUM ({value_list})")


def downgrade() -> None:
    for name, _ in reversed(ENUMS):
        op.execute(f"DROP TYPE IF EXISTS {name}")
