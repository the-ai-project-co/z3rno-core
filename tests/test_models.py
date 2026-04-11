"""Structural tests for the SQLAlchemy 2.0 model classes.

These tests do NOT touch a database — they verify that the declarative
classes are wired up correctly: column names match the spec, foreign keys
point at the right tables, primary keys exist, mixins are applied. Real
SQL exercise lives in the integration test suite (Week 2+).
"""

from __future__ import annotations

from typing import cast

from sqlalchemy import Table, UniqueConstraint, inspect
from sqlalchemy.orm import Mapper

from z3rno_core.models import (
    EMBEDDING_DIMENSION,
    Agent,
    ApiKey,
    AuditLog,
    AuditOperation,
    Base,
    DecayCurve,
    LifecyclePolicy,
    Memory,
    MemoryRelationship,
    MemoryType,
    PlanTier,
    RelationshipType,
    RetentionPolicy,
    Tenant,
)

# ---------------------------------------------------------------------------
# Inheritance + Base
# ---------------------------------------------------------------------------


def test_every_model_inherits_from_base() -> None:
    """All 7 models inherit from the single Base."""
    for model in (Tenant, Agent, ApiKey, LifecyclePolicy, Memory, MemoryRelationship, AuditLog):
        assert issubclass(model, Base), f"{model.__name__} does not inherit from Base"


def test_base_metadata_uses_naming_convention() -> None:
    """The naming convention is set so Alembic produces stable index/FK names."""
    nc = dict(Base.metadata.naming_convention)
    assert nc["pk"] == "pk_%(table_name)s"
    assert "fk_" in str(nc["fk"])
    assert "ix_" in str(nc["ix"])


# ---------------------------------------------------------------------------
# Table names
# ---------------------------------------------------------------------------


def test_table_names_match_spec() -> None:
    """Doc 08 §4.2 names. These are the source of truth — do not rename."""
    assert Tenant.__tablename__ == "tenants"
    assert Agent.__tablename__ == "agents"
    assert ApiKey.__tablename__ == "api_keys"
    assert LifecyclePolicy.__tablename__ == "lifecycle_policies"
    assert Memory.__tablename__ == "memories"
    assert MemoryRelationship.__tablename__ == "memory_relationships"
    assert AuditLog.__tablename__ == "audit_log"


# ---------------------------------------------------------------------------
# Tenant-scoped tables have org_id
# ---------------------------------------------------------------------------


def test_every_tenant_scoped_model_has_org_id() -> None:
    """The OrgScopedMixin applies to every table that needs RLS."""
    for model in (Agent, ApiKey, LifecyclePolicy, Memory, MemoryRelationship, AuditLog):
        mapper = cast(Mapper[object], inspect(model))
        cols = {c.name for c in mapper.columns}
        assert "org_id" in cols, f"{model.__name__} is missing org_id"


def test_tenant_uses_org_id_as_primary_key() -> None:
    """The tenants table itself has org_id as its PK."""
    mapper = cast(Mapper[object], inspect(Tenant))
    pk_cols = [c.name for c in mapper.primary_key]
    assert pk_cols == ["org_id"]


# ---------------------------------------------------------------------------
# Memory schema — the most important table
# ---------------------------------------------------------------------------


def test_memory_has_all_required_columns() -> None:
    """Doc 08 §4.2.1 columns. Locked schema."""
    cols = {c.name for c in cast(Mapper[object], inspect(Memory)).columns}
    expected = {
        "id",
        "org_id",
        "agent_id",
        "user_id",
        "memory_type",
        "content",
        "summary",
        "metadata",  # the underlying column name (Python attr is memory_metadata)
        "embedding",
        "embedding_model",
        "importance_score",
        "recall_count",
        "last_recalled_at",
        "valid_from",
        "valid_to",
        "pinned",
        "ttl_expires_at",
        "deleted_at",
        "quarantined",
        "anomaly_score",
        "created_at",
        "updated_at",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


def test_memory_has_no_is_current_or_is_deleted_booleans() -> None:
    """Locked decision: currency derives from valid_to IS NULL, deletion from
    deleted_at IS NOT NULL. The two boolean columns must NOT exist.
    """
    cols = {c.name for c in cast(Mapper[object], inspect(Memory)).columns}
    assert "is_current" not in cols
    assert "is_deleted" not in cols


def test_memory_uses_recall_count_not_access_count() -> None:
    """Locked decision: column names are recall_count + last_recalled_at."""
    cols = {c.name for c in cast(Mapper[object], inspect(Memory)).columns}
    assert "recall_count" in cols
    assert "last_recalled_at" in cols
    assert "access_count" not in cols
    assert "last_accessed_at" not in cols


def test_memory_valid_to_is_nullable() -> None:
    """SCD Type 2 — NULL means currently valid, not 9999-12-31 sentinel."""
    valid_to = cast(Mapper[object], inspect(Memory)).columns["valid_to"]
    assert valid_to.nullable is True


def test_memory_embedding_dimension_is_1536() -> None:
    """ADR-001: hardcoded 1536 dims for OpenAI text-embedding-3-small."""
    assert EMBEDDING_DIMENSION == 1536


# ---------------------------------------------------------------------------
# AuditLog schema — operation column, hash chain, no memory FK
# ---------------------------------------------------------------------------


def test_audit_log_uses_operation_column_not_action() -> None:
    """Locked decision: column is `operation`, not `action`."""
    cols = {c.name for c in cast(Mapper[object], inspect(AuditLog)).columns}
    assert "operation" in cols
    assert "action" not in cols


def test_audit_log_has_hash_chain_columns() -> None:
    """Tamper-evident hash chain: prev_hash + row_hash."""
    cols = {c.name for c in cast(Mapper[object], inspect(AuditLog)).columns}
    assert "prev_hash" in cols
    assert "row_hash" in cols


def test_audit_log_memory_id_is_not_foreign_key() -> None:
    """Audit rows must survive memory deletion (especially GDPR hard delete).
    The memory_id column exists but is NOT a foreign key.
    """
    memory_id_col = cast(Mapper[object], inspect(AuditLog)).columns["memory_id"]
    assert len(memory_id_col.foreign_keys) == 0, (
        "audit_log.memory_id MUST NOT be a foreign key — see Doc 08 §4.2 audit_log section"
    )


def test_audit_log_id_is_bigint_not_uuid() -> None:
    """BIGSERIAL primary key, not UUID — millions of rows per tenant."""
    id_col = cast(Mapper[object], inspect(AuditLog)).columns["id"]
    assert id_col.type.python_type is int


# ---------------------------------------------------------------------------
# AuditOperation enum has the extended set
# ---------------------------------------------------------------------------


def test_audit_operation_enum_includes_extended_values() -> None:
    """Locked decision: operation enum includes quarantine/pin/unpin/export/gdpr_delete."""
    expected = {
        "store",
        "recall",
        "forget",
        "update",
        "quarantine",
        "pin",
        "unpin",
        "export",
        "gdpr_delete",
    }
    actual = {member.value for member in AuditOperation}
    assert actual == expected


def test_plan_tier_enum_has_four_values() -> None:
    """Locked decision: community/pro/team/enterprise — exactly four tiers."""
    expected = {"community", "pro", "team", "enterprise"}
    actual = {member.value for member in PlanTier}
    assert actual == expected


def test_memory_type_enum_has_four_tiers() -> None:
    """working/episodic/semantic/procedural — the four-tier model."""
    expected = {"working", "episodic", "semantic", "procedural"}
    actual = {member.value for member in MemoryType}
    assert actual == expected


# ---------------------------------------------------------------------------
# MemoryRelationship — FK to memories, no self-edges
# ---------------------------------------------------------------------------


def test_memory_relationship_has_two_fks_to_memories() -> None:
    """source_memory_id + target_memory_id both FK to memories.id."""
    cols = cast(Mapper[object], inspect(MemoryRelationship)).columns
    src_fks = {fk.target_fullname for fk in cols["source_memory_id"].foreign_keys}
    tgt_fks = {fk.target_fullname for fk in cols["target_memory_id"].foreign_keys}
    assert src_fks == {"memories.id"}
    assert tgt_fks == {"memories.id"}


def test_relationship_type_enum_complete() -> None:
    expected = {
        "derived_from",
        "contradicts",
        "supports",
        "supersedes",
        "related_to",
        "caused_by",
    }
    actual = {member.value for member in RelationshipType}
    assert actual == expected


# ---------------------------------------------------------------------------
# LifecyclePolicy — unique on (org_id, memory_type)
# ---------------------------------------------------------------------------


def test_lifecycle_policy_unique_per_tenant_per_type() -> None:
    """Each tenant has at most one policy per memory_type."""
    table = cast(Table, LifecyclePolicy.__table__)
    unique_constraints = [c for c in table.constraints if isinstance(c, UniqueConstraint)]
    assert any(
        {col.name for col in c.columns} == {"org_id", "memory_type"} for c in unique_constraints
    ), "lifecycle_policies must have UNIQUE(org_id, memory_type)"


def test_lifecycle_policy_enums_complete() -> None:
    decay_expected = {"exponential", "logarithmic", "step"}
    assert {m.value for m in DecayCurve} == decay_expected
    retention_expected = {"standard", "gdpr_strict", "hipaa", "financial"}
    assert {m.value for m in RetentionPolicy} == retention_expected


# ---------------------------------------------------------------------------
# Tenant — plan_tier defaults to community
# ---------------------------------------------------------------------------


def test_tenant_default_plan_tier_is_community() -> None:
    plan_tier_col = cast(Mapper[object], inspect(Tenant)).columns["plan_tier"]
    # The default is the Python enum member wrapped in ScalarElementColumnDefault
    default = plan_tier_col.default
    assert default is not None
    assert getattr(default, "arg", None) == PlanTier.COMMUNITY


# ---------------------------------------------------------------------------
# Smoke: Base.metadata sees all 7 tables
# ---------------------------------------------------------------------------


def test_metadata_collects_all_seven_tables() -> None:
    """Importing the models package should register all 7 tables on Base.metadata."""
    table_names = set(Base.metadata.tables.keys())
    expected = {
        "tenants",
        "agents",
        "api_keys",
        "lifecycle_policies",
        "memories",
        "memory_relationships",
        "audit_log",
    }
    assert expected.issubset(table_names), f"missing tables: {expected - table_names}"
