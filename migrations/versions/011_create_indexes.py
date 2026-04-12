"""011 - Create composite, partial, GIN, and HNSW indexes.

All specialised indexes for the memories table live in this dedicated
migration so they can be reviewed and tuned independently.

Revision ID: 011
Revises: 010
Create Date: 2026-04-12
"""

from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Composite B-tree: hot-path tenant+agent recall with "currently valid" filter
    op.create_index(
        "ix_memories_org_agent_valid",
        "memories",
        ["org_id", "agent_id", "valid_to"],
    )

    # Temporal range: point-in-time queries
    op.create_index(
        "ix_memories_valid_range",
        "memories",
        ["valid_from", "valid_to"],
    )

    # Partial index: skip soft-deleted rows on the hot path
    op.execute("""
        CREATE INDEX ix_memories_active
        ON memories (org_id, agent_id, memory_type)
        WHERE deleted_at IS NULL
    """)

    # GIN on JSONB metadata for containment/key-exists queries
    op.execute("""
        CREATE INDEX ix_memories_metadata
        ON memories USING gin (metadata)
    """)

    # HNSW vector index for cosine similarity search (pgvector 0.8+)
    op.execute("""
        CREATE INDEX ix_memories_embedding_hnsw
        ON memories USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
    """)


def downgrade() -> None:
    op.drop_index("ix_memories_embedding_hnsw", table_name="memories")
    op.drop_index("ix_memories_metadata", table_name="memories")
    op.drop_index("ix_memories_active", table_name="memories")
    op.drop_index("ix_memories_valid_range", table_name="memories")
    op.drop_index("ix_memories_org_agent_valid", table_name="memories")
