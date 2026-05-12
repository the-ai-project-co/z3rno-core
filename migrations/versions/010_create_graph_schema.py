"""010 - Create Apache AGE graph schema.

Creates the memory_graph graph with vertex labels (Memory, Agent, Concept)
and edge labels matching the RelationshipType enum.

Revision ID: 010
Revises: 009
Create Date: 2026-04-12
"""

from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None

GRAPH_NAME = "memory_graph"

VERTEX_LABELS = ["Memory", "Agent", "Concept"]

EDGE_LABELS = [
    "RELATES_TO",
    "DERIVED_FROM",
    "CONTRADICTS",
    "SUPPORTS",
    "SUPERSEDES",
    "CAUSED_BY",
]


def upgrade() -> None:
    # Ensure AGE functions are accessible
    op.execute("LOAD 'age'")
    # SET LOCAL — scoped to this migration's transaction only.
    # A bare SET persists for the rest of the alembic connection and
    # caused every subsequent unqualified CREATE TABLE (migration 015's
    # distill_jobs is the famous case) to land in ag_catalog instead
    # of public. transaction_per_migration in env.py keeps each
    # migration in its own transaction so LOCAL is enough.
    op.execute('SET LOCAL search_path = ag_catalog, "$user", public')

    # Create the graph
    op.execute(f"SELECT create_graph('{GRAPH_NAME}')")

    # Create vertex labels
    for label in VERTEX_LABELS:
        op.execute(f"SELECT create_vlabel('{GRAPH_NAME}', '{label}')")

    # Create edge labels
    for label in EDGE_LABELS:
        op.execute(f"SELECT create_elabel('{GRAPH_NAME}', '{label}')")


def downgrade() -> None:
    op.execute("LOAD 'age'")
    # SET LOCAL — scoped to this migration's transaction only.
    # A bare SET persists for the rest of the alembic connection and
    # caused every subsequent unqualified CREATE TABLE (migration 015's
    # distill_jobs is the famous case) to land in ag_catalog instead
    # of public. transaction_per_migration in env.py keeps each
    # migration in its own transaction so LOCAL is enough.
    op.execute('SET LOCAL search_path = ag_catalog, "$user", public')
    op.execute(f"SELECT drop_graph('{GRAPH_NAME}', true)")
