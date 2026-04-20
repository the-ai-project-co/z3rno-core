"""013 - Create SCD Type 2 temporal versioning trigger on memories.

On UPDATE to memories:
  1. Set valid_to = now() on the old row (superseded)
  2. INSERT a new row with valid_from = now(), valid_to = NULL (current)
  3. Copy all fields from the old row, applying the UPDATE changes

On soft DELETE (setting deleted_at):
  - Only sets deleted_at. Does NOT touch valid_to (the row is still
    "the current version", just flagged as deleted).

Hard DELETE (GDPR) is handled by the forget() engine function, not
a trigger.

Revision ID: 013
Revises: 012
Create Date: 2026-04-12
"""

from alembic import op

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create the trigger function
    op.execute("""
        CREATE OR REPLACE FUNCTION memories_scd_trigger()
        RETURNS TRIGGER AS $$
        BEGIN
            -- If only deleted_at changed, allow the update without versioning
            IF NEW.deleted_at IS DISTINCT FROM OLD.deleted_at
               AND NEW.content = OLD.content
               AND NEW.summary IS NOT DISTINCT FROM OLD.summary
               AND NEW.importance_score = OLD.importance_score
               AND NEW.recall_count = OLD.recall_count
               AND NEW.last_recalled_at IS NOT DISTINCT FROM OLD.last_recalled_at
               AND NEW.pinned = OLD.pinned
               AND NEW.quarantined = OLD.quarantined
               AND NEW.anomaly_score = OLD.anomaly_score
               AND NEW.memory_type = OLD.memory_type
            THEN
                RETURN NEW;
            END IF;

            -- DESIGN DECISION (MVP): recall_count / last_recalled_at updates
            -- are intentionally excluded from SCD Type 2 versioning.
            --
            -- Rationale: recall() is a hot-path operation executed on every
            -- query. Creating a new temporal version for each recall would
            -- generate massive table bloat with minimal analytical value for
            -- the MVP use case. Temporal queries therefore will NOT show
            -- recall frequency changes over time.
            --
            -- Revisit for analytics: If per-version recall frequency tracking
            -- is needed (e.g., for "how popular was this memory last month?"
            -- queries), this guard must be removed or replaced with a
            -- periodic rollup strategy that snapshots recall_count into a
            -- separate time-series table instead of creating full SCD rows.
            --
            -- See: z3rno-limitations/z3rno-core.md, issue #4 (SCD Type 2
            -- trigger skips versioning for recall_count updates).
            IF NEW.recall_count IS DISTINCT FROM OLD.recall_count
               AND NEW.content = OLD.content
               AND NEW.summary IS NOT DISTINCT FROM OLD.summary
               AND NEW.importance_score = OLD.importance_score
               AND NEW.memory_type = OLD.memory_type
               AND NEW.pinned = OLD.pinned
               AND NEW.deleted_at IS NOT DISTINCT FROM OLD.deleted_at
            THEN
                RETURN NEW;
            END IF;

            -- SCD Type 2: supersede the old row
            UPDATE memories
            SET valid_to = now()
            WHERE id = OLD.id AND valid_to IS NULL;

            -- Insert the new version
            INSERT INTO memories (
                id, org_id, agent_id, user_id, memory_type,
                content, summary, metadata, embedding, embedding_model,
                importance_score, recall_count, last_recalled_at,
                valid_from, valid_to, pinned, ttl_expires_at,
                deleted_at, quarantined, anomaly_score,
                created_at, updated_at
            ) VALUES (
                OLD.id, NEW.org_id, NEW.agent_id, NEW.user_id, NEW.memory_type,
                NEW.content, NEW.summary, NEW.metadata, NEW.embedding, NEW.embedding_model,
                NEW.importance_score, NEW.recall_count, NEW.last_recalled_at,
                now(), NULL, NEW.pinned, NEW.ttl_expires_at,
                NEW.deleted_at, NEW.quarantined, NEW.anomaly_score,
                OLD.created_at, now()
            );

            -- Return NULL to prevent the original UPDATE from happening
            -- (we did the work manually above)
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Attach the trigger (BEFORE UPDATE so we can return NULL to cancel)
    op.execute("""
        CREATE TRIGGER memories_scd_before_update
        BEFORE UPDATE ON memories
        FOR EACH ROW
        EXECUTE FUNCTION memories_scd_trigger();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS memories_scd_before_update ON memories")
    op.execute("DROP FUNCTION IF EXISTS memories_scd_trigger()")
