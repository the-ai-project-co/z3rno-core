"""035 - v0.21.2: SCD-2 trigger recursion guard + metadata-only short-circuit.

Closes Bug I from V0-21-FULL-BENCH-2026-05-12.

Two latent defects in the migration-013 SCD-2 trigger surfaced during
the v0.21.1 benchmark:

1. **Self-recursion.** The trigger's own
   ``UPDATE memories SET valid_to = now() WHERE valid_to IS NULL``
   re-fires the trigger on the row being superseded. The inner call
   sees ``OLD.valid_to = NULL`` and ``NEW.valid_to = now()`` (only
   ``valid_to`` changed), which doesn't match any of the existing
   short-circuit guards (deleted_at-only, recall_count-only), so it
   falls through to the SCD body again and re-issues the same UPDATE.
   The loop blows ``max_stack_depth`` (Postgres default 2 MB) with
   ``StatementTooComplexError: stack depth limit exceeded``.

   Why it didn't fire across every SCD update in prod: in practice
   most paths that UPDATE memories either match the existing
   short-circuits (forget → deleted_at, recall counter bump →
   recall_count) or are INSERT paths (store, store_batch). The
   benchmark's store + add_turn loop was the first to hit a
   non-short-circuit UPDATE consistently.

2. **add_turn writes weren't whitelisted.** ``add_turn`` writes
   ``conversation_id / turn_index / turn_role`` (Phase G slice 2
   columns added in migration 028). Those columns are *metadata
   about where the Memo lives in the chat*, not content changes — no
   SCD-2 version is wanted. Without a whitelist they fell through
   to the SCD path on every add_turn call.

Fix:

  a. ``IF pg_trigger_depth() > 1 THEN RETURN NEW;`` at the top.
     Standard pattern for any SCD-2 trigger that re-UPDATEs its own
     table. Breaks the recursion cleanly.

  b. New short-circuit guard: when only conversation-linkage columns
     (``conversation_id`` / ``turn_index`` / ``turn_role`` /
     ``updated_at``) change while content+importance+recall_count+
     pinned+quarantined+memory_type+deleted_at stay the same, skip
     the SCD body and return NEW so the UPDATE proceeds normally.

Verified locally: store + add_turn loop completes 1000 iterations
without a single StatementTooComplexError. SCD-2 versioning for
content changes still produces one new row per UPDATE as before.

Revision ID: 035
Revises: 034
Create Date: 2026-05-12
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None


_FIXED_FUNCTION = r"""
CREATE OR REPLACE FUNCTION memories_scd_trigger()
RETURNS TRIGGER AS $$
BEGIN
    -- v0.21.2 — break self-recursion. The SCD body below issues
    -- ``UPDATE memories SET valid_to = now() ...`` which would re-fire
    -- this trigger on the row being superseded and (because the new
    -- guards below don't match a valid_to-only change) recurse until
    -- max_stack_depth blows. pg_trigger_depth() returns 1 for the
    -- outermost call, >1 for any recursive entry.
    IF pg_trigger_depth() > 1 THEN
        RETURN NEW;
    END IF;

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

    -- recall_count / last_recalled_at bumps stay un-versioned (MVP
    -- decision documented in migration 013).
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

    -- v0.21.2 — conversation linkage (Phase G slice 2) is metadata
    -- about where the Memo lives in a chat, not content history.
    -- ``add_turn`` writes conversation_id / turn_index / turn_role on
    -- an already-stored Memo; this should not create a new SCD-2
    -- version. Pre-fix, ``add_turn`` UPDATEs fell through to the SCD
    -- body and (combined with the self-recursion above) blew the
    -- stack on every call. Closes Bug I.
    IF (
            NEW.conversation_id IS DISTINCT FROM OLD.conversation_id
         OR NEW.turn_index IS DISTINCT FROM OLD.turn_index
         OR NEW.turn_role IS DISTINCT FROM OLD.turn_role
       )
       AND NEW.content = OLD.content
       AND NEW.summary IS NOT DISTINCT FROM OLD.summary
       AND NEW.importance_score = OLD.importance_score
       AND NEW.recall_count = OLD.recall_count
       AND NEW.last_recalled_at IS NOT DISTINCT FROM OLD.last_recalled_at
       AND NEW.pinned = OLD.pinned
       AND NEW.quarantined = OLD.quarantined
       AND NEW.anomaly_score = OLD.anomaly_score
       AND NEW.memory_type = OLD.memory_type
       AND NEW.deleted_at IS NOT DISTINCT FROM OLD.deleted_at
    THEN
        RETURN NEW;
    END IF;

    -- SCD Type 2: supersede the old row. The pg_trigger_depth() guard
    -- above prevents this inner UPDATE from re-entering the SCD body.
    UPDATE memories
    SET valid_to = now()
    WHERE id = OLD.id AND valid_to IS NULL;

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

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
"""

_ORIGINAL_FUNCTION = r"""
CREATE OR REPLACE FUNCTION memories_scd_trigger()
RETURNS TRIGGER AS $$
BEGIN
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

    UPDATE memories
    SET valid_to = now()
    WHERE id = OLD.id AND valid_to IS NULL;

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

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    op.execute(_FIXED_FUNCTION)


def downgrade() -> None:
    op.execute(_ORIGINAL_FUNCTION)
