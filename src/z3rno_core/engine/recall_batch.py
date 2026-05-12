"""Batched ``recall_count`` write-back (slice 21.5, shipped v0.22.0 opt-in).

Pre-v0.22, every recall ran one ``UPDATE memories SET recall_count = recall_count + 1``
either inline (default) or as a fire-and-forget task on a fresh
connection (v0.20.5's ``bump_counters_async``). Under recall-heavy
load that's N updates + N connections per N recalls — the audit-drain
tail outliers in the v0.21 benchmark traced back to the same
connection-pressure pattern.

``RecallCountBatcher`` coalesces increments into a sliding window
(default 50 ms). Within the window, repeated hits on the same
``(org_id, memory_id)`` add to a per-key delta. On flush, one
``UPDATE … FROM (SELECT unnest(...))`` per org applies all deltas
atomically — so a memory hit five times in 50 ms goes up by five,
not one.

Design constraints baked in:

- **Per-process singleton.** Two server replicas each maintain their
  own queue. The UPDATE is additive so cross-replica overlap is
  safe; counts converge to the right total.
- **Lazy start.** The drain task isn't spawned until the first
  ``bump()`` call — keeps tests and short-lived scripts from
  leaking background tasks.
- **Best-effort.** A drain failure is logged + dropped; a missed
  counter bump is recoverable (the next recall sees a slightly
  stale count, same trade-off as ``_bump_counters_async``).
- **Graceful shutdown.** ``flush_pending()`` drains the queue
  synchronously so the worker shutdown hook can call it before
  SIGTERM lets pending bumps die.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Final
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

# Default flush cadence. The trade-off: smaller window = fresher
# counters + more UPDATEs; larger window = staler counters + better
# coalescing. 50 ms is the sweet spot from the v0.21 benchmark —
# coalesces ~20 concurrent recalls per org into one UPDATE while
# keeping observable counter lag under a human-perceptible threshold.
DEFAULT_WINDOW_MS: Final[int] = 50

# Safety cap. If the queue ever grows past this, we force-flush
# instead of buffering further. Protects against a stuck drainer
# (network hiccup, replica failover) ballooning memory.
DEFAULT_MAX_QUEUE: Final[int] = 10000


class RecallCountBatcher:
    """Coalesces per-(org, memory) counter bumps and flushes in batches.

    One instance per process. Construct lazily via :func:`get_batcher`
    so tests can replace the singleton.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        window_ms: int = DEFAULT_WINDOW_MS,
        max_queue: int = DEFAULT_MAX_QUEUE,
    ) -> None:
        self._engine = engine
        self._window_seconds = window_ms / 1000.0
        self._max_queue = max_queue
        # (org_id, memory_id) -> pending delta.
        self._pending: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._wake: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task[None] | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bump(self, *, org_id: UUID, memory_ids: list[str]) -> None:
        """Queue +1 increments for each memory_id under ``org_id``.

        Cheap — just updates an in-memory counter and signals the
        drain task. Never blocks on the database.
        """
        if self._closed:
            return
        org_key = str(org_id)
        for mid in memory_ids:
            self._pending[(org_key, mid)] += 1
        # Force-flush if we've grown past the safety cap.
        if len(self._pending) >= self._max_queue:
            self._wake.set()
        # Lazy-start the drain task on first bump. Cheap re-check on
        # subsequent calls because the task reference stays set.
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain_loop())
        # Signal the drainer to wake up early if it's sleeping —
        # otherwise it picks up the new entry on its next cycle.
        self._wake.set()

    async def flush_pending(self) -> None:
        """Drain the current queue synchronously. Call before shutdown
        so in-flight bumps don't die with SIGTERM."""
        await self._flush_once()

    async def aclose(self) -> None:
        """Stop the drain loop. Idempotent."""
        self._closed = True
        self._wake.set()
        if self._drain_task is not None and not self._drain_task.done():
            try:
                await asyncio.wait_for(self._drain_task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                self._drain_task.cancel()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _drain_loop(self) -> None:
        """Wake on either the timer expiring or ``_wake.set()``,
        then flush whatever's queued."""
        while not self._closed:
            try:
                await asyncio.wait_for(
                    self._wake.wait(), timeout=self._window_seconds
                )
            except TimeoutError:
                pass
            self._wake.clear()
            if self._pending:
                await self._flush_once()

    async def _flush_once(self) -> None:
        """Atomically snapshot the queue, group by org, run one UPDATE per org.

        Failures are logged + dropped per the best-effort contract — the
        snapshotted entries are NOT requeued (re-running a failed UPDATE
        with the same deltas after a transient blip could double-bump).
        """
        if not self._pending:
            return
        snapshot, self._pending = self._pending, defaultdict(int)

        per_org: defaultdict[str, dict[str, int]] = defaultdict(dict)
        for (org_id, memory_id), delta in snapshot.items():
            per_org[org_id][memory_id] = delta

        for org_id, deltas in per_org.items():
            ids = list(deltas.keys())
            counts = list(deltas.values())
            try:
                async with self._engine.connect() as conn:
                    await conn.execute(text("SET LOCAL ROLE z3rno_app"))
                    await conn.execute(
                        text(f"SET LOCAL app.current_org_id = '{org_id}'")
                    )
                    await conn.execute(
                        text(
                            """
                            UPDATE public.memories AS m
                            SET recall_count = m.recall_count + bump.delta,
                                last_recalled_at = now(),
                                updated_at = now()
                            FROM (
                                SELECT unnest(CAST(:ids AS uuid[]))  AS id,
                                       unnest(CAST(:deltas AS int[])) AS delta
                            ) AS bump
                            WHERE m.id = bump.id
                            """
                        ),
                        {"ids": ids, "deltas": counts},
                    )
                    await conn.commit()
            except Exception:
                # Log + drop. A missed bump is recoverable; a retry
                # storm on a flapping primary is not.
                logger.warning(
                    "recall_count_batcher.flush_failed",
                    extra={"org_id": org_id, "n_memories": len(ids)},
                    exc_info=True,
                )


# ---------------------------------------------------------------------------
# Process-local singleton
# ---------------------------------------------------------------------------

_BATCHER: RecallCountBatcher | None = None


def get_batcher(
    engine: AsyncEngine,
    *,
    window_ms: int = DEFAULT_WINDOW_MS,
    max_queue: int = DEFAULT_MAX_QUEUE,
) -> RecallCountBatcher:
    """Return the process-wide batcher, constructing it on first call.

    The first engine wins — subsequent calls return the same instance
    regardless of the engine argument. That matches our deployment
    reality (one primary engine per process) and keeps tests simple.
    """
    global _BATCHER  # noqa: PLW0603 — singleton pattern, see module docstring
    if _BATCHER is None:
        _BATCHER = RecallCountBatcher(
            engine, window_ms=window_ms, max_queue=max_queue
        )
    return _BATCHER


async def shutdown_batcher() -> None:
    """Hook for the worker shutdown signal handler."""
    global _BATCHER  # noqa: PLW0603 — singleton pattern, see module docstring
    if _BATCHER is not None:
        await _BATCHER.flush_pending()
        await _BATCHER.aclose()
        _BATCHER = None


def _reset_for_tests() -> None:
    """Clear the singleton. Tests only."""
    global _BATCHER  # noqa: PLW0603 — singleton pattern, see module docstring
    _BATCHER = None
