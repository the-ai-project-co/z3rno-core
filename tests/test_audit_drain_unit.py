"""Unit tests for the audit drain functions (no DB).

Covers:
- ``drain_audit_chain``: lock acquisition, batch fetching, chain computation,
  and pending-row deletion.
- ``flush_audit_chain``: looping until the queue is empty.
- ``list_orgs_with_pending``: returns distinct org_ids.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from z3rno_core.engine.audit import (
    _advisory_lock_key,
    drain_audit_chain,
    flush_audit_chain,
    list_orgs_with_pending,
)


def _mock_lock_result(*, locked: bool) -> MagicMock:
    m = MagicMock()
    m.fetchone.return_value = (locked,)
    return m


def _mock_pending_rows(rows: list[tuple]) -> MagicMock:
    m = MagicMock()
    m.fetchall.return_value = rows
    return m


def _mock_latest_hash(value: bytes | None) -> MagicMock:
    m = MagicMock()
    m.fetchone.return_value = (value,) if value is not None else None
    return m


class TestAdvisoryLockKey:
    def test_returns_int(self) -> None:
        key = _advisory_lock_key(uuid4())
        assert isinstance(key, int)

    def test_in_int64_range(self) -> None:
        for _ in range(100):
            key = _advisory_lock_key(uuid4())
            assert -(2**63) <= key < 2**63

    def test_stable_for_same_uuid(self) -> None:
        org = uuid4()
        assert _advisory_lock_key(org) == _advisory_lock_key(org)


class TestDrainAuditChain:
    async def test_returns_zero_when_lock_contended(self) -> None:
        """If another drainer holds the lock, return 0 without doing work."""
        org_id = uuid4()
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=[_mock_lock_result(locked=False)])

        n = await drain_audit_chain(conn, org_id)

        assert n == 0
        # Only the lock query — no SELECT pending, no INSERT, no DELETE.
        assert conn.execute.call_count == 1

    async def test_returns_zero_when_queue_empty(self) -> None:
        """Lock acquired but no pending rows → no chain work, no delete."""
        org_id = uuid4()
        conn = AsyncMock()
        conn.execute = AsyncMock(
            side_effect=[
                _mock_lock_result(locked=True),
                _mock_pending_rows([]),
            ]
        )

        n = await drain_audit_chain(conn, org_id)

        assert n == 0
        assert conn.execute.call_count == 2

    async def test_chains_single_row_with_no_prev(self) -> None:
        """First-ever audit row for this org: prev_hash is None."""
        org_id = uuid4()
        now = datetime.now(tz=UTC)
        pending_row = (
            1,  # id
            None,  # agent_id
            None,  # user_id
            "store",  # operation
            None,  # memory_id
            None,  # memory_type
            {},  # details
            None,  # api_key_id
            None,  # ip_address
            None,  # user_agent
            None,  # request_id
            now,  # created_at
        )
        conn = AsyncMock()
        conn.execute = AsyncMock(
            side_effect=[
                _mock_lock_result(locked=True),
                _mock_pending_rows([pending_row]),
                _mock_latest_hash(None),  # get_latest_hash → None
                MagicMock(),  # INSERT into audit_log
                MagicMock(),  # DELETE FROM audit_log_pending
            ]
        )

        n = await drain_audit_chain(conn, org_id)

        assert n == 1
        # Lock + SELECT pending + SELECT latest + INSERT + DELETE = 5 calls
        assert conn.execute.call_count == 5
        insert_params = conn.execute.call_args_list[3][0][1]
        assert insert_params["prev_hash"] is None
        assert insert_params["row_hash"] is not None
        assert isinstance(insert_params["row_hash"], bytes)
        assert insert_params["operation"] == "store"

    async def test_chains_multiple_rows_in_order(self) -> None:
        """Each row's prev_hash equals the previous row's row_hash."""
        org_id = uuid4()
        now = datetime.now(tz=UTC)
        pending_rows = [
            (
                1, None, None, "store", None, None, {"i": 1},
                None, None, None, None, now,
            ),
            (
                2, None, None, "recall", None, None, {"i": 2},
                None, None, None, None, now,
            ),
            (
                3, None, None, "forget", None, None, {"i": 3},
                None, None, None, None, now,
            ),
        ]
        conn = AsyncMock()
        conn.execute = AsyncMock(
            side_effect=[
                _mock_lock_result(locked=True),
                _mock_pending_rows(pending_rows),
                _mock_latest_hash(b"\x00" * 32),
                MagicMock(),  # INSERT 1
                MagicMock(),  # INSERT 2
                MagicMock(),  # INSERT 3
                MagicMock(),  # DELETE
            ]
        )

        n = await drain_audit_chain(conn, org_id)

        assert n == 3
        # 1 lock + 1 select pending + 1 select latest + 3 inserts + 1 delete = 7
        assert conn.execute.call_count == 7

        first_params = conn.execute.call_args_list[3][0][1]
        second_params = conn.execute.call_args_list[4][0][1]
        third_params = conn.execute.call_args_list[5][0][1]

        # Chain integrity: each row's prev_hash is the previous row's row_hash.
        assert first_params["prev_hash"] == b"\x00" * 32
        assert second_params["prev_hash"] == first_params["row_hash"]
        assert third_params["prev_hash"] == second_params["row_hash"]

    async def test_deletes_drained_row_ids(self) -> None:
        """The DELETE uses exactly the ids of the rows just inserted."""
        org_id = uuid4()
        now = datetime.now(tz=UTC)
        pending_rows = [
            (10, None, None, "store", None, None, {}, None, None, None, None, now),
            (11, None, None, "store", None, None, {}, None, None, None, None, now),
        ]
        conn = AsyncMock()
        conn.execute = AsyncMock(
            side_effect=[
                _mock_lock_result(locked=True),
                _mock_pending_rows(pending_rows),
                _mock_latest_hash(None),
                MagicMock(),
                MagicMock(),
                MagicMock(),  # DELETE
            ]
        )

        n = await drain_audit_chain(conn, org_id)

        assert n == 2
        delete_params = conn.execute.call_args_list[-1][0][1]
        assert delete_params["ids"] == [10, 11]


class TestFlushAuditChain:
    async def test_loops_until_empty(self) -> None:
        """Calls drain_audit_chain repeatedly until it returns 0."""
        org_id = uuid4()
        now = datetime.now(tz=UTC)
        # Two batches: first drain returns 2 rows, second returns 0.
        first_batch = [
            (1, None, None, "store", None, None, {}, None, None, None, None, now),
            (2, None, None, "store", None, None, {}, None, None, None, None, now),
        ]
        conn = AsyncMock()
        conn.execute = AsyncMock(
            side_effect=[
                # First call to drain_audit_chain
                _mock_lock_result(locked=True),
                _mock_pending_rows(first_batch),
                _mock_latest_hash(None),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                # Second call to drain_audit_chain — empty
                _mock_lock_result(locked=True),
                _mock_pending_rows([]),
            ]
        )

        n = await flush_audit_chain(conn, org_id, batch_size=10)

        assert n == 2

    async def test_stops_at_max_iterations(self) -> None:
        """A pathological non-decreasing queue is bounded by max_iterations."""
        org_id = uuid4()
        now = datetime.now(tz=UTC)
        # Each call returns 1 row — would loop forever without the cap.
        single_row = [
            (1, None, None, "store", None, None, {}, None, None, None, None, now),
        ]
        # Build enough side_effects for 5 iterations of 5 calls each.
        side_effects: list[MagicMock] = []
        for _ in range(5):
            side_effects.extend(
                [
                    _mock_lock_result(locked=True),
                    _mock_pending_rows(single_row),
                    _mock_latest_hash(None),
                    MagicMock(),  # INSERT
                    MagicMock(),  # DELETE
                ]
            )

        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=side_effects)

        n = await flush_audit_chain(conn, org_id, batch_size=1, max_iterations=5)

        assert n == 5


class TestListOrgsWithPending:
    async def test_returns_distinct_uuids(self) -> None:
        ids = [uuid4(), uuid4(), uuid4()]
        result_mock = MagicMock()
        result_mock.fetchall.return_value = [(str(i),) for i in ids]
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value=result_mock)

        orgs = await list_orgs_with_pending(conn)

        assert orgs == ids

    async def test_passes_limit(self) -> None:
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value=result_mock)

        await list_orgs_with_pending(conn, limit=42)

        params = conn.execute.call_args_list[0][0][1]
        assert params["limit"] == 42
