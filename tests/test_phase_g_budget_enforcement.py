"""v0.19.2 — per-tenant budget override + pipeline pre-flight gates."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.usage import Budgets, resolve_budgets


def _mock_conn(row: tuple[object, ...] | None = None) -> MagicMock:
    conn = MagicMock()
    result = MagicMock()
    result.fetchone = MagicMock(return_value=row)
    result.fetchall = MagicMock(return_value=[])
    conn.execute = AsyncMock(return_value=result)
    return conn


# ---------------------------------------------------------------------------
# resolve_budgets — tenant override merge logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_budgets_no_tenant_row_returns_defaults() -> None:
    conn = _mock_conn(row=None)
    defaults = Budgets(daily_tokens=1000)
    out = await resolve_budgets(conn, org_id=uuid4(), defaults=defaults)
    assert out == defaults


@pytest.mark.asyncio
async def test_resolve_budgets_null_jsonb_returns_defaults() -> None:
    conn = _mock_conn(row=(None,))
    defaults = Budgets(daily_tokens=1000, monthly_tokens=10_000)
    out = await resolve_budgets(conn, org_id=uuid4(), defaults=defaults)
    assert out.daily_tokens == 1000
    assert out.monthly_tokens == 10_000


@pytest.mark.asyncio
async def test_resolve_budgets_tenant_override_wins_per_field() -> None:
    """Non-zero override fields beat defaults; zero/missing fall through."""
    conn = _mock_conn(
        row=(
            {
                "daily_tokens": 50_000,  # override
                "monthly_tokens": 0,  # fall through
                # monthly_llm_calls missing → fall through
            },
        )
    )
    defaults = Budgets(daily_tokens=1000, monthly_tokens=10_000, monthly_llm_calls=500)
    out = await resolve_budgets(conn, org_id=uuid4(), defaults=defaults)
    assert out.daily_tokens == 50_000  # override won
    assert out.monthly_tokens == 10_000  # default kept
    assert out.monthly_llm_calls == 500  # default kept


@pytest.mark.asyncio
async def test_resolve_budgets_override_zero_means_inherit() -> None:
    """A tenant explicitly setting 0 doesn't disable the server cap —
    it just means "I have nothing tenant-specific". Defaults win."""
    conn = _mock_conn(row=({"daily_tokens": 0},))
    defaults = Budgets(daily_tokens=1000)
    out = await resolve_budgets(conn, org_id=uuid4(), defaults=defaults)
    assert out.daily_tokens == 1000


# ---------------------------------------------------------------------------
# Forge pipeline — budget pre-flight rejects work before any LLM spend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forge_run_rejects_when_budget_exhausted(monkeypatch) -> None:
    """Driven through ForgePipeline.run() with a fake engine that
    returns counters already past cap. The run must short-circuit
    with status=rejected before insert_distill_job is called."""
    import z3rno_core.forge.pipeline as forge_module
    from z3rno_core.distill.llm_gateway import StubLLMGateway
    from z3rno_core.forge.pipeline import ForgeOptions, ForgePipeline

    options = ForgeOptions(budgets=Budgets(daily_tokens=1000))
    pipeline = ForgePipeline(gateway=StubLLMGateway(), options=options)

    # Stub resolve_budgets to return a non-empty Budgets, then have
    # check_budget raise.
    async def _stub_resolve(_conn, *, org_id, defaults):
        return defaults

    async def _stub_check(_conn, *, org_id, budgets, today=None):
        from z3rno_core.usage import BudgetExceededError
        raise BudgetExceededError(
            kind="tokens", window="daily", used=2000, cap=1000
        )

    monkeypatch.setattr(forge_module, "resolve_budgets", _stub_resolve)
    monkeypatch.setattr(forge_module, "check_budget", _stub_check)

    # The pipeline calls ``engine.begin() as conn`` for its pre-flight.
    # Patch a fake engine that yields a mock conn.
    fake_conn = MagicMock()
    fake_conn.execute = AsyncMock()
    fake_conn.run_sync = AsyncMock()

    class _FakeEngineCtx:
        async def __aenter__(self) -> object:
            return fake_conn

        async def __aexit__(self, *_args: object) -> None:
            return None

    fake_engine = MagicMock()
    fake_engine.begin = _FakeEngineCtx

    summary = await pipeline.run(
        fake_engine,
        org_id=uuid4(),
        agent_id=uuid4(),
        memory_ids=[uuid4()],
    )
    assert summary.status == "rejected"
    assert "tokens" in (summary.error or "")
    # Crucially: the pre-flight ran before the bootstrap path —
    # which would have called fake_conn.execute() for insert/update.
    # We only expect set_org_context to have run (SET LOCAL …).
    update_calls = [
        c
        for c in fake_conn.execute.await_args_list
        if hasattr(c.args[0], "text")
        and "INSERT INTO distill_jobs" in c.args[0].text
    ]
    assert update_calls == [], "Forge should not have written distill_jobs"


# ---------------------------------------------------------------------------
# Refine pipeline — same pre-flight contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refine_run_rejects_when_budget_exhausted(monkeypatch) -> None:
    import z3rno_core.refine.pipeline as refine_module
    from z3rno_core.refine.pipeline import RefineOptions, RefinePipeline

    async def _stub_resolve(_conn, *, org_id, defaults):
        return defaults

    async def _stub_check(_conn, *, org_id, budgets, today=None):
        from z3rno_core.usage import BudgetExceededError
        raise BudgetExceededError(
            kind="llm_calls", window="monthly", used=1500, cap=1000
        )

    monkeypatch.setattr(refine_module, "resolve_budgets", _stub_resolve)
    monkeypatch.setattr(refine_module, "check_budget", _stub_check)

    options = RefineOptions(budgets=Budgets(monthly_llm_calls=1000))
    pipeline = RefinePipeline(options=options)

    conn = MagicMock()
    conn.execute = AsyncMock()

    summary = await pipeline.run(conn, org_id=uuid4())
    assert summary.status == "rejected"
    assert "llm_calls" in (summary.error or "")
    # No insert_refine_job INSERT should have happened.
    insert_calls = [
        c
        for c in conn.execute.await_args_list
        if hasattr(c.args[0], "text")
        and "INSERT INTO refine_jobs" in c.args[0].text
    ]
    assert insert_calls == [], "Refine should not have written refine_jobs"


@pytest.mark.asyncio
async def test_refine_run_skips_preflight_when_budgets_empty(monkeypatch) -> None:
    """Empty Budgets → fast-path no-op: ``resolve_budgets`` is never
    called even before the body runs."""
    import z3rno_core.refine.pipeline as refine_module
    from z3rno_core.refine.pipeline import RefineOptions, RefinePipeline

    called = {"check": 0}

    async def _stub_resolve(*_a, **_k):
        called["check"] += 1
        return Budgets()

    monkeypatch.setattr(refine_module, "resolve_budgets", _stub_resolve)

    # Make the dedupe stage raise so we don't run the rest of the
    # pipeline — we only care about whether resolve_budgets ran
    # before the body.
    async def _raise_dedupe(*_a, **_k):
        raise RuntimeError("stop-test-here")

    monkeypatch.setattr(refine_module, "run_dedupe", _raise_dedupe)

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(refine_module, "insert_refine_job", _noop)
    monkeypatch.setattr(refine_module, "update_refine_job", _noop)

    options = RefineOptions()  # budgets=None
    pipeline = RefinePipeline(options=options)
    conn = MagicMock()
    conn.execute = AsyncMock()

    # Body raises (stub dedupe); we only care that resolve_budgets
    # was skipped before the body ran.
    with pytest.raises(RuntimeError, match="stop-test-here"):
        await pipeline.run(conn, org_id=uuid4())
    assert called["check"] == 0
