"""Unit tests for z3rno_core.distill.llm_gateway.

No network, no LLM provider calls. Constructs gateways and exercises
the resilience + structured-output paths via a stub.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from z3rno_core.distill.llm_gateway import (
    LiteLLMGateway,
    LLMGateway,
    LLMGatewayError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMValidationError,
    StubLLMGateway,
    get_llm_gateway,
)


class _Toy(BaseModel):
    name: str = "default"
    value: int = 0


# ---------------------------------------------------------------------------
# Factory + base interface
# ---------------------------------------------------------------------------


class TestFactory:
    def test_stub_provider_returns_stub_gateway(self) -> None:
        g = get_llm_gateway(provider="stub", model="x/y")
        assert isinstance(g, StubLLMGateway)
        assert g.model_name == "x/y"

    def test_litellm_provider_returns_litellm_gateway(self) -> None:
        g = get_llm_gateway(provider="litellm", model="openai/gpt-4o-mini")
        assert isinstance(g, LiteLLMGateway)
        assert g.model_name == "openai/gpt-4o-mini"

    def test_unknown_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown LLM gateway provider"):
            get_llm_gateway(provider="unknown", model="x")

    def test_litellm_requires_model(self) -> None:
        with pytest.raises(ValueError, match="non-empty model"):
            LiteLLMGateway(model="")

    def test_max_retries_clamped_to_at_least_one(self) -> None:
        g = LiteLLMGateway(model="m", max_retries=0)
        assert g._max_retries == 1


# ---------------------------------------------------------------------------
# StubLLMGateway behavior
# ---------------------------------------------------------------------------


class TestStubGateway:
    def test_default_complete_returns_empty_string(self) -> None:
        g = StubLLMGateway()
        out = asyncio.run(g.complete(system="s", user="u"))
        assert out == ""

    def test_default_structured_returns_default_pydantic(self) -> None:
        g = StubLLMGateway()
        out = asyncio.run(g.complete_structured(system="s", user="u", response_model=_Toy))
        assert isinstance(out, _Toy)
        assert out.name == "default"

    def test_completion_factory_invoked(self) -> None:
        g = StubLLMGateway(completion=lambda s, u: f"sys={s}|user={u}")
        out = asyncio.run(g.complete(system="X", user="Y"))
        assert out == "sys=X|user=Y"

    def test_structured_factory_invoked(self) -> None:
        g = StubLLMGateway(
            structured=lambda s, u, m: m(name=u, value=42),
        )
        out = asyncio.run(g.complete_structured(system="s", user="hello", response_model=_Toy))
        assert out.name == "hello"
        assert out.value == 42

    def test_is_llm_gateway_subclass(self) -> None:
        assert issubclass(StubLLMGateway, LLMGateway)


# ---------------------------------------------------------------------------
# LiteLLMGateway resilience paths (mocking litellm.acompletion)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content: str = "ok") -> None:
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]


class TestLiteLLMGatewayResilience:
    def test_complete_returns_choices_content(self) -> None:
        g = LiteLLMGateway(model="m", timeout_seconds=5, max_retries=1)

        async def fake_acompletion(**kwargs: object) -> _FakeResponse:
            return _FakeResponse("hello")

        with patch("z3rno_core.distill.llm_gateway.litellm.acompletion", new=fake_acompletion):
            out = asyncio.run(g.complete(system="s", user="u"))
        assert out == "hello"

    def test_complete_unexpected_response_shape_raises_provider_error(self) -> None:
        g = LiteLLMGateway(model="m", timeout_seconds=5, max_retries=1)

        async def fake_acompletion(**kwargs: object) -> dict[str, str]:
            return {"unexpected": "shape"}

        with (
            patch("z3rno_core.distill.llm_gateway.litellm.acompletion", new=fake_acompletion),
            pytest.raises(LLMProviderError),
        ):
            asyncio.run(g.complete(system="s", user="u"))

    def test_complete_maps_rate_limit_class_name_to_rate_limit_error(self) -> None:
        # Build the gateway with retries=1 so we don't infinite-loop on transient errors.
        g = LiteLLMGateway(model="m", timeout_seconds=5, max_retries=1)

        class FakeRateLimitError(Exception):
            pass

        async def fake_acompletion(**kwargs: object) -> _FakeResponse:
            raise FakeRateLimitError("rate limited")

        with (
            patch("z3rno_core.distill.llm_gateway.litellm.acompletion", new=fake_acompletion),
            pytest.raises(LLMRateLimitError),
        ):
            asyncio.run(g.complete(system="s", user="u"))

    def test_complete_maps_generic_exception_to_provider_error(self) -> None:
        g = LiteLLMGateway(model="m", timeout_seconds=5, max_retries=1)

        async def fake_acompletion(**kwargs: object) -> _FakeResponse:
            raise RuntimeError("boom")

        with (
            patch("z3rno_core.distill.llm_gateway.litellm.acompletion", new=fake_acompletion),
            pytest.raises(LLMProviderError),
        ):
            asyncio.run(g.complete(system="s", user="u"))

    def test_complete_timeout_raises_llm_timeout_error(self) -> None:
        g = LiteLLMGateway(model="m", timeout_seconds=0.01, max_retries=1)

        async def fake_acompletion(**kwargs: object) -> _FakeResponse:
            await asyncio.sleep(0.5)
            return _FakeResponse("never")

        with (
            patch("z3rno_core.distill.llm_gateway.litellm.acompletion", new=fake_acompletion),
            pytest.raises(LLMTimeoutError),
        ):
            asyncio.run(g.complete(system="s", user="u"))

    def test_complete_passes_temperature_and_messages(self) -> None:
        g = LiteLLMGateway(model="m", api_key="sk-fake", timeout_seconds=5, max_retries=1)
        captured: dict[str, object] = {}

        async def fake_acompletion(**kwargs: object) -> _FakeResponse:
            captured.update(kwargs)
            return _FakeResponse("ok")

        with patch("z3rno_core.distill.llm_gateway.litellm.acompletion", new=fake_acompletion):
            asyncio.run(g.complete(system="SYS", user="USER", max_tokens=128, temperature=0.7))

        assert captured["model"] == "m"
        assert captured["temperature"] == 0.7
        assert captured["max_tokens"] == 128
        assert captured["api_key"] == "sk-fake"
        msgs = captured["messages"]
        assert msgs[0]["role"] == "system"  # type: ignore[index]
        assert msgs[0]["content"] == "SYS"  # type: ignore[index]
        assert msgs[1]["role"] == "user"  # type: ignore[index]
        assert msgs[1]["content"] == "USER"  # type: ignore[index]

    def test_complete_omits_api_key_when_unset(self) -> None:
        g = LiteLLMGateway(model="m", timeout_seconds=5, max_retries=1)
        captured: dict[str, object] = {}

        async def fake_acompletion(**kwargs: object) -> _FakeResponse:
            captured.update(kwargs)
            return _FakeResponse("ok")

        with patch("z3rno_core.distill.llm_gateway.litellm.acompletion", new=fake_acompletion):
            asyncio.run(g.complete(system="s", user="u"))

        assert "api_key" not in captured


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_all_subclasses_of_gateway_error(self) -> None:
        for cls in (LLMTimeoutError, LLMRateLimitError, LLMProviderError, LLMValidationError):
            assert issubclass(cls, LLMGatewayError)
