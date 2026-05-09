"""Provider-agnostic LLM Gateway for the Forge pipeline (Phase A).

The gateway is the single seam through which every Phase A LLM call flows.
It exposes two operations:

  - ``complete(...)``                — free-text completion
  - ``complete_structured(..., response_model=...)`` — structured output
                                        (Pydantic model) via Instructor

Both operations are provider-agnostic: routing happens at the model string
("openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-latest",
"gemini/gemini-2.0-flash-exp", "ollama/llama3.1:8b", etc.) using LiteLLM's
unified interface.

Resilience is built in:
  - per-call timeout (``LLMTimeoutError`` on expiry)
  - bounded retries with exponential backoff via ``tenacity``
  - distinct exception types for transient vs. terminal failures

The gateway is **stateless and import-safe**. Constructing a gateway makes
no network calls; the first network call happens only when ``complete*`` is
invoked. This is critical for Phase A's opt-in design — modules that import
the gateway must not fail to load when ``DISTILL_ENABLED=false``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

import litellm
import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

T = TypeVar("T", bound="BaseModel")
log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMGatewayError(Exception):
    """Base exception for every gateway failure."""


class LLMTimeoutError(LLMGatewayError):
    """The LLM call exceeded its per-call timeout."""


class LLMRateLimitError(LLMGatewayError):
    """Provider rate-limited us; transient, retried by the gateway."""


class LLMProviderError(LLMGatewayError):
    """Terminal provider error (auth, model-not-found, malformed request)."""


class LLMValidationError(LLMGatewayError):
    """Structured output failed Pydantic validation after retries."""


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class LLMGateway(ABC):
    """Abstract LLM gateway interface.

    Implementations are stateless and import-safe. Construction does no I/O.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier in LiteLLM's namespaced format."""

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a free-text completion."""

    @abstractmethod
    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> T:
        """Generate a structured (Pydantic-validated) completion."""


# ---------------------------------------------------------------------------
# LiteLLM + Instructor implementation
# ---------------------------------------------------------------------------


class LiteLLMGateway(LLMGateway):
    """Default Phase A gateway: LiteLLM for routing, Instructor for structured output.

    Construction parameters mirror ``z3rno_server.config.Settings`` Phase A
    fields. Pass them explicitly so this module never imports the server
    settings (clean dependency direction: server → core).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        instructor_mode: str | None = None,
    ) -> None:
        if not model:
            raise ValueError("LiteLLMGateway requires a non-empty model identifier")
        self._model = model
        self._api_key = api_key or None
        self._timeout = timeout_seconds
        self._max_retries = max(1, max_retries)
        self._instructor_mode = instructor_mode

    @property
    def model_name(self) -> str:
        return self._model

    # ---- public API ------------------------------------------------------

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Free-text completion with timeout + retries."""
        kwargs = self._build_kwargs(system, user, max_tokens, temperature)
        response = await self._with_resilience(litellm.acompletion, **kwargs)
        try:
            return str(response.choices[0].message.content or "")
        except (AttributeError, IndexError, KeyError) as exc:
            raise LLMProviderError(f"unexpected LiteLLM response shape: {exc}") from exc

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> T:
        """Structured completion via Instructor; returns a validated Pydantic model."""
        # Lazy import: instructor's import-time setup touches OpenAI client classes
        # we don't want loaded unless distillation is actually invoked. This keeps
        # `import z3rno_core.distill` safe with DISTILL_ENABLED=false.
        import instructor  # noqa: PLC0415
        from pydantic import ValidationError  # noqa: PLC0415

        client = instructor.from_litellm(litellm.acompletion)
        kwargs = self._build_kwargs(system, user, max_tokens, temperature)
        kwargs["response_model"] = response_model
        # Instructor exposes its own retry parameter; we still wrap in tenacity
        # for transient transport errors.
        kwargs["max_retries"] = 1

        async def _call() -> T:
            return await client.create(**kwargs)  # type: ignore[no-any-return]

        try:
            result = await self._with_resilience(_call)
        except ValidationError as exc:
            raise LLMValidationError(str(exc)) from exc
        return result  # type: ignore[no-any-return]

    # ---- internals -------------------------------------------------------

    def _build_kwargs(
        self,
        system: str,
        user: str,
        max_tokens: int | None,
        temperature: float,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "timeout": self._timeout,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return kwargs

    async def _with_resilience(self, func: Any, **kwargs: Any) -> Any:
        """Run ``func(**kwargs)`` with timeout + bounded retry on transient errors."""
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                retry=retry_if_exception_type(
                    (LLMRateLimitError, LLMTimeoutError, ConnectionError)
                ),
                reraise=True,
            ):
                with attempt:
                    return await self._run_once(func, **kwargs)
        except RetryError as exc:  # pragma: no cover — tenacity reraises by default
            raise LLMGatewayError("retry budget exhausted") from exc
        # Unreachable: AsyncRetrying always returns or raises inside the loop.
        raise LLMGatewayError("retry loop exited without result")  # pragma: no cover

    async def _run_once(self, func: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.wait_for(func(**kwargs), timeout=self._timeout)
        except TimeoutError as exc:
            log.warning("llm_gateway.timeout", model=self._model, timeout=self._timeout)
            raise LLMTimeoutError(f"LLM call timed out after {self._timeout}s") from exc
        except Exception as exc:
            # Map common LiteLLM exceptions onto our taxonomy.
            cls = type(exc).__name__
            if "RateLimit" in cls or "TooManyRequests" in cls:
                raise LLMRateLimitError(str(exc)) from exc
            if isinstance(exc, LLMGatewayError):
                raise
            # Anything else is terminal — let the caller see the original cause.
            raise LLMProviderError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Test double — useful for unit tests and the NoOp / dry-run path
# ---------------------------------------------------------------------------


class StubLLMGateway(LLMGateway):
    """Deterministic gateway for tests. No network calls.

    Configure responses by passing ``completion`` and ``structured`` factories.
    Each factory receives ``(system, user)`` and returns the response payload.
    """

    def __init__(
        self,
        *,
        model: str = "stub/test",
        completion: Any = None,
        structured: Any = None,
    ) -> None:
        self._model = model
        self._completion = completion
        self._structured = structured

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        if self._completion is None:
            return ""
        return str(self._completion(system, user))

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> T:
        if self._structured is None:
            return response_model()
        result: T = self._structured(system, user, response_model)
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm_gateway(
    *,
    provider: str = "litellm",
    model: str,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
) -> LLMGateway:
    """Construct an LLM gateway by provider name.

    ``provider="stub"`` returns an empty :class:`StubLLMGateway` for tests
    that do not configure response factories.
    """
    if provider == "stub":
        return StubLLMGateway(model=model)
    if provider == "litellm":
        return LiteLLMGateway(
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    raise ValueError(f"unknown LLM gateway provider: {provider!r}")
