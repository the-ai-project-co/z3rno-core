"""Pluggable embedding provider abstraction.

The engine uses an EmbeddingProvider to generate vector embeddings for
memory content. The default provider uses LiteLLM which routes to any
supported backend (OpenAI, Cohere, local models, etc.).

Configuration:
  EMBEDDING_MODEL env var (default: text-embedding-3-small)
  EMBEDDING_DIMENSION env var (default: 1536)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import litellm

from z3rno_core.models.memory import EMBEDDING_DIMENSION


class EmbeddingProvider(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return EMBEDDING_DIMENSION


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using LiteLLM (routes to OpenAI, Cohere, etc.).

    Uses the EMBEDDING_MODEL environment variable to select the model.
    Default: text-embedding-3-small (OpenAI, 1536 dims per ADR-001).
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding via LiteLLM."""
        response = await litellm.aembedding(model=self._model, input=[text])
        return response.data[0]["embedding"]  # type: ignore[no-any-return]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch via LiteLLM."""
        response = await litellm.aembedding(model=self._model, input=texts)
        return [item["embedding"] for item in response.data]


class NoOpEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that returns None (for testing without API calls).

    Use this in tests or when embeddings will be generated asynchronously
    by the worker.
    """

    @property
    def model_name(self) -> str:
        return "noop"

    async def embed_text(self, text: str) -> list[float]:
        """Return an empty list (no embedding)."""
        return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return empty lists."""
        return [[] for _ in texts]


def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    """Factory function to get the configured embedding provider.

    Args:
        provider: Override the provider type. If None, uses EMBEDDING_PROVIDER
                  env var (default: "litellm").
    """
    provider_type = provider or os.environ.get("EMBEDDING_PROVIDER", "litellm")
    if provider_type == "noop":
        return NoOpEmbeddingProvider()
    return LiteLLMEmbeddingProvider()
