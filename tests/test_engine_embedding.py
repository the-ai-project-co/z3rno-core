"""Unit tests for z3rno_core.engine.embedding.

Tests the NoOpEmbeddingProvider and get_embedding_provider factory.
No database or API calls needed.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

from z3rno_core.engine.embedding import (
    EmbeddingProvider,
    LiteLLMEmbeddingProvider,
    NoOpEmbeddingProvider,
    get_embedding_provider,
)

# ---------------------------------------------------------------------------
# NoOpEmbeddingProvider
# ---------------------------------------------------------------------------


class TestNoOpEmbeddingProvider:
    """Test the no-op embedding provider used in tests."""

    def test_model_name_is_noop(self) -> None:
        provider = NoOpEmbeddingProvider()
        assert provider.model_name == "noop"

    def test_embed_text_returns_empty_list(self) -> None:
        provider = NoOpEmbeddingProvider()
        result = asyncio.run(provider.embed_text("hello world"))
        assert result == []

    def test_embed_batch_returns_empty_lists(self) -> None:
        provider = NoOpEmbeddingProvider()
        texts = ["hello", "world", "test"]
        result = asyncio.run(provider.embed_batch(texts))
        assert result == [[], [], []]

    def test_embed_batch_empty_input(self) -> None:
        provider = NoOpEmbeddingProvider()
        result = asyncio.run(provider.embed_batch([]))
        assert result == []

    def test_dimension_property(self) -> None:
        provider = NoOpEmbeddingProvider()
        assert isinstance(provider.dimension, int)
        assert provider.dimension > 0

    def test_is_embedding_provider_subclass(self) -> None:
        provider = NoOpEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)


# ---------------------------------------------------------------------------
# LiteLLMEmbeddingProvider (construction only, no API calls)
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingProvider:
    """Test LiteLLMEmbeddingProvider construction (no API calls)."""

    def test_default_model_name(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EMBEDDING_MODEL", None)
            provider = LiteLLMEmbeddingProvider()
            assert provider.model_name == "text-embedding-3-small"

    def test_custom_model_name(self) -> None:
        provider = LiteLLMEmbeddingProvider(model="text-embedding-ada-002")
        assert provider.model_name == "text-embedding-ada-002"

    def test_model_from_env(self) -> None:
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "cohere-embed-v3"}, clear=False):
            provider = LiteLLMEmbeddingProvider()
            assert provider.model_name == "cohere-embed-v3"

    def test_explicit_model_overrides_env(self) -> None:
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "from-env"}, clear=False):
            provider = LiteLLMEmbeddingProvider(model="explicit")
            assert provider.model_name == "explicit"

    def test_is_embedding_provider_subclass(self) -> None:
        provider = LiteLLMEmbeddingProvider(model="test")
        assert isinstance(provider, EmbeddingProvider)


# ---------------------------------------------------------------------------
# get_embedding_provider factory
# ---------------------------------------------------------------------------


class TestGetEmbeddingProvider:
    """Test the factory function."""

    def test_noop_provider_by_arg(self) -> None:
        provider = get_embedding_provider("noop")
        assert isinstance(provider, NoOpEmbeddingProvider)

    def test_litellm_provider_by_arg(self) -> None:
        provider = get_embedding_provider("litellm")
        assert isinstance(provider, LiteLLMEmbeddingProvider)

    def test_default_is_litellm(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EMBEDDING_PROVIDER", None)
            provider = get_embedding_provider()
            assert isinstance(provider, LiteLLMEmbeddingProvider)

    def test_noop_from_env(self) -> None:
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "noop"}, clear=False):
            provider = get_embedding_provider()
            assert isinstance(provider, NoOpEmbeddingProvider)

    def test_unknown_provider_returns_litellm(self) -> None:
        provider = get_embedding_provider("unknown-provider")
        assert isinstance(provider, LiteLLMEmbeddingProvider)
