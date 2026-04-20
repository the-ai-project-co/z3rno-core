"""Unit tests for z3rno_core.engine.embedding — LiteLLMEmbeddingProvider.

Tests embed_text() and embed_batch() by mocking litellm.aembedding().
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from z3rno_core.engine.embedding import LiteLLMEmbeddingProvider

# ---------------------------------------------------------------------------
# LiteLLMEmbeddingProvider.embed_text
# ---------------------------------------------------------------------------


class TestLiteLLMEmbedText:
    """Test embed_text() calls litellm.aembedding correctly."""

    async def test_embed_text_returns_embedding(self) -> None:
        """embed_text() returns the embedding vector from the response."""
        provider = LiteLLMEmbeddingProvider(model="test-model")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]

        with patch(
            "z3rno_core.engine.embedding.litellm.aembedding",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_aembedding:
            result = await provider.embed_text("hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_aembedding.assert_awaited_once_with(model="test-model", input=["hello world"])

    async def test_embed_text_wraps_text_in_list(self) -> None:
        """embed_text() passes the text as a single-element list."""
        provider = LiteLLMEmbeddingProvider(model="m")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [1.0]}]

        with patch(
            "z3rno_core.engine.embedding.litellm.aembedding",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_aembedding:
            await provider.embed_text("test")

        # Verify input is a list with a single string
        call_kwargs = mock_aembedding.call_args[1]
        assert call_kwargs["input"] == ["test"]

    async def test_embed_text_uses_configured_model(self) -> None:
        """embed_text() passes the configured model name."""
        provider = LiteLLMEmbeddingProvider(model="text-embedding-ada-002")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.5]}]

        with patch(
            "z3rno_core.engine.embedding.litellm.aembedding",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_aembedding:
            await provider.embed_text("x")

        call_kwargs = mock_aembedding.call_args[1]
        assert call_kwargs["model"] == "text-embedding-ada-002"


# ---------------------------------------------------------------------------
# LiteLLMEmbeddingProvider.embed_batch
# ---------------------------------------------------------------------------


class TestLiteLLMEmbedBatch:
    """Test embed_batch() calls litellm.aembedding correctly."""

    async def test_embed_batch_returns_list_of_embeddings(self) -> None:
        """embed_batch() returns a list of embedding vectors."""
        provider = LiteLLMEmbeddingProvider(model="test-model")
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]

        with patch(
            "z3rno_core.engine.embedding.litellm.aembedding",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.embed_batch(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    async def test_embed_batch_passes_all_texts(self) -> None:
        """embed_batch() passes all texts in a single call."""
        provider = LiteLLMEmbeddingProvider(model="m")
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [1.0]},
            {"embedding": [2.0]},
            {"embedding": [3.0]},
        ]

        with patch(
            "z3rno_core.engine.embedding.litellm.aembedding",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_aembedding:
            await provider.embed_batch(["a", "b", "c"])

        mock_aembedding.assert_awaited_once_with(model="m", input=["a", "b", "c"])

    async def test_embed_batch_empty_list(self) -> None:
        """embed_batch() with empty list returns empty list."""
        provider = LiteLLMEmbeddingProvider(model="m")
        mock_response = MagicMock()
        mock_response.data = []

        with patch(
            "z3rno_core.engine.embedding.litellm.aembedding",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.embed_batch([])

        assert result == []


# ---------------------------------------------------------------------------
# LiteLLMEmbeddingProvider properties
# ---------------------------------------------------------------------------


class TestLiteLLMProperties:
    """Test model_name and default model selection."""

    def test_model_name_returns_configured_model(self) -> None:
        provider = LiteLLMEmbeddingProvider(model="custom-model")
        assert provider.model_name == "custom-model"

    def test_default_model(self) -> None:
        provider = LiteLLMEmbeddingProvider()
        assert provider.model_name == "text-embedding-3-small"
