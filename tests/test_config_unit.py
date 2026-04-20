"""Unit tests for z3rno_core.config — ConnectionConfig and engine factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from z3rno_core.config import ConnectionConfig, create_async_engine_from_config


class TestConnectionConfig:
    """Test ConnectionConfig dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = ConnectionConfig()
        assert cfg.pool_size == 5
        assert cfg.max_overflow == 10
        assert cfg.pool_timeout == 30
        assert cfg.pool_recycle == 1800
        assert cfg.pool_pre_ping is True
        assert cfg.echo is False

    def test_custom_values(self) -> None:
        cfg = ConnectionConfig(pool_size=20, echo=True)
        assert cfg.pool_size == 20
        assert cfg.echo is True


class TestCreateAsyncEngineFromConfig:
    """Test create_async_engine_from_config factory."""

    @patch("z3rno_core.config.create_async_engine")
    def test_uses_default_config_when_none(self, mock_create: MagicMock) -> None:
        create_async_engine_from_config("postgresql+asyncpg://localhost/test")

        assert mock_create.called
        _, kwargs = mock_create.call_args
        assert kwargs["pool_size"] == 5
        assert kwargs["pool_pre_ping"] is True

    @patch("z3rno_core.config.create_async_engine")
    def test_uses_provided_config(self, mock_create: MagicMock) -> None:
        cfg = ConnectionConfig(pool_size=42, max_overflow=100)
        create_async_engine_from_config("postgresql+asyncpg://localhost/test", cfg)
        _, kwargs = mock_create.call_args
        assert kwargs["pool_size"] == 42
        assert kwargs["max_overflow"] == 100
