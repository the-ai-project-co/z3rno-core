"""Connection and engine configuration for z3rno-core.

Provides ``ConnectionConfig`` — a dataclass that exposes the SQLAlchemy
pool tuning knobs (pool_size, max_overflow, pool_timeout, pool_recycle)
and a helper to create async engines with those settings.

Usage::

    from z3rno_core.config import ConnectionConfig, create_async_engine_from_config

    cfg = ConnectionConfig(pool_size=10, max_overflow=20)
    engine = create_async_engine_from_config("postgresql+asyncpg://...", cfg)
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


@dataclass(frozen=True)
class ConnectionConfig:
    """Database connection-pool configuration.

    All values mirror the corresponding SQLAlchemy ``create_engine`` /
    ``create_async_engine`` keyword arguments.

    Attributes:
        pool_size: Number of persistent connections to keep in the pool.
            Default ``5`` (SQLAlchemy default).
        max_overflow: Maximum number of connections allowed *above*
            ``pool_size`` during bursts.  Default ``10``.
        pool_timeout: Seconds to wait for a connection from the pool
            before raising ``TimeoutError``.  Default ``30``.
        pool_recycle: Seconds after which an idle connection is recycled
            (helps avoid stale connections behind PgBouncer or cloud
            proxies).  Default ``1800`` (30 minutes).  Set to ``-1``
            to disable recycling.
        pool_pre_ping: If ``True``, each connection is tested with a
            lightweight ``SELECT 1`` before being handed out.
            Default ``True``.
        echo: If ``True``, emit SQL statements to the log (useful for
            debugging).  Default ``False``.
    """

    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800
    pool_pre_ping: bool = True
    echo: bool = False


def create_async_engine_from_config(
    url: str,
    config: ConnectionConfig | None = None,
) -> AsyncEngine:
    """Create a SQLAlchemy ``AsyncEngine`` using the given connection config.

    Args:
        url: Database URL (e.g. ``postgresql+asyncpg://user:pass@host/db``).
        config: Optional ``ConnectionConfig``.  When ``None``, the
            SQLAlchemy/z3rno defaults from ``ConnectionConfig()`` are used.

    Returns:
        A configured ``AsyncEngine`` instance.
    """
    if config is None:
        config = ConnectionConfig()

    return create_async_engine(
        url,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_timeout=config.pool_timeout,
        pool_recycle=config.pool_recycle,
        pool_pre_ping=config.pool_pre_ping,
        echo=config.echo,
    )
