"""Shared pytest fixtures for the z3rno-core test suite.

Populated as the engine code lands. For now this file exists so pytest
discovers the tests/ directory as a test package and so we have a single
known location for future fixtures (database, redis, factories, etc.).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def sentinel() -> str:
    """Trivial fixture proving conftest.py is loaded.

    Remove once real fixtures land.
    """
    return "z3rno-core test suite"
