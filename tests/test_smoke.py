"""Smoke tests proving the package imports cleanly.

These tests run on every CI build and act as the first line of defense
against accidental breakage of the package layout. As real engine code lands,
this file gets supplemented (not replaced) by per-module test files.
"""

from __future__ import annotations


def test_package_imports() -> None:
    """The top-level package and every subpackage import without errors."""
    import z3rno_core
    import z3rno_core.engine
    import z3rno_core.graph
    import z3rno_core.models
    import z3rno_core.security
    import z3rno_core.temporal

    # __version__ is the only attribute exported at this stage
    assert hasattr(z3rno_core, "__version__")
    assert isinstance(z3rno_core.__version__, str)
    assert z3rno_core.__version__.count(".") >= 2


def test_subpackages_have_all() -> None:
    """Every subpackage declares an explicit ``__all__`` (even if empty)."""
    import z3rno_core.engine
    import z3rno_core.graph
    import z3rno_core.models
    import z3rno_core.security
    import z3rno_core.temporal

    for module in (
        z3rno_core.engine,
        z3rno_core.graph,
        z3rno_core.models,
        z3rno_core.security,
        z3rno_core.temporal,
    ):
        assert hasattr(module, "__all__"), f"{module.__name__} is missing __all__"
        assert isinstance(module.__all__, list)


def test_sentinel_fixture(sentinel: str) -> None:
    """Conftest is wired up correctly."""
    assert sentinel == "z3rno-core test suite"
