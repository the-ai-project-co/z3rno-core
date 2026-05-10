"""Unified ``MissingExtraError`` for "this feature needs an extra".

Subsystems that depend on opt-in pip extras (Playwright, multimodal-
local) used to raise different exception classes for the same conceptual
failure — operators had to learn two different signals. This module
centralises the contract:

  * One ``MissingExtraError`` class with structured fields
    (``extra_name``, ``dependency``, ``install_command``) so
    callers can render a precise message or programmatically install.
  * Subsystems still raise their own typed errors, but those errors
    now *also* inherit from ``MissingExtraError`` (multiple inheritance
    of two ``Exception`` subclasses is fine — no shared state).
    Existing ``except MultimodalProviderError`` catches keep working;
    new ``except MissingExtraError`` catches both subsystems uniformly.

Usage from a subsystem::

    from z3rno_core.extras import MissingExtraError

    class MyMissingExtra(MyDomainError, MissingExtraError):
        pass

    raise MyMissingExtra.for_extra(
        extra_name="multimodal-local",
        dependency="sentence-transformers",
        install_command="pip install 'z3rno-core[multimodal-local]'",
    )
"""

from __future__ import annotations


class MissingExtraError(Exception):
    """Raised when an opt-in pip extra is required but not installed.

    The structured fields let callers build precise diagnostics:

      * ``extra_name`` — the extra name as it appears in pyproject.toml,
        e.g. ``"multimodal-local"`` or ``"playwright"``.
      * ``dependency`` — the specific package that failed to import.
      * ``install_command`` — a shell-ready command the operator can copy.
    """

    extra_name: str = ""
    dependency: str = ""
    install_command: str = ""

    def __init__(
        self,
        message: str,
        *,
        extra_name: str = "",
        dependency: str = "",
        install_command: str = "",
    ) -> None:
        super().__init__(message)
        self.extra_name = extra_name
        self.dependency = dependency
        self.install_command = install_command

    @classmethod
    def for_extra(
        cls,
        *,
        extra_name: str,
        dependency: str,
        install_command: str | None = None,
        action: str = "",
    ) -> MissingExtraError:
        """Build a properly-shaped instance for the common case.

        ``action`` is an optional trailing instruction (e.g. "and run
        ``playwright install chromium`` once.").
        """
        cmd = install_command or f"pip install 'z3rno-core[{extra_name}]'"
        msg = (
            f"{dependency} is required (extra '{extra_name}'); "
            f"install with `{cmd}`"
        )
        if action:
            msg = f"{msg} {action}"
        return cls(
            msg,
            extra_name=extra_name,
            dependency=dependency,
            install_command=cmd,
        )
