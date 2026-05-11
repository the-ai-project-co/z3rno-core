#!/usr/bin/env python3
"""v0.20.1 — CI drift guard for PyPI.

Asserts ``pyproject.toml`` version is **strictly greater** than the
latest version published on PyPI. Fails the CI build otherwise so the
next tag push can't repeat the v0.11.0 silent-failure scenario (where
``pyproject.toml`` was never bumped past 0.11.0 and every release
workflow run rejected with HTTP 400 "file already exists").

Exit codes:
  0   local version > PyPI latest  (safe to publish)
  1   local version <= PyPI latest (drift — bump pyproject before merging)
  2   PyPI lookup failed / network error

Drop-in for every repo that publishes to PyPI. The TS SDK has its own
npm-side equivalent in ``check_npm_drift.py``.
"""

from __future__ import annotations

import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


def _load_local_version(pyproject_path: Path) -> str:
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return str(data["project"]["version"])


def _fetch_pypi_latest(package: str) -> str | None:
    """Return the highest version on PyPI for ``package``.

    Returns ``None`` if the package isn't on PyPI yet (first publish
    is a no-op for the guard — there's nothing to drift against).
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            import json

            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    versions = list(data.get("releases", {}).keys())
    if not versions:
        return None
    # PyPI returns dict keys unordered; sort by tuple-of-ints.
    return max(versions, key=_version_tuple)


def _version_tuple(v: str) -> tuple[int, ...]:
    """Coerce ``"0.19.0"`` → ``(0, 19, 0)``. Non-numeric segments
    (alphas / betas / rc) sort *before* the bare release."""
    parts: list[int] = []
    for raw in v.split("."):
        try:
            parts.append(int(raw))
        except ValueError:
            # ``0.19.0rc1`` → split numeric prefix
            num = ""
            for ch in raw:
                if ch.isdigit():
                    num += ch
                else:
                    break
            parts.append(int(num) if num else 0)
            break
    return tuple(parts)


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: check_pypi_drift.py <package-name> [pyproject.toml]", file=sys.stderr)
        return 2

    package = argv[0]
    pyproject_path = Path(argv[1] if len(argv) > 1 else "pyproject.toml")
    if not pyproject_path.is_file():
        print(f"FAIL: {pyproject_path} not found", file=sys.stderr)
        return 2

    local = _load_local_version(pyproject_path)
    try:
        remote = _fetch_pypi_latest(package)
    except Exception as exc:  # noqa: BLE001 — surface network issues clearly
        print(f"FAIL: PyPI lookup for {package!r} crashed: {exc}", file=sys.stderr)
        return 2

    if remote is None:
        print(f"OK: {package} not yet on PyPI; local={local}")
        return 0

    if _version_tuple(local) > _version_tuple(remote):
        print(f"OK: {package} local={local} > PyPI={remote}")
        return 0

    print(
        f"FAIL: {package} local={local} is NOT strictly greater than "
        f"PyPI={remote}. Bump ``version`` in {pyproject_path} before "
        f"merging — the release workflow will 400 on the next tag.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
