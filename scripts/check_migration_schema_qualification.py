#!/usr/bin/env python3
"""Reject migrations that issue unqualified DDL.

Apache AGE puts ``ag_catalog`` first on the database's ``search_path``.
An unqualified ``CREATE TABLE foo`` in an Alembic migration lands in
``ag_catalog``, not ``public``. The application role ``z3rno_app`` has
no ``USAGE`` on ``ag_catalog``, so RLS-correct CRUD against the
mis-placed table fails with ``relation "foo" does not exist`` — which
is exactly how the v0.7.0 incident surfaced (Migrations 015 + 016
created ``distill_jobs``, ``entity_provenance``, ``datasets``,
``ingest_jobs`` in ``ag_catalog`` and the bug stayed hidden until
``SET LOCAL ROLE z3rno_app`` integration tests ran).

This script scans ``migrations/versions/*.py`` for the bug pattern:

  * ``CREATE TABLE foo (...)`` — must be ``public.foo``
  * ``CREATE INDEX … ON foo (...)`` — must be ``public.foo``
  * ``ALTER TABLE foo …`` — must be ``public.foo`` (or ``ag_catalog.foo``
    in a deliberate cleanup migration)

The check tolerates Alembic conventions:
  - ``IF [NOT] EXISTS`` between the keyword and the identifier
  - ``ONLY`` (for partitioned tables)
  - Python f-string interpolation as the identifier slot —
    ``{table}`` is treated like a bare identifier and must still be
    schema-qualified at the source level (``public.{table}``).

Allowed identifiers (system catalogues + Alembic bookkeeping) are
exempt from the check; see ``_ALLOWED_BARE``.

Exit code 0 = clean, 1 = at least one offence found (with line refs).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "migrations" / "versions"

# An "identifier slot" is what should sit where a schema-qualified table
# name belongs: either a bare ident or a Python f-string interpolation.
# We capture the optional schema prefix separately so we can verify it.
#
# Pattern shape:
#   CREATE TABLE  [IF NOT EXISTS]   <ident>
#   CREATE INDEX  [name] ON [ONLY]  <ident>
#   ALTER  TABLE  [IF EXISTS] [ONLY] <ident>
#
# Where <ident> = (schema . )?  (bare-ident | {f-string-expr})
#
# Group 1 = schema (or None), group 2 = bare ident (or None when group 3 set),
# group 3 = f-string expr text (or None when group 2 set).
_IDENT = (
    r"(?:([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*)?"
    r"(?:([A-Za-z_][A-Za-z0-9_]*)|\{([^}]*)\})"
)

_PATTERNS = [
    # CREATE TABLE [IF NOT EXISTS] <ident>
    (
        "CREATE TABLE",
        re.compile(
            r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?" + _IDENT,
            re.IGNORECASE,
        ),
    ),
    # CREATE [UNIQUE] INDEX [CONCURRENTLY] [IF NOT EXISTS] [name] ON [ONLY] <ident>
    (
        "CREATE INDEX",
        re.compile(
            r"\bCREATE\s+(?:UNIQUE\s+)?INDEX\s+"
            r"(?:CONCURRENTLY\s+)?"
            r"(?:IF\s+NOT\s+EXISTS\s+)?"
            r"(?:[A-Za-z_][A-Za-z0-9_]*\s+)?"
            r"ON\s+(?:ONLY\s+)?" + _IDENT,
            re.IGNORECASE,
        ),
    ),
    # ALTER TABLE [IF EXISTS] [ONLY] <ident>
    (
        "ALTER TABLE",
        re.compile(
            r"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?" + _IDENT,
            re.IGNORECASE,
        ),
    ),
]

# Bare names that legitimately live outside ``public`` and can be referenced
# unqualified. Add to this list (with reasoning) as new exemptions emerge.
_ALLOWED_BARE = {
    "alembic_version",  # alembic's bookkeeping table; it does its own thing
}


def _scan_one_file(path: Path) -> list[tuple[int, str, str, str]]:
    """Return ``(lineno, ddl_kind, identifier, snippet)`` per offence."""
    findings: list[tuple[int, str, str, str]] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        # Drop trailing Python comments cheaply (good enough for migrations;
        # anything inside an actual string literal that contains '#' is rare
        # and benign for this scanner).
        stripped = re.sub(r"(?<![\"'])#.*$", "", line)
        for kind, pat in _PATTERNS:
            for m in pat.finditer(stripped):
                schema = m.group(1)
                bare = m.group(2)
                fstring = m.group(3)
                ident_repr = bare if bare is not None else "{" + (fstring or "") + "}"
                if schema:
                    # Schema explicit (e.g. ``public.foo`` or ``ag_catalog.foo``)
                    # — author made an intentional choice; trust it.
                    continue
                if bare and bare.lower() in _ALLOWED_BARE:
                    continue
                findings.append((lineno, kind, ident_repr, line.strip()))
    return findings


def main() -> int:
    if not MIGRATIONS_DIR.is_dir():
        # No migrations directory — pre-commit runs on multiple repos; not
        # every repo has migrations. Skip silently.
        return 0

    total = 0
    for path in sorted(MIGRATIONS_DIR.glob("*.py")):
        if path.name.startswith("__"):
            continue
        findings = _scan_one_file(path)
        if not findings:
            continue
        rel = path.relative_to(REPO_ROOT)
        for lineno, kind, ident, snippet in findings:
            total += 1
            print(
                f"{rel}:{lineno}: unqualified {kind} on '{ident}'\n"
                f"    {snippet}",
                file=sys.stderr,
            )

    if total:
        print(
            "\n"
            f"{total} unqualified DDL statement(s) in migrations.\n"
            "\n"
            "Apache AGE puts ag_catalog first on search_path. Unqualified\n"
            "CREATE TABLE / CREATE INDEX / ALTER TABLE will land there\n"
            "instead of public, breaking RLS-correct CRUD via z3rno_app.\n"
            "\n"
            "Fix: prefix the identifier with a schema. For application\n"
            "tables this is always 'public.':\n"
            "\n"
            "    op.execute('CREATE TABLE public.foo (...)')\n"
            "    op.execute('CREATE INDEX ix_foo_bar ON public.foo (bar)')\n"
            "    op.execute(f'ALTER TABLE public.{table} ENABLE RLS')\n"
            "\n"
            "See migrations/versions/019_move_misplaced_tables_to_public.py\n"
            "for the cleanup pattern if a table has already shipped to\n"
            "the wrong schema.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
