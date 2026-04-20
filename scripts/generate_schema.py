"""Auto-generate docs/SCHEMA.md from SQLAlchemy model metadata.

Usage:
    python scripts/generate_schema.py

Iterates all tables registered in Base.metadata and writes a Markdown
file documenting each table's columns, constraints, and indexes.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from z3rno_core.models import Base


def _col_type_str(col_type: object) -> str:
    """Render a column type as a human-readable string."""
    try:
        return str(col_type)
    except Exception:
        return type(col_type).__name__


def generate_schema_md() -> str:
    """Generate Markdown content from Base.metadata.tables."""
    lines: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# Z3rno Database Schema")
    lines.append("")
    lines.append(f"> Auto-generated from SQLAlchemy models on {now}.")
    lines.append("> Do not edit manually. Run `python scripts/generate_schema.py` to regenerate.")
    lines.append("")

    sorted_tables = sorted(Base.metadata.tables.values(), key=lambda t: t.name)

    # Table of contents
    lines.append("## Tables")
    lines.append("")
    for table in sorted_tables:
        lines.append(f"- [{table.name}](#{table.name})")
    lines.append("")

    # Table details
    for table in sorted_tables:
        lines.append("---")
        lines.append("")
        lines.append(f"## {table.name}")
        lines.append("")

        # Columns
        lines.append("### Columns")
        lines.append("")
        lines.append("| Column | Type | Nullable | Default |")
        lines.append("|--------|------|----------|---------|")

        for col in table.columns:
            col_type = _col_type_str(col.type)
            nullable = "Yes" if col.nullable else "No"
            default = ""
            if col.server_default is not None:
                default = (
                    str(col.server_default.arg)
                    if hasattr(col.server_default, "arg")
                    else str(col.server_default)
                )
            elif col.default is not None:
                default = str(col.default.arg) if hasattr(col.default, "arg") else str(col.default)
            lines.append(f"| `{col.name}` | `{col_type}` | {nullable} | {default} |")

        lines.append("")

        # Primary key
        if table.primary_key:
            pk_cols = ", ".join(f"`{c.name}`" for c in table.primary_key.columns)
            lines.append(f"**Primary Key:** {pk_cols}")
            lines.append("")

        # Foreign keys
        fks = list(table.foreign_keys)
        if fks:
            lines.append("### Foreign Keys")
            lines.append("")
            for fk in fks:
                lines.append(f"- `{fk.parent.name}` -> `{fk.column.table.name}.{fk.column.name}`")
            lines.append("")

        # Indexes
        if table.indexes:
            lines.append("### Indexes")
            lines.append("")
            for idx in sorted(table.indexes, key=lambda i: i.name or ""):
                idx_cols = ", ".join(f"`{c.name}`" for c in idx.columns)
                unique = " (UNIQUE)" if idx.unique else ""
                lines.append(f"- `{idx.name}`: {idx_cols}{unique}")
            lines.append("")

        # Unique constraints
        unique_constraints = [
            c for c in table.constraints if c.__class__.__name__ == "UniqueConstraint" and c.columns
        ]
        if unique_constraints:
            lines.append("### Unique Constraints")
            lines.append("")
            for uc in unique_constraints:
                uc_cols = ", ".join(f"`{c.name}`" for c in uc.columns)
                lines.append(f"- `{uc.name}`: {uc_cols}")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    """Write docs/SCHEMA.md."""
    output_dir = Path(__file__).resolve().parent.parent / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "SCHEMA.md"

    content = generate_schema_md()
    output_path.write_text(content)
    print(f"Schema written to {output_path}")


if __name__ == "__main__":
    main()
