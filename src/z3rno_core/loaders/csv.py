"""CSV loader — header-aware row stringification.

Decodes a CSV file (auto-detecting common dialects + BOM), reads up to
``max_rows`` data rows, and emits one logical paragraph per row of the
form ``key1: value1; key2: value2; ...``. The first emitted paragraph
is a header banner so downstream summaries can distinguish "schema"
from "data".

Phase A's chunker handles the resulting paragraphs naturally. The
``max_rows`` cap is hard-coded at construction (sourced from the
``INGEST_MAX_CSV_ROWS`` server setting upstream) — it prevents a stray
1M-row dump from blowing past LLM context and storage budgets.
"""

from __future__ import annotations

import csv as _csv
import io
import logging

from z3rno_core.loaders.base import (
    Loader,
    LoaderInputError,
    LoaderResult,
)

logger = logging.getLogger(__name__)


class CsvLoader(Loader):
    """Header-aware CSV-to-text loader."""

    def __init__(self, *, max_rows: int = 10_000) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be > 0")
        self._max_rows = max_rows

    @property
    def name(self) -> str:
        return "csv"

    async def load(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> LoaderResult:
        if not content:
            return LoaderResult(
                text="",
                metadata={
                    "loader": self.name,
                    "mime_type": "text/csv",
                    "filename": filename,
                    "byte_size": 0,
                    "char_count": 0,
                    "row_count": 0,
                    "column_count": 0,
                    "truncated": False,
                },
            )

        # Decode with BOM tolerance.
        text = content.decode("utf-8-sig", errors="replace")

        # Sniff the dialect so comma / semicolon / tab CSVs all work.
        try:
            dialect = _csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except _csv.Error:
            dialect = _csv.excel

        reader = _csv.reader(io.StringIO(text), dialect)
        try:
            header = next(reader)
        except StopIteration:
            return LoaderResult(
                text="",
                metadata={
                    "loader": self.name,
                    "mime_type": "text/csv",
                    "filename": filename,
                    "byte_size": len(content),
                    "char_count": 0,
                    "row_count": 0,
                    "column_count": 0,
                    "truncated": False,
                },
            )
        except _csv.Error as exc:
            raise LoaderInputError(f"malformed CSV: {exc}") from exc

        header = [h.strip() for h in header]
        column_count = len(header)

        paragraphs: list[str] = []
        # Header banner so the summary stage has obvious context.
        paragraphs.append("CSV columns: " + ", ".join(header))

        truncated = False
        row_count = 0
        for row in reader:
            if row_count >= self._max_rows:
                truncated = True
                break
            # Pad short rows / truncate long rows so the zip aligns with header.
            cells = (row + [""] * column_count)[:column_count]
            pairs = "; ".join(f"{h}: {v.strip()}" for h, v in zip(header, cells, strict=False))
            paragraphs.append(pairs)
            row_count += 1

        body = "\n\n".join(paragraphs)

        return LoaderResult(
            text=body,
            metadata={
                "loader": self.name,
                "mime_type": "text/csv",
                "filename": filename,
                "byte_size": len(content),
                "char_count": len(body),
                "row_count": row_count,
                "column_count": column_count,
                "truncated": truncated,
            },
        )
