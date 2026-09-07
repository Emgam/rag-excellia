"""Helpers for turning tabular data into search-friendly markdown text.
The heavy lifting already happens in ingestion/parsers/parser.py
(`_rows_to_markdown`); this module exposes it plus a row-summary helper
used by metadata extraction to keep column headers searchable even when
a table gets split across chunks."""
from __future__ import annotations

from typing import List


def summarize_columns(markdown_table: str) -> List[str]:
    lines = [l for l in markdown_table.splitlines() if l.strip().startswith("|")]
    if not lines:
        return []
    header = lines[0].strip("|").split("|")
    return [h.strip() for h in header if h.strip()]
