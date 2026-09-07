"""Metadata helpers for code chunks — surfaces function/class names so
they're filterable/searchable independent of the BM25/dense score."""
from __future__ import annotations

from typing import Optional


def function_signature_hint(text: str) -> Optional[str]:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line.strip() or None
