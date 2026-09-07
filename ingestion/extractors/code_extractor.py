"""
Language-aware code splitting so functions/classes are never cut mid-body.

Python uses the stdlib `ast` module for exact boundaries. Other languages
(JS/TS/Java/Go/Rust/C/C++) use brace-matching against common function/class
signatures — not a full parser, but reliable enough to keep a `{ ... }`
body intact, which is the property chunking depends on.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

BRACE_LANG_SIGNATURE_PATTERNS = {
    ".js": r"^\s*(export\s+)?(async\s+)?function\s+\w+|^\s*(export\s+)?(default\s+)?class\s+\w+|^\s*const\s+\w+\s*=\s*(async\s*)?\(.*\)\s*=>",
    ".ts": r"^\s*(export\s+)?(async\s+)?function\s+\w+|^\s*(export\s+)?(default\s+)?class\s+\w+|^\s*(export\s+)?interface\s+\w+",
    ".java": r"^\s*(public|private|protected|static|\s)+[\w<>\[\]]+\s+\w+\s*\([^;]*\)\s*\{|^\s*(public|private|protected)?\s*class\s+\w+",
    ".go": r"^\s*func\s+(\(\w+\s+\*?\w+\)\s+)?\w+",
    ".rs": r"^\s*(pub\s+)?(async\s+)?fn\s+\w+|^\s*(pub\s+)?struct\s+\w+|^\s*(pub\s+)?impl\b",
    ".cpp": r"^[\w:<>\*&\s]+\s+\w+\s*\([^;{]*\)\s*\{|^\s*class\s+\w+",
    ".c": r"^[\w\*\s]+\s+\w+\s*\([^;{]*\)\s*\{",
    ".h": r"^[\w\*\s]+\s+\w+\s*\([^;{]*\)\s*\{",
    ".hpp": r"^[\w:<>\*&\s]+\s+\w+\s*\([^;{]*\)\s*\{|^\s*class\s+\w+",
}


@dataclass
class CodeUnit:
    name: Optional[str]
    unit_type: str  # "function" | "class" | "module_chunk"
    text: str
    start_line: int
    end_line: int


def extract_code_units(path: Path) -> List[CodeUnit]:
    source = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()
    if ext == ".py":
        units = _extract_python(source)
        if units:
            return units
    elif ext in BRACE_LANG_SIGNATURE_PATTERNS:
        units = _extract_brace_language(source, ext)
        if units:
            return units
    # Fallback: whole file as one unit (still never "splits" anything)
    return [CodeUnit(name=path.name, unit_type="module_chunk", text=source, start_line=1, end_line=len(source.splitlines()))]


def _extract_python(source: str) -> List[CodeUnit]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    units: List[CodeUnit] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            text = "\n".join(lines[start - 1:end])
            unit_type = "class" if isinstance(node, ast.ClassDef) else "function"
            units.append(CodeUnit(name=node.name, unit_type=unit_type, text=text, start_line=start, end_line=end))
    return units


def _extract_brace_language(source: str, ext: str) -> List[CodeUnit]:
    pattern = re.compile(BRACE_LANG_SIGNATURE_PATTERNS[ext])
    lines = source.splitlines()
    units: List[CodeUnit] = []
    i = 0
    while i < len(lines):
        if pattern.match(lines[i]):
            start = i
            brace_open_idx = None
            for j in range(i, min(i + 5, len(lines))):
                if "{" in lines[j]:
                    brace_open_idx = j
                    break
            if brace_open_idx is None:
                i += 1
                continue
            depth = 0
            end = brace_open_idx
            for j in range(brace_open_idx, len(lines)):
                depth += lines[j].count("{") - lines[j].count("}")
                if depth <= 0 and j >= brace_open_idx:
                    end = j
                    break
            else:
                end = len(lines) - 1
            name_match = re.search(r"\b(\w+)\s*\(", lines[start]) or re.search(r"class\s+(\w+)", lines[start])
            name = name_match.group(1) if name_match else None
            units.append(CodeUnit(
                name=name, unit_type="function",
                text="\n".join(lines[start:end + 1]),
                start_line=start + 1, end_line=end + 1,
            ))
            i = end + 1
        else:
            i += 1
    return units
