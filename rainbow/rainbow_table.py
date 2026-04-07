"""
rainbow_table.py — Pre-computed hash → compressed form lookup.
Common stdlib patterns (import json, #include <stdio.h>, etc.) get
compressed once and cached forever. Zero embedding cost on hits.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

_DEFAULT_TABLES = [
    Path(__file__).parent.parent / "data" / "rainbow_stdlib_python.jsonl",
    Path(__file__).parent.parent / "data" / "rainbow_stdlib_c.jsonl",
]


class RainbowTable:
    def __init__(self, table_paths: List[str | Path] = None):
        self._table: Dict[str, str] = {}
        paths = [Path(p) for p in (table_paths or _DEFAULT_TABLES)]
        for p in paths:
            if p.exists():
                self._load(p)

    def _load(self, path: Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                self._table[rec["hash"]] = rec["compressed"]
            except (json.JSONDecodeError, KeyError):
                continue

    def lookup(self, sha256: str) -> Optional[str]:
        return self._table.get(sha256)

    def size(self) -> int:
        return len(self._table)

    def add(self, sha256: str, compressed: str) -> None:
        self._table[sha256] = compressed
