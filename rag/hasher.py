"""
hasher.py — SHA-256 content hasher + rainbow table lookup
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rainbow.rainbow_table import RainbowTable


class Hasher:
    def __init__(self, rainbow_paths: list = None):
        self._rainbow = RainbowTable(rainbow_paths or [])

    def hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def lookup(self, text: str) -> Optional[str]:
        """Check rainbow table. Returns compressed form or None."""
        h = self.hash(text)
        return self._rainbow.lookup(h)

    def hash_chunk(self, chunk) -> dict:
        h = self.hash(chunk.source)
        cached = self._rainbow.lookup(h)
        return {
            "chunk_id":    chunk.chunk_id,
            "sha256":      h,
            "rainbow_hit": cached is not None,
            "cached_compressed": cached,
        }
