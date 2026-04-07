"""
taxonomy.py — Shared vocabulary/ontology for agent slices
==========================================================
Loads data/taxonomy.json.
Every slice MUST carry tags that exist in this taxonomy.
Fail fast: invalid tags raise ValueError at publish time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Set

_DEFAULT_TAXONOMY = Path(__file__).parent.parent / "data" / "taxonomy.json"


class Taxonomy:
    """Validates and resolves taxonomy tags for agent feature slices."""

    def __init__(self, taxonomy_path: str | Path = _DEFAULT_TAXONOMY):
        self.path = Path(taxonomy_path)
        self._data: dict = {}
        self._valid_tags: Set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            # Graceful degradation — no taxonomy file means no validation
            self._valid_tags = set()
            return
        with open(self.path, encoding="utf-8") as f:
            self._data = json.load(f)
        self._valid_tags = set(self._data.get("all_valid_tags", []))

    def validate(self, tags: List[str]) -> None:
        """Raise ValueError if any tag is not in the taxonomy."""
        if not self._valid_tags:
            return  # no taxonomy loaded — skip validation
        unknown = [t for t in tags if t not in self._valid_tags]
        if unknown:
            raise ValueError(
                f"Unknown taxonomy tags: {unknown}. "
                f"Valid tags: {sorted(self._valid_tags)}"
            )

    def lookup(self, tag: str) -> Optional[dict]:
        """Return the category info for a tag, or None if not found."""
        for cat_name, cat_data in self._data.get("categories", {}).items():
            if tag in cat_data.get("tags", []):
                return {"category": cat_name, **cat_data}
        return None

    def tags_for_category(self, category: str) -> List[str]:
        """Return all valid tags for a given category."""
        return self._data.get("categories", {}).get(category, {}).get("tags", [])

    @property
    def all_tags(self) -> Set[str]:
        return set(self._valid_tags)

    def reload(self) -> None:
        """Hot-reload the taxonomy file."""
        self._load()
