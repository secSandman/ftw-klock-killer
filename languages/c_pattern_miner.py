"""
c_pattern_miner.py — Per-repo C pattern dictionary builder
===========================================================
Scans a C/C++ repository and builds a pattern dictionary of known-quantity
(KQ) patterns: things that appear in 3+ files and can be safely masked or
replaced with a short placeholder token.

Pattern types mined:
  - include      : #include <x.h> / #include "x.h"
  - define       : #define NAME value  (single-line, small)
  - comment_block: repeated block comment fingerprints (copyright headers)
  - pragma       : #pragma once, #pragma pack(n), etc.

Output: {repo}/.kloc/c_pattern_dict.json

Design notes:
  - Per-repo first; global accumulation is a future extension (see TODO below).
  - Threshold KQ_THRESHOLD=3 mirrors ChromatophoricMasker's "Known Quantity" rule.
  - Hash keys are full SHA-256 for collision safety; display uses first 8 chars.

TODO (future global extension):
  - Accumulate pattern dicts across repos into a global registry.
  - Weight by repo count, not just file count (avoid single-repo bias).
  - Add LRU decay: patterns not seen in N new repos lose weight over time.
  - Performance: index by hash prefix for O(1) lookup at scale.
  - Storage: consider SQLite or LMDB for global registry > 100k patterns.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

# ── Constants ──────────────────────────────────────────────────────────────────

import os as _os
KQ_THRESHOLD  = int(_os.environ.get("KLOC_CPM_KQ_THRESHOLD",  "3"))
MAX_DEFINE_LEN = int(_os.environ.get("KLOC_CPM_MAX_DEFINE_LEN", "120"))
COMMENT_WORDS  = int(_os.environ.get("KLOC_CPM_COMMENT_WORDS", "12"))

C_EXTENSIONS: Set[str] = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh"}

DICT_PATH_RELATIVE = Path(".kloc") / "c_pattern_dict.json"

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PatternEntry:
    hash:         str   # full SHA-256
    original:     str   # exact original text
    compressed:   str   # replacement token, e.g. [§:INC_a1b2c3d4]
    frequency:    int   # number of files this appears in
    pattern_type: str   # include | define | comment_block | pragma


@dataclass
class CPatternDict:
    repo_path:   str
    total_files: int
    total_lines: int
    patterns:    List[PatternEntry]          = field(default_factory=list)
    # Fast lookup: sha256 -> PatternEntry (not serialised — rebuilt on load)
    _by_hash:    Dict[str, PatternEntry]     = field(default_factory=dict, repr=False)
    # Fast lookup: original_text -> PatternEntry
    _by_original: Dict[str, PatternEntry]   = field(default_factory=dict, repr=False)

    def add(self, entry: PatternEntry) -> None:
        self.patterns.append(entry)
        self._by_hash[entry.hash]         = entry
        self._by_original[entry.original] = entry

    def lookup_original(self, text: str) -> Optional[PatternEntry]:
        return self._by_original.get(text.strip())

    def lookup_hash(self, sha: str) -> Optional[PatternEntry]:
        return self._by_hash.get(sha)

    # ── Serialisation ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "repo_path":   self.repo_path,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "patterns":    [asdict(p) for p in self.patterns],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CPatternDict":
        obj = cls(
            repo_path   = d.get("repo_path", ""),
            total_files = d.get("total_files", 0),
            total_lines = d.get("total_lines", 0),
        )
        for p in d.get("patterns", []):
            entry = PatternEntry(**{k: v for k, v in p.items()})
            # Migrate old 5-token format [§:PREFIX_hash] → new 2-token §hash
            # Old: [§:INC_a1b2c3d4] = 5 tokens.  New: §a1b2c3d4 = 2 tokens.
            if entry.compressed.startswith("[\u00a7:"):
                m = re.search(r'_([a-f0-9]{8})\]$', entry.compressed)
                if not m:
                    m = re.search(r':([a-f0-9]{8})\]$', entry.compressed)
                if m:
                    entry = PatternEntry(
                        hash=entry.hash, original=entry.original,
                        compressed=f"\u00a7{m.group(1)}",
                        frequency=entry.frequency, pattern_type=entry.pattern_type,
                    )
            obj._by_hash[entry.hash]          = entry
            obj._by_original[entry.original]  = entry
            obj.patterns.append(entry)
        return obj

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = Path(self.repo_path) / DICT_PATH_RELATIVE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "CPatternDict":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def load_for_repo(cls, repo_path: Path) -> Optional["CPatternDict"]:
        p = repo_path / DICT_PATH_RELATIVE
        if p.exists():
            return cls.load(p)
        return None

    def __len__(self) -> int:
        return len(self.patterns)

    def summary(self) -> str:
        by_type: Dict[str, int] = defaultdict(int)
        for e in self.patterns:
            by_type[e.pattern_type] += 1
        parts = [f"{v} {k}" for k, v in sorted(by_type.items())]
        return (
            f"{len(self.patterns)} patterns "
            f"({', '.join(parts)}) "
            f"from {self.total_files} files / {self.total_lines:,} lines"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _sha8(text: str) -> str:
    return _sha256(text)[:8]


# ── Miner ──────────────────────────────────────────────────────────────────────

class CPatternMiner:
    """
    Scans a C/C++ repository and builds a per-repo CPatternDict.

    Usage:
        miner  = CPatternMiner()
        pdict  = miner.mine(Path("doom/"))
        saved  = pdict.save()          # writes doom/.kloc/c_pattern_dict.json
        print(pdict.summary())
    """

    _RE_INCLUDE   = re.compile(r'^\s*#\s*include\s*[<"][^>"]+[>"]', re.MULTILINE)
    _RE_DEFINE    = re.compile(r'^\s*#\s*define\s+\w+(?:\([^)]*\))?\s+\S[^\n]*', re.MULTILINE)
    _RE_PRAGMA    = re.compile(r'^\s*#\s*pragma\s+\S[^\n]*', re.MULTILINE)
    _RE_CMT_BLOCK = re.compile(r'/\*.*?\*/', re.DOTALL)
    # ALL_CAPS identifiers (macros, enum constants) and long_underscore_names
    _RE_BODY_ID   = re.compile(r'\b([A-Z][A-Z0-9_]{4,}|[a-z_]\w*_[a-z]\w{4,})\b')
    _C_KEYWORDS   = frozenset([
        'NULL','EOF','TRUE','FALSE','true','false','static','extern','const',
        'void','int','char','float','double','return','struct','if','else',
        'for','while','do','switch','case','break','continue','typedef',
        'unsigned','signed','long','short','sizeof','goto','inline',
    ])

    def __init__(self, kq_threshold: int = KQ_THRESHOLD):
        self.kq_threshold = kq_threshold

    def mine(self, repo_path: Path) -> CPatternDict:
        """
        Walk all C/C++ files in repo_path, extract patterns,
        and return a CPatternDict of known-quantity entries.
        """
        repo_path = Path(repo_path)
        c_files = [
            f for f in repo_path.rglob("*")
            if f.suffix.lower() in C_EXTENSIONS and f.is_file()
        ]

        # Maps: pattern_text -> set of file paths it appears in
        include_files:  Dict[str, Set[str]] = defaultdict(set)
        define_files:   Dict[str, Set[str]] = defaultdict(set)
        pragma_files:   Dict[str, Set[str]] = defaultdict(set)
        comment_files:  Dict[str, Set[str]] = defaultdict(set)

        total_lines = 0

        for fpath in c_files:
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            total_lines += text.count("\n") + 1
            fkey = str(fpath)

            for m in self._RE_INCLUDE.finditer(text):
                include_files[m.group(0).strip()].add(fkey)

            for m in self._RE_DEFINE.finditer(text):
                pat = m.group(0).strip()
                if len(pat) <= MAX_DEFINE_LEN:
                    define_files[pat].add(fkey)

            for m in self._RE_PRAGMA.finditer(text):
                pragma_files[m.group(0).strip()].add(fkey)

            for m in self._RE_CMT_BLOCK.finditer(text):
                blk = m.group(0).strip()
                if len(blk) > 40:
                    # Fingerprint = first COMMENT_WORDS words (stable across minor edits)
                    fp = " ".join(blk.split()[:COMMENT_WORDS])
                    comment_files[fp].add(fkey)

        # Identifier frequency: mine ALL_CAPS and long_underscore names
        # that appear 4+ times within a single file (per-file count, then union)
        identifier_files: Dict[str, Set[str]] = defaultdict(set)
        for fpath in c_files:
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            fkey = str(fpath)
            per_file: Dict[str, int] = defaultdict(int)
            for m in self._RE_BODY_ID.finditer(text):
                ident = m.group(1)
                if ident not in self._C_KEYWORDS and len(ident) >= 6:
                    per_file[ident] += 1
            for ident, cnt in per_file.items():
                if cnt >= 4:
                    identifier_files[ident].add(fkey)

        pdict = CPatternDict(
            repo_path   = str(repo_path),
            total_files = len(c_files),
            total_lines = total_lines,
        )

        self._add_entries(pdict, include_files,    "include",       "INC")
        self._add_entries(pdict, define_files,     "define",        "DEF")
        self._add_entries(pdict, pragma_files,     "pragma",        "PRA")
        self._add_entries(pdict, comment_files,    "comment_block", "CMT")
        # Only add identifiers where the compressed token is actually shorter
        # A placeholder [§:ID_xxxxxxxx] is ~4 tokens; skip if original is <= 4 tokens
        filtered_id: Dict[str, Set[str]] = {
            k: v for k, v in identifier_files.items()
            if len(k) > 12   # rough heuristic: only mask long identifiers
        }
        self._add_entries(pdict, filtered_id, "identifier", "ID")

        return pdict

    def _add_entries(
        self,
        pdict:    CPatternDict,
        freq_map: Dict[str, Set[str]],
        ptype:    str,
        prefix:   str,
    ) -> None:
        # Sort descending by frequency so the most common patterns are first
        for pat, files in sorted(freq_map.items(), key=lambda x: -len(x[1])):
            freq = len(files)
            if freq >= self.kq_threshold:
                h          = _sha256(pat)
                compressed = f"\u00a7{_sha8(pat)}"   # 2 tokens (§ + 8-hex) vs old 5-token [§:PREFIX_hash]
                pdict.add(PatternEntry(
                    hash         = h,
                    original     = pat,
                    compressed   = compressed,
                    frequency    = freq,
                    pattern_type = ptype,
                ))


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("Usage: python languages/c_pattern_miner.py <repo_path> [--dry-run]")
        sys.exit(1)

    repo   = Path(sys.argv[1])
    dry    = "--dry-run" in sys.argv
    miner  = CPatternMiner()
    pdict  = miner.mine(repo)

    print(f"\n  Repo:     {repo}")
    print(f"  {pdict.summary()}")
    print(f"\n  Top 10 patterns by frequency:")
    for e in pdict.patterns[:10]:
        print(f"    [{e.frequency:3d} files] {e.pattern_type:<14} {e.original[:70]}")

    if not dry:
        saved = pdict.save()
        print(f"\n  Saved -> {saved}")
    else:
        print("\n  (dry-run: not saved)")
