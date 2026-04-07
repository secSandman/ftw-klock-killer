"""
metadata_index.py — Function-level metadata index for enhanced RAG retrieval
=============================================================================
Addresses the core BM25 weakness: large functions win by vocabulary mass, not
relevance.  This index adds field-weighted scoring on top of BM25:

  Field weights (empirically tuned):
    function_name  ×5   — exact / partial name match is the strongest signal
    summary        ×3   — LLM one-liner cached at build time (optional)
    callee_names   ×2   — called functions share domain vocabulary
    signature      ×2   — parameter names carry intent
    body           ×1   — baseline BM25 (length-normalized)

  Symbol routing:
    If any query token exactly matches a known function name, that chunk's
    score jumps by SYMBOL_BOOST (4.0) before normalisation — zero ambiguity.

  Call-chain expansion:
    Top-K result set is expanded to also include direct callers/callees of
    selected chunks (up to CHAIN_EXPAND_K extra slots), so the LLM gets
    context surrounding the answer function.

  Complexity dampening:
    Functions with line_count > LARGE_FN_LINES get a size penalty
    (÷ log(line_count)) to counteract BM25's raw-count bias.

Storage: {repo}/.kloc/metadata_index.json  (built by build_metadata.py)

Build cost: ~1-2s per 1000 functions (pure Python, no GPU/API needed).
Optional T1 summaries: ~$0.002 per 100 functions via Haiku if available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────

import os as _os

FIELD_WEIGHTS: Dict[str, float] = {
    "name":      float(_os.environ.get("KLOC_RAG_NAME_WEIGHT",      "5.0")),
    "summary":   float(_os.environ.get("KLOC_RAG_SUMMARY_WEIGHT",   "3.0")),
    "callees":   float(_os.environ.get("KLOC_RAG_CALLEE_WEIGHT",    "2.0")),
    "signature": float(_os.environ.get("KLOC_RAG_SIGNATURE_WEIGHT", "2.0")),
    "body":      float(_os.environ.get("KLOC_RAG_BODY_WEIGHT",      "1.0")),
}

SYMBOL_BOOST   = float(_os.environ.get("KLOC_RAG_SYMBOL_BOOST",    "4.0"))
CHAIN_EXPAND_K = int(  _os.environ.get("KLOC_RAG_CHAIN_EXPAND_K",  "2"))
LARGE_FN_LINES = int(  _os.environ.get("KLOC_RAG_LARGE_FN_LINES",  "100"))

META_PATH_RELATIVE = Path(".kloc") / "metadata_index.json"

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class FunctionMeta:
    chunk_id:   str
    name:       str
    file_path:  str
    start_line: int
    end_line:   int
    signature:  str          # return_type + name + params (best-effort)
    callee_names: List[str]  = field(default_factory=list)
    caller_names: List[str]  = field(default_factory=list)
    summary:    str          = ""          # T1 one-liner (optional)
    complexity: str          = "unknown"   # simple | moderate | complex
    line_count: int          = 0


@dataclass
class MetadataIndex:
    repo_path:  str
    functions:  List[FunctionMeta]                 = field(default_factory=list)
    _by_chunk:  Dict[str, FunctionMeta]            = field(default_factory=dict, repr=False)
    _by_name:   Dict[str, List[FunctionMeta]]      = field(default_factory=dict, repr=False)
    _all_names: Set[str]                           = field(default_factory=set,  repr=False)

    # ── Build index ────────────────────────────────────────────────────

    def add(self, meta: FunctionMeta) -> None:
        self.functions.append(meta)
        self._by_chunk[meta.chunk_id] = meta
        self._by_name.setdefault(meta.name.lower(), []).append(meta)
        self._all_names.add(meta.name.lower())

    def _rebuild_indices(self) -> None:
        self._by_chunk.clear()
        self._by_name.clear()
        self._all_names.clear()
        for m in self.functions:
            self._by_chunk[m.chunk_id] = m
            self._by_name.setdefault(m.name.lower(), []).append(m)
            self._all_names.add(m.name.lower())

    # ── Serialisation ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "repo_path": self.repo_path,
            "functions": [asdict(f) for f in self.functions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetadataIndex":
        obj = cls(repo_path=d.get("repo_path", ""))
        for f in d.get("functions", []):
            meta = FunctionMeta(**{k: v for k, v in f.items()})
            obj.functions.append(meta)
        obj._rebuild_indices()
        return obj

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = Path(self.repo_path) / META_PATH_RELATIVE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "MetadataIndex":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def load_for_repo(cls, repo_path: Path) -> Optional["MetadataIndex"]:
        p = repo_path / META_PATH_RELATIVE
        if p.exists():
            return cls.load(p)
        return None

    def __len__(self) -> int:
        return len(self.functions)

    def summary(self) -> str:
        with_summary = sum(1 for f in self.functions if f.summary)
        complex_cnt  = sum(1 for f in self.functions if f.complexity == "complex")
        return (
            f"{len(self.functions)} functions indexed, "
            f"{with_summary} with summaries, "
            f"{complex_cnt} complex"
        )


# ── Builder ────────────────────────────────────────────────────────────────────

class MetadataIndexBuilder:
    """
    Builds a MetadataIndex from CodeChunk objects.

    Usage:
        builder = MetadataIndexBuilder()
        index   = builder.build(chunks, repo_path=Path("doom/"))
        index.save()
    """

    # Regex to extract callee names: word followed by ( — crude but fast
    _RE_CALL = re.compile(r'\b([A-Za-z_]\w+)\s*\(')
    # Regex to extract C function signature header (first line pattern)
    _RE_SIG  = re.compile(
        r'^(?:(?:static|extern|inline|const|unsigned|signed|long|short|void|int|char|float|double|\w+)\s+)*'
        r'(\*?\w+)\s*\(([^)]*)\)',
        re.MULTILINE,
    )
    _C_BUILTINS = frozenset([
        'if', 'for', 'while', 'switch', 'return', 'sizeof', 'printf', 'fprintf',
        'malloc', 'free', 'memcpy', 'memset', 'strcmp', 'strlen', 'sprintf',
        'fopen', 'fclose', 'assert', 'exit', 'abort',
    ])

    def build(self, chunks: list, repo_path: Path) -> MetadataIndex:
        """Build index from a list of CodeChunk objects."""
        index = MetadataIndex(repo_path=str(repo_path))

        # First pass: extract basic metadata for all function/struct chunks
        for chunk in chunks:
            name       = getattr(chunk, "name", "") or ""
            chunk_type = getattr(chunk, "chunk_type", "") or ""
            if chunk_type not in ("function", "struct", "class"):
                continue

            source     = getattr(chunk, "source", "") or ""
            start_line = getattr(chunk, "start_line", 0)
            end_line   = getattr(chunk, "end_line", 0)
            line_count = end_line - start_line + 1

            callee_names = [
                c for c in self._extract_callees(source)
                if c.lower() != name.lower() and c not in self._C_BUILTINS
            ]

            meta = FunctionMeta(
                chunk_id     = getattr(chunk, "chunk_id", ""),
                name         = name,
                file_path    = getattr(chunk, "file_path", ""),
                start_line   = start_line,
                end_line     = end_line,
                signature    = self._extract_signature(name, source),
                callee_names = callee_names,
                line_count   = line_count,
                complexity   = self._classify_complexity(source, line_count),
            )
            index.add(meta)

        # Second pass: backfill caller_names
        # For each function A that calls B, add A to B's caller_names
        name_to_metas: Dict[str, List[FunctionMeta]] = defaultdict(list)
        for meta in index.functions:
            name_to_metas[meta.name.lower()].append(meta)

        for meta in index.functions:
            for callee in meta.callee_names:
                for callee_meta in name_to_metas.get(callee.lower(), []):
                    if meta.name not in callee_meta.caller_names:
                        callee_meta.caller_names.append(meta.name)

        return index

    def _extract_callees(self, source: str) -> List[str]:
        seen   = set()
        result = []
        for m in self._RE_CALL.finditer(source):
            name = m.group(1)
            if name not in seen and len(name) >= 2:
                seen.add(name)
                result.append(name)
        return result[:30]  # cap at 30 to keep index small

    def _extract_signature(self, name: str, source: str) -> str:
        """Best-effort: return first line that looks like a function header."""
        first_lines = source[:300]
        m = self._RE_SIG.search(first_lines)
        if m:
            # Return up to 120 chars of the full match
            return first_lines[m.start():m.end()][:120].strip()
        # Fallback: first non-blank line
        for line in source.splitlines():
            line = line.strip()
            if line and not line.startswith("//") and not line.startswith("/*"):
                return line[:120]
        return name

    def _classify_complexity(self, source: str, line_count: int) -> str:
        """Simple heuristic: count loop/branch keywords."""
        text = source.lower()
        branches = (
            text.count(" if ") + text.count("\tif(") + text.count("\nif(")
            + text.count(" for(") + text.count(" while(")
            + text.count(" switch(")
        )
        if branches >= 8 or line_count > 150:
            return "complex"
        if branches >= 3 or line_count > 40:
            return "moderate"
        return "simple"


# ── Enhanced retriever ─────────────────────────────────────────────────────────

class MetadataRetriever:
    """
    Field-weighted BM25 retriever that uses MetadataIndex when available.

    Falls back gracefully to plain BM25 if no index is loaded.

    Usage:
        index    = MetadataIndex.load_for_repo(Path("doom/"))
        ret      = MetadataRetriever(index)
        top      = ret.retrieve("how does A_Chase work", chunks, top_k=5)
    """

    def __init__(self, index: Optional[MetadataIndex] = None):
        self.index = index

    def retrieve(self, question: str, chunks: list, top_k: int = 5) -> list:
        if not chunks:
            return []
        if len(chunks) <= top_k:
            return list(chunks)

        if self.index is None:
            return self._plain_bm25(question, chunks, top_k)

        return self._metadata_rank(question, chunks, top_k)

    # ── Metadata-augmented ranking ─────────────────────────────────────

    def _metadata_rank(self, question: str, chunks: list, top_k: int) -> list:
        query_tokens = set(_tokenize(question))
        if not query_tokens:
            return list(chunks[:top_k])

        # Detect exact symbol names in query for direct boost
        symbol_matches: Set[str] = set()
        for tok in query_tokens:
            if tok in self.index._all_names:
                symbol_matches.add(tok)

        scored: List[Tuple[float, int]] = []  # (score, original_index)
        for i, chunk in enumerate(chunks):
            meta = self.index._by_chunk.get(getattr(chunk, "chunk_id", ""))
            score = self._score_chunk(query_tokens, symbol_matches, chunk, meta)
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_indices = [i for _, i in scored[:top_k]]

        # Call-chain expansion: add direct callers/callees of top results
        if CHAIN_EXPAND_K > 0:
            expansion_names: Set[str] = set()
            for idx in top_indices:
                chunk = chunks[idx]
                meta  = self.index._by_chunk.get(getattr(chunk, "chunk_id", ""))
                if meta:
                    expansion_names.update(meta.callee_names[:3])
                    expansion_names.update(meta.caller_names[:2])

            # Find chunks matching expansion names that aren't already selected
            selected_set = set(top_indices)
            extras: List[Tuple[float, int]] = []
            for j, (s, i) in enumerate(scored):
                if i in selected_set:
                    continue
                chunk = chunks[i]
                meta  = self.index._by_chunk.get(getattr(chunk, "chunk_id", ""))
                if meta and meta.name.lower() in expansion_names:
                    extras.append((s, i))
                if len(extras) >= CHAIN_EXPAND_K:
                    break

            # Merge: top_k slots + up to CHAIN_EXPAND_K expansion (total capped at top_k + CHAIN_EXPAND_K)
            final_indices = top_indices + [i for _, i in extras]
        else:
            final_indices = top_indices

        return [chunks[i] for i in final_indices]

    def _score_chunk(
        self,
        query_tokens: Set[str],
        symbol_matches: Set[str],
        chunk,
        meta: Optional[FunctionMeta],
    ) -> float:
        """Compute field-weighted BM25 score for one chunk."""
        score = 0.0

        if meta is None:
            # No metadata: plain BM25 on source
            source = _chunk_text(chunk)
            tokens = _tokenize(source.lower())
            count  = sum(tokens.count(t) for t in query_tokens)
            return count / math.log(len(tokens) + 2)

        # Symbol routing: name exact match
        name_lower = meta.name.lower()
        if name_lower in symbol_matches:
            score += SYMBOL_BOOST

        # Field: name (partial token overlap)
        name_tokens = set(_tokenize(name_lower))
        score += FIELD_WEIGHTS["name"] * _overlap(query_tokens, name_tokens)

        # Field: signature
        sig_tokens = set(_tokenize(meta.signature.lower()))
        score += FIELD_WEIGHTS["signature"] * _overlap(query_tokens, sig_tokens)

        # Field: callee names (domain vocabulary)
        callee_text = " ".join(meta.callee_names).lower()
        callee_tokens = set(_tokenize(callee_text))
        score += FIELD_WEIGHTS["callees"] * _overlap(query_tokens, callee_tokens)

        # Field: summary (if available)
        if meta.summary:
            sum_tokens = set(_tokenize(meta.summary.lower()))
            score += FIELD_WEIGHTS["summary"] * _overlap(query_tokens, sum_tokens)

        # Field: body (length-normalised BM25)
        body = _chunk_text(chunk).lower()
        body_toks = _tokenize(body)
        body_count = sum(body_toks.count(t) for t in query_tokens)
        body_score = FIELD_WEIGHTS["body"] * (body_count / math.log(len(body_toks) + 2))

        # Size penalty for huge functions
        if meta.line_count > LARGE_FN_LINES:
            body_score /= math.log(meta.line_count)

        score += body_score
        return score

    # ── Plain BM25 fallback ────────────────────────────────────────────

    def _plain_bm25(self, question: str, chunks: list, k: int) -> list:
        query_tokens = set(_tokenize(question))
        if not query_tokens:
            return list(chunks[:k])
        scored = []
        for chunk in chunks:
            text   = _chunk_text(chunk).lower()
            tokens = _tokenize(text)
            count  = sum(tokens.count(t) for t in query_tokens)
            score  = count / math.log(len(tokens) + 2)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z_]\w*", text.lower())


def _overlap(a: Set[str], b: Set[str]) -> float:
    if not b:
        return 0.0
    return len(a & b) / math.sqrt(len(b))


def _chunk_text(chunk) -> str:
    if hasattr(chunk, "source"):
        return chunk.source or ""
    if isinstance(chunk, dict):
        return chunk.get("source", chunk.get("compressed_src", ""))
    return str(chunk)
