"""
retriever.py — RAG retrieval layer for the KLOC-KILLER pipeline.

Ranks code chunks against a natural-language query and returns the top-K
most relevant ones. Three strategies, selected automatically:

  0. Metadata-augmented field-weighted BM25 (best offline quality)
     Activated when a MetadataIndex exists for the repo (.kloc/metadata_index.json).
     Boosts function name matches ×5, signature ×2, callees ×2 over body text.
     Adds call-chain expansion and large-function size penalty.

  1. Embedding cosine similarity (default with sentence-transformers)
     Uses the local sentence-transformers model (no API needed, free).
     Falls back to strategy 2 if the model is not installed.

  2. BM25-style keyword matching (offline fallback)
     Activated when:
       - ACTIVE_TIER_MAX=0  (fully offline / CI mode), OR
       - sentence-transformers is not installed,  AND no metadata index.
     Counts overlapping question tokens in each chunk's source text.
     Not as accurate but zero-dependency and deterministic.

Usage:
    from rag.retriever import RAGRetriever
    retriever = RAGRetriever()
    top_chunks = retriever.retrieve(question="how does A_Chase work", chunks=all_chunks, top_k=3)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_TIER_MAX = int(os.environ.get("ACTIVE_TIER_MAX", "2"))


class RAGRetriever:
    """
    Chunk retriever with three strategies (best-first):

    0. Metadata field-weighted BM25 (if .kloc/metadata_index.json present)
    1. Embedding cosine similarity (if sentence-transformers installed)
    2. Plain BM25 keyword matching (always available, zero deps)

    Parameters
    ----------
    top_k        : default number of chunks to return
    force_bm25   : bypass embedding and always use keyword matching
    repo_path    : path to repo root for metadata index lookup
    metadata_index : pre-loaded MetadataIndex (overrides repo_path lookup)
    """

    def __init__(
        self,
        top_k: int = 3,
        force_bm25: bool = False,
        repo_path: Optional[Path] = None,
        metadata_index=None,
    ):
        self.top_k = top_k
        self._force_bm25 = force_bm25
        self._embedder = None          # lazy-loaded
        self._meta_retriever = None    # lazy-loaded

        # Try to load metadata index
        if metadata_index is not None:
            from rag.metadata_index import MetadataRetriever
            self._meta_retriever = MetadataRetriever(metadata_index)
        elif repo_path is not None:
            self._try_load_metadata(Path(repo_path))

    def _try_load_metadata(self, repo_path: Path) -> None:
        try:
            from rag.metadata_index import MetadataIndex, MetadataRetriever
            idx = MetadataIndex.load_for_repo(repo_path)
            if idx is not None:
                self._meta_retriever = MetadataRetriever(idx)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def retrieve(self, question: str, chunks: list, top_k: int = None) -> list:
        """
        Rank *chunks* against *question* and return the top-k most relevant.

        chunks must be objects/dicts that expose at minimum a ``source`` field
        (CodeChunk objects from the parser, or plain dicts with a 'source' key).

        Returns the same objects (not copies), ordered best-first.
        """
        k = top_k if top_k is not None else self.top_k
        if not chunks:
            return []
        if len(chunks) <= k:
            return list(chunks)

        # Strategy 0: metadata-augmented field-weighted BM25
        if self._meta_retriever is not None:
            return self._meta_retriever.retrieve(question, chunks, k)

        use_bm25 = self._should_use_bm25()

        # Strategy 1: hybrid RRF (BM25 + embedding) when both are available
        if not use_bm25 and not self._force_bm25:
            return self._rrf_rank(question, chunks, k)

        # Strategy 2: plain BM25 fallback
        return self._bm25_rank(question, chunks, k)

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _should_use_bm25(self) -> bool:
        if self._force_bm25:
            return True
        tier_max = int(os.environ.get("ACTIVE_TIER_MAX", "2"))
        if tier_max == 0:
            return True
        if not self._embedding_available():
            return True
        return False

    def _embedding_available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Strategy 1 — cosine similarity on sentence-transformer embeddings
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            from rag.embedder import Embedder
            self._embedder = Embedder(provider="local")
        return self._embedder

    def _embedding_rank(self, question: str, chunks: list, k: int) -> list:
        import numpy as np

        embedder = self._get_embedder()

        sources = [_chunk_text(c) for c in chunks]
        # Batch-embed chunks + query together for efficiency
        all_texts = sources + [question]
        all_vecs = embedder.embed_batch(all_texts)

        chunk_vecs = np.array(all_vecs[:-1], dtype=np.float32)
        query_vec  = np.array(all_vecs[-1],  dtype=np.float32)

        # Cosine similarity: dot / (||a|| * ||b||)
        norms  = np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
        norms  = np.where(norms == 0, 1e-9, norms)
        normed = chunk_vecs / norms
        q_norm = query_vec / (np.linalg.norm(query_vec) or 1e-9)
        scores = normed @ q_norm          # shape: (n_chunks,)

        top_idx = _argtopk(scores, k)
        return [chunks[i] for i in top_idx]

    # ------------------------------------------------------------------
    # Strategy 1b — RRF hybrid merge (BM25 + embedding, best of both)
    # Reciprocal Rank Fusion: score = Σ 1/(k + rank) across both lists.
    # k=60 is the standard RRF constant (Cormack et al. 2009).
    # Consistently +5-10% recall vs either alone on BEIR benchmarks.
    # ------------------------------------------------------------------

    _RRF_K = 60

    def _rrf_rank(self, question: str, chunks: list, k: int) -> list:
        # Get ranked lists from both strategies (fetch 2×k from each)
        fetch_k = min(k * 2, len(chunks))
        bm25_ranked    = self._bm25_rank(question, chunks, fetch_k)
        dense_ranked   = self._embedding_rank(question, chunks, fetch_k)

        # Build chunk_id → chunk map for dedup
        def _cid(chunk):
            if hasattr(chunk, "chunk_id"):
                return chunk.chunk_id
            if isinstance(chunk, dict):
                return chunk.get("chunk_id", id(chunk))
            return id(chunk)

        rrf_scores: dict = {}
        seen: dict = {}

        for rank, chunk in enumerate(bm25_ranked):
            cid = _cid(chunk)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._RRF_K + rank + 1)
            seen[cid] = chunk

        for rank, chunk in enumerate(dense_ranked):
            cid = _cid(chunk)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._RRF_K + rank + 1)
            seen[cid] = chunk

        sorted_cids = sorted(rrf_scores, key=lambda c: -rrf_scores[c])
        return [seen[cid] for cid in sorted_cids[:k]]

    # ------------------------------------------------------------------
    # Strategy 2 — BM25-style keyword scoring (zero dependencies)
    # ------------------------------------------------------------------

    def _bm25_rank(self, question: str, chunks: list, k: int) -> list:
        query_tokens = set(_tokenize(question))
        if not query_tokens:
            return list(chunks[:k])

        scored = []
        for chunk in chunks:
            text   = _chunk_text(chunk).lower()
            tokens = _tokenize(text)
            count  = sum(tokens.count(t) for t in query_tokens)
            # Slight length normalisation: divide by log(len+2) to avoid
            # huge functions always winning purely due to size.
            import math
            norm_score = count / math.log(len(tokens) + 2)
            scored.append((norm_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text(chunk) -> str:
    """Extract source text from either a CodeChunk object or a dict."""
    if hasattr(chunk, "source"):
        return chunk.source or ""
    if isinstance(chunk, dict):
        return chunk.get("source", chunk.get("compressed_src", ""))
    return str(chunk)


def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z_]\w*", text.lower())


def _argtopk(scores, k: int) -> list:
    """Return indices of the top-k scores, highest first. Pure Python/numpy."""
    import numpy as np
    if k >= len(scores):
        return list(range(len(scores)))
    idx = int(k)
    # argpartition gives unsorted top-k, then we sort the slice
    part = np.argpartition(scores, -idx)[-idx:]
    # sort within the partition by descending score
    part = part[np.argsort(scores[part])[::-1]]
    return part.tolist()
