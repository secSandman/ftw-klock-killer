"""
embedder.py — Embedding generation
Uses sentence-transformers/all-MiniLM-L6-v2 locally (free, 384-dim).
Falls back to OpenAI text-embedding-3-small if EMBEDDING_PROVIDER=openai.

Upgrade path for better code retrieval (+10-15% MRR vs all-MiniLM):
  EMBEDDING_MODEL=flax-sentence-embeddings/st-codesearch-distilroberta-base
  Note: that model is 768-dim — set QDRANT_VECTOR_SIZE=768 and re-run ETL.

  Or the lighter code-specific option:
  EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2  (768-dim, general)
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
import numpy as np

_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

# Override with EMBEDDING_MODEL env-var for code-specific models.
# Default: all-MiniLM-L6-v2 (384-dim, fast, generic NLI/STS trained)
# Recommended code upgrade: flax-sentence-embeddings/st-codesearch-distilroberta-base (768-dim)
_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# Model → dimension mapping (auto-detected; QDRANT_VECTOR_SIZE overrides)
_MODEL_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "flax-sentence-embeddings/st-codesearch-distilroberta-base": 768,
    "sentence-transformers/all-mpnet-base-v2": 768,
}

_local_model = None

def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(_MODEL_NAME)
    return _local_model


def _infer_dim() -> int:
    """Infer vector dimension from model name or QDRANT_VECTOR_SIZE env-var."""
    env_dim = os.getenv("QDRANT_VECTOR_SIZE")
    if env_dim:
        return int(env_dim)
    return _MODEL_DIMS.get(_MODEL_NAME, 384)


class Embedder:
    """Generates embeddings for code chunks.

    Default model: all-MiniLM-L6-v2 (384-dim, fast, generic)
    Override:  EMBEDDING_MODEL=<hf-model-id>  (must also set QDRANT_VECTOR_SIZE if != 384)
    """

    def __init__(self, provider: str = _PROVIDER):
        self.provider = provider
        self.dim = _infer_dim()

    def embed(self, text: str) -> List[float]:
        """Embed a single text string. Returns list[float] of length self.dim."""
        if self.provider == "openai":
            return self._embed_openai(text)
        return self._embed_local(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            return [self._embed_openai(t) for t in texts]
        return self._embed_local_batch(texts)

    def _embed_local(self, text: str) -> List[float]:
        model = _get_local_model()
        vec = model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def _embed_local_batch(self, texts: List[str]) -> List[List[float]]:
        model = _get_local_model()
        vecs = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        return vecs.tolist()

    def _embed_openai(self, text: str) -> List[float]:
        import openai
        resp = openai.embeddings.create(model="text-embedding-3-small", input=text[:8000])
        return resp.data[0].embedding
