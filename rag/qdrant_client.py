"""
qdrant_client.py — Thin Qdrant wrapper
Collection: kloc_killer_chunks  |  Vector size: 384  |  Distance: Cosine
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

QDRANT_URL   = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION   = os.getenv("QDRANT_COLLECTION", "kloc_killer_chunks")
VECTOR_SIZE  = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))


class QdrantStore:
    def __init__(self, url: str = QDRANT_URL, collection: str = COLLECTION):
        self.url = url
        self.collection = collection
        self._client = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self.url)
        return self._client

    def ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams
        client = self._get_client()
        existing = [c.name for c in client.get_collections().collections]
        if self.collection not in existing:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def upsert(self, chunks_with_vectors: List[Dict[str, Any]]) -> int:
        """
        chunks_with_vectors: list of {chunk_id, vector, payload}
        payload: {file_path, language, chunk_type, sha256, original_tokens, ...}
        Returns number of points upserted.
        """
        from qdrant_client.models import PointStruct
        client = self._get_client()
        self.ensure_collection()
        points = []
        for i, item in enumerate(chunks_with_vectors):
            import hashlib
            point_id = int(hashlib.md5(item["chunk_id"].encode()).hexdigest()[:8], 16)
            points.append(PointStruct(
                id      = point_id,
                vector  = item["vector"],
                payload = item.get("payload", {}),
            ))
        client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def query(self, vector: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = self._get_client()
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)
        results = client.search(
            collection_name = self.collection,
            query_vector    = vector,
            limit           = top_k,
            query_filter    = qdrant_filter,
            with_payload    = True,
        )
        return [
            {"score": r.score, "chunk_id": r.payload.get("chunk_id"), **r.payload}
            for r in results
        ]

    def delete_collection(self):
        self._get_client().delete_collection(self.collection)

    def count(self) -> int:
        return self._get_client().count(self.collection).count
