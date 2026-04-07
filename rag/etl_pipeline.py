"""
etl_pipeline.py — Full ETL: scan → chunk → hash → embed → upsert
Run standalone: python rag/etl_pipeline.py --repo ./data/test_corpus/chocolate-doom --language c
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunker import Chunker
from rag.hasher import Hasher
from rag.embedder import Embedder
from rag.qdrant_client import QdrantStore


def run_etl(
    repo_path: str,
    language: str = None,
    rainbow_paths: list = None,
    batch_size: int = 32,
    dry_run: bool = False,
) -> dict:
    """
    Full ETL pipeline.
    Returns stats dict: {files, chunks, rainbow_hits, upserted, skipped, elapsed_s}
    """
    t0      = time.time()
    chunker = Chunker()
    hasher  = Hasher(rainbow_paths or [])
    embedder= Embedder()
    store   = QdrantStore()

    exts = {".py": ["python"], ".c": ["c"], ".h": ["c"], ".go": ["go"], ".rs": ["rust"]}
    if language:
        target_exts = [k for k, v in exts.items() if language in v]
    else:
        target_exts = list(exts.keys())

    print(f"  Scanning {repo_path} for {target_exts}...")
    chunks = chunker.chunk_repo(repo_path, extensions=target_exts)
    print(f"  {len(chunks)} chunks found")

    rainbow_hits = 0
    to_embed     = []
    for chunk in chunks:
        h_info = hasher.hash_chunk(chunk)
        if h_info["rainbow_hit"]:
            rainbow_hits += 1
        else:
            to_embed.append((chunk, h_info["sha256"]))

    print(f"  Rainbow cache: {rainbow_hits} hits, {len(to_embed)} to embed")

    if dry_run:
        return {
            "files": len(set(c.file_path for c in chunks)),
            "chunks": len(chunks),
            "rainbow_hits": rainbow_hits,
            "upserted": 0,
            "skipped": rainbow_hits,
            "elapsed_s": round(time.time() - t0, 2),
            "dry_run": True,
        }

    # Embed in batches
    upserted = 0
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i:i + batch_size]
        texts  = [c.source[:2000] for c, _ in batch]  # truncate huge chunks
        try:
            vectors = embedder.embed_batch(texts)
        except Exception as e:
            print(f"  Embedding error batch {i//batch_size}: {e}")
            continue

        points = []
        for (chunk, sha), vec in zip(batch, vectors):
            points.append({
                "chunk_id": chunk.chunk_id,
                "vector":   vec,
                "payload":  {
                    "chunk_id":       chunk.chunk_id,
                    "file_path":      chunk.file_path,
                    "language":       chunk.language,
                    "chunk_type":     chunk.chunk_type,
                    "name":           chunk.name,
                    "start_line":     chunk.start_line,
                    "end_line":       chunk.end_line,
                    "sha256":         sha,
                    "original_tokens": chunk.original_tokens,
                    "rainbow_hit":    False,
                },
            })
        try:
            n = store.upsert(points)
            upserted += n
        except Exception as e:
            print(f"  Upsert error: {e}")

        print(f"  Batch {i//batch_size + 1}: {len(points)} upserted (total: {upserted})")

    elapsed = round(time.time() - t0, 2)
    stats = {
        "files":        len(set(c.file_path for c in chunks)),
        "chunks":       len(chunks),
        "rainbow_hits": rainbow_hits,
        "upserted":     upserted,
        "skipped":      rainbow_hits,
        "elapsed_s":    elapsed,
        "dry_run":      False,
    }
    print(f"\n  ETL complete: {stats}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FTW-KLOC-KILLER ETL Pipeline")
    parser.add_argument("--repo",     required=True, help="Path to source repo")
    parser.add_argument("--language", default=None,  help="Filter: python|c|go|rust")
    parser.add_argument("--rainbow",  nargs="*",     help="Rainbow table JSONL files")
    parser.add_argument("--batch",    type=int, default=32, help="Embedding batch size")
    parser.add_argument("--dry-run",  action="store_true", help="Chunk+hash only, no embed/upsert")
    args = parser.parse_args()

    run_etl(
        repo_path     = args.repo,
        language      = args.language,
        rainbow_paths = args.rainbow,
        batch_size    = args.batch,
        dry_run       = args.dry_run,
    )
