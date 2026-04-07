"""
c_etl.py — C/C++ repo ETL pipeline
====================================
Full pipeline for an arbitrary C/C++ repo:

  Step 1  Mine patterns  → build per-repo CPatternDict (.kloc/c_pattern_dict.json)
  Step 2  Chunk          → extract functions/structs via CParser
  Step 3  Compress       → run CCompressor (F1+F3+F4) on every chunk
  Step 4  Hash           → SHA-256 + rainbow table lookup
  Step 5  Embed          → sentence-transformer vectors (384-dim)
  Step 6  Upsert         → store in Qdrant with compressed source + metadata

The compressed source (not original) is what gets embedded and stored.
This means RAG retrieval returns already-compressed context — direct LLM input
without a second compression pass at query time.

Usage:
    python rag/c_etl.py --repo doom/ --dry-run
    python rag/c_etl.py --repo doom/ --force-remine
    python rag/c_etl.py --repo doom/ --file doom/src/strife/p_enemy.c
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunker import Chunker
from rag.hasher import Hasher
from src.c_compressor import CCompressor, CTokenReport
from languages.c_pattern_miner import CPatternMiner, CPatternDict

C_EXTENSIONS = [".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"]

# ── Stats dataclass ────────────────────────────────────────────────────────────

@dataclass
class CETLStats:
    repo_path:      str
    files:          int    = 0
    chunks:         int    = 0
    rainbow_hits:   int    = 0
    to_embed:       int    = 0
    upserted:       int    = 0
    skipped:        int    = 0
    total_orig_tok: int    = 0
    total_comp_tok: int    = 0
    pattern_count:  int    = 0
    elapsed_s:      float  = 0.0
    dry_run:        bool   = False
    errors:         List[str] = field(default_factory=list)

    @property
    def ter_pct(self) -> float:
        if self.total_orig_tok == 0:
            return 0.0
        return round((1 - self.total_comp_tok / self.total_orig_tok) * 100, 1)

    def __str__(self) -> str:
        lines = [
            f"  C ETL complete {'(DRY RUN) ' if self.dry_run else ''}",
            f"  Files:    {self.files}",
            f"  Chunks:   {self.chunks}",
            f"  Patterns: {self.pattern_count} KQ patterns mined",
            f"  TER:      {self.ter_pct:.1f}%  ({self.total_orig_tok:,} -> {self.total_comp_tok:,} tok)",
            f"  Rainbow:  {self.rainbow_hits} cache hits",
            f"  Upserted: {self.upserted}",
            f"  Elapsed:  {self.elapsed_s:.2f}s",
        ]
        if self.errors:
            lines.append(f"  Errors:   {len(self.errors)}")
        return "\n".join(lines)


# ── Main ETL function ──────────────────────────────────────────────────────────

def run_c_etl(
    repo_path:    str,
    force_remine: bool = False,
    dry_run:      bool = False,
    single_file:  Optional[str] = None,
    batch_size:   int  = 32,
    rainbow_paths: Optional[List[str]] = None,
    verbose:      bool = True,
) -> CETLStats:
    """
    Run the full C ETL pipeline on a repo (or a single file within it).

    Args:
        repo_path:    Root of the C/C++ repository.
        force_remine: Rebuild pattern dict even if .kloc/c_pattern_dict.json exists.
        dry_run:      Stop after compress+hash — no embed/upsert.
        single_file:  If set, only process this one file (must be inside repo_path).
        batch_size:   Embedding batch size.
        rainbow_paths: Extra rainbow table JSONL files.
        verbose:      Print progress.

    Returns:
        CETLStats with full metrics.
    """
    t0        = time.time()
    repo_path = Path(repo_path)
    stats     = CETLStats(repo_path=str(repo_path))

    def _log(msg: str):
        if verbose:
            print(msg)

    # ── Step 1: Mine patterns ──────────────────────────────────────────
    _log(f"\n  [1/5] Mining C patterns in {repo_path.name}...")
    compressor = CCompressor.for_repo(repo_path, force_remine=force_remine)
    pdict      = compressor.pattern_dict
    stats.pattern_count = len(pdict)
    _log(f"       {pdict.summary()}")

    # ── Step 2: Chunk ──────────────────────────────────────────────────
    _log(f"  [2/5] Chunking C files...")
    chunker = Chunker()

    if single_file:
        chunks = chunker.chunk_file(Path(single_file))
        stats.files = 1
    else:
        chunks = chunker.chunk_repo(repo_path, extensions=C_EXTENSIONS)
        stats.files = len(set(c.file_path for c in chunks))

    stats.chunks = len(chunks)
    _log(f"       {stats.chunks} chunks from {stats.files} files")

    if not chunks:
        _log("  No chunks found — aborting.")
        stats.elapsed_s = round(time.time() - t0, 2)
        return stats

    # ── Step 3: Compress ───────────────────────────────────────────────
    _log(f"  [3/5] Compressing chunks (F1+F3+F4)...")
    compressed_chunks = []

    for chunk in chunks:
        try:
            compressed_src, report = compressor.compress(
                source = chunk.source,
                zone   = "FOREGROUND",   # TODO: wire PRUNER zone here
            )
            stats.total_orig_tok += report.original
            stats.total_comp_tok += report.final
            compressed_chunks.append((chunk, compressed_src, report))
        except Exception as exc:
            stats.errors.append(f"{chunk.chunk_id}: {exc}")
            # Fall back to uncompressed
            from src.inference_bridge import count_tokens
            orig_t = count_tokens(chunk.source)
            stats.total_orig_tok += orig_t
            stats.total_comp_tok += orig_t
            compressed_chunks.append((chunk, chunk.source, None))

    ter = stats.ter_pct
    _log(f"       TER={ter:.1f}%  ({stats.total_orig_tok:,} -> {stats.total_comp_tok:,} tok)")

    # ── Step 4: Hash ───────────────────────────────────────────────────
    _log(f"  [4/5] Hashing chunks...")
    hasher = Hasher(rainbow_paths or [])

    to_embed: List[tuple] = []
    for chunk, comp_src, report in compressed_chunks:
        try:
            h_info = hasher.hash_chunk(chunk)
            if h_info["rainbow_hit"]:
                stats.rainbow_hits += 1
                stats.skipped      += 1
            else:
                to_embed.append((chunk, comp_src, h_info["sha256"]))
        except Exception as exc:
            stats.errors.append(f"hash {chunk.chunk_id}: {exc}")
            to_embed.append((chunk, comp_src, "unknown"))

    stats.to_embed = len(to_embed)
    _log(f"       {stats.rainbow_hits} rainbow hits, {stats.to_embed} to embed")

    if dry_run:
        stats.elapsed_s = round(time.time() - t0, 2)
        _log(f"\n  [DRY RUN] Stopping before embed/upsert.")
        _log(str(stats))
        return stats

    # ── Step 5+6: Embed + Upsert ───────────────────────────────────────
    _log(f"  [5/5] Embedding and upserting {stats.to_embed} chunks...")
    try:
        from rag.embedder import Embedder
        from rag.qdrant_client import QdrantStore
        embedder = Embedder()
        store    = QdrantStore()
    except ImportError as exc:
        _log(f"  [SKIP] Embedding not available: {exc}")
        stats.elapsed_s = round(time.time() - t0, 2)
        return stats

    # Load metadata index once — used to prepend summaries to embedded text.
    # If no index exists this is a no-op (meta_idx stays None).
    meta_idx = None
    try:
        from rag.metadata_index import MetadataIndex
        meta_idx = MetadataIndex.load_for_repo(repo_path)
        if meta_idx:
            _log(f"       Metadata index loaded: {meta_idx.summary()}")
    except Exception:
        pass

    def _embed_text(chunk, comp_src: str) -> str:
        """Build text to embed: summary (if available) prepended to compressed source."""
        if meta_idx:
            meta = meta_idx._by_chunk.get(chunk.chunk_id)
            if meta and meta.summary:
                # Summary at the front: transformers are positionally biased
                return f"{meta.summary}\n{comp_src[:1800]}"
        return comp_src[:2000]

    upserted = 0
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i : i + batch_size]
        texts = [_embed_text(chunk, comp_src) for chunk, comp_src, _ in batch]

        try:
            vectors = embedder.embed_batch(texts)
        except Exception as exc:
            stats.errors.append(f"embed batch {i//batch_size}: {exc}")
            continue

        points = []
        for (chunk, comp_src, sha), vec in zip(batch, vectors):
            # Include summary in payload for downstream retrieval scoring
            summary = ""
            if meta_idx:
                meta = meta_idx._by_chunk.get(chunk.chunk_id)
                if meta:
                    summary = meta.summary or ""
            points.append({
                "chunk_id": chunk.chunk_id,
                "vector":   vec,
                "payload":  {
                    "chunk_id":         chunk.chunk_id,
                    "file_path":        chunk.file_path,
                    "language":         chunk.language,
                    "chunk_type":       chunk.chunk_type,
                    "name":             chunk.name,
                    "start_line":       chunk.start_line,
                    "end_line":         chunk.end_line,
                    "sha256":           sha,
                    "original_tokens":  chunk.original_tokens,
                    "compressed_src":   comp_src,
                    "summary":          summary,
                    "rainbow_hit":      False,
                    "repo_path":        str(repo_path),
                },
            })

        try:
            n = store.upsert(points)
            upserted += n
        except Exception as exc:
            stats.errors.append(f"upsert batch {i//batch_size}: {exc}")

        _log(f"       Batch {i//batch_size + 1}: {len(points)} points (total: {upserted})")

    stats.upserted  = upserted
    stats.elapsed_s = round(time.time() - t0, 2)

    _log(str(stats))
    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="C/C++ ETL pipeline for FTW-KLOC-KILLER")
    parser.add_argument("--repo",         required=True, help="Path to C/C++ repo root")
    parser.add_argument("--file",         default=None,  help="Single file to process (optional)")
    parser.add_argument("--force-remine", action="store_true", help="Rebuild pattern dict")
    parser.add_argument("--dry-run",      action="store_true", help="Stop after compress, no embed/upsert")
    parser.add_argument("--batch",        type=int, default=32, help="Embedding batch size")
    parser.add_argument("--rainbow",      nargs="*", help="Extra rainbow JSONL paths")
    args = parser.parse_args()

    stats = run_c_etl(
        repo_path     = args.repo,
        force_remine  = args.force_remine,
        dry_run       = args.dry_run,
        single_file   = args.file,
        batch_size    = args.batch,
        rainbow_paths = args.rainbow,
    )

    if stats.errors:
        print(f"\n  Errors ({len(stats.errors)}):")
        for e in stats.errors[:10]:
            print(f"    {e}")

    sys.exit(0 if not stats.errors else 1)
