"""
build_metadata.py — Pre-compute function metadata index for a C/Python repo
============================================================================
Builds a MetadataIndex and writes it to {repo}/.kloc/metadata_index.json.

Steps:
  1. Chunk all C/Python files via Chunker
  2. Extract signatures, callee names, complexity tier via MetadataIndexBuilder
  3. (Optional) Generate one-line summaries via T1 Haiku — gated on --summaries flag
  4. Save index to .kloc/metadata_index.json

Usage:
    python rag/build_metadata.py --repo doom/
    python rag/build_metadata.py --repo doom/ --summaries    # adds T1 Haiku summaries
    python rag/build_metadata.py --repo doom/ --file doom/src/p_enemy.c
    python rag/build_metadata.py --repo . --dry-run          # print index, don't save
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunker import Chunker
from rag.metadata_index import MetadataIndex, MetadataIndexBuilder, FunctionMeta

C_EXTENSIONS = [".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"]
PY_EXTENSIONS = [".py"]


def build_metadata(
    repo_path: Path,
    single_file: Path = None,
    add_summaries: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> MetadataIndex:
    """
    Build MetadataIndex for repo_path and optionally enrich with summaries.
    Returns the completed MetadataIndex.
    """
    t0 = time.time()

    def _log(msg: str):
        if verbose:
            print(msg)

    # Step 1: chunk
    _log(f"\n  [1/3] Chunking {repo_path.name}...")
    chunker = Chunker()

    if single_file:
        chunks = chunker.chunk_file(single_file)
        _log(f"       {len(chunks)} chunks from {single_file.name}")
    else:
        chunks = chunker.chunk_repo(
            repo_path,
            extensions=C_EXTENSIONS + PY_EXTENSIONS,
        )
        file_count = len(set(getattr(c, "file_path", "") for c in chunks))
        _log(f"       {len(chunks)} chunks from {file_count} files")

    # Step 2: build index
    _log(f"  [2/3] Building metadata index...")
    builder = MetadataIndexBuilder()
    index   = builder.build(chunks, repo_path)
    _log(f"       {index.summary()}")

    # Step 3: optional T1 summaries
    if add_summaries:
        _log(f"  [3/3] Generating summaries (T1 Haiku)...")
        _enrich_summaries(index, verbose)
    else:
        _log(f"  [3/3] Skipping summaries (pass --summaries to enable)")

    elapsed = round(time.time() - t0, 2)
    _log(f"\n  Done in {elapsed}s")

    if dry_run:
        _log(f"\n  [DRY RUN] Not saving.")
        _print_sample(index)
        return index

    saved = index.save()
    _log(f"  Saved -> {saved}")
    _print_sample(index)
    return index


def _enrich_summaries(index: MetadataIndex, verbose: bool) -> None:
    """
    Generate one-line summaries for complex functions via T1 Haiku.
    Skips functions that already have summaries, caps at 50 per run.
    """
    tier_max = int(__import__("os").environ.get("ACTIVE_TIER_MAX", "2"))
    if tier_max == 0:
        if verbose:
            print("       ACTIVE_TIER_MAX=0 — skipping LLM summaries")
        return

    to_summarise = [
        f for f in index.functions
        if not f.summary and f.complexity in ("complex", "moderate")
    ][:50]

    if not to_summarise:
        if verbose:
            print("       No functions to summarise")
        return

    if verbose:
        print(f"       Summarising {len(to_summarise)} functions...")

    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        if verbose:
            print("       anthropic SDK not installed — skipping summaries")
        return

    for meta in to_summarise:
        # Build a compact prompt
        prompt = (
            f"In one sentence (max 20 words), describe what this function does:\n\n"
            f"Function: {meta.signature or meta.name}\n"
            f"Calls: {', '.join(meta.callee_names[:5]) or 'none'}\n"
        )
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            meta.summary = resp.content[0].text.strip()
        except Exception as exc:
            if verbose:
                print(f"       Summary failed for {meta.name}: {exc}")

    summarised = sum(1 for f in to_summarise if f.summary)
    if verbose:
        print(f"       Added {summarised} summaries")


def _print_sample(index: MetadataIndex) -> None:
    top = sorted(index.functions, key=lambda f: -f.line_count)[:10]
    print("\n  Top 10 functions by size:")
    for f in top:
        callee_preview = ", ".join(f.callee_names[:3])
        print(f"    [{f.line_count:4d} lines] {f.complexity:<8} {f.name:<30} calls=[{callee_preview}]")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Build function metadata index for a C/Python repo"
    )
    parser.add_argument("--repo",      required=True, help="Repo root path")
    parser.add_argument("--file",      default=None,  help="Single file to index (optional)")
    parser.add_argument("--summaries", action="store_true", help="Generate T1 Haiku summaries for complex functions")
    parser.add_argument("--dry-run",   action="store_true", help="Print index but don't save")
    args = parser.parse_args()

    repo = Path(args.repo)
    single = Path(args.file) if args.file else None

    build_metadata(
        repo_path     = repo,
        single_file   = single,
        add_summaries = args.summaries,
        dry_run       = args.dry_run,
    )
