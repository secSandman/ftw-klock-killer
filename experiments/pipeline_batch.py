"""
pipeline_batch.py — Measure the combined (blended) cost of compress → T0.5 → remote.
======================================================================================
Runs a batch of questions through the full local pipeline:
  compress (F1+F3+F4) → T0.5 quality gate → record tier used

Measures:
  - acceptance_rate: % answered by T0.5 without calling Anthropic
  - blended_cost_per_query: (accepted × $0) + (rejected × compressed_cost)
  - projected_savings: vs sending full file to Anthropic every time

NOTE (PROTO-001): This is a prototype. Quality of T0.5 answers vs Anthropic answers
is NOT yet compared (requires Anthropic API calls for ground truth). That is M6.

Usage:
  python experiments/pipeline_batch.py --file doom/src/strife/p_enemy.c --repo doom/ --n 10
  python kloc.py experiment pipeline-batch --file doom/src/strife/p_enemy.c --repo doom/
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import _check_ollama at module level so tests can patch it via
# "experiments.pipeline_batch._check_ollama"
from experiments.local_llm_sweep import _check_ollama  # noqa: E402

COST_T1_PER_1K = 0.00025   # Haiku input tokens

DEFAULT_C_QUESTIONS = [
    "What does this function do?",
    "Explain the main control flow",
    "What are the key data structures used?",
    "What functions modify enemy state?",
    "How does the AI pathfinding work?",
]

_REFUSAL_PHRASES = [
    "i cannot",
    "i don't have",
    "i'm unable",
    "no context provided",
]


def _quality_gate(response: str, threshold: int = 20) -> bool:
    """
    Return True if the response passes the quality gate (good enough to use).
    Reject if:
      - fewer than `threshold` words
      - contains a known refusal phrase
    Gate is strictly less-than: len < threshold → reject, len == threshold → pass.
    """
    if not response:
        return False
    lower = response.lower()
    for phrase in _REFUSAL_PHRASES:
        if phrase in lower:
            return False
    words = response.split()
    return len(words) >= threshold


def _compress_c_file(
    file_path: str,
    repo_path: str,
    cpi_threshold: float = 0.40,
) -> tuple[str, int, int, float]:
    """
    Compress a C file with CPI_THRESHOLD override, returning
    (compressed_text, orig_tokens, compressed_tokens, ter_pct).

    Uses the same staged-pass pattern as generate_c_report.
    """
    from src.c_compressor import CCompressor, CSkeletonizer, C_IncludeMasker, C_CavemanCompressor
    from src.inference_bridge import count_tokens

    fp = Path(file_path)
    rp = Path(repo_path) if repo_path else fp.parent

    source      = fp.read_text(encoding="utf-8", errors="replace")
    orig_tokens = count_tokens(source)

    # Temporarily override the CPI threshold via env var so CSkeletonizer picks it up
    prev_cpi = os.environ.get("KLOC_CPI_THRESHOLD")
    os.environ["KLOC_CPI_THRESHOLD"] = str(cpi_threshold)
    try:
        compressor   = CCompressor.for_repo(rp)
        pdict        = compressor.pattern_dict
        skeletonizer = CSkeletonizer()
        masker       = C_IncludeMasker(pdict)
        caveman      = C_CavemanCompressor()

        after_skel, _ = skeletonizer.skeletonize(source)
        after_mask, _ = masker.mask(after_skel)
        compressed    = caveman.compress(after_mask)
    finally:
        if prev_cpi is None:
            os.environ.pop("KLOC_CPI_THRESHOLD", None)
        else:
            os.environ["KLOC_CPI_THRESHOLD"] = prev_cpi

    compressed_tokens = count_tokens(compressed)
    ter_pct = round((1 - compressed_tokens / orig_tokens) * 100, 1) if orig_tokens else 0.0
    return compressed, orig_tokens, compressed_tokens, ter_pct


def _compress_py_file(file_path: str) -> tuple[str, int, int, float]:
    """
    Compress a Python file using InferenceBridge F1+F3+F4.
    Returns (compressed_text, orig_tokens, compressed_tokens, ter_pct).
    """
    import tempfile
    from src.inference_bridge import InferenceBridge, count_tokens

    fp     = Path(file_path)
    source = fp.read_text(encoding="utf-8", errors="replace")

    with tempfile.TemporaryDirectory() as tmp:
        reg = str(Path(tmp) / "registry.json")
        inb = InferenceBridge(repo_path=str(fp.parent),
                              registry_path=reg,
                              enable_delta=False)
        compressed, report = inb.process_file(str(fp))

    orig_tokens       = report.original
    compressed_tokens = report.final
    ter_pct = round((1 - compressed_tokens / orig_tokens) * 100, 1) if orig_tokens else 0.0
    return compressed, orig_tokens, compressed_tokens, ter_pct


def run_pipeline_batch(
    file_path:          str,
    questions:          List[str],
    repo_path:          str          = "",
    cpi_threshold:      float        = 0.40,
    max_ctx_words:      int          = 8000,
    quality_threshold:  int          = 20,
    model:              str          = "qwen2.5-coder:14b",
    temperature:        float        = 0.1,
    save:               bool         = True,
) -> dict:
    """
    Run a batch of questions through compress → T0.5 → record tier.

    Returns a dict with aggregate metrics and per-query results list.
    If Ollama is not reachable, returns {"error": "...", "total_queries": 0}.
    """
    from experiments.local_llm import LocalLLMHarness

    if not _check_ollama():
        return {
            "error":         "Ollama not running — start with: python kloc.py experiment llm-setup",
            "total_queries": 0,
        }

    fp   = Path(file_path)
    ext  = fp.suffix.lower()
    is_c = ext in (".c", ".cpp", ".h", ".cc")

    # ── Compress once, reuse for all queries ─────────────────────────────────
    try:
        if is_c:
            compressed_text, orig_tokens, compressed_tokens, ter_pct = _compress_c_file(
                file_path, repo_path, cpi_threshold=cpi_threshold
            )
        else:
            compressed_text, orig_tokens, compressed_tokens, ter_pct = _compress_py_file(
                file_path
            )
    except Exception as exc:
        return {
            "error":         f"Compression failed: {exc}",
            "total_queries": 0,
        }

    harness = LocalLLMHarness(model=model, verbose=False)

    per_query   = []
    accepted    = 0
    rejected    = 0
    latencies   = []
    ctx_exceeded = 0

    for q in questions:
        q_result: dict = {"question": q}

        # Context word-count guard
        ctx_words = len(compressed_text.split())
        if ctx_words > max_ctx_words:
            q_result["tier"]              = "ctx_exceeded"
            q_result["tokens_to_remote"]  = 0
            q_result["latency_ms"]        = None
            q_result["response_words"]    = 0
            ctx_exceeded += 1
            per_query.append(q_result)
            continue

        t_start = time.time()
        response = harness.ask(
            q,
            context     = compressed_text,
            temperature = temperature,
        )
        latency_ms = harness._last_latency_ms

        passed = _quality_gate(response, threshold=quality_threshold)

        if passed:
            accepted += 1
            q_result["tier"]             = "local"
            q_result["tokens_to_remote"] = 0
        else:
            rejected += 1
            q_result["tier"]             = "remote_compressed"
            q_result["tokens_to_remote"] = compressed_tokens

        q_result["latency_ms"]     = latency_ms
        q_result["response_words"] = len(response.split()) if response else 0
        q_result["response_preview"] = response[:120] if response else ""
        if latency_ms is not None:
            latencies.append(latency_ms)
        per_query.append(q_result)

    total = len(questions)
    if total == 0:
        return {"error": "No questions provided", "total_queries": 0}

    acceptance_rate = round(accepted / total, 4) if total else 0.0
    avg_latency_ms  = round(sum(latencies) / len(latencies)) if latencies else None

    cost_uncompressed_t1 = orig_tokens       * COST_T1_PER_1K / 1000
    cost_compressed_t1   = compressed_tokens * COST_T1_PER_1K / 1000
    # Blended: rejected queries pay compressed_cost, accepted pay $0
    cost_blended_t1 = (rejected * compressed_tokens * COST_T1_PER_1K / 1000) / total if total else 0.0
    savings_pct = (
        round((1 - cost_blended_t1 / cost_uncompressed_t1) * 100, 1)
        if cost_uncompressed_t1 > 0 else 0.0
    )

    result = {
        "proto":                     "PROTO-001",
        "file":                      file_path,
        "model":                     model,
        "cpi_threshold":             cpi_threshold,
        "total_queries":             total,
        "accepted":                  accepted,
        "rejected":                  rejected,
        "ctx_exceeded":              ctx_exceeded,
        "acceptance_rate":           acceptance_rate,
        "orig_tokens":               orig_tokens,
        "compressed_tokens":         compressed_tokens,
        "ter_pct":                   ter_pct,
        "avg_latency_ms":            avg_latency_ms,
        "cost_uncompressed_t1":      round(cost_uncompressed_t1, 8),
        "cost_compressed_t1":        round(cost_compressed_t1, 8),
        "cost_blended_t1":           round(cost_blended_t1, 8),
        "savings_vs_uncompressed_pct": savings_pct,
        "per_query":                 per_query,
    }

    if save:
        _save_result(result)

    return result


def _save_result(result: dict) -> None:
    """Append result to experiments/results/YYYY-MM-DD/batch_results.jsonl"""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    out_dir = ROOT / "experiments" / "results" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "batch_results.jsonl"
    with open(out_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, default=str) + "\n")


def print_batch_results(result: dict) -> None:
    """Print per-query tier table + aggregate metrics to terminal."""
    W = 72

    print()
    print("=" * W)
    print("  BLENDED PIPELINE BATCH  (PROTO-001)")
    print("=" * W)

    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        print("=" * W)
        return

    print(f"  File:         {result.get('file', '?')}")
    print(f"  Model (T0.5): {result.get('model', '?')}")
    print(f"  CPI thresh:   {result.get('cpi_threshold', '?')}")
    print()

    # Per-query table
    header = f"  {'#':>3}  {'Tier':<20}  {'Lat(ms)':>8}  {'Words':>5}  Question"
    print(header)
    print("  " + "-" * (W - 2))
    for i, q in enumerate(result.get("per_query", []), 1):
        tier    = q.get("tier", "?")
        lat     = q.get("latency_ms")
        lat_s   = f"{lat}" if lat is not None else "—"
        words   = q.get("response_words", "?")
        q_text  = q.get("question", "")[:42]
        print(f"  {i:>3}  {tier:<20}  {lat_s:>8}  {words:>5}  {q_text}")
    print("  " + "-" * (W - 2))

    # Aggregate
    total   = result.get("total_queries", 0)
    accepted = result.get("accepted", 0)
    rejected = result.get("rejected", 0)
    ar       = result.get("acceptance_rate", 0.0)
    lat_avg  = result.get("avg_latency_ms")
    lat_s    = f"{lat_avg}ms" if lat_avg is not None else "—"
    print()
    print(f"  Total queries:    {total}")
    print(f"  Accepted (T0.5):  {accepted}  ({ar*100:.0f}%)")
    print(f"  Rejected (remote):{rejected}  ({(1-ar)*100:.0f}%)")
    print(f"  Avg latency:      {lat_s}")
    print()

    # Cost table
    orig_tok  = result.get("orig_tokens", 0)
    comp_tok  = result.get("compressed_tokens", 0)
    ter       = result.get("ter_pct", 0.0)
    cu_t1     = result.get("cost_uncompressed_t1", 0.0)
    cc_t1     = result.get("cost_compressed_t1", 0.0)
    cb_t1     = result.get("cost_blended_t1", 0.0)
    sav_pct   = result.get("savings_vs_uncompressed_pct", 0.0)

    W2 = 72
    print(f"  {'Layer':<26} {'Tokens':>7}  {'TER':>6}  {'Cost/query (T1)':>16}  {'vs Uncompressed'}")
    print("  " + "-" * (W2 - 2))
    print(f"  {'Uncompressed (none)':<26} {orig_tok:>7,}  {'0.0%':>6}  ${cu_t1:.5f}{'':>8}  baseline")
    print(f"  {'Compressed only':<26} {comp_tok:>7,}  {ter:>5.1f}%  ${cc_t1:.5f}{'':>8}  -{ter:.1f}%")
    print(f"  {'ACTUAL blended result':<26} {'':>7}  {'':>6}  ${cb_t1:.5f}{'':>8}  -{sav_pct:.1f}%")
    print("  " + "-" * (W2 - 2))
    print()
    print("  NOTE (PROTO-001): T0.5 answer quality vs Anthropic ground truth is NOT")
    print("  yet compared. Acceptance = response length >= threshold, no refusals.")
    print("  Run with ACTIVE_TIER_MAX=1 + Anthropic API key for M6 ground-truth eval.")
    print("=" * W)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="pipeline_batch — PROTO-001 blended pipeline measurement"
    )
    parser.add_argument("--file",       required=True,  help="Source file (C or Python)")
    parser.add_argument("--repo",       default="",     help="Repo root (for C pattern dict)")
    parser.add_argument("--n",          type=int, default=5,
                        help="Number of questions (default: 5)")
    parser.add_argument("--questions",  default="",
                        help="Path to .txt file with questions (one per line)")
    parser.add_argument("--model",      default="qwen2.5-coder:14b",
                        help="Ollama model to use (default: qwen2.5-coder:14b)")
    parser.add_argument("--threshold",  type=int, default=20,
                        help="Minimum word count to pass quality gate (default: 20)")
    parser.add_argument("--max-ctx",    type=int, default=8000, dest="max_ctx",
                        help="Max word count of compressed text before ctx_exceeded (default: 8000)")
    parser.add_argument("--cpi",        type=float, default=0.40, dest="cpi_threshold",
                        help="CPI threshold for C skeletonizer (default: 0.40)")
    args = parser.parse_args()

    questions = DEFAULT_C_QUESTIONS[: args.n]
    if args.questions:
        qp = Path(args.questions)
        if qp.exists():
            lines = qp.read_text(encoding="utf-8").strip().splitlines()
            questions = [l.strip() for l in lines if l.strip()][: args.n]

    result = run_pipeline_batch(
        file_path         = args.file,
        questions         = questions,
        repo_path         = args.repo,
        cpi_threshold     = args.cpi_threshold,
        max_ctx_words     = args.max_ctx,
        quality_threshold = args.threshold,
        model             = args.model,
    )
    print_batch_results(result)
