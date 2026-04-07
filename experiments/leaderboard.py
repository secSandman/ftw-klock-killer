"""
leaderboard.py — Read all experiment results and show ranked table
===================================================================
Reads all JSONL files in experiments/results/ and prints a sorted leaderboard.

Usage:
    python experiments/leaderboard.py                    # all time
    python experiments/leaderboard.py --date 2026-04-06  # one day
    python experiments/leaderboard.py --top 20           # top N
    python experiments/leaderboard.py --param KLOC_PI_THRESHOLD  # filter by param
    python experiments/leaderboard.py --diff             # show delta from baseline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"

# Baseline numbers to compute delta against
BASELINE = {
    "python_ter_pct": 0.793,
    "rqs_l1":         0.650,
    "true_value":     0.516,
    "c_ter_pct":      0.351,
    "c_rqs_l1":       0.862,
    "rqs_llm":        0.630,   # baseline from 14B benchmark (2026-04-06)
    "true_value_llm": 0.0,
    # TV×TIME baseline: TV / elapsed_s at default config (~2s subprocess)
    "tv_time":        0.258,   # 0.516 / 2.0s  (approximate; recalculated from median)
}


def load_results(
    date: str = None,
    param_filter: str = None,
    type_filter: str = "all",
) -> List[dict]:
    """
    Load experiment results from JSONL files.

    Args:
        date:         Filter by date string YYYY-MM-DD (None = all dates).
        param_filter: Filter by param name substring.
        type_filter:  'all' | 'sim' (simulation-only) | 'llm' (real-LLM results only).
    """
    results = []
    if not RESULTS_DIR.exists():
        return results

    for day_dir in sorted(RESULTS_DIR.iterdir()):
        if not day_dir.is_dir():
            continue
        if date and day_dir.name != date:
            continue
        jsonl = day_dir / "results.jsonl"
        if not jsonl.exists():
            continue
        for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if param_filter:
                    if not any(param_filter.upper() in k.upper()
                               for k in r.get("env", {}).keys()):
                        continue
                # Apply type filter
                has_rqs_llm = r.get("metrics", {}).get("rqs_llm") is not None
                if type_filter == "llm" and not has_rqs_llm:
                    continue
                if type_filter == "sim" and has_rqs_llm:
                    continue
                results.append(r)
            except json.JSONDecodeError:
                continue

    return results


def _compute_tv_time(results: List[dict]) -> None:
    """
    Inject tv_time into each result's metrics in-place.

    tv_time = TV × (ref_elapsed_ms / elapsed_ms)
    where ref_elapsed_ms = median elapsed_ms across all results with elapsed > 0.

    Interpretation:
      tv_time > TV  → faster than median (speed bonus)
      tv_time < TV  → slower than median (speed penalty)
      tv_time = TV  → exactly median speed

    Configs with high TER×CCC that are also fast score highest.
    Clamped to [0.1× TV, 3.0× TV] to prevent outliers from dominating.
    """
    import statistics
    elapsed_vals = [r.get("elapsed_ms", 0) for r in results if r.get("elapsed_ms", 0) > 0]
    if not elapsed_vals:
        return
    ref_ms = statistics.median(elapsed_vals)
    for r in results:
        m        = r.get("metrics", {})
        tv       = m.get("true_value") or 0.0
        elapsed  = r.get("elapsed_ms", 0) or ref_ms
        ratio    = ref_ms / elapsed
        # Clamp: don't let extreme speed ratios overwhelm the quality signal
        ratio    = max(0.1, min(3.0, ratio))
        m["tv_time"] = round(tv * ratio, 4)


def print_leaderboard(
    results: List[dict],
    top: int = 20,
    show_diff: bool = False,
    sort_by: str = "tv",          # "tv" | "tv_time" | "ter" | "ccc"
) -> None:
    if not results:
        print("  No results found.")
        return

    # Inject tv_time into all results
    _compute_tv_time(results)

    # Check if any results have LLM metrics
    has_llm     = any(r.get("metrics", {}).get("rqs_llm") for r in results)
    has_latency = any(r.get("metrics", {}).get("llm_latency_ms") for r in results)
    has_time    = any(r.get("elapsed_ms", 0) > 0 for r in results)

    def _sort_key(r):
        m      = r.get("metrics", {})
        if sort_by == "tv_time":
            return (m.get("tv_time") or 0.0, m.get("true_value") or 0.0)
        if sort_by == "ter":
            return (m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0, m.get("true_value") or 0.0)
        if sort_by == "ccc":
            return (m.get("rqs_l1") or m.get("c_rqs_l1") or 0.0, m.get("true_value") or 0.0)
        # Default: tv (or tv_llm if present)
        tv_llm = m.get("true_value_llm")
        if tv_llm is not None:
            return (tv_llm, m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0)
        tv  = m.get("true_value") or 0.0
        ter = m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0
        rqs = m.get("rqs_l1") or m.get("c_rqs_l1") or 0.0
        return (tv, ter * rqs if not tv else tv)

    ranked = sorted(results, key=_sort_key, reverse=True)[:top]

    sort_label = {"tv": "TV", "tv_time": "TV×TIME", "ter": "TER", "ccc": "CCC"}.get(sort_by, sort_by.upper())
    hdr = f"  {'#':>3}  {'ID':>8}  {'Label':<45}  {'TER':>6}  {'CCC':>6}  {'TV':>6}  {'TV×T':>6}  {'ms':>6}"
    if has_llm:
        hdr += f"  {'RQS-LLM':>8}  {'TV-LLM':>7}"
        if has_latency:
            hdr += f"  {'LLMms':>7}"
    if show_diff:
        hdr += f"  {'ΔTER':>6}  {'ΔTV':>6}  {'ΔTV×T':>6}"
    hdr += f"  {'Date':<10}"
    print(f"  Sort: {sort_label}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for rank, r in enumerate(ranked, 1):
        m       = r.get("metrics", {})
        ter     = (m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0)
        rqs     = m.get("rqs_l1") or m.get("c_rqs_l1") or 0.0
        tv      = m.get("true_value") or (ter * rqs if rqs else ter)
        # Infer CCC from TV/TER when the CCC metric was not captured by the parser
        if not rqs and tv and ter:
            rqs = round(tv / ter, 3) if ter > 0 else 0.0
        tv_time = m.get("tv_time") or 0.0
        elapsed = r.get("elapsed_ms", 0)
        ts      = r.get("timestamp", "")[:10]
        eid     = r.get("experiment_id", "?")[:8]
        lbl     = (r.get("label", "") or "")[:45]
        elapsed_str = f"{elapsed}" if elapsed else "—"

        row = (
            f"  {rank:>3}  {eid:>8}  {lbl:<45}  "
            f"{ter*100:>5.1f}%  {rqs:>6.3f}  {tv:>6.3f}  {tv_time:>6.3f}  {elapsed_str:>6}"
        )
        if has_llm:
            rqs_llm = m.get("rqs_llm") or 0.0
            tv_llm  = m.get("true_value_llm") or 0.0
            row += f"  {rqs_llm:>7.3f}  {tv_llm:>6.3f}"
            if has_latency:
                lat = m.get("llm_latency_ms")
                lat_str = f"{lat}" if lat is not None else "—"
                row += f"  {lat_str:>7}"
        if show_diff:
            b_ter   = BASELINE.get("python_ter_pct") or BASELINE.get("c_ter_pct") or 0.0
            b_tv    = BASELINE.get("true_value", 0.0)
            b_tvt   = BASELINE.get("tv_time", 0.0)
            dter    = (ter - b_ter) * 100
            dtv     = tv - b_tv
            dtvt    = tv_time - b_tvt
            sign_ter = "+" if dter >= 0 else ""
            sign_tv  = "+" if dtv  >= 0 else ""
            sign_tvt = "+" if dtvt >= 0 else ""
            row += f"  {sign_ter}{dter:>4.1f}%  {sign_tv}{dtv:>5.3f}  {sign_tvt}{dtvt:>5.3f}"
        row += f"  {ts:<10}"
        print(row)

    print(f"\n  Total experiments: {len(results)}  |  Shown: {len(ranked)}  |  Sort: {sort_label}")
    print(f"  TV×T = TV × (median_elapsed / elapsed)  — rewards fast configs proportionally")


def best_config(results: List[dict]) -> Optional[dict]:
    if not results:
        return None
    return max(results,
               key=lambda r: r.get("metrics", {}).get("true_value") or 0.0)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Experiment leaderboard")
    parser.add_argument("--date",  default=None, help="Filter by date YYYY-MM-DD")
    parser.add_argument("--top",   type=int, default=20, help="Show top N")
    parser.add_argument("--param", default=None, help="Filter by param name substring")
    parser.add_argument("--diff",  action="store_true", help="Show delta from baseline")
    parser.add_argument("--type",  default="all", choices=["all", "sim", "llm"],
                        help="Show all/sim-only/llm-only results")
    parser.add_argument("--sort",  default="tv",
                        choices=["tv", "tv_time", "ter", "ccc"],
                        help="Sort by: tv (default), tv_time (TER×CCC×speed), ter, ccc")
    args = parser.parse_args()

    results = load_results(date=args.date, param_filter=args.param,
                           type_filter=args.type)
    print(f"\n  EXPERIMENT LEADERBOARD  ({len(results)} total)\n")
    print_leaderboard(results, top=args.top, show_diff=args.diff, sort_by=args.sort)

    best = best_config(results)
    if best:
        print(f"\n  Best config: {best['label']}")
        print(f"  Env: {json.dumps(best['env'], indent=4)}")
