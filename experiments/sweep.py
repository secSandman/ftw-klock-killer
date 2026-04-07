"""
sweep.py — Parallel grid search over parameter space
======================================================
Generates all permutations of a param grid and runs them in parallel
via ThreadPoolExecutor (each experiment is a subprocess, so GIL doesn't matter).

Usage:
    # Sweep a single param
    python experiments/sweep.py \\
        --param KLOC_PI_THRESHOLD 0.60 0.65 0.70 0.75 0.80

    # Sweep multiple params (full grid: 3×3 = 9 experiments)
    python experiments/sweep.py \\
        --param KLOC_PI_THRESHOLD 0.65 0.70 0.75 \\
        --param KLOC_PI_BOILERPLATE_BONUS 0.10 0.15 0.20

    # Sweep against a C file instead of synthetic Python
    python experiments/sweep.py \\
        --corpus doom/src/strife/p_enemy.c \\
        --repo doom/ \\
        --param KLOC_CPI_THRESHOLD 0.60 0.65 0.70 0.75

    # Load param grid from a JSON file
    python experiments/sweep.py --grid experiments/grids/pi_quick.json

    # Control parallelism
    python experiments/sweep.py --workers 4 --param ...

Results saved to experiments/results/YYYY-MM-DD/results.jsonl.
View with: python experiments/leaderboard.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.runner import run_experiment


def sweep(
    param_grid: Dict[str, List[str]],
    corpus:  str = "synthetic",
    c_repo:  str = "",
    workers: int = 4,
    label_prefix: str = "",
) -> List[dict]:
    """
    Run all combinations of param_grid in parallel.

    param_grid: {env_key: [val1, val2, ...], ...}
    Returns list of result dicts, sorted by primary metric descending.
    """
    # Generate all combinations
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    total = len(combos)
    print(f"\n  Sweep: {total} experiments  ({workers} parallel workers)")
    print(f"  Params: {', '.join(keys)}")
    print(f"  Corpus: {corpus}")
    print()

    jobs = []
    for combo in combos:
        env_overrides = {k: str(v) for k, v in zip(keys, combo)}
        jobs.append(env_overrides)

    results   = []
    completed = 0
    t0        = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_experiment,
                        env_overrides = env,
                        label         = label_prefix + _short_label(env),
                        corpus        = corpus,
                        c_repo        = c_repo,
                        save          = True): env
            for env in jobs
        }

        for fut in as_completed(futures):
            completed += 1
            env = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                m   = r["metrics"]
                ter = m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0
                rqs = m.get("rqs_l1") or m.get("c_rqs_l1") or 0.0
                tv  = m.get("true_value") or (ter * rqs if rqs else ter)
                # Fallback: infer CCC from TV/TER when parser missed the CCC line
                if not rqs and tv and ter:
                    rqs = round(tv / ter, 3) if ter > 0 else 0.0
                elapsed = time.time() - t0
                print(
                    f"  [{completed:>3}/{total}] {r['label']:<50} "
                    f"TER={ter*100:.1f}%  RQS={rqs:.3f}  TV={tv:.3f}  ({elapsed:.1f}s)"
                )
            except Exception as exc:
                print(f"  [{completed:>3}/{total}] ERROR: {exc}")

    # Sort by true_value descending, then by TER
    def _sort_key(r):
        m  = r["metrics"]
        tv = m.get("true_value", 0.0) or 0.0
        ter = m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0
        return (tv, ter)

    results.sort(key=_sort_key, reverse=True)

    total_time = time.time() - t0
    print(f"\n  Done in {total_time:.1f}s  ({total_time/total:.1f}s/experiment)")
    print(f"  Results saved to experiments/results/")
    _print_top(results, n=5)

    return results


def _short_label(env: dict) -> str:
    parts = [f"{k.replace('KLOC_','').replace('_','.')}={v}"
             for k, v in sorted(env.items())]
    return " | ".join(parts[:3])


def _print_top(results: list, n: int = 5) -> None:
    print(f"\n  Top {min(n, len(results))} by True Value:")
    print(f"  {'Label':<50} {'TER':>6} {'RQS':>6} {'TV':>6}")
    print(f"  {'-'*50} {'------':>6} {'------':>6} {'------':>6}")
    for r in results[:n]:
        m   = r["metrics"]
        ter = (m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0) * 100
        rqs = m.get("rqs_l1") or m.get("c_rqs_l1") or 0.0
        tv  = m.get("true_value") or (ter/100 * rqs if rqs else ter/100)
        print(f"  {r['label'][:50]:<50} {ter:>5.1f}%  {rqs:>5.3f}  {tv:>5.3f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Parallel param sweep")
    parser.add_argument("--param",   nargs="+", action="append", metavar=("KEY", "VAL"),
                        help="--param ENV_KEY val1 val2 val3 (repeatable)")
    parser.add_argument("--grid",    default=None,
                        help="JSON file with param grid: {ENV_KEY: [val1, val2, ...]}")
    parser.add_argument("--corpus",  default="synthetic",
                        help="'synthetic' or path to a .c file")
    parser.add_argument("--repo",    default="",
                        help="Repo root (for C file corpus)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--label",   default="", help="Label prefix for this sweep")
    args = parser.parse_args()

    param_grid: Dict[str, List[str]] = {}

    # Load from --grid file
    if args.grid:
        g = json.loads(Path(args.grid).read_text())
        for k, vals in g.items():
            if not k.startswith("_"):         # skip _doc, _corpus, _repo metadata keys
                param_grid[k] = [str(v) for v in vals]

    # Merge --param flags
    if args.param:
        for parts in args.param:
            if len(parts) < 2:
                print(f"  ⚠  --param needs KEY val1 [val2 ...]: got {parts}")
                continue
            key  = parts[0]
            vals = parts[1:]
            param_grid[key] = vals

    if not param_grid:
        print("  No params specified. Use --param KEY val1 val2 or --grid FILE.")
        sys.exit(1)

    sweep(
        param_grid   = param_grid,
        corpus       = args.corpus,
        c_repo       = args.repo,
        workers      = args.workers,
        label_prefix = args.label,
    )
