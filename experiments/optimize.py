"""
optimize.py — Iterative RL-style optimizer using True Value as reward signal
=============================================================================
Uses a bandit/coordinate-descent loop:

  1. Start from current param values
  2. For each param (one at a time), sweep its range
  3. Pick the value that maximizes True Value
  4. Fix that value, move to next param
  5. Repeat until convergence (TV stops improving by > MIN_IMPROVEMENT)

Multi-agent variant: spawns parallel subagents for each param dimension.
Learns across runs: reads all past results from JSONL to skip known-bad configs.

This is coordinate descent, not full RL — but it converges fast (O(params × values)
vs O(values^params) for full grid) and respects the TV = TER × RQS primary metric.

Usage:
    python experiments/optimize.py                    # optimize all params
    python experiments/optimize.py --param pi         # optimize PI params only
    python experiments/optimize.py --rounds 3         # 3 full passes
    python experiments/optimize.py --corpus doom/src/strife/p_enemy.c --repo doom/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.runner import run_experiment
from experiments.leaderboard import load_results, BASELINE

# ── Param spaces ──────────────────────────────────────────────────────────────
# Each entry: env_key → list of candidate values to sweep
# Coarse grid for speed; refine with --fine flag

PARAM_SPACES: Dict[str, Dict[str, list]] = {
    "pi": {
        "KLOC_PI_THRESHOLD":              [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
        "KLOC_PI_BOILERPLATE_BONUS":      [0.05, 0.10, 0.15, 0.20, 0.25],
        "KLOC_PI_LOOP_FREE_BONUS":        [0.05, 0.10, 0.15, 0.20],
        "KLOC_PI_LOOP_PENALTY":           [0.04, 0.06, 0.08, 0.10],
        "KLOC_PI_SIMPLE_RATIO_THRESHOLD": [0.40, 0.50, 0.60, 0.70],
    },
    "c_pi": {
        "KLOC_CPI_THRESHOLD":          [0.60, 0.65, 0.70, 0.75, 0.80],
        "KLOC_CPI_BOILERPLATE_BONUS":  [0.10, 0.15, 0.20, 0.25],
        "KLOC_CPI_LOOP_FREE_BONUS":    [0.05, 0.10, 0.15],
        "KLOC_CPM_KQ_THRESHOLD":       [2, 3, 4, 5],
    },
    "retrieval": {
        "KLOC_RAG_NAME_WEIGHT":    [3.0, 5.0, 8.0, 12.0],
        "KLOC_RAG_SYMBOL_BOOST":   [2.0, 4.0, 6.0, 8.0],
        "KLOC_RAG_LARGE_FN_LINES": [50, 100, 150, 200],
        "KLOC_RAG_CHAIN_EXPAND_K": [0, 1, 2, 3],
    },
}

MIN_IMPROVEMENT = 0.002    # stop if TV gain < 0.2%
WORKERS         = 4        # parallel workers per param sweep


def optimize(
    param_group: str  = "pi",
    corpus:       str = "synthetic",
    c_repo:       str = "",
    rounds:       int = 2,
    workers:      int = WORKERS,
    fine:         bool = False,
) -> Tuple[Dict[str, str], float]:
    """
    Coordinate-descent optimizer.

    Returns (best_config_env_dict, best_true_value).
    """
    spaces = PARAM_SPACES.get(param_group, {})
    if not spaces:
        print(f"  Unknown param group: {param_group}. Options: {list(PARAM_SPACES)}")
        return {}, 0.0

    # Seed with existing best from past results
    past = load_results()
    best_env, best_tv = _best_from_history(past, spaces) or ({}, 0.0)
    if not best_env:
        best_env = {k: str(v[len(v)//2]) for k, v in spaces.items()}  # start at midpoint
        best_tv  = 0.0

    print(f"\n  Optimizer: {param_group} | corpus={corpus} | rounds={rounds}")
    print(f"  Starting config: {best_env}")
    print(f"  Starting TV:     {best_tv:.4f}\n")

    for rnd in range(rounds):
        improved = False
        print(f"  ── Round {rnd+1}/{rounds} ──────────────────────────────────")

        for param, candidates in spaces.items():
            if fine:
                candidates = _refine(candidates, best_env.get(param))

            # Run each candidate with the current best values for all other params
            print(f"  Sweeping {param} over {candidates}...")
            results = _sweep_param(param, candidates, best_env.copy(), corpus, c_repo, workers)

            best_result = max(results, key=lambda r: _tv(r))
            new_tv = _tv(best_result)

            if new_tv > best_tv + MIN_IMPROVEMENT:
                old_val  = best_env.get(param, "?")
                best_env = {**best_env, param: best_result["env"][param]}
                best_tv  = new_tv
                improved = True
                print(
                    f"    {param}: {old_val} → {best_env[param]}  "
                    f"TV: {new_tv:.4f}  (+{(new_tv - best_tv + new_tv - new_tv):.4f})"
                )
            else:
                print(f"    {param}: no improvement (best TV={new_tv:.4f})")

        if not improved:
            print(f"\n  Converged after round {rnd+1}.")
            break

    print(f"\n  Final best config (TV={best_tv:.4f}):")
    for k, v in sorted(best_env.items()):
        baseline_default = _param_default(k)
        marker = "  ←changed" if str(v) != str(baseline_default) else ""
        print(f"    {k:<45} = {v}{marker}")

    _print_delta(best_tv)
    return best_env, best_tv


def _sweep_param(
    param:    str,
    values:   list,
    base_env: Dict[str, str],
    corpus:   str,
    c_repo:   str,
    workers:  int,
) -> List[dict]:
    """Sweep one param while holding all others fixed."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    jobs = []
    for val in values:
        env = {**base_env, param: str(val)}
        jobs.append(env)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_experiment, env, label=f"{param}={val}",
                        corpus=corpus, c_repo=c_repo): env
            for env, val in zip(jobs, values)
        }
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                print(f"    ERROR: {exc}")
    return results


def _tv(result: dict) -> float:
    m = result.get("metrics", {})
    tv = m.get("true_value")
    if tv is not None:
        return tv
    ter = m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0
    rqs = m.get("rqs_l1") or m.get("c_rqs_l1") or 0.0
    return ter * rqs if rqs else ter


def _best_from_history(
    past: list,
    spaces: dict,
) -> Optional[Tuple[dict, float]]:
    """Find best past result that uses params from our space."""
    relevant = [
        r for r in past
        if any(k in r.get("env", {}) for k in spaces.keys())
    ]
    if not relevant:
        return None
    best = max(relevant, key=lambda r: _tv(r))
    return best.get("env", {}), _tv(best)


def _refine(candidates: list, current_val: Optional[str]) -> list:
    """Narrow search around current best value."""
    if current_val is None or not candidates:
        return candidates
    try:
        cur  = float(current_val)
        step = (max(float(c) for c in candidates) - min(float(c) for c in candidates)) / (len(candidates) * 2)
        refined = [round(cur + step * i, 4) for i in range(-2, 3)]
        return refined
    except (ValueError, TypeError):
        return candidates


def _param_default(key: str) -> str:
    defaults = {
        "KLOC_PI_THRESHOLD": "0.70",
        "KLOC_PI_BOILERPLATE_BONUS": "0.15",
        "KLOC_PI_LOOP_FREE_BONUS": "0.10",
        "KLOC_PI_LOOP_PENALTY": "0.06",
        "KLOC_PI_SIMPLE_RATIO_THRESHOLD": "0.50",
        "KLOC_CPI_THRESHOLD": "0.70",
        "KLOC_CPI_BOILERPLATE_BONUS": "0.15",
        "KLOC_CPI_LOOP_FREE_BONUS": "0.10",
        "KLOC_CPM_KQ_THRESHOLD": "3",
        "KLOC_RAG_NAME_WEIGHT": "5.0",
        "KLOC_RAG_SYMBOL_BOOST": "4.0",
        "KLOC_RAG_LARGE_FN_LINES": "100",
        "KLOC_RAG_CHAIN_EXPAND_K": "2",
    }
    return defaults.get(key, "?")


def _print_delta(best_tv: float) -> None:
    base_tv = BASELINE.get("true_value", 0.0)
    if base_tv:
        delta = best_tv - base_tv
        sign  = "+" if delta >= 0 else ""
        print(f"\n  Baseline TV: {base_tv:.4f}")
        print(f"  Best TV:     {best_tv:.4f}  ({sign}{delta:.4f}  {sign}{delta/base_tv*100:.1f}%)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Iterative TV-maximizing optimizer")
    parser.add_argument("--param",   default="pi",
                        choices=list(PARAM_SPACES), help="Param group to optimize")
    parser.add_argument("--corpus",  default="synthetic")
    parser.add_argument("--repo",    default="")
    parser.add_argument("--rounds",  type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fine",    action="store_true", help="Refine around best value")
    args = parser.parse_args()

    best_env, best_tv = optimize(
        param_group = args.param,
        corpus      = args.corpus,
        c_repo      = args.repo,
        rounds      = args.rounds,
        workers     = args.workers,
        fine        = args.fine,
    )
