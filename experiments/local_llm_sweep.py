"""
local_llm_sweep.py — Sweep local LLM T0.5 tier parameters
==========================================================
Measures how local LLM configuration affects:
  - Acceptance rate (% of queries served by local vs escalated to Anthropic)
  - Response quality (rqs_llm via compute_rqs_llm)
  - Latency per query

This sweep runs AGAINST A LIVE OLLAMA INSTANCE. Start Ollama before running:
  docker start kloc-ollama   (or python kloc.py experiment llm-setup)

Usage:
  python experiments/local_llm_sweep.py --grid local_llm_quick
  python experiments/local_llm_sweep.py --grid local_llm_full --workers 2
  python experiments/local_llm_sweep.py --corpus c --file doom/src/strife/p_enemy.c
  python kloc.py experiment llm-sweep --grid local_llm_quick
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GRIDS_DIR  = Path(__file__).parent / "grids"
PARAMS_DIR = Path(__file__).parent / "params"


def _check_ollama() -> bool:
    """Return True if Ollama is reachable."""
    try:
        import httpx
        port = int(os.getenv("KLOC_LLM_PORT", "11434"))
        r = httpx.get(f"http://localhost:{port}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _load_grid(grid_name: str) -> Dict[str, List]:
    """Load a grid JSON file, stripping _doc/_corpus/_real_llm metadata keys."""
    # Accept with or without .json extension
    name = grid_name if grid_name.endswith(".json") else grid_name + ".json"
    # Look in grids/ first, then treat as absolute path
    candidates = [GRIDS_DIR / name, Path(name)]
    for p in candidates:
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            return {k: v for k, v in raw.items() if not k.startswith("_")}
    raise FileNotFoundError(f"Grid not found: {grid_name}  (looked in {GRIDS_DIR})")


def run_llm_sweep(
    param_grid: Dict[str, List],
    corpus:     str = "synthetic",
    c_file:     str = "",
    c_repo:     str = "",
    workers:    int = 2,
    label_prefix: str = "llm",
) -> List[dict]:
    """
    Run all combinations of param_grid with real_llm=True.
    Each combination is a subprocess with env overrides + LocalLLMHarness.
    Returns list of result dicts sorted by rqs_llm descending.
    """
    from experiments.local_llm import LocalLLMHarness
    from experiments.runner import run_experiment

    if not _check_ollama():
        print("  [llm-sweep] Ollama not running. Start with: python kloc.py experiment llm-setup")
        return []

    harness = LocalLLMHarness()
    if not harness.health_check():
        print("  [llm-sweep] Ollama health check failed.")
        return []

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    print(f"\n  LLM Sweep: {total} experiments  ({workers} parallel workers)")
    print(f"  Params:  {', '.join(keys)}")
    print(f"  Corpus:  {c_file if c_file else corpus}")
    print(f"  Model:   {os.getenv('KLOC_LLM_MODEL', 'qwen2.5-coder:14b')}")
    print()

    jobs = [{k: str(v) for k, v in zip(keys, combo)} for combo in combos]

    results   = []
    completed = 0
    t0        = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_experiment,
                env_overrides = env,
                label         = label_prefix + " | " + _short_label(env),
                corpus        = c_file if c_file else corpus,
                c_repo        = c_repo,
                real_llm      = True,
                llm_harness   = harness,
                save          = True,
            ): env
            for env in jobs
        }

        for fut in as_completed(futures):
            completed += 1
            env = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                m        = r.get("metrics", {})
                rqs_llm  = m.get("rqs_llm",  "—")
                rqs_l1   = m.get("rqs_l1",   "—")
                ter      = (m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0) * 100
                tv_llm   = m.get("true_value_llm", "—")
                elapsed  = time.time() - t0
                rqs_str  = f"{rqs_llm:.3f}" if isinstance(rqs_llm, float) else rqs_llm
                tv_str   = f"{tv_llm:.3f}"  if isinstance(tv_llm,  float) else tv_llm
                lat      = m.get("llm_latency_ms")
                lat_str  = f"{lat}ms" if lat is not None else "—"
                print(
                    f"  [{completed:>3}/{total}] {r['label'][:45]:<45} "
                    f"TER={ter:>5.1f}%  RQS-L1={rqs_l1 if isinstance(rqs_l1,str) else f'{rqs_l1:.3f}':>6}  "
                    f"RQS-LLM={rqs_str:>6}  TV-LLM={tv_str:>6}  lat={lat_str:>7}  ({elapsed:.1f}s)"
                )
            except Exception as exc:
                print(f"  [{completed:>3}/{total}] ERROR: {exc}")

    # Sort by rqs_llm descending, then by TER
    def _sort_key(r):
        m = r.get("metrics", {})
        return (
            m.get("rqs_llm") or 0.0,
            (m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0),
        )

    results.sort(key=_sort_key, reverse=True)

    total_time = time.time() - t0
    print(f"\n  Done in {total_time:.1f}s  ({total_time/max(total,1):.1f}s/experiment)")
    _print_top_llm(results, n=5)

    return results


def _print_top_llm(results: list, n: int = 5) -> None:
    """Print top-N results with LLM-specific columns."""
    has_latency = any(r.get("metrics", {}).get("llm_latency_ms") for r in results)
    print(f"\n  Top {min(n, len(results))} by RQS-LLM:")
    hdr = f"  {'Label':<50} {'TER':>6} {'RQS-L1':>7} {'RQS-LLM':>8} {'TV-LLM':>7}"
    sep = f"  {'-'*50} {'------':>6} {'-------':>7} {'--------':>8} {'-------':>7}"
    if has_latency:
        hdr += f" {'Latency':>9}"
        sep += f" {'-'*9:>9}"
    print(hdr)
    print(sep)
    for r in results[:n]:
        m       = r.get("metrics", {})
        ter     = (m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0) * 100
        rqs_l1  = m.get("rqs_l1",  0.0) or 0.0
        rqs_llm = m.get("rqs_llm", 0.0) or 0.0
        tv_llm  = m.get("true_value_llm", 0.0) or 0.0
        row = (
            f"  {r.get('label','')[:50]:<50} {ter:>5.1f}%  {rqs_l1:>6.3f}  {rqs_llm:>7.3f}  {tv_llm:>6.3f}"
        )
        if has_latency:
            lat = m.get("llm_latency_ms")
            lat_str = f"{lat}ms" if lat is not None else "—"
            row += f" {lat_str:>9}"
        print(row)


def _short_label(env: dict) -> str:
    parts = [
        f"{k.replace('KLOC_LOCAL_LLM_','').replace('KLOC_','').replace('_','.')}={v}"
        for k, v in sorted(env.items())
    ]
    return " | ".join(parts[:4])


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Sweep local LLM T0.5 parameters")
    parser.add_argument("--grid",    default="local_llm_quick",
                        help="Grid name (file in experiments/grids/) or path to JSON")
    parser.add_argument("--param",   nargs="+", action="append", metavar=("KEY", "VAL"),
                        help="Additional --param ENV_KEY val1 val2 (repeatable)")
    parser.add_argument("--corpus",  default="synthetic",
                        help="'synthetic' or path to .c file")
    parser.add_argument("--file",    default="",  help="C source file path")
    parser.add_argument("--repo",    default="",  help="Repo root for C benchmarks")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers (keep low — each job calls Ollama)")
    parser.add_argument("--label",   default="llm", help="Label prefix")
    args = parser.parse_args()

    param_grid: Dict[str, List] = {}
    try:
        param_grid = _load_grid(args.grid)
    except FileNotFoundError as e:
        print(f"  Warning: {e}")

    if args.param:
        for parts in args.param:
            if len(parts) >= 2:
                param_grid[parts[0]] = parts[1:]

    if not param_grid:
        print("  No params. Specify --grid or --param KEY val1 val2")
        sys.exit(1)

    run_llm_sweep(
        param_grid   = param_grid,
        corpus       = args.corpus,
        c_file       = args.file,
        c_repo       = args.repo,
        workers      = args.workers,
        label_prefix = args.label,
    )
