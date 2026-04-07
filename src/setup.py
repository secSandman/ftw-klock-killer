#!/usr/bin/env python3
"""
setup.py — Inference-Bridge (InB) Setup & Test Harness
=======================================================
One-command setup, dependency check, test repo cloning, and config validation.

Usage:
  python setup.py                        # check deps + run synthetic test
  python setup.py --clone flask          # clone Flask and run pipeline on it
  python setup.py --clone flask --bench  # clone + full benchmark + graph
  python setup.py --clone all            # clone all repos in inb_config.json
  python setup.py --config              # print current config and exit
  python setup.py --reset               # delete registry + test repos, start fresh
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

CONFIG_PATH = Path("inb_config.json")
REGISTRY_PATH = Path("inb_registry.json")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str):
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")


def ok(text: str):  print(f"  \033[32m✓\033[0m  {text}")
def warn(text: str): print(f"  \033[33m!\033[0m  {text}")
def err(text: str):  print(f"  \033[31m✗\033[0m  {text}")
def info(text: str): print(f"     {text}")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        err(f"{CONFIG_PATH} not found. Run from the InB project directory.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Dependency check
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED = {
    "git":        ("git", "gitpython"),
    "matplotlib": ("matplotlib", "matplotlib"),
    "numpy":      ("numpy", "numpy"),
}

OPTIONAL = {
    "tiktoken": ("tiktoken", "tiktoken"),  # Not required — InB has its own counter
}


def check_deps() -> bool:
    banner("Dependency Check")
    all_ok = True

    # Python version
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 9):
        ok(f"Python {major}.{minor}")
    else:
        err(f"Python {major}.{minor} — need 3.9+")
        all_ok = False

    # Required packages
    for name, (import_name, pip_name) in REQUIRED.items():
        try:
            importlib.import_module(import_name)
            ok(f"{name}")
        except ImportError:
            err(f"{name} not found — run: pip install {pip_name}")
            all_ok = False

    # Optional packages
    for name, (import_name, pip_name) in OPTIONAL.items():
        try:
            importlib.import_module(import_name)
            ok(f"{name} (optional — present)")
        except ImportError:
            warn(f"{name} (optional — not installed, InB uses its own token counter)")

    # Git binary
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True)
        ok(f"git binary: {result.stdout.strip()}")
    except FileNotFoundError:
        err("git binary not found — needed for Retinal Delta (F2)")
        warn("Retinal Delta will be disabled automatically")

    # Core InB files
    for fname in ["inference_bridge.py", "benchmark.py", "inb_config.json"]:
        if Path(fname).exists():
            ok(f"{fname} present")
        else:
            err(f"{fname} missing — download all InB files to the same directory")
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Config display / validation
# ─────────────────────────────────────────────────────────────────────────────

def show_config():
    banner("Current Configuration")
    cfg = load_config()
    info(f"PI threshold:          {cfg.get('pi_threshold')}  (bodies above this → skeleton)")
    info(f"Entropy threshold:     {cfg.get('entropy_threshold')} bit/byte  (imports below this → mask)")
    info(f"Min import frequency:  {cfg.get('known_quantity_min_freq')}  (appears in N+ files → Known Quantity)")
    info(f"HZS cold cutoff:       {cfg.get('hzs_cold_cutoff')}  (below this → Cold zone)")
    info(f"Commits to diff:       {cfg.get('n_commits')}")
    info(f"Vowel prune min len:   {cfg.get('vowel_prune_min_len')} chars")
    info("")
    info("Features enabled:")
    for feat, enabled in cfg.get("features", {}).items():
        status = "\033[32mon\033[0m" if enabled else "\033[33moff\033[0m"
        info(f"  {feat:<12} {status}")
    info("")
    info("Available test repos:")
    for name, url in cfg.get("test_repos", {}).items():
        clone_path = Path(cfg.get("clone_dir", "./test_repos")) / name
        status = "cloned ✓" if clone_path.exists() else "not cloned"
        info(f"  {name:<10} {status:<12} {url}")
    info("")
    info(f"Registry:     {cfg.get('registry_path')}")
    info(f"Clone dir:    {cfg.get('clone_dir')}")
    info(f"Clone depth:  {cfg.get('clone_depth')} commits")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Clone a test repo
# ─────────────────────────────────────────────────────────────────────────────

def clone_repo(name: str, cfg: dict) -> Path | None:
    repos = cfg.get("test_repos", {})
    clone_dir = Path(cfg.get("clone_dir", "./test_repos"))
    depth = cfg.get("clone_depth", 5)

    if name == "all":
        cloned = []
        for repo_name in repos:
            result = clone_repo(repo_name, cfg)
            if result:
                cloned.append(result)
        return cloned[0] if cloned else None

    if name not in repos:
        err(f"Unknown repo '{name}'. Available: {', '.join(repos.keys())}")
        err("Add custom repos to inb_config.json under 'test_repos'.")
        return None

    url = repos[name]
    target = clone_dir / name

    if target.exists():
        ok(f"{name} already cloned at {target}")
        return target

    banner(f"Cloning {name}")
    info(f"Source:  {url}")
    info(f"Target:  {target}")
    info(f"Depth:   {depth} commits (enough for Retinal Delta F2)")
    info("")

    clone_dir.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            ["git", "clone", f"--depth={depth}", url, str(target)],
            capture_output=False,  # show git output live
            timeout=300,
        )
        if proc.returncode == 0:
            ok(f"Cloned successfully → {target}")
            return target
        else:
            err(f"git clone failed (exit {proc.returncode})")
            return None
    except subprocess.TimeoutExpired:
        err("Clone timed out after 5 minutes")
        return None
    except FileNotFoundError:
        err("git binary not found — install git and try again")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Run pipeline on a cloned repo
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(repo_path: Path, cfg: dict):
    banner(f"Running InB Pipeline on {repo_path.name}")

    # Patch config into inference_bridge at runtime
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from inference_bridge import InferenceBridge, SKELETAL_PI_THRESHOLD
    except ImportError as e:
        err(f"Cannot import inference_bridge: {e}")
        return

    feats = cfg.get("features", {})
    reg   = cfg.get("registry_path", "inb_registry.json")

    inb = InferenceBridge(
        repo_path=str(repo_path),
        registry_path=reg,
        n_commits=cfg.get("n_commits", 1),
        enable_masking=feats.get("masking", True),
        enable_delta=feats.get("delta", True),
        enable_skeleton=feats.get("skeleton", True),
        enable_caveman=feats.get("caveman", True),
    )

    info(f"Processing all .py files (this may take 10–30 seconds for large repos)...")
    results = inb.process_repo(
        target_extensions=tuple(cfg.get("target_extensions", [".py"]))
    )

    info("")
    info(f"Files processed:    {results['files_processed']}")
    info(f"Original tokens:    {results['total_original_tokens']:,}")
    info(f"Final tokens:       {results['total_final_tokens']:,}")
    info(f"Reduction:          {results['overall_reduction_pct']:.1f}%")

    if results["target_met"]:
        ok(f">80% target MET — {results['overall_reduction_pct']:.1f}% reduction")
    else:
        warn(f">80% target not met — achieved {results['overall_reduction_pct']:.1f}%")
        warn("Try lowering pi_threshold in inb_config.json (e.g. 0.60)")

    # Show top 10 files by absolute token saving
    info("")
    info("Top files by token saving:")
    files_sorted = sorted(
        results["files"],
        key=lambda f: f["original"] - f["final"],
        reverse=True
    )
    for f in files_sorted[:10]:
        saving = f["original"] - f["final"]
        info(f"  {f['reduction_pct']:>5.1f}%  {saving:>5} tokens saved  {f['file']}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Quick synthetic self-test
# ─────────────────────────────────────────────────────────────────────────────

def run_synthetic_test():
    banner("Synthetic Self-Test (no network required)")
    result = subprocess.run(
        [sys.executable, "benchmark.py", "--mode", "synthetic"],
        capture_output=False,
    )
    if result.returncode == 0:
        ok("Synthetic test passed")
    else:
        err("Synthetic test failed — check benchmark.py and inference_bridge.py")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Reset
# ─────────────────────────────────────────────────────────────────────────────

def reset(cfg: dict):
    banner("Reset")
    reg = Path(cfg.get("registry_path", "inb_registry.json"))
    clone_dir = Path(cfg.get("clone_dir", "./test_repos"))
    bench_dir = Path(cfg.get("benchmark", {}).get("output_dir", "./benchmark_results"))

    targets = []
    if reg.exists():
        targets.append(reg)
    if clone_dir.exists():
        targets.append(clone_dir)
    if bench_dir.exists():
        targets.append(bench_dir)

    if not targets:
        info("Nothing to reset — already clean.")
        return

    info("The following will be deleted:")
    for t in targets:
        info(f"  {t}")
    confirm = input("\n  Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        info("Aborted.")
        return

    for t in targets:
        if t.is_file():
            t.unlink()
            ok(f"Deleted {t}")
        elif t.is_dir():
            shutil.rmtree(t)
            ok(f"Deleted {t}/")

    info("Reset complete. Run setup.py again to start fresh.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="InB Setup & Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                        check deps + run synthetic test
  python setup.py --clone flask          clone Flask repo
  python setup.py --clone flask --bench  clone + benchmark + graph
  python setup.py --clone all            clone all configured repos
  python setup.py --config               show current config
  python setup.py --reset                clean slate
        """
    )
    parser.add_argument("--clone",  metavar="REPO",  help="Clone a test repo by name (or 'all')")
    parser.add_argument("--bench",  action="store_true", help="Run full benchmark after cloning")
    parser.add_argument("--config", action="store_true", help="Print current config and exit")
    parser.add_argument("--reset",  action="store_true", help="Delete registry + cloned repos")
    parser.add_argument("--skip-test", action="store_true", help="Skip the synthetic self-test")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║   Inference-Bridge (InB) Setup       ║")
    print("  ║   Semantic Token Compressor v1.0     ║")
    print("  ╚══════════════════════════════════════╝")

    cfg = load_config()

    if args.config:
        show_config()
        return

    if args.reset:
        reset(cfg)
        return

    # Always check deps first
    deps_ok = check_deps()
    if not deps_ok:
        err("Fix the above issues then re-run setup.py")
        sys.exit(1)

    # Synthetic self-test (unless skipped)
    if not args.skip_test:
        run_synthetic_test()

    # Clone if requested
    repo_path = None
    if args.clone:
        repo_path = clone_repo(args.clone, cfg)

    # Run pipeline on cloned repo
    if repo_path and Path(repo_path).exists():
        run_pipeline(Path(repo_path), cfg)

    # Full benchmark
    if args.bench:
        if not repo_path:
            warn("--bench specified but no repo was cloned. Use --clone NAME --bench")
        else:
            banner("Running Full Benchmark + Graph")
            subprocess.run([
                sys.executable, "benchmark.py", "--mode", "full"
            ])

    # Final summary
    banner("Setup Complete")
    ok("InB is ready to use")
    info("")
    info("Quick reference:")
    info("  Compress a file:    python inference_bridge.py mask path/to/file.py")
    info("  Repo stats:         python inference_bridge.py stats ./my-repo/")
    info("  Unmask output:      python inference_bridge.py unmask masked_output.txt")
    info("  Tune settings:      edit inb_config.json")
    info("  Clone test repos:   python setup.py --clone flask")
    info("  Full benchmark:     python setup.py --clone flask --bench")
    info("")


if __name__ == "__main__":
    main()
