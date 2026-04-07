#!/usr/bin/env python3
"""
kloc.py — FTW-KLOC-KILLER Master CLI ☠️
========================================
One entry point to run everything — or just one thing.

COMMANDS
  setup      Build rainbow tables, verify deps, print environment summary
  doctor     Health-check: all deps, Docker, API keys, paths
  benchmark  TER + RQS combined report on any file, corpus, or synthetic suite
  pipeline   Run the full 4-agent pipeline on a file and print the routing plan
  ter        Token count only — fast, no quality scoring
  rqs        Quality check only — compare two source variants
  test       Run the full pytest suite
  rag        ETL operations: chunk, hash, embed, upsert to Qdrant

QUICK START (new user with a C game)
  python kloc.py setup
  python kloc.py benchmark --file path/to/game.c
  python kloc.py pipeline  --file path/to/game.c --question "explain the render loop"

OFFLINE MODE (zero API keys, zero Docker)
  ACTIVE_TIER_MAX=0 python kloc.py benchmark --file game.c
  python kloc.py benchmark --mode synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Fix Windows console UTF-8 (emoji + box-drawing characters)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

W = 66   # report width

def _bar(val: float, width: int = 30, fill: str = "█", empty: str = "░") -> str:
    n = round(val * width)
    return fill * n + empty * (width - n)

def _header(title: str) -> str:
    pad = (W - len(title) - 2) // 2
    return ("═" * pad) + f" {title} " + ("═" * (W - pad - len(title) - 2))

def _divider() -> str:
    return "─" * W

def _score_icon(val: float, good: float, warn: float) -> str:
    if val < 0:      return "⬜ N/A"
    if val >= good:  return f"✅ {val:.3f}"
    if val >= warn:  return f"⚠️  {val:.3f}"
    return              f"❌ {val:.3f}"

def _ter_icon(pct: float) -> str:
    if pct >= 80:  return f"✅ {pct:.1f}%"
    if pct >= 65:  return f"⚠️  {pct:.1f}%"
    return              f"❌ {pct:.1f}%"

def _print(msg: str = "") -> None:
    print(msg)

# ─────────────────────────────────────────────────────────────────────────────
# setup
# ─────────────────────────────────────────────────────────────────────────────

def cmd_setup(args) -> int:
    _print(_header("☠️  FTW-KLOC-KILLER  SETUP"))
    ok = True

    # ── Rainbow tables ───────────────────────────────────────────────────────
    _print("\n  [1/3] Building rainbow tables...")
    script = ROOT / "rainbow" / "build_rainbow.py"
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode == 0:
        _print("       ✅ rainbow_stdlib_python.jsonl + rainbow_stdlib_c.jsonl written")
    else:
        _print(f"       ❌ Rainbow build failed:\n{r.stderr}")
        ok = False

    # ── Dep check ────────────────────────────────────────────────────────────
    _print("\n  [2/3] Checking dependencies...")
    deps = {
        "ast":               "built-in",
        "hashlib":           "built-in",
        "pathlib":           "built-in",
        "numpy":             "pip install numpy",
        "sentence_transformers": "pip install sentence-transformers",
        "qdrant_client":     "pip install qdrant-client",
        "anthropic":         "pip install anthropic",
    }
    for mod, install in deps.items():
        try:
            __import__(mod)
            _print(f"       ✅ {mod}")
        except ImportError:
            tag = "⬜ optional" if mod not in ("ast", "hashlib", "pathlib") else "❌ required"
            _print(f"       {tag}  {mod}  ({install})")

    # ── Env vars ─────────────────────────────────────────────────────────────
    _print("\n  [3/3] Environment variables:")
    env_vars = {
        "ANTHROPIC_API_KEY": "T1-T3 tier routing",
        "OPENAI_API_KEY":    "OpenAI fallback routing",
        "ACTIVE_TIER_MAX":   f"tier ceiling (current: {os.environ.get('ACTIVE_TIER_MAX', '2')})",
        "QDRANT_URL":        f"vector DB (current: {os.environ.get('QDRANT_URL', 'http://localhost:6333')})",
    }
    for var, desc in env_vars.items():
        val = os.environ.get(var)
        icon = "✅" if val else "⬜"
        _print(f"       {icon} {var:<22} {desc}")

    _print()
    _print(_divider())
    _print("  Next steps:")
    _print("    python kloc.py benchmark --mode synthetic     # run Python TER baseline")
    _print("    python kloc.py benchmark --file path/to/game.c  # benchmark your C game")
    _print("    python kloc.py test                           # run full test suite")
    _print(_divider())
    return 0 if ok else 1


# ─────────────────────────────────────────────────────────────────────────────
# doctor
# ─────────────────────────────────────────────────────────────────────────────

def cmd_doctor(args) -> int:
    _print(_header("☠️  DOCTOR  health check"))
    all_ok = True

    checks = [
        ("src/inference_bridge.py", ROOT / "src" / "inference_bridge.py", True),
        ("src/benchmark.py",        ROOT / "src" / "benchmark.py",        True),
        ("src/quality.py",          ROOT / "src" / "quality.py",          True),
        ("data/taxonomy.json",      ROOT / "data" / "taxonomy.json",      True),
        ("agents/pruner.py",        ROOT / "agents" / "pruner.py",        True),
        ("agents/grug.py",          ROOT / "agents" / "grug.py",          True),
        ("agents/balancer.py",      ROOT / "agents" / "balancer.py",      True),
        ("agents/zippy.py",         ROOT / "agents" / "zippy.py",         True),
        ("rainbow stdlib python",   ROOT / "data" / "rainbow_stdlib_python.jsonl", False),
        ("rainbow stdlib c",        ROOT / "data" / "rainbow_stdlib_c.jsonl",      False),
        ("test corpus doom",        ROOT / "data" / "test_corpus" / "chocolate-doom", False),
        ("test corpus quake",       ROOT / "data" / "test_corpus" / "quakespasm",     False),
    ]

    _print()
    for label, path, required in checks:
        exists = path.exists()
        if required and not exists:
            all_ok = False
        icon = "✅" if exists else ("❌" if required else "⬜")
        note = "" if exists else (" (run: python kloc.py setup)" if "rainbow" in label else " (run: bash scripts/download_test_corpus.sh)" if "corpus" in label else " MISSING")
        _print(f"  {icon} {label}{note}")

    _print()
    # Docker check
    r = subprocess.run(["docker", "info"], capture_output=True, text=True)
    icon = "✅" if r.returncode == 0 else "⬜"
    _print(f"  {icon} Docker  (needed for Qdrant RAG; offline mode works without)")

    # Qdrant ping
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:6333/healthz", timeout=2) as resp:
            icon = "✅"
    except Exception:
        icon = "⬜"
    _print(f"  {icon} Qdrant at localhost:6333  (run: bash scripts/start_qdrant.sh)")

    # API keys
    _print()
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        icon = "✅" if os.environ.get(key) else "⬜"
        _print(f"  {icon} {key}  (optional — T1-T3 tier routing)")

    _print()
    _print(_divider())
    status = "ALL SYSTEMS NOMINAL ✅" if all_ok else "ISSUES FOUND — see above ❌"
    _print(f"  {status}")
    _print(_divider())
    return 0 if all_ok else 1


# ─────────────────────────────────────────────────────────────────────────────
# ter  (token count only — fast)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ter(args) -> int:
    """Run only TER on one file. No quality scoring."""
    from inference_bridge import InferenceBridge, count_tokens

    if not args.file:
        _print("  ❌  --file required for ter command")
        return 1

    path = Path(args.file)
    if not path.exists():
        _print(f"  ❌  File not found: {path}")
        return 1

    _print(_header(f"☠️  TER  {path.name}"))

    with tempfile.TemporaryDirectory() as tmp:
        reg = str(Path(tmp) / "registry.json")
        inb = InferenceBridge(repo_path=str(path.parent), registry_path=reg,
                              enable_delta=False)
        compressed, report = inb.process_file(str(path))

    orig = report.original
    _print()
    _print(f"  File:            {path}")
    _print(f"  Language:        {_detect_lang(path)}")
    _print()
    _print(f"  Original:        {orig:>8,} tokens")
    _print(f"  After skeleton:  {report.after_skeleton:>8,}  ({_pct(orig, report.after_skeleton):>5.1f}%↓)  {_bar((orig - report.after_skeleton) / orig if orig else 0, 20)}")
    _print(f"  After masking:   {report.after_masking:>8,}  ({_pct(orig, report.after_masking):>5.1f}%↓)  {_bar((orig - report.after_masking) / orig if orig else 0, 20)}")
    _print(f"  After caveman:   {report.after_caveman:>8,}  ({_pct(orig, report.after_caveman):>5.1f}%↓)  {_bar((orig - report.after_caveman) / orig if orig else 0, 20)}")
    _print()
    _print(f"  TER:             {_ter_icon(report.reduction_pct)}")
    _print(f"  Target (>80%):   {'✅ MET' if report.reduction_pct >= 80 else '⚠️  NOT MET (target: 80%)'}")
    _print()
    _print(_divider())
    return 0


def _pct(orig: int, after: int) -> float:
    return (1 - after / orig) * 100 if orig else 0.0

def _detect_lang(path: Path) -> str:
    return {".py": "Python", ".c": "C", ".h": "C", ".cc": "C++",
            ".cpp": "C++", ".go": "Go", ".rs": "Rust"}.get(path.suffix.lower(), "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# rqs  (quality check only)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_rqs(args) -> int:
    """Run only CCC (Code Consistency Comparison) on one file by comparing original vs compressed text."""
    from inference_bridge import InferenceBridge
    from quality import compute_ccc, compute_code_overlap, syntax_check

    if not args.file:
        _print("  ❌  --file required for rqs command")
        return 1

    path = Path(args.file)
    if not path.exists():
        _print(f"  ❌  File not found: {path}")
        return 1

    _print(_header(f"☠️  CCC  {path.name}"))

    original = path.read_text(encoding="utf-8", errors="replace")

    with tempfile.TemporaryDirectory() as tmp:
        reg = str(Path(tmp) / "registry.json")
        inb = InferenceBridge(repo_path=str(path.parent), registry_path=reg,
                              enable_delta=False)
        compressed, report = inb.process_file(str(path))

    question = getattr(args, "question", "") or ""
    l1_ccc  = compute_ccc(original, compressed, question=question)
    l1_code = compute_code_overlap(original, compressed)
    ok_syntax, _ = syntax_check(compressed) if path.suffix == ".py" else (True, "")

    composite = round((l1_ccc * 0.6 + l1_code * 0.4), 4)

    _print()
    _print(f"  File:              {path}")
    _print(f"  TER (context):     {report.reduction_pct:.1f}%")
    _print()
    _print(f"  CCC (consistency): {_score_icon(l1_ccc, 0.85, 0.70)}  {_bar(l1_ccc, 20)}")
    _print(f"  L1 Code Overlap:   {_score_icon(l1_code, 0.80, 0.60)}  {_bar(l1_code, 20)}")
    _print(f"  L2 Functional:     ⬜ N/A  (provide --tests to enable)")
    _print(f"  L3 LLM Judge:      ⬜ N/A  (set ANTHROPIC_API_KEY to enable)")
    _print()
    if path.suffix == ".py":
        _print(f"  Syntax valid:      {'✅ Yes' if ok_syntax else '❌ BROKEN'}")
    _print(f"  Composite CCC:     {_score_icon(composite, 0.85, 0.70)}")
    _print(f"  Consistency floor: 0.85  {'✅ MET' if composite >= 0.85 else '❌ BELOW FLOOR'}")
    _print()
    _print(f"  NOTE: CCC measures compression consistency, not answer correctness.")
    _print(f"  True Value (TER × CCC) = {report.reduction_pct/100:.3f} × {composite:.3f} = {report.reduction_pct/100 * composite:.3f}")
    _print()
    _print(_divider())
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# benchmark  (TER + RQS combined — the main event)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_benchmark(args) -> int:
    mode = args.mode or ("file" if args.file else "synthetic")

    if mode == "synthetic":
        return _benchmark_synthetic(args)
    elif mode == "file" or args.file:
        return _benchmark_file(args)
    elif mode == "corpus" or args.corpus:
        return _benchmark_corpus(args)
    else:
        _print("  Specify --mode synthetic|corpus, --file FILE, or --corpus DIR")
        return 1


def _benchmark_file(args) -> int:
    from inference_bridge import InferenceBridge
    from quality import compute_ccc, compute_code_overlap, syntax_check

    path = Path(args.file)
    if not path.exists():
        _print(f"  ❌  File not found: {path}")
        return 1

    lang = _detect_lang(path)

    # C files require the C compressor — redirect to report-c
    if lang == "C":
        _print()
        _print(f"  ⚠️   C files use the dedicated C pipeline.")
        _print(f"  Run: python kloc.py report-c --file {path} --repo <repo_root>")
        _print(f"  The generic benchmark only covers Python (InferenceBridge F1-F4).")
        return 1

    _print()
    _print(_header(f"☠️  BENCHMARK  {path.name}"))
    _print(f"  Language: {lang}  |  {time.strftime('%Y-%m-%d %H:%M')}")
    _print()

    original_src = path.read_text(encoding="utf-8", errors="replace")

    with tempfile.TemporaryDirectory() as tmp:
        reg = str(Path(tmp) / "registry.json")
        inb = InferenceBridge(repo_path=str(path.parent), registry_path=reg,
                              enable_delta=False)
        t0 = time.time()
        compressed, report = inb.process_file(str(path))
        elapsed = time.time() - t0

    orig = report.original
    ter  = report.reduction_pct

    # Quality — CCC (Code Consistency Comparison)
    question = getattr(args, "question", "") or ""
    l1s = compute_ccc(original_src, compressed, question=question)
    l1c = compute_code_overlap(original_src, compressed)
    composite = round(l1s * 0.6 + l1c * 0.4, 4)
    true_value = round((ter / 100) * composite, 4)

    _print("  TOKEN EFFICIENCY RATIO (TER)")
    _print(_divider())
    _print(f"  {'Stage':<18} {'Tokens':>8}  {'Saved':>7}  Progress")
    _print(f"  {'Original':<18} {orig:>8,}  {'—':>7}")
    _print(f"  {'F1 Skeleton':<18} {report.after_skeleton:>8,}  {_pct(orig, report.after_skeleton):>6.1f}%  {_bar((orig - report.after_skeleton) / orig if orig else 0, 22)}")
    _print(f"  {'F3 Masking':<18} {report.after_masking:>8,}  {_pct(orig, report.after_masking):>6.1f}%  {_bar((orig - report.after_masking) / orig if orig else 0, 22)}")
    _print(f"  {'F4 Caveman':<18} {report.after_caveman:>8,}  {_pct(orig, report.after_caveman):>6.1f}%  {_bar((orig - report.after_caveman) / orig if orig else 0, 22)}")
    _print(_divider())
    _print(f"  TER:    {_ter_icon(ter)}   target >80%")
    _print()

    _print("  CODE CONSISTENCY COMPARISON (CCC)")
    _print(f"  NOTE: CCC measures token overlap consistency, not answer correctness.")
    _print(_divider())
    _print(f"  CCC (consistency)       {_score_icon(l1s, 0.85, 0.70)}  {_bar(l1s, 22)}  floor 0.85")
    _print(f"  L1 Code Overlap         {_score_icon(l1c, 0.80, 0.60)}  {_bar(l1c, 22)}")
    _print(f"  L2 Functional           ⬜ N/A  (pass --tests FILE to enable)")
    _print(f"  L3 LLM Judge            ⬜ N/A  (set ANTHROPIC_API_KEY to enable)")
    _print(_divider())
    _print(f"  Composite CCC:  {_score_icon(composite, 0.85, 0.70)}")
    _print()

    _print("  COMBINED VERDICT")
    _print(_divider())
    _print(f"  True Value  =  TER × CCC  =  {ter/100:.3f} × {composite:.3f}  =  {true_value:.3f}")
    verdict = _combined_verdict(ter, composite)
    _print(f"  Verdict:    {verdict}")
    _print(f"  Elapsed:    {elapsed:.3f}s")
    _print(_divider())
    _print()

    if args.output:
        _write_json(args.output, {
            "file": str(path), "language": lang,
            "ter": ter, "rqs_l1_semantic": l1s, "rqs_l1_code": l1c,
            "composite_rqs": composite, "true_value": true_value,
            "tokens": {"original": orig, "final": report.after_caveman},
            "stages": {"skeleton": report.after_skeleton,
                       "masking": report.after_masking, "caveman": report.after_caveman},
            "elapsed_s": round(elapsed, 3),
        })
        _print(f"  Results saved → {args.output}")
        _print()

    return 0 if ter >= 70 and composite >= 0.80 else 1


def _benchmark_corpus(args) -> int:
    corpus = Path(args.corpus or args.file)
    if not corpus.exists():
        _print(f"  ❌  Corpus path not found: {corpus}")
        return 1

    from inference_bridge import InferenceBridge
    from quality import compute_ccc, compute_code_overlap

    lang_filter = args.lang or None
    ext_map = {"python": [".py"], "c": [".c", ".h"], "go": [".go"], "rust": [".rs"]}
    exts = set(ext_map.get(lang_filter, [".py", ".c", ".h", ".go", ".rs"]))

    files = sorted(f for f in corpus.rglob("*") if f.suffix.lower() in exts and f.is_file()
                   and "__pycache__" not in str(f) and ".git" not in str(f))

    max_files = args.max_files or 50
    if len(files) > max_files:
        import random; random.shuffle(files)
        files = files[:max_files]

    _print()
    _print(_header(f"☠️  CORPUS BENCHMARK  {corpus.name}"))
    _print(f"  Files: {len(files)}  |  Lang filter: {lang_filter or 'all'}  |  {time.strftime('%Y-%m-%d %H:%M')}")
    _print()

    rows = []
    total_orig = total_final = 0
    l1s_scores = []

    _print(f"  {'File':<28} {'Orig':>6} {'Final':>6} {'TER':>7} {'CCC':>8}")
    _print(_divider())

    for fpath in files:
        try:
            original = fpath.read_text(encoding="utf-8", errors="replace")
            with tempfile.TemporaryDirectory() as tmp:
                inb = InferenceBridge(repo_path=str(fpath.parent),
                                      registry_path=str(Path(tmp) / "r.json"),
                                      enable_delta=False)
                compressed, report = inb.process_file(str(fpath))

            l1s = compute_ccc(original, compressed)
            total_orig  += report.original
            total_final += report.final
            l1s_scores.append(l1s)

            ter_icon = "✅" if report.reduction_pct >= 70 else "⚠️ "
            ccc_icon = "✅" if l1s >= 0.85 else "⚠️ "
            name = fpath.name[:28]
            _print(f"  {name:<28} {report.original:>6,} {report.final:>6,} "
                   f"{ter_icon}{report.reduction_pct:>4.0f}%  {ccc_icon}{l1s:.3f}")
            rows.append({"file": str(fpath.relative_to(corpus)),
                         "ter": report.reduction_pct, "ccc": l1s,
                         "orig": report.original, "final": report.final})
        except Exception as e:
            _print(f"  ⚠️  {fpath.name[:28]} — skipped ({e})")

    if not rows:
        _print("  No files processed.")
        return 1

    overall_ter = (1 - total_final / total_orig) * 100 if total_orig else 0
    mean_rqs    = sum(l1s_scores) / len(l1s_scores)
    true_value  = (overall_ter / 100) * mean_rqs

    _print(_divider())
    _print(f"  {'TOTAL':<28} {total_orig:>6,} {total_final:>6,} "
           f"{_ter_icon(overall_ter):>7}  {_score_icon(mean_rqs, 0.85, 0.70):>8}")
    _print()
    _print(f"  Files benchmarked:  {len(rows)}")
    _print(f"  Overall TER:        {_ter_icon(overall_ter)}")
    _print(f"  Mean CCC:           {_score_icon(mean_rqs, 0.85, 0.70)}")
    _print(f"  NOTE: CCC = Code Consistency Comparison (token overlap, not quality)")
    _print(f"  True Value (TER×CCC): {true_value:.3f}")
    _print()
    _print(_divider())

    if args.output:
        _write_json(args.output, {"corpus": str(corpus), "files": rows,
                                   "overall_ter": overall_ter, "mean_ccc": mean_rqs,
                                   "true_value": true_value})
        _print(f"  Results saved → {args.output}")
    return 0


def _benchmark_synthetic(args) -> int:
    _print()
    _print(_header("☠️  SYNTHETIC BENCHMARK  Python CRUD corpus"))
    _print(f"  3 boilerplate-heavy service files  |  {time.strftime('%Y-%m-%d %H:%M')}")
    _print()

    # Delegate to existing benchmark.py for the TER part (canonical output)
    script = ROOT / "src" / "benchmark.py"
    r = subprocess.run(
        [sys.executable, str(script), "--mode", "synthetic"],
        capture_output=True, text=True, cwd=str(ROOT / "src"),
    )
    if r.returncode != 0 and not r.stdout:
        _print(f"  ❌  benchmark.py failed:\n{r.stderr}")
        return 1
    _print(r.stdout)

    # Compute CCC inline on the same synthetic corpus
    try:
        from inference_bridge import InferenceBridge
        from quality import rqs as _rqs_fn
        sys.path.insert(0, str(ROOT / "src"))
        from benchmark import SYNTHETIC_FILES  # type: ignore

        rqs_version = os.environ.get("KLOC_RQS_VERSION", "v1")
        col_label   = f"CCC ({rqs_version})"
        _print()
        _print(f"  CODE CONSISTENCY COMPARISON ({col_label}) on synthetic corpus")
        _print(f"  NOTE: CCC = token overlap consistency, not answer correctness.")
        _print(_divider())
        _print(f"  {'File':<28} {col_label:>12}  {'Status'}")
        _print(_divider())

        scores = []
        for fname, source in SYNTHETIC_FILES.items():
            with tempfile.TemporaryDirectory() as tmp:
                inb = InferenceBridge(repo_path=tmp, registry_path=str(Path(tmp) / "r.json"),
                                      enable_delta=False)
                src_file = Path(tmp) / fname
                src_file.write_text(source)
                compressed, report = inb.process_file(str(src_file))
            l1 = _rqs_fn(source, compressed)
            scores.append(l1)
            icon = "✅" if l1 >= 0.85 else ("⚠️ " if l1 >= 0.70 else "❌")
            _print(f"  {fname:<28} {l1:.4f}   {icon}")

        mean = sum(scores) / len(scores)
        _print(_divider())
        _print(f"  {f'Mean {col_label}':<28} {mean:.4f}   {_score_icon(mean, 0.85, 0.70)}")
        _print()

        # Extract TER from benchmark.py output
        ter = None
        for line in r.stdout.splitlines():
            m = re.search(r"(\d+\.\d+)%\s*\)", line)
            if m:
                ter = float(m.group(1))
        if ter:
            tv = (ter / 100) * mean
            _print(f"  True Value (TER × CCC) = {ter/100:.3f} × {mean:.3f} = {tv:.3f}")
            _print(f"  NOTE: True Value here measures compression consistency, not correctness.")
            _print()

    except Exception as e:
        _print(f"  ⚠️  CCC scoring skipped: {e}")

    _print(_divider())
    return 0


def _combined_verdict(ter: float, ccc: float) -> str:
    if ter >= 80 and ccc >= 0.90: return "🏆 EXCELLENT — max compression, max consistency"
    if ter >= 80 and ccc >= 0.85: return "✅ PASSED — hits both TER and CCC targets"
    if ter >= 70 and ccc >= 0.85: return "✅ GOOD — consistency solid, push TER higher"
    if ter >= 80 and ccc < 0.85:  return "⚠️  CONSISTENCY RISK — TER great, CCC below floor"
    if ter < 70  and ccc >= 0.85: return "⚠️  LOW COMPRESSION — raise pi_threshold or add masking"
    return                               "❌ BELOW TARGETS — review inference_bridge settings"


# ─────────────────────────────────────────────────────────────────────────────
# pipeline
# ─────────────────────────────────────────────────────────────────────────────

def cmd_pipeline(args) -> int:
    if not args.file:
        _print("  ❌  --file required")
        return 1

    path = Path(args.file)
    if not path.exists():
        _print(f"  ❌  File not found: {path}")
        return 1

    # --question enables RAG filtering automatically (top_k default 5).
    # Legacy --rag flag is still honoured for backward compatibility.
    question  = args.question or "explain this code"
    task_type = args.task_type or "explain"
    use_rag   = getattr(args, "rag", False)
    # top_k applies when --question is explicitly supplied or --rag is set
    top_k_arg = getattr(args, "top_k", 5) or 5
    rag_active = (args.question is not None) or use_rag

    _print()
    _print(_header(f"☠️  PIPELINE  {path.name}"))
    _print(f"  Question:  {question}")
    _print(f"  Task type: {task_type}")
    _print(f"  Tier max:  T{os.environ.get('ACTIVE_TIER_MAX', '2')}")
    if rag_active:
        # Check if a metadata index exists for the file's repo
        _meta_hint = path.parent
        _has_meta  = False
        while _meta_hint != _meta_hint.parent:
            if (_meta_hint / ".kloc" / "metadata_index.json").exists():
                _has_meta = True
                break
            _meta_hint = _meta_hint.parent
        _rag_strategy = "metadata+BM25" if _has_meta else "BM25"
        _print(f"  RAG mode:  ON ({_rag_strategy}, top-{top_k_arg} chunks)")
    _print()

    os.environ.setdefault("ACTIVE_TIER_MAX", "2")   # default to T2 (Sonnet ceiling)
    from orchestrator.pipeline import Pipeline
    pipe = Pipeline(session_id="kloc_cli")
    t0   = time.time()

    # ── Pipeline run ─────────────────────────────────────────────────────────
    # When a question is provided (or --rag), pass top_k into pipeline.run()
    # so the inline BM25 RAG filter fires inside the orchestrator.
    # The legacy --rag path (_apply_rag_filter) is kept for backward compat
    # but now delegates to the same top_k mechanism.
    if rag_active:
        result = pipe.run(
            file_path=str(path),
            question=question,
            task_type=task_type,
            top_k=top_k_arg,
        )
    else:
        result = pipe.run(file_path=str(path), question=question, task_type=task_type)

    elapsed = time.time() - t0

    if "error" in result:
        _print(f"  ❌  Pipeline error: {result['error']}")
        return 1

    rd = result.get("routing_decision", {})
    _print("  AGENT ROUTING DECISIONS")
    _print(_divider())
    _print(f"  PRUNER  → {rd.get('pruner_summary',  'classified chunks')}")
    _print(f"  GRUG    → TER {result.get('overall_ter', 0):.1f}%  |  CCC {result.get('rqs_l1_score', 0):.3f}")
    _print(f"  BALANCER→ Tier {rd.get('tier', 'T0')}  model: {rd.get('model_id') or 'local'}  est. cost: ${rd.get('estimated_cost_usd', 0):.6f}")
    _print(f"  ZIPPY   → history zipped to {rd.get('zipped_tokens', 0)} tokens")
    _print()

    chunks = result.get("compressed_chunks", [])
    _print(f"  Compressed chunks: {len(chunks)}")
    _print(f"  Decision audit:    {len(result.get('decision_audit', []))} entries")
    _print(f"  Elapsed:           {elapsed:.3f}s")
    _print()

    _print("  ROUTING PLAN (what would be sent to LLM)")
    _print(_divider())
    tier_int = rd.get("tier", 0)
    tier_label_map = {0: "LOCAL — no API call (rainbow cache hit or simple task)",
                      1: "HAIKU — cheap model, fast",
                      2: "SONNET — mid-tier, complex tasks",
                      3: "OPUS — full model, crisis/refactor"}
    _print(f"  Tier:    T{tier_int}  {tier_label_map.get(tier_int, '')}")
    _print(f"  Model:   {rd.get('model_id') or 'none (T0/local)'}")
    compressed_tokens = sum(c.get("final_tokens", c.get("tokens", 0)) for c in chunks)
    _print(f"  Context: ~{compressed_tokens} compressed tokens")

    # ── RAG FILTER savings summary ───────────────────────────────────────────
    rag_used  = result.get("rag_chunks_used")
    rag_total = result.get("rag_total_chunks")
    if rag_used is not None and rag_total is not None:
        _print()
        _print("  RAG FILTER")
        _print(_divider())
        _rag_strat = "metadata+BM25 field-weighted" if _has_meta else "BM25 keyword (offline)"
        _print(f"  Strategy:          {_rag_strat}")
        _print(f"  Total chunks:      {rag_total}")
        _print(f"  Selected chunks:   {rag_used}  (top-{top_k_arg})")
        chunks_saved = rag_total - rag_used
        if rag_total > 0:
            saving_pct = chunks_saved / rag_total * 100
            _print(f"  Chunk savings:     {chunks_saved} chunks dropped  ({saving_pct:.0f}% of file filtered out)")
        _print(f"  Compressed tokens: ~{compressed_tokens}  (vs full-file compression)")

    _print(_divider())
    _print()

    llm_resp = result.get("llm_response", "")
    if llm_resp:
        _print("  LLM RESPONSE")
        _print(_divider())
        if llm_resp.startswith("[T0/LOCAL]") or llm_resp.startswith("[LLM ERROR]"):
            _print(f"  {llm_resp}")
        else:
            # Print first 40 lines, then truncate
            resp_lines = llm_resp.splitlines()
            for ln in resp_lines[:40]:
                _print(f"  {ln}")
            if len(resp_lines) > 40:
                _print(f"  ... ({len(resp_lines) - 40} more lines — use --output to save full response)")
        _print(_divider())
        _print()

    # ── Prompt cache savings report ──────────────────────────────────────────
    cache_savings = result.get("cache_savings")
    if cache_savings and cache_savings.get("cache_read_tokens", 0) > 0:
        _print("  PROMPT CACHE SAVINGS")
        _print(_divider())
        _print(f"  Cache read tokens:   {cache_savings['cache_read_tokens']:,}")
        _print(f"  Cache write tokens:  {cache_savings.get('cache_write_tokens', 0):,}")
        _print(f"  Estimated savings:   ${cache_savings['savings_usd']:.6f}  (90% discount on cached tokens)")
        _print(_divider())
        _print()

    if args.output:
        _write_json(args.output, result)
        _print(f"  Full result saved → {args.output}")

    return 0


def _apply_rag_filter(pipe, path: Path, question: str, top_k: int = 3) -> dict:
    """
    Chunk the file, rank chunks against the question, return metadata dict
    including the set of chunk_ids to keep.

    This runs BEFORE the pipeline so the orchestrator can filter the payload.
    """
    from rag.retriever import RAGRetriever

    retriever = RAGRetriever(top_k=top_k)
    strategy  = "bm25" if retriever._should_use_bm25() else "embedding"

    # Reuse the pipeline's already-instantiated chunker
    all_chunks = pipe.chunker.chunk_file(path)
    if not all_chunks:
        src = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        from languages.language_registry import LanguageRegistry
        lang = LanguageRegistry().language_for(path) or "python"
        all_chunks = pipe.chunker.chunk_source(src, str(path), lang)

    # Rough token count: use whitespace split as a fast approximation
    def _raw_tokens(c) -> int:
        src = c.source if hasattr(c, "source") else c.get("source", "")
        return len(src.split())

    tokens_full = sum(_raw_tokens(c) for c in all_chunks)

    selected = retriever.retrieve(question=question, chunks=all_chunks, top_k=top_k)
    selected_ids = {
        (c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id", ""))
        for c in selected
    }
    tokens_rag = sum(_raw_tokens(c) for c in selected)

    return {
        "strategy":       strategy,
        "total_chunks":   len(all_chunks),
        "selected_count": len(selected),
        "top_k":          top_k,
        "selected_ids":   selected_ids,
        "tokens_full":    tokens_full,
        "tokens_rag":     tokens_rag,
    }


# ─────────────────────────────────────────────────────────────────────────────
# test
# ─────────────────────────────────────────────────────────────────────────────

def cmd_test(args) -> int:
    _print(_header("☠️  TEST SUITE"))
    _print()

    pytest_args = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]

    if args.test_filter:
        pytest_args += ["-k", args.test_filter]
    if args.fast:
        pytest_args += ["-x", "--timeout=10"]

    r = subprocess.run(pytest_args, cwd=str(ROOT))
    return r.returncode


# ─────────────────────────────────────────────────────────────────────────────
# rag
# ─────────────────────────────────────────────────────────────────────────────

def cmd_rag(args) -> int:
    action = args.action or "status"

    if action == "build-rainbow":
        _print(_header("☠️  RAG  build rainbow tables"))
        r = subprocess.run([sys.executable, str(ROOT / "rainbow" / "build_rainbow.py")],
                           cwd=str(ROOT))
        return r.returncode

    if action == "etl":
        if not args.corpus:
            _print("  ❌  --corpus required for rag etl")
            return 1
        _print(_header("☠️  RAG  ETL pipeline"))
        cmd = [sys.executable, str(ROOT / "rag" / "etl_pipeline.py"),
               "--repo", args.corpus]
        if args.lang:
            cmd += ["--language", args.lang]
        if args.dry_run:
            cmd += ["--dry-run"]
        r = subprocess.run(cmd, cwd=str(ROOT))
        return r.returncode

    if action == "status":
        _print(_header("☠️  RAG  status"))
        py_table = ROOT / "data" / "rainbow_stdlib_python.jsonl"
        c_table  = ROOT / "data" / "rainbow_stdlib_c.jsonl"
        _print()
        _print(f"  rainbow_stdlib_python: {'✅ ' + str(sum(1 for _ in py_table.open())) + ' entries' if py_table.exists() else '❌ missing — run: python kloc.py rag --action build-rainbow'}")
        _print(f"  rainbow_stdlib_c:      {'✅ ' + str(sum(1 for _ in c_table.open())) + ' entries' if c_table.exists() else '❌ missing — run: python kloc.py rag --action build-rainbow'}")
        corpus = ROOT / "data" / "test_corpus"
        if corpus.exists():
            c_files = len(list(corpus.rglob("*.c")))
            py_files = len(list(corpus.rglob("*.py")))
            _print(f"  test corpus:           ✅ {c_files} .c files, {py_files} .py files")
        else:
            _print("  test corpus:           ⬜ not downloaded — run: bash scripts/download_test_corpus.sh")
        _print()
        return 0

    _print(f"  Unknown rag action '{action}'. Use: build-rainbow | etl | status")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# etl-c
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build_metadata(args) -> int:
    from pathlib import Path as _Path
    repo = _Path(args.repo)
    if not repo.exists():
        _print(f"  ❌  Repo not found: {repo}")
        return 1

    _print()
    _print(_header(f"☠️  BUILD METADATA  {repo.name}"))
    if args.summaries:
        _print("  Summaries: T1 Haiku (--summaries enabled)")
    if args.dry_run:
        _print("  Mode: DRY RUN (not saving)")
    _print()

    from rag.build_metadata import build_metadata
    single = _Path(args.file) if getattr(args, "file", None) else None
    index  = build_metadata(
        repo_path     = repo,
        single_file   = single,
        add_summaries = args.summaries,
        dry_run       = args.dry_run,
    )
    _print()
    _print(f"  {index.summary()}")
    return 0


def cmd_etl_c(args) -> int:
    from pathlib import Path as _Path
    repo = _Path(args.repo)
    if not repo.exists():
        _print(f"  ❌  Repo not found: {repo}")
        return 1

    _print()
    _print(_header(f"☠️  C ETL  {repo.name}"))
    if args.dry_run:
        _print("  Mode: DRY RUN (no embed/upsert)")
    if args.force_remine:
        _print("  Force remine: yes")
    _print()

    from rag.c_etl import run_c_etl
    stats = run_c_etl(
        repo_path    = str(repo),
        force_remine = args.force_remine,
        dry_run      = args.dry_run,
        single_file  = args.file,
    )

    _print()
    _print(f"  TER: {stats.ter_pct:.1f}%  "
           f"({stats.total_orig_tok:,} -> {stats.total_comp_tok:,} tokens)")
    _print(f"  Patterns mined: {stats.pattern_count}")
    if stats.errors:
        _print(f"  Errors: {len(stats.errors)}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# gen-c
# ─────────────────────────────────────────────────────────────────────────────

def cmd_experiment(args) -> int:
    exp_cmd = getattr(args, "exp_cmd", None)

    if exp_cmd == "sweep" or exp_cmd is None:
        import json as _json
        from pathlib import Path as _Path
        from experiments.sweep import sweep

        param_grid: dict = {}

        if getattr(args, "grid", None):
            grid_path = _Path(args.grid)
            if not grid_path.exists():
                grid_path = _Path(__file__).parent / "experiments" / "grids" / args.grid
            g = _json.loads(grid_path.read_text())
            for k, v in g.items():
                if not k.startswith("_"):
                    param_grid[k] = [str(x) for x in v]

        if getattr(args, "param", None):
            for parts in (args.param or []):
                if len(parts) >= 2:
                    param_grid[parts[0]] = parts[1:]

        if not param_grid:
            _print("  Specify --param KEY val1 val2 or --grid GRID_FILE")
            return 1

        sweep(
            param_grid   = param_grid,
            corpus       = getattr(args, "corpus", "synthetic"),
            c_repo       = getattr(args, "repo", ""),
            workers      = getattr(args, "workers", 4),
            label_prefix = getattr(args, "label", ""),
        )

    elif exp_cmd == "optimize":
        from experiments.optimize import optimize
        optimize(
            param_group = getattr(args, "param", "pi"),
            corpus      = getattr(args, "corpus", "synthetic"),
            c_repo      = getattr(args, "repo", ""),
            rounds      = getattr(args, "rounds", 2),
            workers     = getattr(args, "workers", 4),
            fine        = getattr(args, "fine", False),
        )

    elif exp_cmd == "leaderboard":
        from experiments.leaderboard import load_results, print_leaderboard
        results = load_results(
            date         = args.date,
            param_filter = getattr(args, "param", None),
            type_filter  = getattr(args, "type", "all"),
        )
        _print(f"\n  EXPERIMENT LEADERBOARD  ({len(results)} total)\n")
        print_leaderboard(results, top=args.top, show_diff=args.diff,
                          sort_by=getattr(args, "sort", "tv"))

    elif exp_cmd == "run":
        from experiments.runner import run_experiment
        env_overrides = {}
        for kv in (args.env or []):
            k, _, v = kv.partition("=")
            env_overrides[k.strip()] = v.strip()
        result = run_experiment(
            env_overrides = env_overrides,
            label         = args.label,
            corpus        = args.corpus,
            c_repo        = args.repo,
        )
        _print(f"  ID:      {result['experiment_id']}")
        _print(f"  Metrics: {result['metrics']}")
    elif exp_cmd == "bayesian":
        from experiments.bayesian import run_bayesian
        run_bayesian(
            param_group = getattr(args, "param", "pi+tfidf"),
            corpus      = getattr(args, "corpus", "synthetic"),
            c_repo      = getattr(args, "repo", ""),
            n_trials    = getattr(args, "trials", 50),
            storage     = getattr(args, "storage", None),
            study_name  = getattr(args, "study", "kloc_tv_maximize"),
        )

    elif exp_cmd == "llm-setup":
        from experiments.local_llm import LocalLLMHarness
        if not LocalLLMHarness.docker_available():
            _print("  ERROR: Docker is not running. Start Docker Desktop first.")
            return 1
        h = LocalLLMHarness(model=getattr(args, "model", None) or "qwen2.5-coder:14b")
        ok = h.setup(fast=getattr(args, "fast", False))
        return 0 if ok else 1

    elif exp_cmd == "llm-benchmark":
        from experiments.local_llm import LocalLLMHarness, compute_rqs_llm, BENCHMARK_QUESTIONS
        from src.inference_bridge import InferenceBridge
        h = LocalLLMHarness(model=getattr(args, "model", None) or "qwen2.5-coder:14b")
        if not h.health_check():
            _print("  Ollama not running. Run: kloc experiment llm-setup")
            return 1
        _print(f"\n  Real-LLM RQS benchmark  |  model: {h.model}\n")
        try:
            import src.benchmark as _bm
            bridge = InferenceBridge()
            scores = []
            import tempfile, os as _os2
            sf = getattr(_bm, "SYNTHETIC_FILES", {})
            items = sf.items() if isinstance(sf, dict) else sf
            for fname, src in items:
                # write source to temp file so process_file can read it
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                                 delete=False, encoding="utf-8") as tf:
                    tf.write(src)
                    tmp_path = tf.name
                try:
                    compressed, report = bridge.process_file(tmp_path)
                finally:
                    _os2.unlink(tmp_path)
                ter = round(report.reduction_pct / 100.0, 3)
                result = compute_rqs_llm(src, compressed, harness=h)
                rqs = result.get("rqs_llm")
                if rqs is not None:
                    scores.append(rqs)
                    tv = round(ter * rqs, 4)
                    _print(f"  {fname:<35} TER={ter*100:.1f}%  RQS-LLM={rqs:.4f}  TV={tv:.4f}")
            if scores:
                mean_rqs = round(sum(scores) / len(scores), 4)
                _print(f"\n  Mean RQS-LLM: {mean_rqs:.4f}")
                _print(f"  Simulation baseline (TF cosine): 0.650")
                _print(f"  Delta: {mean_rqs - 0.650:+.4f}")
        except Exception as exc:
            _print(f"  ERROR: {exc}")
            return 1

    elif exp_cmd == "llm-status":
        from experiments.local_llm import LocalLLMHarness
        h = LocalLLMHarness()
        running = h._container_running()
        healthy = h.health_check() if running else False
        model   = getattr(args, "model", None) or "qwen2.5-coder:14b"
        cached  = h._model_cached(model) if running else False
        _print(f"  Container: {'running' if running else 'stopped'}")
        _print(f"  API:       {'healthy' if healthy else 'not responding'}")
        _print(f"  Model:     {model} ({'cached' if cached else 'not pulled'})")
        if running and healthy:
            proc = h.ollama_processor()
            _print(f"  Processor: {proc}")

    elif exp_cmd == "llm-gpu-check":
        from experiments.local_llm import LocalLLMHarness
        if not LocalLLMHarness.docker_available():
            _print("  ERROR: Docker not running.")
            return 1
        h = LocalLLMHarness()
        _print("\n  Checking NVIDIA GPU visibility inside Docker...")
        _print("  (may pull nvidia/cuda base image ~200MB on first run)\n")
        status = h.gpu_status()
        if not status["available"]:
            _print(f"  GPU NOT visible. Error: {status.get('error', 'unknown')}")
            _print("\n  Run setup steps:")
            _print("    python experiments/local_llm.py --gpu-check")
            _print("  (prints the exact WSL commands to fix it)")
            return 1
        for g in status.get("gpus", []):
            _print(f"  {g['name']}  |  VRAM: {g['vram']}  |  Driver: {g['driver']}")
        return 0

    elif exp_cmd == "llm-sweep":
        from experiments.local_llm_sweep import run_llm_sweep, _load_grid
        grid_name = getattr(args, "grid", "local_llm_quick")
        workers   = getattr(args, "workers", 2)
        corpus    = getattr(args, "corpus", "synthetic")
        c_file    = getattr(args, "file", "")
        c_repo    = getattr(args, "repo", "")
        try:
            param_grid = _load_grid(grid_name)
        except FileNotFoundError as e:
            _print(f"  Grid not found: {e}")
            return 1
        run_llm_sweep(
            param_grid   = param_grid,
            corpus       = corpus,
            c_file       = c_file,
            c_repo       = c_repo,
            workers      = workers,
        )

    elif exp_cmd == "pipeline-batch":
        from experiments.pipeline_batch import (
            run_pipeline_batch,
            print_batch_results,
            DEFAULT_C_QUESTIONS,
        )
        questions = DEFAULT_C_QUESTIONS[: getattr(args, "n", 5)]
        q_file = getattr(args, "questions", "")
        if q_file and Path(q_file).exists():
            lines = Path(q_file).read_text(encoding="utf-8").strip().splitlines()
            questions = [l.strip() for l in lines if l.strip()][: getattr(args, "n", 5)]
        result = run_pipeline_batch(
            file_path         = args.file,
            questions         = questions,
            repo_path         = getattr(args, "repo", ""),
            cpi_threshold     = getattr(args, "cpi_threshold", 0.40),
            max_ctx_words     = getattr(args, "max_ctx", 8000),
            quality_threshold = getattr(args, "threshold", 20),
            model             = getattr(args, "model", "qwen2.5-coder:14b"),
        )
        print_batch_results(result)

    else:
        _print("  Usage: kloc experiment sweep|leaderboard|run|optimize|bayesian|llm-setup|llm-benchmark|llm-status|llm-sweep|pipeline-batch ...")

    return 0


def cmd_report_c(args) -> int:
    from pathlib import Path as _Path
    f = _Path(args.file)
    r = _Path(args.repo)
    if not f.exists():
        _print(f"  ❌  File not found: {f}")
        return 1
    if not r.exists():
        _print(f"  ❌  Repo not found: {r}")
        return 1

    from reports.c_report import generate_c_report
    generate_c_report(
        file_path    = str(f),
        repo_path    = str(r),
        output_html  = args.html,
        force_remine = args.force_remine,
        question     = getattr(args, "question", None),
        top_k        = getattr(args, "top_k", 5),
    )
    return 0


def cmd_gen_c(args) -> int:
    from pathlib import Path as _Path
    from synthetic.gen_c_game import generate, MODULES

    out_dir = _Path(args.out)
    _print()
    _print(_header("☠️  GEN-C  TOXOID synthetic corpus"))
    _print(f"  Output: {out_dir}")
    _print(f"  Seed:   {args.seed}")
    _print()

    files = generate(out_dir, seed=args.seed)
    total_lines = total_bytes = 0
    for f in files:
        lines = f.read_text(encoding="utf-8").count("\n")
        size  = f.stat().st_size
        total_lines += lines
        total_bytes += size
        _print(f"    {f.name:<20} {lines:>4} lines  {size:>6} bytes")

    _print(f"\n    {'TOTAL':<20} {total_lines:>4} lines  {total_bytes:>6} bytes")
    _print()
    _print("  Next:")
    _print(f"    python kloc.py etl-c --repo {out_dir} --dry-run")
    _print(f"    python kloc.py pipeline --file {out_dir}/enemy.c --question \"explain enemy AI\"")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# CLI wiring
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="kloc",
        description="FTW-KLOC-KILLER ☠️  — Token compressor + quality benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python kloc.py setup
  python kloc.py doctor
  python kloc.py benchmark --mode synthetic
  python kloc.py benchmark --file doom/src/r_plane.c
  python kloc.py benchmark --corpus data/test_corpus/chocolate-doom --lang c --max-files 30
  python kloc.py ter       --file main.go
  python kloc.py rqs       --file service.py
  python kloc.py pipeline  --file doom/src/r_plane.c --question "explain BSP traversal"
  python kloc.py test
  python kloc.py test      --filter "test_skeleton or test_caveman"
  python kloc.py rag       --action build-rainbow
  python kloc.py rag       --action etl --corpus data/test_corpus/chocolate-doom --lang c
        """,
    )

    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    # setup
    sub.add_parser("setup", help="build rainbow tables + verify deps")

    # doctor
    sub.add_parser("doctor", help="health-check environment, deps, Docker, keys")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="TER + RQS combined report")
    p_bench.add_argument("--mode",      choices=["synthetic", "corpus", "file"],
                         help="synthetic=Python CRUD baseline | file=single file | corpus=directory")
    p_bench.add_argument("--file",      help="path to source file (any supported language)")
    p_bench.add_argument("--corpus",    help="path to source code directory for corpus benchmark")
    p_bench.add_argument("--lang",      choices=["python", "c", "go", "rust"],
                         help="filter language in corpus mode")
    p_bench.add_argument("--max-files", type=int, default=50,
                         help="max files to sample in corpus mode (default: 50)")
    p_bench.add_argument("--output",    help="save JSON results to file")

    # ter
    p_ter = sub.add_parser("ter", help="token count only — fast, no quality scoring")
    p_ter.add_argument("--file", required=True, help="source file to analyse")

    # rqs
    p_rqs = sub.add_parser("rqs", help="consistency check — CCC (Code Consistency Comparison)")

    p_rqs.add_argument("--file",  required=True, help="source file to analyse")
    p_rqs.add_argument("--tests", help="pytest test file for L2 functional correctness")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="run full 4-agent pipeline and print routing plan")
    p_pipe.add_argument("--file",      required=True, help="source file to process")
    p_pipe.add_argument("--question",  default=None,
                        help="question about the code; enables RAG chunk filtering automatically")
    p_pipe.add_argument("--top-k",     dest="top_k", type=int, default=5,
                        help="number of top RAG chunks to use when --question is given (default: 5)")
    p_pipe.add_argument("--task-type", dest="task_type",
                        choices=["explain", "codegen", "review", "refactor", "search"],
                        default="explain")
    p_pipe.add_argument("--output",    help="save full pipeline result to JSON file")
    p_pipe.add_argument("--rag",       action="store_true",
                        help="retrieve top-3 relevant chunks (cosine or BM25) instead of whole file (legacy; --question is preferred)")

    # test
    p_test = sub.add_parser("test", help="run pytest suite")
    p_test.add_argument("--filter", dest="test_filter", help="pytest -k expression")
    p_test.add_argument("--fast",   action="store_true", help="stop on first failure, 10s timeout")

    # rag
    p_rag = sub.add_parser("rag", help="RAG operations: build-rainbow | etl | status")
    p_rag.add_argument("--action", choices=["build-rainbow", "etl", "status"], default="status")
    p_rag.add_argument("--corpus",   help="corpus path for etl action")
    p_rag.add_argument("--lang",     choices=["python", "c", "go", "rust"])
    p_rag.add_argument("--dry-run",  action="store_true", dest="dry_run")

    p_bmeta = sub.add_parser("build-metadata", help="build function metadata index for enhanced RAG retrieval")
    p_bmeta.add_argument("--repo",      required=True, help="path to repo root")
    p_bmeta.add_argument("--file",      default=None,  help="index a single file only")
    p_bmeta.add_argument("--summaries", action="store_true", help="generate T1 Haiku one-line summaries")
    p_bmeta.add_argument("--dry-run",   action="store_true", dest="dry_run")

    p_cetl = sub.add_parser("etl-c", help="C/C++ ETL: mine patterns, compress, embed, upsert")
    p_cetl.add_argument("--repo",         required=True, help="path to C/C++ repo root")
    p_cetl.add_argument("--file",         default=None,  help="process single file only")
    p_cetl.add_argument("--force-remine", action="store_true", dest="force_remine",
                        help="rebuild per-repo pattern dict even if it exists")
    p_cetl.add_argument("--dry-run",      action="store_true", dest="dry_run",
                        help="compress + hash only, skip embed/upsert")

    p_rep = sub.add_parser("report-c", help="professional C compression analysis report (terminal + HTML)")
    p_rep.add_argument("--file",         required=True, help="C source file to analyse")
    p_rep.add_argument("--repo",         required=True, help="repo root (for pattern dict)")
    p_rep.add_argument("--html",         default=None,  help="save HTML report to this path")
    p_rep.add_argument("--force-remine", action="store_true", dest="force_remine")
    p_rep.add_argument("--question",     default=None,  help="filter chunks by question (KISS RAG / BM25)")
    p_rep.add_argument("--top-k",        type=int, default=5, dest="top_k",
                       help="number of top chunks to return for --question (default: 5)")

    p_synth = sub.add_parser("gen-c", help="generate synthetic TOXOID C game corpus")
    p_synth.add_argument("--out",  default="synthetic/toxoid", help="output directory")
    p_synth.add_argument("--seed", type=int, default=7, help="RNG seed")

    p_exp = sub.add_parser("experiment", help="run or view parameter sweep experiments")
    p_exp_sub = p_exp.add_subparsers(dest="exp_cmd")

    p_sweep = p_exp_sub.add_parser("sweep", help="parallel grid search over param space")
    p_sweep.add_argument("--param",   nargs="+", action="append", metavar=("KEY","VAL"),
                         help="--param ENV_KEY val1 val2 val3  (repeatable)")
    p_sweep.add_argument("--grid",    default=None, help="JSON grid file in experiments/grids/")
    p_sweep.add_argument("--corpus",  default="synthetic")
    p_sweep.add_argument("--repo",    default="")
    p_sweep.add_argument("--workers", type=int, default=4)
    p_sweep.add_argument("--label",   default="")

    p_lb = p_exp_sub.add_parser("leaderboard", help="show experiment leaderboard")
    p_lb.add_argument("--date",  default=None)
    p_lb.add_argument("--top",   type=int, default=20)
    p_lb.add_argument("--param", default=None, help="filter by param name")
    p_lb.add_argument("--diff",  action="store_true")
    p_lb.add_argument("--type",  default="all", choices=["all", "sim", "llm"],
                      help="Show all/sim-only/llm-only results")
    p_lb.add_argument("--sort",  default="tv", choices=["tv", "tv_time", "ter", "ccc"],
                      help="Sort by tv (default), tv_time (TER×CCC×speed), ter, ccc")

    p_opt = p_exp_sub.add_parser("optimize", help="coordinate-descent optimizer (TV reward signal)")
    p_opt.add_argument("--param",   default="pi", help="pi | c_pi | retrieval")
    p_opt.add_argument("--corpus",  default="synthetic")
    p_opt.add_argument("--repo",    default="")
    p_opt.add_argument("--rounds",  type=int, default=2)
    p_opt.add_argument("--workers", type=int, default=4)
    p_opt.add_argument("--fine",    action="store_true")

    p_erun = p_exp_sub.add_parser("run", help="run single experiment with env overrides")
    p_erun.add_argument("--env",    action="append", default=[], metavar="KEY=VALUE")
    p_erun.add_argument("--corpus", default="synthetic")
    p_erun.add_argument("--repo",   default="")
    p_erun.add_argument("--label",  default="")

    p_bay = p_exp_sub.add_parser("bayesian", help="Bayesian TPE optimizer (Optuna)")
    p_bay.add_argument("--param",   default="pi+tfidf",
                       choices=["pi", "f4", "tfidf", "sig", "all", "pi+tfidf", "pi+sig"],
                       help="Param group to optimize (default: pi+tfidf)")
    p_bay.add_argument("--trials",  type=int, default=50, help="Number of TPE trials")
    p_bay.add_argument("--corpus",  default="synthetic")
    p_bay.add_argument("--repo",    default="")
    p_bay.add_argument("--storage", default=None,
                       help="Optuna storage URL (e.g. sqlite:///experiments/optuna.db)")
    p_bay.add_argument("--study",   default="kloc_tv_maximize")

    p_llm_setup = p_exp_sub.add_parser("llm-setup",
                                        help="Start Ollama container + pull model")
    p_llm_setup.add_argument("--model", default="qwen2.5-coder:14b",
                              help="Model to pull (default: qwen2.5-coder:14b)")
    p_llm_setup.add_argument("--fast",  action="store_true",
                              help="Pull smaller 1.5B model instead")

    p_llm_bench = p_exp_sub.add_parser("llm-benchmark",
                                        help="Run real-LLM RQS benchmark (requires llm-setup)")
    p_llm_bench.add_argument("--model", default="qwen2.5-coder:14b")

    p_llm_status = p_exp_sub.add_parser("llm-status",
                                         help="Show Ollama container + model status")
    p_llm_status.add_argument("--model", default="qwen2.5-coder:14b")

    p_exp_sub.add_parser("llm-gpu-check",
                          help="Verify NVIDIA GPU is visible inside Docker")

    p_llm_sweep = p_exp_sub.add_parser("llm-sweep",
                                        help="Sweep local LLM T0.5 parameters with real Ollama")
    p_llm_sweep.add_argument("--grid",    default="local_llm_quick",
                              help="Grid name in experiments/grids/ (default: local_llm_quick)")
    p_llm_sweep.add_argument("--workers", type=int, default=2,
                              help="Parallel workers (keep low — each job calls Ollama)")
    p_llm_sweep.add_argument("--corpus",  default="synthetic",
                              help="'synthetic' or path to .c file")
    p_llm_sweep.add_argument("--file",    default="",
                              help="C source file path")
    p_llm_sweep.add_argument("--repo",    default="",
                              help="Repo root for C benchmarks")

    pb_parser = p_exp_sub.add_parser("pipeline-batch",
                                      help="Blended pipeline batch measurement (PROTO-001)")
    pb_parser.add_argument("--file",      required=True, help="C source file")
    pb_parser.add_argument("--repo",      default="",    help="Repo root")
    pb_parser.add_argument("--n",         type=int, default=5, help="Number of questions")
    pb_parser.add_argument("--questions", default="",
                           help="Path to .txt file with questions (one per line)")
    pb_parser.add_argument("--model",     default="qwen2.5-coder:14b")
    pb_parser.add_argument("--threshold", type=int, default=20)
    pb_parser.add_argument("--max-ctx",   type=int, default=8000, dest="max_ctx")
    pb_parser.add_argument("--cpi",       type=float, default=0.40, dest="cpi_threshold")

    args = parser.parse_args()

    dispatch = {
        "setup":     cmd_setup,
        "doctor":    cmd_doctor,
        "benchmark": cmd_benchmark,
        "ter":       cmd_ter,
        "rqs":       cmd_rqs,
        "pipeline":  cmd_pipeline,
        "test":      cmd_test,
        "experiment":       cmd_experiment,
        "rag":              cmd_rag,
        "build-metadata":   cmd_build_metadata,
        "etl-c":            cmd_etl_c,
        "gen-c":     cmd_gen_c,
        "report-c":  cmd_report_c,
    }

    if not args.cmd:
        parser.print_help()
        return 0

    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
