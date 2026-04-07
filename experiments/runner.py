"""
runner.py — Single-experiment executor
=======================================
Runs a benchmark with a given config dict, captures metrics, saves to JSONL.

Design:
  - Each experiment runs in a subprocess with env vars set from config.
  - Subprocess isolation: no module-level constant bleed between experiments.
  - Results appended to experiments/results/<YYYY-MM-DD>/results.jsonl.

Usage:
    from experiments.runner import run_experiment
    result = run_experiment({
        "KLOC_PI_THRESHOLD": "0.65",
        "KLOC_PI_BOILERPLATE_BONUS": "0.20",
    })
    print(result["metrics"])

CLI:
    python experiments/runner.py --env KLOC_PI_THRESHOLD=0.65 --env KLOC_PI_BOILERPLATE_BONUS=0.20
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"


def run_experiment(
    env_overrides: Dict[str, str],
    label: str = "",
    corpus: str = "synthetic",          # "synthetic" | path to a .c file
    c_repo: str = "",                   # repo root for C benchmarks
    save: bool = True,
    timeout: int = 120,
    real_llm: bool = False,             # use local Ollama for RQS instead of TF cosine
    llm_harness: Optional["LocalLLMHarness"] = None,
) -> dict:
    """
    Run one benchmark experiment with the given environment overrides.

    Returns a dict with:
      experiment_id, timestamp, label, env, metrics, stdout, success
    """
    experiment_id = uuid.uuid4().hex[:8]
    timestamp     = datetime.datetime.now().isoformat()

    # Build env for subprocess: inherit current env + overrides
    env = {**os.environ, **{k: str(v) for k, v in env_overrides.items()}}

    if corpus == "synthetic":
        cmd = [sys.executable, str(ROOT / "kloc.py"), "benchmark", "--mode", "synthetic"]
    elif corpus.endswith(".c") or corpus.endswith(".cpp"):
        # C file benchmark via report-c, parsing stdout
        repo = c_repo or str(Path(corpus).parent.parent.parent)
        cmd  = [sys.executable, str(ROOT / "kloc.py"), "report-c",
                "--file", corpus, "--repo", repo]
    else:
        cmd = [sys.executable, str(ROOT / "src" / "benchmark.py"), "--mode", "synthetic"]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env            = env,
            capture_output = True,
            text           = True,
            encoding       = "utf-8",
            errors         = "replace",
            timeout        = timeout,
            cwd            = str(ROOT),
        )
        stdout  = (proc.stdout or "") + (proc.stderr or "")
        success = proc.returncode == 0
    except subprocess.TimeoutExpired:
        stdout  = f"TIMEOUT after {timeout}s"
        success = False
    elapsed_ms = round((time.time() - t0) * 1000)

    metrics = _parse_metrics(stdout, corpus)

    # Optional: augment with real-LLM RQS if harness is provided
    if real_llm:
        if corpus == "synthetic":
            llm_result = _compute_llm_rqs(env_overrides, llm_harness)
            ter_key = "python_ter_pct"
        elif corpus.endswith((".c", ".cpp")):
            llm_result = _compute_c_llm_rqs(corpus, env_overrides, llm_harness)
            ter_key = "c_ter_pct"
        else:
            llm_result = {"rqs": None, "llm_latency_ms": None}
            ter_key = "python_ter_pct"
        rqs_llm = llm_result.get("rqs") if isinstance(llm_result, dict) else llm_result
        if rqs_llm is not None:
            metrics["rqs_llm"] = rqs_llm
            ter = metrics.get(ter_key) or 0.0
            metrics["true_value_llm"] = round(ter * rqs_llm, 4)
        if isinstance(llm_result, dict) and llm_result.get("llm_latency_ms") is not None:
            metrics["llm_latency_ms"] = llm_result["llm_latency_ms"]

    result = {
        "experiment_id": experiment_id,
        "timestamp":     timestamp,
        "label":         label or _label_from_env(env_overrides),
        "corpus":        corpus,
        "env":           env_overrides,
        "metrics":       metrics,
        "elapsed_ms":    elapsed_ms,
        "success":       success,
        "stdout":        stdout[-3000:],   # last 3000 chars to cap size
    }

    if save:
        _save_result(result)

    return result


def run_llm_experiment(
    env_overrides: Dict[str, str] = None,
    label: str = "",
    corpus: str = "synthetic",
    c_repo: str = "",
    llm_harness: Optional["LocalLLMHarness"] = None,
    save: bool = True,
    timeout: int = 180,
) -> dict:
    """
    Convenience wrapper: run a benchmark experiment with real_llm=True.
    Defaults to a longer timeout and always enables the LLM harness.
    """
    return run_experiment(
        env_overrides = env_overrides or {},
        label         = label,
        corpus        = corpus,
        c_repo        = c_repo,
        save          = save,
        timeout       = timeout,
        real_llm      = True,
        llm_harness   = llm_harness,
    )


def run_c_llm_experiment(
    c_file:       str,
    c_repo:       str,
    env_overrides: Dict[str, str] = None,
    label:         str = "",
    llm_harness   = None,
    save:          bool = True,
    timeout:       int = 180,
) -> dict:
    """
    Convenience wrapper: benchmark a C file with real_llm=True.
    Combines C corpus TER measurement with Ollama RQS measurement.
    """
    return run_experiment(
        env_overrides = env_overrides or {},
        label         = label or f"c_llm | {Path(c_file).name}",
        corpus        = c_file,
        c_repo        = c_repo,
        save          = save,
        timeout       = timeout,
        real_llm      = True,
        llm_harness   = llm_harness,
    )


def _compute_llm_rqs(
    env_overrides: Dict[str, str],
    harness: Optional["LocalLLMHarness"],
) -> Optional[float]:
    """
    Run the synthetic corpus through InferenceBridge with the given env overrides,
    then compare full vs compressed LLM responses for each file.
    Returns mean RQS-LLM across all files, or None if harness unavailable.
    """
    if harness is None:
        try:
            from experiments.local_llm import LocalLLMHarness
            harness = LocalLLMHarness()
            if not harness.health_check():
                return None
        except Exception:
            return None

    try:
        from experiments.local_llm import compute_rqs_llm, BENCHMARK_QUESTIONS

        # Apply env overrides in-process for the compression step
        import importlib
        import tempfile
        saved = {k: os.environ.get(k) for k in env_overrides}
        os.environ.update(env_overrides)

        # Re-import inference_bridge with new env values
        import src.inference_bridge as _ib
        importlib.reload(_ib)
        bridge = _ib.InferenceBridge()

        # Import synthetic corpus — SYNTHETIC_FILES is a dict {name: source_str}
        import src.benchmark as _bm
        importlib.reload(_bm)
        synthetic_files = getattr(_bm, "SYNTHETIC_FILES", {})
        if isinstance(synthetic_files, dict):
            file_items = synthetic_files.items()
        else:
            file_items = synthetic_files  # legacy list-of-tuples fallback

        scores = []
        latencies = []
        rejected = 0
        for fname, src in file_items:
            # Compress via process_file (InferenceBridge API)
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", encoding="utf-8", delete=False
            ) as tf:
                tf.write(src)
                tmp_path = tf.name
            try:
                compressed, _ = bridge.process_file(tmp_path)
            except Exception:
                continue
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            t_llm = time.time()
            result = compute_rqs_llm(src, compressed,
                                     questions=BENCHMARK_QUESTIONS[:2],  # 2 Qs for speed
                                     harness=harness)
            result["llm_latency_ms"] = round((time.time() - t_llm) * 1000)
            rqs = result.get("rqs_llm")
            if rqs is None:
                continue

            # P0 quality gate: discard question pairs where either response
            # was too short (< 20 words) — mirrors _local_response_quality_ok.
            # compute_rqs_llm reports word counts in per_question[*]["resp_*_len"].
            per_q = result.get("per_question", [])
            gate_ok = all(
                q.get("resp_full_len", 0) >= 20 and q.get("resp_bridge_len", 0) >= 20
                for q in per_q
            )
            if not gate_ok:
                rejected += 1
                continue

            scores.append(rqs)
            latencies.append(result["llm_latency_ms"])

        if rejected:
            print(f"  [llm_rqs] P0 quality gate rejected {rejected} response(s)")

        # Restore env
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        rqs_mean = round(sum(scores) / len(scores), 4) if scores else None
        lat_mean = round(sum(latencies) / len(latencies)) if latencies else None
        return {"rqs": rqs_mean, "llm_latency_ms": lat_mean}

    except Exception as exc:
        print(f"  [llm_rqs] error: {exc}")
        return {"rqs": None, "llm_latency_ms": None}


def _compute_c_llm_rqs(
    c_file: str,
    env_overrides: Dict[str, str],
    harness: Optional["LocalLLMHarness"],
) -> Optional[float]:
    """
    Compute real-LLM RQS for a C source file.

    Reads the C file, compresses it via InferenceBridge (C path), then asks
    the LLM two focused questions about the code and computes TF cosine
    between full-source and compressed-source responses.

    C-specific questions probe function behaviour and control flow — things
    that compression degrades most for game/systems code.
    """
    if harness is None:
        try:
            from experiments.local_llm import LocalLLMHarness
            harness = LocalLLMHarness()
            if not harness.health_check():
                return None
        except Exception:
            return None

    C_QUESTIONS = [
        "What are the main functions in this code and what do they do?",
        "Describe the control flow of the most complex function.",
    ]

    try:
        from experiments.local_llm import compute_rqs_llm

        import importlib
        import tempfile
        saved = {k: os.environ.get(k) for k in env_overrides}
        os.environ.update(env_overrides)

        import src.inference_bridge as _ib
        importlib.reload(_ib)
        bridge = _ib.InferenceBridge()

        c_src = Path(c_file).read_text(encoding="utf-8", errors="replace")

        # Write to temp .c file so InferenceBridge picks the C path
        with tempfile.NamedTemporaryFile(
            suffix=".c", mode="w", encoding="utf-8", delete=False
        ) as tf:
            tf.write(c_src)
            tmp_path = tf.name
        try:
            compressed, _ = bridge.process_file(tmp_path)
        except Exception as exc:
            print(f"  [c_llm_rqs] compression failed: {exc}")
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Restore env before LLM calls (env only needed for compression)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        t_llm = time.time()
        result = compute_rqs_llm(
            c_src, compressed,
            questions=C_QUESTIONS,
            harness=harness,
        )
        llm_latency_ms = round((time.time() - t_llm) * 1000)
        result["llm_latency_ms"] = llm_latency_ms
        rqs = result.get("rqs_llm")
        if rqs is None:
            return {"rqs": None, "llm_latency_ms": None}

        # P0 quality gate
        per_q = result.get("per_question", [])
        gate_ok = all(
            q.get("resp_full_len", 0) >= 20 and q.get("resp_bridge_len", 0) >= 20
            for q in per_q
        )
        if not gate_ok:
            print(f"  [c_llm_rqs] quality gate rejected response")
            return {"rqs": None, "llm_latency_ms": None}

        return {"rqs": rqs, "llm_latency_ms": llm_latency_ms}

    except Exception as exc:
        print(f"  [c_llm_rqs] error: {exc}")
        return {"rqs": None, "llm_latency_ms": None}


def _parse_metrics(stdout: str, corpus: str) -> dict:
    """Extract TER, RQS, True Value from benchmark stdout."""
    metrics: dict = {}

    for line in stdout.splitlines():
        # Synthetic Python benchmark
        if "Original tokens:" in line:
            metrics.setdefault("original_tokens", _extract_int(line))
        if "After skeleton:" in line:
            metrics.setdefault("tokens_after_skeleton", _extract_int(line))
            if "%" in line:
                metrics.setdefault("ter_skeleton_pct", _extract_pct(line))
        if "Mean RQS-L1" in line or "Mean CCC" in line:
            metrics["rqs_l1"] = _extract_float(line)
        if "True Value" in line and ("x" in line.lower() or "×" in line):
            # Format: "True Value (TER × CCC) = 0.793 × 0.650 = 0.516" → last float
            metrics["true_value"] = _extract_last_float(line)
        # C report-c benchmark
        if "TOTAL TER" in line:
            metrics["c_ter_pct"] = _extract_pct(line)
        if ("RQS-L1" in line or "CCC" in line) and "threshold" in line:
            metrics.setdefault("c_rqs_l1", _extract_float(line))
        # Generic TER line from synthetic
        if line.strip().startswith("After caveman:") or "After masking:" in line:
            pct = _extract_pct(line)
            if pct is not None:
                metrics["python_ter_pct"] = pct

    return metrics


def _extract_last_float(line: str) -> Optional[float]:
    """Extract the last decimal float from a line (e.g. True Value = 0.793 × 0.650 = 0.516 → 0.516)."""
    import re
    matches = re.findall(r"\d+\.\d+", line)
    for m in reversed(matches):
        try:
            v = float(m)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass
    return None


def _extract_int(line: str) -> Optional[int]:
    import re
    m = re.search(r"[\d,]+", line)
    if m:
        try:
            return int(m.group(0).replace(",", ""))
        except ValueError:
            pass
    return None


def _extract_float(line: str) -> Optional[float]:
    import re
    # Match numbers like 0.793 or 79.3
    matches = re.findall(r"\d+\.\d+", line)
    for m in matches:
        try:
            v = float(m)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass
    return None


def _extract_pct(line: str) -> Optional[float]:
    import re
    m = re.search(r"([\d.]+)%", line)
    if m:
        try:
            return float(m.group(1)) / 100.0
        except ValueError:
            pass
    return None


def _label_from_env(env: dict) -> str:
    """Build a short human-readable label from env overrides."""
    parts = []
    for k, v in sorted(env.items()):
        short_k = k.replace("KLOC_", "").replace("_", ".")
        parts.append(f"{short_k}={v}")
    return " | ".join(parts[:3])   # cap at 3 for readability


def _save_result(result: dict) -> Path:
    """Append result to today's results JSONL file."""
    today = datetime.date.today().isoformat()
    out_dir = RESULTS_DIR / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return out_file


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run one experiment")
    parser.add_argument("--env",    action="append", default=[],
                        metavar="KEY=VALUE", help="Env override (repeatable)")
    parser.add_argument("--label",  default="", help="Human-readable label")
    parser.add_argument("--corpus", default="synthetic",
                        help="'synthetic' or path/to/file.c")
    parser.add_argument("--repo",   default="", help="Repo root for C benchmarks")
    parser.add_argument("--no-save", action="store_true", dest="no_save")
    args = parser.parse_args()

    env_overrides = {}
    for kv in args.env:
        k, _, v = kv.partition("=")
        env_overrides[k.strip()] = v.strip()

    result = run_experiment(
        env_overrides = env_overrides,
        label         = args.label,
        corpus        = args.corpus,
        c_repo        = args.repo,
        save          = not args.no_save,
    )

    print(f"\n  Experiment: {result['experiment_id']}")
    print(f"  Label:      {result['label']}")
    print(f"  Metrics:    {result['metrics']}")
    print(f"  Success:    {result['success']}")
