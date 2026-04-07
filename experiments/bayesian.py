"""
bayesian.py — Bayesian hyperparameter optimization using Optuna TPE
====================================================================
Uses Tree-structured Parzen Estimators (TPE) to search the joint parameter
space without exhaustive grid enumeration. TPE builds two density estimates
over the parameter space — one for "good" configs (high TV) and one for
"bad" configs — then samples from the ratio.

Key advantage over coordinate descent: explores joint interactions between
parameters (e.g. PI_THRESHOLD × F3_KQ_THRESHOLD) that coordinate descent
misses because it fixes all but one param at a time.

Expected gain: TV ≈ 0.59–0.60 from joint search over {threshold, F3, F4}.
Combined with Strategy 3 (TF-IDF RQS): TV ≈ 0.65+.

Search space:
  KLOC_PI_THRESHOLD:            [0.55, 0.90]
  KLOC_PI_BOILERPLATE_BONUS:    [0.05, 0.30]
  KLOC_PI_LOOP_FREE_BONUS:      [0.03, 0.20]
  KLOC_PI_LOOP_PENALTY:         [0.02, 0.12]
  KLOC_PI_SIMPLE_RATIO_THRESHOLD: [0.30, 0.70]
  KLOC_F4_VOWEL_PRUNE_MIN_LEN:  [4, 8]        (int)
  KLOC_TFIDF_IDF_FLOOR:         [0.001, 0.10]

Usage:
    python experiments/bayesian.py                     # 50 trials
    python experiments/bayesian.py --trials 100        # more trials
    python experiments/bayesian.py --corpus doom/src/strife/p_enemy.c --repo doom/
    python experiments/bayesian.py --param pi          # restrict to pi group
    python experiments/bayesian.py --storage sqlite:///experiments/optuna.db  # persist
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.runner import run_experiment
from experiments.leaderboard import load_results, BASELINE

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("  optuna not installed. Run: pip install optuna")
    sys.exit(1)


# ── Search spaces per group ────────────────────────────────────────────────────

def _suggest_pi(trial: "optuna.Trial") -> Dict[str, str]:
    return {
        "KLOC_PI_THRESHOLD":              str(round(trial.suggest_float("pi_threshold",    0.55, 0.90, step=0.01), 2)),
        "KLOC_PI_BOILERPLATE_BONUS":      str(round(trial.suggest_float("pi_bplate_bonus", 0.05, 0.30, step=0.01), 2)),
        "KLOC_PI_LOOP_FREE_BONUS":        str(round(trial.suggest_float("pi_lf_bonus",     0.03, 0.20, step=0.01), 2)),
        "KLOC_PI_LOOP_PENALTY":           str(round(trial.suggest_float("pi_loop_pen",     0.02, 0.12, step=0.01), 2)),
        "KLOC_PI_SIMPLE_RATIO_THRESHOLD": str(round(trial.suggest_float("pi_sr_thr",       0.30, 0.70, step=0.05), 2)),
    }


def _suggest_f4(trial: "optuna.Trial") -> Dict[str, str]:
    return {
        "KLOC_F4_VOWEL_PRUNE_MIN_LEN": str(trial.suggest_int("f4_vp_min_len", 4, 8)),
    }


def _suggest_tfidf(trial: "optuna.Trial") -> Dict[str, str]:
    return {
        "KLOC_USE_TFIDF_RQS":   "1",
        "KLOC_TFIDF_IDF_FLOOR": str(round(trial.suggest_float("tfidf_idf_floor", 0.001, 0.10, log=True), 4)),
    }


def _suggest_sig_retain(trial: "optuna.Trial") -> Dict[str, str]:
    return {
        "KLOC_SKELETON_SIG_RETAIN":  "1",
        "KLOC_SKELETON_SIG_MAX_IDS": str(trial.suggest_int("sig_max_ids", 5, 25)),
    }


_SUGGEST_GROUPS = {
    "pi":         [_suggest_pi],
    "f4":         [_suggest_f4],
    "tfidf":      [_suggest_tfidf],
    "sig":        [_suggest_sig_retain],
    "all":        [_suggest_pi, _suggest_f4, _suggest_tfidf, _suggest_sig_retain],
    "pi+tfidf":   [_suggest_pi, _suggest_tfidf],
    "pi+sig":     [_suggest_pi, _suggest_sig_retain],
}


def _build_env(trial: "optuna.Trial", groups: list) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for fn in groups:
        env.update(fn(trial))
    return env


def _tv(result: dict) -> float:
    m  = result.get("metrics", {})
    tv = m.get("true_value")
    if tv is not None:
        return tv
    ter = m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0
    rqs = m.get("rqs_l1") or m.get("c_rqs_l1") or 0.0
    return ter * rqs if rqs else ter


def run_bayesian(
    param_group: str  = "pi+tfidf",
    corpus:      str  = "synthetic",
    c_repo:      str  = "",
    n_trials:    int  = 50,
    storage:     Optional[str] = None,
    study_name:  str  = "kloc_tv_maximize",
    show_progress: bool = True,
) -> Dict[str, str]:
    """
    Run Optuna TPE study to maximize True Value.

    Returns the best env-override dict found.
    """
    groups = _SUGGEST_GROUPS.get(param_group)
    if groups is None:
        print(f"  Unknown param group: {param_group}. Options: {list(_SUGGEST_GROUPS)}")
        return {}

    base_tv = BASELINE.get("true_value", 0.516)
    print(f"\n  Bayesian optimizer: group={param_group} | trials={n_trials} | corpus={corpus}")
    print(f"  Baseline TV: {base_tv:.4f}  |  Target: beat it by >{0.002:.3f}")
    print()

    # Seed pruner with existing results to skip known-bad configs
    past_results = load_results()
    n_startup    = min(10, n_trials // 5)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials = n_startup,
        seed             = 42,
    )

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = sampler,
        storage    = storage,
        study_name = study_name,
        load_if_exists = True,
    )

    # Warm-start: add best past result as initial trial
    if past_results:
        best_past = max(past_results, key=_tv)
        if _tv(best_past) > 0:
            past_env = best_past.get("env", {})
            _maybe_enqueue(study, groups, past_env)

    trial_count = [0]

    def objective(trial: "optuna.Trial") -> float:
        env = _build_env(trial, groups)
        result = run_experiment(
            env_overrides = env,
            label         = f"tpe-{param_group}-t{trial.number}",
            corpus        = corpus,
            c_repo        = c_repo,
            save          = True,
        )
        tv = _tv(result)
        trial_count[0] += 1
        if show_progress:
            m   = result.get("metrics", {})
            ter = (m.get("python_ter_pct") or m.get("c_ter_pct") or 0.0) * 100
            rqs = m.get("rqs_l1") or m.get("c_rqs_l1") or 0.0
            sign = "+" if tv >= base_tv else " "
            print(
                f"  [{trial_count[0]:>3}/{n_trials}] "
                f"TER={ter:>5.1f}%  RQS={rqs:.3f}  TV={tv:.4f}  "
                f"({sign}{tv - base_tv:+.4f} vs baseline)"
            )
        return tv

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    best_tv = best.value or 0.0
    best_env = _build_env(best, groups)

    print(f"\n  Best TV: {best_tv:.4f}  (delta: {best_tv - base_tv:+.4f} = {(best_tv - base_tv)/base_tv*100:+.1f}%)")
    print(f"  Best params:")
    for k, v in sorted(best_env.items()):
        print(f"    {k:<45} = {v}")

    # Print top-5
    top5 = sorted(study.trials, key=lambda t: t.value or 0.0, reverse=True)[:5]
    print(f"\n  Top 5 trials:")
    print(f"  {'Trial':>5}  {'TV':>6}  {'ΔTER':>7}")
    for t in top5:
        print(f"  {t.number:>5}  {t.value or 0.0:>6.4f}  {(t.value or 0.0) - base_tv:>+7.4f}")

    return best_env


def _maybe_enqueue(study: "optuna.Study", groups: list, past_env: dict) -> None:
    """Try to enqueue a past best result as an initial trial."""
    try:
        # Build param dict mapping trial param names to values
        params: dict = {}
        for fn in groups:
            if fn == _suggest_pi:
                params.update({
                    "pi_threshold":    float(past_env.get("KLOC_PI_THRESHOLD", 0.70)),
                    "pi_bplate_bonus": float(past_env.get("KLOC_PI_BOILERPLATE_BONUS", 0.15)),
                    "pi_lf_bonus":     float(past_env.get("KLOC_PI_LOOP_FREE_BONUS", 0.10)),
                    "pi_loop_pen":     float(past_env.get("KLOC_PI_LOOP_PENALTY", 0.06)),
                    "pi_sr_thr":       float(past_env.get("KLOC_PI_SIMPLE_RATIO_THRESHOLD", 0.50)),
                })
            elif fn == _suggest_f4:
                params["f4_vp_min_len"] = int(float(past_env.get("KLOC_F4_VOWEL_PRUNE_MIN_LEN", 5)))
            elif fn == _suggest_tfidf:
                params["tfidf_idf_floor"] = float(past_env.get("KLOC_TFIDF_IDF_FLOOR", 0.01))
            elif fn == _suggest_sig_retain:
                params["sig_max_ids"] = int(float(past_env.get("KLOC_SKELETON_SIG_MAX_IDS", 15)))
        if params:
            study.enqueue_trial(params)
    except Exception:
        pass  # warm-start is optional — never block the study


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Bayesian TV optimizer (Optuna TPE)")
    parser.add_argument("--param",    default="pi+tfidf",
                        choices=list(_SUGGEST_GROUPS),
                        help="Param group (default: pi+tfidf)")
    parser.add_argument("--trials",   type=int, default=50, help="Number of trials")
    parser.add_argument("--corpus",   default="synthetic")
    parser.add_argument("--repo",     default="")
    parser.add_argument("--storage",  default=None,
                        help="Optuna storage URL (e.g. sqlite:///experiments/optuna.db)")
    parser.add_argument("--study",    default="kloc_tv_maximize")
    args = parser.parse_args()

    best = run_bayesian(
        param_group = args.param,
        corpus      = args.corpus,
        c_repo      = args.repo,
        n_trials    = args.trials,
        storage     = args.storage,
        study_name  = args.study,
    )

    print(f"\n  To use this config:")
    for k, v in sorted(best.items()):
        print(f"    set {k}={v}")
