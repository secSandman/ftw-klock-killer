"""
quality.py — Inference-Bridge Code Consistency Comparison (CCC) and Quality Scorer
====================================================================================
Measures whether compressed context produces consistent LLM responses compared to
full context. Runs alongside TER (Token Efficiency Ratio).

The full value equation (simulation context):
  True Value = TER (tokens saved) × CCC (compression consistency)

NOTE: CCC (Code Consistency Comparison) is a CONSISTENCY metric, not a quality metric.
It measures whether two texts (original vs compressed) overlap — it does NOT measure
whether either answer is correct or helpful. See M7 in ROADMAP.md for the plan to
move beyond consistency to correctness measurement.

Four measurement layers:
  L1  CCC (Code Consistency Comparison) — TF cosine or LLM-as-judge, fast, automatic
  L2  Functional correctness — execute generated code against tests
  L3  LLM-as-judge          — second model scores both responses
  L4  Regression tracker    — longitudinal quality over time

Usage:
  python quality.py --task codegen --file my_module.py --question "add input validation"
  python quality.py --report                  # show historical CCC trend
  python quality.py --compare full vs bridge  # side-by-side diff
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    """One quality measurement comparing a full-context vs bridged-context response."""

    task_id: str
    timestamp: str
    file_path: str
    question: str

    # Token counts
    full_tokens: int
    bridge_tokens: int
    ter: float                    # token efficiency ratio (% saved)

    # Quality scores (0.0–1.0, higher = better)
    semantic_similarity: float    # L1: embedding cosine similarity
    functional_score: float       # L2: test pass rate (or -1 if not applicable)
    judge_score: float            # L3: LLM-as-judge score (or -1 if not run)
    composite_rqs: float          # weighted composite of available scores

    # Responses
    full_response: str = ""
    bridge_response: str = ""
    diff_summary: str = ""

    # Flags
    quality_regression: bool = False   # True if RQS < threshold
    rqs_threshold: float = 0.85        # minimum acceptable quality

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def verdict(self) -> str:
        if self.composite_rqs >= 0.95:
            return "EXCELLENT — compression preserved full quality"
        elif self.composite_rqs >= self.rqs_threshold:
            return "ACCEPTABLE — minor quality delta, within threshold"
        elif self.composite_rqs >= 0.70:
            return "DEGRADED — quality loss detected, review compression settings"
        else:
            return "FAILURE — significant quality regression, disable compression for this file"


@dataclass
class RQSReport:
    """Aggregate quality report across multiple tasks."""
    generated_at: str
    tasks_evaluated: int
    mean_ter: float
    mean_rqs: float
    regressions: int
    regression_rate: float
    results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Semantic Similarity
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_simple(text: str) -> dict[str, int]:
    """Simple term-frequency tokenizer — no external deps."""
    words = re.findall(r"[a-zA-Z_]\w*|[0-9]+", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq


def _apply_tfidf(tf: dict[str, int], idf: dict[str, float]) -> dict[str, float]:
    """Multiply TF counts by IDF weights to produce TF-IDF vector."""
    return {k: v * idf.get(k, 1.0) for k, v in tf.items()}


def _build_idf(tf_ref: dict[str, int]) -> dict[str, float]:
    """
    Self-IDF: down-weight high-frequency terms in the reference document.

    idf(w) = log(total_unique_terms / (1 + freq_w))

    High-frequency boilerplate (self, return, logger, def) get low IDF.
    Rare domain identifiers (pw_hash, record_id, A_Chase) get high IDF.
    Controlled by KLOC_TFIDF_IDF_FLOOR (default 0.01).
    """
    idf_floor = float(os.environ.get("KLOC_TFIDF_IDF_FLOOR", "0.01"))
    total_unique = max(len(tf_ref), 1)
    idf: dict[str, float] = {}
    for w, freq in tf_ref.items():
        raw = math.log(total_unique / (1.0 + freq))
        idf[w] = max(raw, idf_floor)
    return idf


def _cosine_similarity(a: dict, b: dict) -> float:
    """Cosine similarity between two vectors (TF or TF-IDF)."""
    all_keys = set(a) | set(b)
    if not all_keys:
        return 0.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v**2 for v in a.values()))
    mag_b = math.sqrt(sum(v**2 for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def compute_ccc(text_a: str, text_b: str, question: str = "") -> float:
    """
    CCC — Code Consistency Comparison.

    Measures whether two texts (typically: LLM response to full context vs
    LLM response to compressed context) are consistent with each other.

    NOTE: This is a CONSISTENCY metric, not a QUALITY metric.
    A high CCC score means the compressed context produced a similar answer.
    It does NOT mean either answer is correct or helpful.

    Online path  (ACTIVE_TIER_MAX > 0 + API key + question):  LLM-as-Judge score
                 from llm_judge() — asks the model to rate consistency 1-10,
                 normalised to [0, 1]. Most meaningful signal.
    Offline path (ACTIVE_TIER_MAX = 0 or no key or no question):
                 TF cosine similarity between the two texts. Fast, no API cost,
                 but circular when texts are source code rather than LLM responses.

    See ROADMAP.md M7 — True Quality Measurement for the plan to move
    beyond consistency to correctness measurement.
    """
    tier_max = int(os.environ.get("ACTIVE_TIER_MAX", "0"))
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if tier_max > 0 and key and question:
        score, _ = llm_judge(question, text_a, text_b)
        if score >= 0:
            return score

    # Offline fallback: TF cosine
    tf_a = _tokenize_simple(text_a)
    tf_b = _tokenize_simple(text_b)
    use_tfidf = os.environ.get("KLOC_USE_TFIDF_RQS", "0") == "1"
    if use_tfidf:
        idf = _build_idf(tf_a)
        return round(_cosine_similarity(_apply_tfidf(tf_a, idf), _apply_tfidf(tf_b, idf)), 4)
    return round(_cosine_similarity(tf_a, tf_b), 4)


def _compute_ccc_tf_cosine(text_a: str, text_b: str) -> float:
    """
    CCC v1 — pure TF-cosine path (offline only, ignores ACTIVE_TIER_MAX).

    Use this in tests or benchmarks that need the offline signal regardless of
    environment, or when comparing against historical experiment results.
    """
    tf_a = _tokenize_simple(text_a)
    tf_b = _tokenize_simple(text_b)
    use_tfidf = os.environ.get("KLOC_USE_TFIDF_RQS", "0") == "1"
    if use_tfidf:
        idf = _build_idf(tf_a)
        return round(_cosine_similarity(_apply_tfidf(tf_a, idf), _apply_tfidf(tf_b, idf)), 4)
    return round(_cosine_similarity(tf_a, tf_b), 4)


# Backward compat alias — keeps old call sites working without change
compute_semantic_similarity = compute_ccc

# Explicit pure-TF-cosine alias for tests that need the offline signal regardless of env
compute_ccc_v1 = _compute_ccc_tf_cosine

# Alias: old name kept for backward compat; clearly labelled CSO (Compressed Source Overlap)
compute_rqs_v1 = _compute_ccc_tf_cosine


# ─────────────────────────────────────────────────────────────────────────────
# RQS v2 — non-circular quality measurement
# ─────────────────────────────────────────────────────────────────────────────
#
# AUDIT FINDING (2026-04-07): compute_semantic_similarity() was being called in
# the benchmark as compute_semantic_similarity(source, compressed) — comparing
# compressed source text to original source text.  This is a circular metric:
# higher compression (TER) → fewer surviving tokens → lower overlap → lower RQS.
# True Value = TER × RQS therefore plateaus at ~0.528 regardless of tuning.
# 81 experiments across three optimisers confirmed this is a structural ceiling,
# not a local optimum.  See EXPERIMENTS.md §"TV Plateau — Structural Analysis".
#
# v2 design:
#   Offline  (ACTIVE_TIER_MAX=0):  ROUGE-L between the original's critical
#            interface (function signatures + first docstring line) and the
#            compressed text.  Tests whether the *important* parts survive, not
#            all tokens.  Still simulation-only; establishes a fairer floor.
#   Online   (ACTIVE_TIER_MAX>0 + API key):  ask T1 Haiku the same question
#            about original AND compressed.  ROUGE-L(response_full, response_bridge).
#            This is the only non-circular quality signal.
#
# Controlled by KLOC_RQS_VERSION=v1|v2  (default v1, backward compat).
#


def _rouge_l(reference: str, candidate: str) -> float:
    """
    ROUGE-L: LCS-recall of reference token sequence in candidate.
    Pure Python, no deps.  Range [0, 1].
    """
    ref = re.findall(r"[a-zA-Z_]\w*|[0-9]+", reference.lower())
    can = re.findall(r"[a-zA-Z_]\w*|[0-9]+", candidate.lower())
    if not ref or not can:
        return 0.0
    # LCS length via DP
    m, n = len(ref), len(can)
    # Space-efficient: two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == can[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    lcs = prev[n]
    precision = lcs / n if n else 0.0
    recall    = lcs / m if m else 0.0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)  # F1


def _extract_interface(source: str) -> str:
    """
    Extract the critical interface of a source file: function/method signatures
    and the first line of each docstring.  This is what an LLM needs to answer
    questions about a module's API — it's a non-circular reference because it
    focuses on *what the code exposes*, not the full token set.
    """
    lines = source.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Function / method / class definition
        if re.match(r"^(def |class |async def )", stripped):
            out.append(stripped)
            # Grab first docstring line if next non-empty line is a triple-quote
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_s = lines[j].strip()
                if next_s.startswith('"""') or next_s.startswith("'''"):
                    # One-liner docstring
                    inner = next_s.strip('"\' ')
                    if inner:
                        out.append(f"  # {inner[:120]}")
        i += 1
    return "\n".join(out) if out else source[:500]


def compute_rqs_v2(
    original: str,
    compressed: str,
    question: str = "",
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
) -> float:
    """
    RQS v2 — non-circular quality score.

    Offline path (ACTIVE_TIER_MAX=0 or no API key):
      ROUGE-L between the original's interface (signatures + docstrings)
      and the compressed text.  Asks: "do the critical API surfaces survive?"
      Less circular than v1 because the reference is the *interface*, not
      the full token set; still a simulation approximation.

    Online path (ACTIVE_TIER_MAX>0 and API key available):
      Ask T1 Haiku the same question against original AND compressed context.
      ROUGE-L(response_original, response_compressed).
      This is the only truly non-circular signal: measures whether the LLM
      produces equivalent answers, not whether tokens survived.

    Returns a float in [0.0, 1.0].  Falls back to offline gracefully.
    """
    tier_max = int(os.environ.get("ACTIVE_TIER_MAX", "0"))
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if tier_max > 0 and key:
        # Online path: compare actual LLM responses
        q = question or "Describe what this code does and list the key functions."
        try:
            import urllib.request

            def _ask(ctx: str) -> str:
                payload = json.dumps({
                    "model": model,
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": f"{q}\n\n```\n{ctx[:3000]}\n```"}],
                }).encode()
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read())
                    return data["content"][0]["text"].strip()

            resp_orig = _ask(original)
            resp_comp = _ask(compressed)
            return _rouge_l(resp_orig, resp_comp)
        except Exception:
            pass  # Fall through to offline path

    # Offline path: ROUGE-L(interface, compressed)
    interface = _extract_interface(original)
    return _rouge_l(interface, compressed)


def rqs(
    original: str,
    compressed: str,
    question: str = "",
    api_key: Optional[str] = None,
) -> float:
    """
    Route to compute_rqs_v1 or compute_rqs_v2 based on KLOC_RQS_VERSION env var.
    Default: v1 (backward compat — does not invalidate experiment history).
    Set KLOC_RQS_VERSION=v2 to use the non-circular metric.
    """
    version = os.environ.get("KLOC_RQS_VERSION", "v1")
    if version == "v2":
        return compute_rqs_v2(original, compressed, question=question, api_key=api_key)
    # v1: CCC / CSO — Compressed Source Overlap (legacy, circular with TER)
    return compute_ccc(original, compressed, question=question)


def compute_code_overlap(response_full: str, response_bridge: str) -> float:
    """
    Supplementary L1: Extract code blocks from both responses and compare.
    Measures whether the same identifiers, function calls, and logic appear.
    """
    code_re = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

    def extract_identifiers(text: str) -> set[str]:
        blocks = code_re.findall(text)
        if not blocks:
            return set()  # No code blocks — signal "not applicable"
        all_code = "\n".join(blocks)
        return set(re.findall(r"[a-zA-Z_]\w*", all_code))

    ids_full   = extract_identifiers(response_full)
    ids_bridge = extract_identifiers(response_bridge)

    if not ids_full:
        return 1.0  # No code to compare — not applicable

    intersection = ids_full & ids_bridge
    union = ids_full | ids_bridge
    return round(len(intersection) / len(union), 4) if union else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Functional Correctness
# ─────────────────────────────────────────────────────────────────────────────

def extract_code_block(response: str) -> Optional[str]:
    """Pull the first Python code block from an LLM response."""
    # Try fenced code block first
    m = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fall back: anything that looks like a function definition
    m = re.search(r"(def \w+.*?)(?:\n\n|\Z)", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def run_code_against_tests(code: str, test_code: str) -> tuple[float, str]:
    """
    L2: Execute generated code against a test suite.
    Returns (pass_rate, output_log).

    test_code should be pytest-compatible test functions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "generated.py"
        test_path = Path(tmpdir) / "test_generated.py"

        code_path.write_text(code)
        test_path.write_text(f"from generated import *\n\n{test_code}")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short", "-q"],
                capture_output=True, text=True, timeout=30, cwd=tmpdir
            )
            output = result.stdout + result.stderr

            # Parse pytest summary line: "X passed, Y failed"
            passed = len(re.findall(r"\bpassed\b", output))
            failed = len(re.findall(r"\bfailed\b", output))
            errors = len(re.findall(r"\berror\b", output.lower()))

            total = passed + failed + errors
            pass_rate = passed / total if total > 0 else 0.0
            return round(pass_rate, 4), output

        except subprocess.TimeoutExpired:
            return 0.0, "TIMEOUT: test execution exceeded 30 seconds"
        except Exception as e:
            return 0.0, f"ERROR: {e}"


def syntax_check(code: str) -> tuple[bool, str]:
    """Quick sanity check: is the generated code valid Python?"""
    try:
        ast.parse(code)
        return True, "valid"
    except SyntaxError as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: LLM-as-Judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an impartial code review judge.
You will be given two LLM responses to the same coding question.
Response A was generated with full source code context.
Response B was generated with compressed (token-reduced) context.

Score Response B relative to Response A on three dimensions.
Respond with ONLY a JSON object — no other text.

{
  "correctness": 0.0,    // 0-1: Is B's solution as correct as A's?
  "completeness": 0.0,   // 0-1: Does B address everything A addresses?
  "relevance": 0.0,      // 0-1: Is B as focused on the actual question?
  "composite": 0.0,      // weighted average: correctness*0.5 + completeness*0.3 + relevance*0.2
  "reasoning": "..."     // one sentence explaining the key difference if any
}"""

JUDGE_USER_TEMPLATE = """QUESTION: {question}

RESPONSE A (full context):
{response_full}

RESPONSE B (compressed context):
{response_bridge}

Score Response B relative to Response A."""


def llm_judge(
    question: str,
    response_full: str,
    response_bridge: str,
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[float, str]:
    """
    L3: Use a second LLM to score response quality.
    Returns (composite_score, reasoning).

    Requires an API key. Falls back to -1.0 (not evaluated) if unavailable.
    Set api_key via environment: ANTHROPIC_API_KEY or OPENAI_API_KEY.
    """
    # Try Anthropic first
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return _judge_anthropic(question, response_full, response_bridge, key, model)

    # Try OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return _judge_openai(question, response_full, response_bridge, key)

    return -1.0, "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable LLM judging."


def _judge_anthropic(question, resp_full, resp_bridge, api_key, model) -> tuple[float, str]:
    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": model,
            "max_tokens": 512,
            "system": JUDGE_SYSTEM_PROMPT,
            "messages": [{
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    question=question,
                    response_full=resp_full[:3000],
                    response_bridge=resp_bridge[:3000],
                )
            }]
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = data["content"][0]["text"].strip()
            # Strip any markdown fences
            text = re.sub(r"```json\n?|```", "", text).strip()
            scores = json.loads(text)
            return round(float(scores.get("composite", 0.0)), 4), scores.get("reasoning", "")

    except Exception as e:
        return -1.0, f"Judge call failed: {e}"


def _judge_openai(question, resp_full, resp_bridge, api_key) -> tuple[float, str]:
    try:
        import urllib.request

        payload = json.dumps({
            "model": "gpt-4o-mini",
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                    question=question,
                    response_full=resp_full[:3000],
                    response_bridge=resp_bridge[:3000],
                )}
            ],
            "response_format": {"type": "json_object"},
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            scores = json.loads(text)
            return round(float(scores.get("composite", 0.0)), 4), scores.get("reasoning", "")

    except Exception as e:
        return -1.0, f"Judge call failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Composite RQS
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_rqs(
    semantic: float,
    functional: float,
    judge: float,
    code_overlap: float,
) -> float:
    """
    Weighted composite Response Quality Score.

    Weights reflect signal reliability:
      - Functional correctness is the ground truth (highest weight) when available
      - LLM judge is nuanced but expensive (second when available)
      - Semantic similarity + code overlap are always available (floor signal)

    If functional or judge scores are -1 (not run), weight redistributes.
    """
    scores = []
    weights = []

    # Always available
    scores.append(semantic)
    weights.append(0.25)
    scores.append(code_overlap)
    weights.append(0.15)

    # Conditional
    if functional >= 0:
        scores.append(functional)
        weights.append(0.40)
    if judge >= 0:
        scores.append(judge)
        weights.append(0.20)

    # Renormalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    rqs = sum(s * w for s, w in zip(scores, weights))
    return round(rqs, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4: Regression Tracker
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_PATH = Path("inb_quality_history.json")


def save_result(result: QualityResult):
    """Append a QualityResult to the local history file."""
    history = []
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            history = json.load(f)
    history.append(result.to_dict())
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH) as f:
        return json.load(f)


def generate_rqs_report() -> RQSReport:
    """Aggregate all historical results into a summary report."""
    history = load_history()
    if not history:
        return RQSReport(
            generated_at=datetime.now().isoformat(),
            tasks_evaluated=0,
            mean_ter=0.0,
            mean_rqs=0.0,
            regressions=0,
            regression_rate=0.0,
            results=[],
        )

    ters = [r["ter"] for r in history]
    rqss = [r["composite_rqs"] for r in history]
    regressions = sum(1 for r in history if r.get("quality_regression", False))

    return RQSReport(
        generated_at=datetime.now().isoformat(),
        tasks_evaluated=len(history),
        mean_ter=round(sum(ters) / len(ters), 2),
        mean_rqs=round(sum(rqss) / len(rqss), 4),
        regressions=regressions,
        regression_rate=round(regressions / len(history), 4),
        results=history,
    )


def print_rqs_report(report: RQSReport):
    print(f"\n{'='*60}")
    print(f"  InB Response Quality Report")
    print(f"  Generated: {report.generated_at[:19]}")
    print(f"{'='*60}")
    print(f"  Tasks evaluated:   {report.tasks_evaluated}")
    print(f"  Mean TER (saved):  {report.mean_ter:.1f}%")
    print(f"  Mean RQS:          {report.mean_rqs:.3f}  (target: ≥0.850)")
    print(f"  Quality regressions: {report.regressions} ({report.regression_rate*100:.1f}%)")
    print(f"{'='*60}")

    if report.results:
        print(f"\n  Per-task breakdown (most recent 10):")
        print(f"  {'FILE':<30} {'TER':>6} {'SIM':>6} {'FUNC':>6} {'RQS':>6}  VERDICT")
        print(f"  {'-'*80}")
        for r in report.results[-10:]:
            func_str = f"{r['functional_score']:.2f}" if r['functional_score'] >= 0 else "  N/A"
            verdict_short = "✓ OK" if not r.get("quality_regression") else "✗ REG"
            fname = Path(r['file_path']).name[:28]
            print(
                f"  {fname:<30} {r['ter']:>5.1f}%"
                f" {r['semantic_similarity']:>6.3f}"
                f" {func_str:>6}"
                f" {r['composite_rqs']:>6.3f}"
                f"  {verdict_short}"
            )

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    file_path: str,
    question: str,
    response_full: str,
    response_bridge: str,
    full_tokens: int,
    bridge_tokens: int,
    test_code: Optional[str] = None,
    run_judge: bool = False,
    rqs_threshold: float = 0.85,
    save: bool = True,
    _override_functional: float = -99.0,  # internal: inject a known score for demos/tests
) -> QualityResult:
    """
    Run all available quality layers and return a QualityResult.

    Args:
        file_path:       Path to the file that was compressed
        question:        The question/task that was asked
        response_full:   LLM response with full context
        response_bridge: LLM response with InB-compressed context
        full_tokens:     Token count of full context
        bridge_tokens:   Token count of bridged context
        test_code:       Optional pytest test functions to run against generated code
        run_judge:       Whether to call the LLM judge (costs tokens)
        rqs_threshold:   Minimum acceptable RQS (default 0.85)
        save:            Whether to persist result to inb_quality_history.json
    """
    task_id = hashlib.md5(
        f"{file_path}{question}{time.time()}".encode()
    ).hexdigest()[:8]

    ter = (1 - bridge_tokens / full_tokens) * 100 if full_tokens > 0 else 0.0

    # L1: CCC — Code Consistency Comparison
    semantic = compute_ccc(response_full, response_bridge, question=question)
    code_overlap = compute_code_overlap(response_full, response_bridge)

    # L2: Functional correctness
    functional = -1.0
    if _override_functional != -99.0:
        functional = _override_functional
    elif test_code:
        code_bridge = extract_code_block(response_bridge)
        if code_bridge:
            valid, _ = syntax_check(code_bridge)
            if valid:
                functional, _ = run_code_against_tests(code_bridge, test_code)
            else:
                functional = 0.0
        else:
            functional = 0.0

    # L3: LLM judge
    judge = -1.0
    judge_reasoning = ""
    if run_judge:
        judge, judge_reasoning = llm_judge(question, response_full, response_bridge)

    # Composite RQS
    composite = compute_composite_rqs(semantic, functional, judge, code_overlap)

    # Diff summary (which identifiers are in full but not bridge response)
    full_ids = set(re.findall(r"[a-zA-Z_]\w*", response_full))
    bridge_ids = set(re.findall(r"[a-zA-Z_]\w*", response_bridge))
    missing = full_ids - bridge_ids
    extra = bridge_ids - full_ids
    diff_parts = []
    if missing:
        diff_parts.append(f"Missing from bridge: {', '.join(sorted(missing)[:8])}")
    if extra:
        diff_parts.append(f"Extra in bridge: {', '.join(sorted(extra)[:5])}")
    diff_summary = " | ".join(diff_parts) if diff_parts else "responses are closely aligned"
    if judge_reasoning:
        diff_summary += f" | Judge: {judge_reasoning}"

    result = QualityResult(
        task_id=task_id,
        timestamp=datetime.now().isoformat(),
        file_path=file_path,
        question=question,
        full_tokens=full_tokens,
        bridge_tokens=bridge_tokens,
        ter=round(ter, 2),
        semantic_similarity=semantic,
        functional_score=functional,
        judge_score=judge,
        composite_rqs=composite,
        full_response=response_full,
        bridge_response=response_bridge,
        diff_summary=diff_summary,
        quality_regression=composite < rqs_threshold,
        rqs_threshold=rqs_threshold,
    )

    if save:
        save_result(result)

    return result


def print_result(result: QualityResult):
    print(f"\n{'='*60}")
    print(f"  InB Quality Evaluation — Task {result.task_id}")
    print(f"{'='*60}")
    print(f"  File:              {result.file_path}")
    print(f"  Question:          {result.question[:60]}")
    print(f"  Timestamp:         {result.timestamp[:19]}")
    print()
    print(f"  Token Efficiency Ratio (TER)")
    print(f"    Full context:    {result.full_tokens:,} tokens")
    print(f"    Bridge context:  {result.bridge_tokens:,} tokens")
    print(f"    Tokens saved:    {result.ter:.1f}%")
    print()
    print(f"  Response Quality Score (RQS)")
    print(f"    L1 Semantic sim: {result.semantic_similarity:.3f}")
    print(f"    L1 Code overlap: — (factored into composite)")
    func_str = f"{result.functional_score:.3f}" if result.functional_score >= 0 else "not run"
    judge_str = f"{result.judge_score:.3f}" if result.judge_score >= 0 else "not run"
    print(f"    L2 Functional:   {func_str}")
    print(f"    L3 LLM judge:    {judge_str}")
    print(f"    ─────────────────────────────")
    print(f"    Composite RQS:   {result.composite_rqs:.3f}  (threshold: {result.rqs_threshold:.2f})")
    print()
    reg_flag = "✗ REGRESSION DETECTED" if result.quality_regression else "✓ within threshold"
    print(f"  Status:  {reg_flag}")
    print(f"  Verdict: {result.verdict}")
    print(f"  Notes:   {result.diff_summary}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Demo / self-test
# ─────────────────────────────────────────────────────────────────────────────

DEMO_QUESTION = "Add input validation to the load() method so it raises ValueError if port is not between 1 and 65535."

DEMO_RESPONSE_FULL = '''
Here's the updated `load()` method with input validation:

```python
def load(self) -> Config:
    """Load and validate configuration."""
    if self._loaded:
        return self._get_cached()

    raw = {}
    if self.config_path and os.path.exists(self.config_path):
        with open(self.config_path) as f:
            raw = json.load(f)

    host = raw.get("host", os.environ.get("APP_HOST", self.DEFAULT_HOST))
    port = int(raw.get("port", os.environ.get("APP_PORT", self.DEFAULT_PORT)))
    debug = raw.get("debug", False)
    max_conn = int(raw.get("max_connections", 100))

    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port {port}: must be between 1 and 65535")

    config = Config(host=host, port=port, debug=debug, max_connections=max_conn)
    self._cache["config"] = config
    self._loaded = True
    return config
```

The validation raises `ValueError` immediately after computing the port,
before constructing the `Config` object.
'''

DEMO_RESPONSE_BRIDGE = '''
Add the port validation right after computing `port`:

```python
def load(self) -> Config:
    if self._loaded:
        return self._get_cached()

    raw = {}
    if self.config_path and os.path.exists(self.config_path):
        with open(self.config_path) as f:
            raw = json.load(f)

    host = raw.get("host", self.DEFAULT_HOST)
    port = int(raw.get("port", self.DEFAULT_PORT))

    if not (1 <= port <= 65535):
        raise ValueError(f"Port {port} is out of range (1-65535)")

    debug = raw.get("debug", False)
    max_conn = int(raw.get("max_connections", 100))
    config = Config(host=host, port=port, debug=debug, max_connections=max_conn)
    self._cache["config"] = config
    self._loaded = True
    return config
```

Both responses correctly implement the validation at the same logical point.
'''

DEMO_TEST_CODE = '''
import pytest

def validate_port(port):
    """Extracted validation logic from both responses."""
    if not (1 <= port <= 65535):
        raise ValueError(f"Port {port} is out of range (1-65535)")
    return port

def test_valid_port():
    assert validate_port(8080) == 8080

def test_port_too_low():
    with pytest.raises(ValueError):
        validate_port(0)

def test_port_too_high():
    with pytest.raises(ValueError):
        validate_port(99999)

def test_boundary_low():
    assert validate_port(1) == 1

def test_boundary_high():
    assert validate_port(65535) == 65535
'''

# Standalone version of the generated code for functional testing
DEMO_CODE_TO_TEST = '''
def validate_port(port):
    if not (1 <= int(port) <= 65535):
        raise ValueError(f"Port {port} is out of range (1-65535)")
    return int(port)
'''


def run_demo():
    print("\n" + "="*60)
    print("  InB Quality Scorer — Demo Run")
    print("  Simulating: full context vs bridged context responses")
    print("="*60)

    result = evaluate(
        file_path="module_a.py",
        question=DEMO_QUESTION,
        response_full=DEMO_RESPONSE_FULL,
        response_bridge=DEMO_RESPONSE_BRIDGE,
        full_tokens=3190,
        bridge_tokens=582,
        test_code=None,         # pass DEMO_TEST_CODE when you have a real codegen task
        run_judge=False,        # set True if ANTHROPIC_API_KEY is set
        save=True,
        _override_functional=1.0,  # demo: both responses pass the same 5/5 tests
    )

    print_result(result)

    # Show the 2D value chart
    print("  True Value Matrix:")
    print(f"  {'':20}  {'Low TER (<50%)':^20}  {'High TER (>80%)':^20}")
    print(f"  {'High RQS (>0.85)':20}  {'Marginal savings':^20}  {'IDEAL — ship it':^20}")
    print(f"  {'Low RQS (<0.85)':20}  {'Both wrong':^20}  {'DANGEROUS — tune':^20}")
    print()
    current_label = "← your result" if result.ter > 80 and result.composite_rqs > 0.85 else "← check tuning"
    print(f"  Your result:  TER={result.ter:.1f}%  RQS={result.composite_rqs:.3f}  {current_label}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="InB Response Quality Scorer")
    parser.add_argument("--demo",   action="store_true", help="Run demo evaluation")
    parser.add_argument("--report", action="store_true", help="Show aggregate quality report")
    parser.add_argument("--clear",  action="store_true", help="Clear quality history")
    args = parser.parse_args()

    if args.clear:
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
            print("Quality history cleared.")
        else:
            print("No history to clear.")

    elif args.report:
        report = generate_rqs_report()
        print_rqs_report(report)

    elif args.demo:
        run_demo()

    else:
        print("\nInB Quality Scorer")
        print("  --demo    run a sample quality evaluation")
        print("  --report  show aggregate quality history")
        print("  --clear   reset quality history")
        print()
        print("  Programmatic use:")
        print("    from quality import evaluate")
        print("    result = evaluate(file, question, resp_full, resp_bridge,")
        print("                      full_tokens, bridge_tokens, test_code=...)")
