"""
test_pipeline_batch.py — Tests for PROTO-001 blended pipeline metric
======================================================================
All Ollama / network calls are mocked — no real inference happens.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.pipeline_batch import (
    DEFAULT_C_QUESTIONS,
    _quality_gate,
    run_pipeline_batch,
)


# ── Quality gate unit tests ────────────────────────────────────────────────────

def test_quality_gate_short_response():
    """10 words, threshold=20 → rejected (False)."""
    response = "This function handles enemy movement in the game world today."
    assert len(response.split()) == 10
    assert _quality_gate(response, threshold=20) is False


def test_quality_gate_passes_long_response():
    """25 words, threshold=20 → passes (True)."""
    response = (
        "This function handles enemy AI movement. It checks the current state of the enemy "
        "and transitions between chase, retreat, and idle modes based on proximity to the player."
    )
    words = response.split()
    assert len(words) >= 25
    assert _quality_gate(response, threshold=20) is True


def test_quality_gate_refusal():
    """Refusal phrase → rejected regardless of length."""
    long_refusal = "I cannot help with this question because " + "word " * 30
    assert _quality_gate(long_refusal, threshold=20) is False


def test_quality_gate_refusal_i_dont_have():
    """'i don't have' phrase → rejected."""
    response = "I don't have enough context provided to answer this question accurately."
    assert _quality_gate(response, threshold=20) is False


def test_quality_gate_refusal_unable():
    """'i'm unable' phrase → rejected."""
    response = "I'm unable to determine the answer from the code " + "word " * 20
    assert _quality_gate(response, threshold=20) is False


def test_quality_gate_threshold_boundary():
    """Exactly threshold words → passes (gate is strict <, not <=)."""
    threshold = 20
    response = " ".join(["word"] * threshold)
    assert len(response.split()) == threshold
    assert _quality_gate(response, threshold=threshold) is True


def test_quality_gate_empty_response():
    """Empty string → rejected."""
    assert _quality_gate("", threshold=20) is False


def test_quality_gate_no_context_provided():
    """'no context provided' phrase → rejected."""
    response = "No context provided so I cannot answer your question properly here."
    assert _quality_gate(response, threshold=5) is False


# ── Compression helper test ────────────────────────────────────────────────────

def test_compress_returns_fewer_tokens():
    """Mock CCompressor chain — verify compressed_tokens < orig_tokens in result."""
    mock_source = "int foo() {\n    int x = 0;\n    return x;\n}\n" * 50
    mock_compressed = "int foo() { ... }\n" * 50

    # We mock at the module level so run_pipeline_batch uses the mocks
    with (
        patch("experiments.pipeline_batch._check_ollama", return_value=True),
        patch("experiments.pipeline_batch._compress_c_file",
              return_value=(mock_compressed, 500, 100, 80.0)),
        patch("experiments.local_llm.LocalLLMHarness") as MockHarness,
    ):
        mock_harness = MagicMock()
        mock_harness.ask.return_value = "word " * 25
        mock_harness._last_latency_ms = 200
        MockHarness.return_value = mock_harness

        result = run_pipeline_batch(
            file_path   = "dummy.c",
            questions   = ["What does this do?"],
            repo_path   = "",
            save        = False,
        )

    assert result.get("orig_tokens", 0) == 500
    assert result.get("compressed_tokens", 0) == 100
    assert result.get("compressed_tokens", 0) < result.get("orig_tokens", 1)


# ── run_pipeline_batch integration tests (all mocked) ─────────────────────────

_LONG_RESPONSE  = "This function manages the enemy AI state machine. " \
                  "It evaluates proximity to the player, current health status, " \
                  "and ambient noise levels before transitioning states. " \
                  "The transition logic uses a priority queue for decisions. " \
                  "Multiple enemies can share the same decision graph."

_SHORT_RESPONSE = "Enemy moves."   # < 20 words → rejected


def _make_mock_harness(response: str, latency: int = 300):
    mock = MagicMock()
    mock.ask.return_value = response
    mock._last_latency_ms = latency
    return mock


def test_run_pipeline_batch_all_accepted():
    """All long responses → acceptance_rate=1.0, blended cost = $0."""
    questions = ["Q1?", "Q2?", "Q3?"]

    with (
        patch("experiments.pipeline_batch._check_ollama", return_value=True),
        patch("experiments.pipeline_batch._compress_c_file",
              return_value=("compressed text " * 100, 1000, 500, 50.0)),
        patch("experiments.local_llm.LocalLLMHarness",
              return_value=_make_mock_harness(_LONG_RESPONSE)),
    ):
        result = run_pipeline_batch(
            file_path  = "dummy.c",
            questions  = questions,
            save       = False,
        )

    assert result["total_queries"] == 3
    assert result["accepted"] == 3
    assert result["rejected"] == 0
    assert result["acceptance_rate"] == 1.0
    assert result["cost_blended_t1"] == 0.0


def test_run_pipeline_batch_all_rejected():
    """All short responses → acceptance_rate=0.0, blended cost = compressed cost."""
    questions = ["Q1?", "Q2?"]

    with (
        patch("experiments.pipeline_batch._check_ollama", return_value=True),
        patch("experiments.pipeline_batch._compress_c_file",
              return_value=("c text " * 100, 2000, 800, 60.0)),
        patch("experiments.local_llm.LocalLLMHarness",
              return_value=_make_mock_harness(_SHORT_RESPONSE)),
    ):
        result = run_pipeline_batch(
            file_path  = "dummy.c",
            questions  = questions,
            save       = False,
        )

    assert result["total_queries"] == 2
    assert result["accepted"] == 0
    assert result["rejected"] == 2
    assert result["acceptance_rate"] == 0.0
    # Blended cost should equal compressed cost per query (all rejected)
    expected_blended = 2 * 800 * 0.00025 / 1000 / 2   # (rejected * tok * rate / 1000) / total
    assert abs(result["cost_blended_t1"] - expected_blended) < 1e-10


def test_run_pipeline_batch_mixed():
    """50% acceptance → acceptance_rate ≈ 0.5."""
    answers = [_LONG_RESPONSE, _SHORT_RESPONSE, _LONG_RESPONSE, _SHORT_RESPONSE]

    call_count = [0]

    def _ask_side_effect(*args, **kwargs):
        resp = answers[call_count[0] % len(answers)]
        call_count[0] += 1
        return resp

    mock_harness = MagicMock()
    mock_harness.ask.side_effect = _ask_side_effect
    mock_harness._last_latency_ms = 200

    with (
        patch("experiments.pipeline_batch._check_ollama", return_value=True),
        patch("experiments.pipeline_batch._compress_c_file",
              return_value=("text " * 200, 1000, 500, 50.0)),
        patch("experiments.local_llm.LocalLLMHarness", return_value=mock_harness),
    ):
        result = run_pipeline_batch(
            file_path  = "dummy.c",
            questions  = ["Q1?", "Q2?", "Q3?", "Q4?"],
            save       = False,
        )

    assert result["total_queries"] == 4
    assert result["accepted"] == 2
    assert result["rejected"] == 2
    assert abs(result["acceptance_rate"] - 0.5) < 1e-6


# ── Cost formula verification ──────────────────────────────────────────────────

def test_blended_cost_formula():
    """Verify: blended_cost = (1 - acceptance_rate) × compressed_cost."""
    questions = ["Q1?", "Q2?", "Q3?", "Q4?"]
    # 1 accepted, 3 rejected
    answers = [_LONG_RESPONSE, _SHORT_RESPONSE, _SHORT_RESPONSE, _SHORT_RESPONSE]
    call_count = [0]

    def _ask(self_unused, *a, **kw):
        resp = answers[call_count[0] % len(answers)]
        call_count[0] += 1
        return resp

    mock_harness = MagicMock()
    mock_harness.ask.side_effect = lambda *a, **kw: answers[
        (call_count[0] - 1 + len(answers)) % len(answers)
    ]

    # Simpler: use side_effect with a counter
    idx = [0]
    def side_eff(*a, **kw):
        r = answers[idx[0] % len(answers)]
        idx[0] += 1
        return r

    mock_harness2 = MagicMock()
    mock_harness2.ask.side_effect = side_eff
    mock_harness2._last_latency_ms = 100

    compressed_tokens = 600

    with (
        patch("experiments.pipeline_batch._check_ollama", return_value=True),
        patch("experiments.pipeline_batch._compress_c_file",
              return_value=("c " * 300, 1200, compressed_tokens, 50.0)),
        patch("experiments.local_llm.LocalLLMHarness", return_value=mock_harness2),
    ):
        result = run_pipeline_batch(
            file_path = "dummy.c",
            questions = questions,
            save      = False,
        )

    total    = result["total_queries"]
    rejected = result["rejected"]
    ar       = result["acceptance_rate"]
    blended  = result["cost_blended_t1"]
    comp_cost_per_q = compressed_tokens * 0.00025 / 1000

    # Formula: blended = (1 - acceptance_rate) × compressed_cost_per_query
    expected = (1 - ar) * comp_cost_per_q
    assert abs(blended - expected) < 1e-10, (
        f"Blended={blended:.10f}, expected={expected:.10f}"
    )


# ── No Ollama → empty result ───────────────────────────────────────────────────

def test_no_ollama_returns_empty():
    """When Ollama not available, returns dict with error key and total_queries=0."""
    with patch("experiments.pipeline_batch._check_ollama", return_value=False):
        result = run_pipeline_batch(
            file_path = "dummy.c",
            questions = DEFAULT_C_QUESTIONS[:3],
            save      = False,
        )

    assert "error" in result
    assert result["total_queries"] == 0
    assert "ollama" in result["error"].lower() or "Ollama" in result["error"]
