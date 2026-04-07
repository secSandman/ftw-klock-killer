"""
test_llm_caller.py — Unit tests for P0: T0.5 local LLM shim
=============================================================
Tests the _try_local_llm / _local_response_quality_ok / _local_llm_available
functions in isolation. No Anthropic API key required. No Ollama required.

All tests run fully offline:
  - KLOC_LOCAL_LLM_ENABLED=0  disables the Ollama probe entirely
  - Monkeypatching replaces the harness for acceptance-path tests
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Disable local LLM by default so tests never hit a real Ollama instance
os.environ["KLOC_LOCAL_LLM_ENABLED"] = "0"
os.environ["ACTIVE_TIER_MAX"] = "0"

# Import after env is set
import orchestrator.llm_caller as caller


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_cache():
    """Reset the module-level availability cache between tests."""
    caller._local_available_cache = None


# ── _local_response_quality_ok ────────────────────────────────────────────────

class TestLocalResponseQualityOk:
    def test_empty_string_rejected(self):
        assert caller._local_response_quality_ok("") is False

    def test_none_rejected(self):
        assert caller._local_response_quality_ok(None) is False

    def test_too_short_rejected(self):
        assert caller._local_response_quality_ok("hello world") is False

    def test_refusal_i_cannot_rejected(self):
        long_refusal = "i cannot answer this because " + "word " * 20
        assert caller._local_response_quality_ok(long_refusal) is False

    def test_refusal_unable_rejected(self):
        long_refusal = "i'm unable to process " + "word " * 20
        assert caller._local_response_quality_ok(long_refusal) is False

    def test_refusal_no_context_rejected(self):
        long_refusal = "no context provided to answer " + "word " * 20
        assert caller._local_response_quality_ok(long_refusal) is False

    def test_good_response_accepted(self):
        good = (
            "The fibonacci function computes the nth Fibonacci number using "
            "an iterative approach. It initialises a and b to 0 and 1, then "
            "loops n-1 times swapping values. Time complexity is O(n), space O(1)."
        )
        assert caller._local_response_quality_ok(good) is True

    def test_boundary_19_words_rejected(self):
        # check is `< 20`, so 19 words is rejected
        text = " ".join(["word"] * 19)
        assert caller._local_response_quality_ok(text) is False

    def test_boundary_20_words_accepted(self):
        # exactly 20 words passes (not < 20)
        text = " ".join(["word"] * 20)
        assert caller._local_response_quality_ok(text) is True


# ── _local_llm_available ──────────────────────────────────────────────────────

class TestLocalLlmAvailable:
    def setup_method(self):
        _reset_cache()
        os.environ["KLOC_LOCAL_LLM_ENABLED"] = "0"

    def teardown_method(self):
        _reset_cache()
        os.environ["KLOC_LOCAL_LLM_ENABLED"] = "0"

    def test_disabled_returns_false(self):
        # KLOC_LOCAL_LLM_ENABLED=0 — should return False without network probe
        caller._LOCAL_LLM_ENABLED = False
        result = caller._local_llm_available()
        assert result is False

    def test_result_is_cached(self):
        caller._LOCAL_LLM_ENABLED = False
        caller._local_llm_available()
        first = caller._local_available_cache
        caller._local_llm_available()   # second call hits cache
        assert caller._local_available_cache is first

    def test_enabled_but_no_server_returns_false(self):
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = True
        # Patch the module constant directly — it's read at import time, not call time
        original_port = caller._LOCAL_LLM_PORT
        caller._LOCAL_LLM_PORT = 19999  # nothing listening here
        try:
            result = caller._local_llm_available()
        finally:
            caller._LOCAL_LLM_PORT = original_port
        assert result is False

    def test_enabled_mocked_server_returns_true(self):
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = True
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("httpx.get", return_value=mock_response):
            result = caller._local_llm_available()
        assert result is True


# ── _try_local_llm ────────────────────────────────────────────────────────────

class TestTryLocalLlm:
    def setup_method(self):
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = False  # default: offline

    def teardown_method(self):
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = False

    def test_unavailable_returns_none(self):
        pr = {}
        result = caller._try_local_llm("some context", "question?", "explain", pr)
        assert result is None

    def test_unavailable_sets_attempted_false(self):
        pr = {}
        caller._try_local_llm("ctx", "q?", "explain", pr)
        assert pr["local_llm_attempted"] is True
        assert pr["local_llm_accepted"] is False

    def test_unavailable_sets_skip_reason(self):
        pr = {}
        caller._try_local_llm("ctx", "q?", "explain", pr)
        assert pr.get("local_llm_skip_reason")  # non-empty string

    def test_context_too_large_skipped(self):
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = True
        mock_response = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_response):
            caller._local_llm_available()  # prime cache as True

        # Build context larger than _LOCAL_LLM_MAX_CTX (default 8000 words)
        big_ctx = " ".join(["token"] * 9000)
        pr = {}
        result = caller._try_local_llm(big_ctx, "q?", "explain", pr)
        assert result is None
        assert "too large" in pr.get("local_llm_skip_reason", "")

    def test_quality_gate_failure_returns_none(self):
        """Harness returns a short/refusal response → quality gate rejects it."""
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = True

        mock_http = MagicMock(status_code=200)
        mock_harness = MagicMock()
        mock_harness.ask.return_value = "i cannot answer this."

        with patch("httpx.get", return_value=mock_http), \
             patch("experiments.local_llm.LocalLLMHarness", return_value=mock_harness):
            pr = {}
            result = caller._try_local_llm("context", "question?", "explain", pr)

        assert result is None
        assert pr["local_llm_accepted"] is False
        assert pr.get("local_llm_skip_reason") == "quality gate failed"

    def test_acceptance_path(self):
        """Harness returns a good response → accepted, metadata populated."""
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = True

        good_response = (
            "This function computes the nth Fibonacci number iteratively. "
            "It uses O(n) time and O(1) space by maintaining two variables "
            "a and b and swapping them in a loop. Very efficient for large n."
        )

        mock_http = MagicMock(status_code=200)
        mock_harness = MagicMock()
        mock_harness.ask.return_value = good_response

        with patch("httpx.get", return_value=mock_http), \
             patch("experiments.local_llm.LocalLLMHarness", return_value=mock_harness):
            pr = {}
            result = caller._try_local_llm("context", "explain fibonacci", "explain", pr)

        assert result == good_response
        assert pr["local_llm_accepted"] is True
        assert pr["local_llm_model"] == caller._LOCAL_LLM_MODEL
        assert pr["local_llm_tier"] == "T0.5/LOCAL_LLM"

    def test_harness_exception_returns_none(self):
        """If harness.ask() raises, return None gracefully."""
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = True

        mock_http = MagicMock(status_code=200)
        mock_harness = MagicMock()
        mock_harness.ask.side_effect = ConnectionError("Ollama down")

        with patch("httpx.get", return_value=mock_http), \
             patch("experiments.local_llm.LocalLLMHarness", return_value=mock_harness):
            pr = {}
            result = caller._try_local_llm("context", "question?", "explain", pr)

        assert result is None
        assert pr["local_llm_accepted"] is False
        assert "harness error" in pr.get("local_llm_skip_reason", "")


# ── call_llm T0 path (unchanged) ─────────────────────────────────────────────

class TestCallLlmT0Path:
    def test_t0_tier_skips_llm(self):
        """T0 routing should never reach the local shim or Anthropic."""
        pr = {
            "routing_decision": {"tier": 0, "model_id": None},
            "question": "what is this?",
            "task_type": "explain",
            "compressed_chunks": [],
        }
        response = caller.call_llm(pr)
        assert "T0/LOCAL" in response
        # local_llm_attempted should NOT be set — we returned before the shim
        assert "local_llm_attempted" not in pr


# ── Pipeline integration: local_llm keys in result ───────────────────────────

class TestPipelineLocalLlmKeys:
    """
    Verify pipeline.run() always returns local_llm_attempted + local_llm_accepted
    even when running fully offline (ACTIVE_TIER_MAX=0 → T0 → never reaches shim).
    """

    def setup_method(self):
        import tempfile
        from orchestrator.pipeline import Pipeline
        self._tmpdir = tempfile.mkdtemp()
        self.py_file = Path(self._tmpdir) / "sample.py"
        self.py_file.write_text(
            "def add(a, b):\n    return a + b\n"
        )
        self.pipeline = Pipeline(session_id="test_local_llm_keys")

    def test_local_llm_keys_present_offline(self):
        result = self.pipeline.run(str(self.py_file), question="explain add")
        # Keys must always be present so callers can inspect them safely
        assert "local_llm_attempted" in result
        assert "local_llm_accepted" in result

    def test_local_llm_not_accepted_in_t0(self):
        result = self.pipeline.run(str(self.py_file), question="explain add")
        assert result["local_llm_accepted"] is False


# ── P1: RAG context injection ─────────────────────────────────────────────────

class TestBuildRagContext:
    def test_disabled_when_top_k_zero(self):
        original_k = caller._LOCAL_LLM_RAG_TOP_K
        caller._LOCAL_LLM_RAG_TOP_K = 0
        try:
            result = caller._build_rag_context("question?", "my context", "")
            assert result == "my context"
        finally:
            caller._LOCAL_LLM_RAG_TOP_K = original_k

    def test_disabled_when_no_question(self):
        original_k = caller._LOCAL_LLM_RAG_TOP_K
        caller._LOCAL_LLM_RAG_TOP_K = 3
        try:
            result = caller._build_rag_context("", "my context", "")
            assert result == "my context"
        finally:
            caller._LOCAL_LLM_RAG_TOP_K = original_k

    def test_rag_exception_returns_original(self):
        original_k = caller._LOCAL_LLM_RAG_TOP_K
        caller._LOCAL_LLM_RAG_TOP_K = 3
        try:
            with patch("rag.retriever.RAGRetriever", side_effect=ImportError("no rag")):
                result = caller._build_rag_context("explain fibonacci", "ctx", "")
            assert result == "ctx"
        finally:
            caller._LOCAL_LLM_RAG_TOP_K = original_k


# ── P2: Confidence scorer ─────────────────────────────────────────────────────

class TestScoreResponseConfidence:
    def test_empty_question_returns_zero(self):
        assert caller._score_response_confidence("", "some long response here") == 0.0

    def test_empty_response_returns_zero(self):
        assert caller._score_response_confidence("what is fibonacci?", "") == 0.0

    def test_on_topic_response_scores_higher_than_off_topic(self):
        q = "explain fibonacci recursive algorithm"
        on_topic  = "fibonacci uses recursive calls to compute values by summing previous fibonacci numbers"
        off_topic = "the weather today is sunny with clear blue skies"
        assert caller._score_response_confidence(q, on_topic) > \
               caller._score_response_confidence(q, off_topic)

    def test_score_in_valid_range(self):
        score = caller._score_response_confidence("explain this", "this is an explanation")
        assert 0.0 <= score <= 1.0

    def test_quality_threshold_env_respected(self):
        original = caller._LOCAL_LLM_QUALITY_THRESHOLD
        caller._LOCAL_LLM_QUALITY_THRESHOLD = 5
        try:
            # 5-word response should now pass the gate
            assert caller._local_response_quality_ok("one two three four five") is True
        finally:
            caller._LOCAL_LLM_QUALITY_THRESHOLD = original

    def test_acceptance_path_sets_confidence(self):
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = True

        good_response = " ".join(["word"] * 30)
        mock_http = MagicMock(status_code=200)
        mock_harness = MagicMock()
        mock_harness.ask.return_value = good_response

        with patch("httpx.get", return_value=mock_http), \
             patch("experiments.local_llm.LocalLLMHarness", return_value=mock_harness):
            pr = {}
            caller._try_local_llm("context", "explain something", "explain", pr)

        assert "local_llm_confidence" in pr
        assert isinstance(pr["local_llm_confidence"], float)
        caller._LOCAL_LLM_ENABLED = False


# ── P3: Two-stage ─────────────────────────────────────────────────────────────

class TestTryTwoStage:
    def setup_method(self):
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = False
        caller._LOCAL_LLM_TWO_STAGE = False

    def teardown_method(self):
        _reset_cache()
        caller._LOCAL_LLM_ENABLED = False
        caller._LOCAL_LLM_TWO_STAGE = False

    def test_disabled_returns_none(self):
        caller._LOCAL_LLM_TWO_STAGE = False
        pr = {}
        result = caller._try_two_stage("ctx", "refactor this", "refactor", "claude-haiku-4-5", pr)
        assert result is None

    def test_wrong_task_type_returns_none(self):
        caller._LOCAL_LLM_TWO_STAGE = True
        caller._LOCAL_LLM_ENABLED = True
        pr = {}
        result = caller._try_two_stage("ctx", "explain this", "explain", "claude-haiku-4-5", pr)
        assert result is None

    def test_no_local_llm_returns_none(self):
        caller._LOCAL_LLM_TWO_STAGE = True
        caller._LOCAL_LLM_ENABLED = False
        pr = {}
        result = caller._try_two_stage("ctx", "refactor this", "refactor", "claude-haiku-4-5", pr)
        assert result is None
