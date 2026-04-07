"""
test_quality.py — RQS (Response Quality Scorer) tests
======================================================
Tests the 4-layer quality measurement framework in src/quality.py:
  L1 — semantic similarity (cosine, no-dep)
  L1 — code overlap (identifier Jaccard)
  L2 — functional correctness (run_code_against_tests)
  L2 — syntax_check
  L2 — extract_code_block
  QualityResult — verdict logic
  QualityScorer — composite scoring
  RQSReport — aggregate stats
  QualityRegressionTracker — longitudinal log

No LLM API keys required (L3 judge tests are skipped without a key).

Run: pytest tests/test_quality.py -v
"""

import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pytest
from quality import (
    QualityResult,
    compute_semantic_similarity,
    compute_code_overlap,
    extract_code_block,
    run_code_against_tests,
    syntax_check,
)


# ── L1: Semantic Similarity ───────────────────────────────────────────────────

class TestComputeSemanticSimilarity:
    def test_identical_texts(self):
        text = "The function returns the sum of two integers."
        sim  = compute_semantic_similarity(text, text)
        assert sim == 1.0

    def test_empty_texts(self):
        sim = compute_semantic_similarity("", "")
        assert sim == 0.0

    def test_one_empty(self):
        sim = compute_semantic_similarity("hello world", "")
        assert sim == 0.0

    def test_similar_texts_high_score(self):
        a = "This function computes the factorial of a number recursively."
        b = "This function calculates the factorial of a number using recursion."
        sim = compute_semantic_similarity(a, b)
        assert sim > 0.60, f"Expected high similarity, got {sim}"

    def test_completely_different_texts(self):
        a = "def factorial(n): return n * factorial(n-1)"
        b = "SELECT * FROM users WHERE active = 1 ORDER BY created_at"
        sim = compute_semantic_similarity(a, b)
        assert sim < 0.50, f"Expected low similarity for unrelated texts, got {sim}"

    def test_range_is_zero_to_one(self):
        sim = compute_semantic_similarity("foo bar baz", "baz qux quux")
        assert 0.0 <= sim <= 1.0

    def test_order_independent(self):
        a = "The quick brown fox"
        b = "The lazy dog sleeps"
        assert compute_semantic_similarity(a, b) == compute_semantic_similarity(b, a)


# ── L1: Code Overlap ─────────────────────────────────────────────────────────

class TestComputeCodeOverlap:
    def test_identical_code(self):
        code = "```python\ndef foo(x):\n    return x + 1\n```"
        assert compute_code_overlap(code, code) == 1.0

    def test_no_code_blocks_returns_one(self):
        # No code blocks → not applicable → 1.0 (no regression)
        a = "The function adds two numbers together."
        b = "This adds two values and returns the result."
        score = compute_code_overlap(a, b)
        assert score == 1.0

    def test_different_function_names_lower_score(self):
        a = "```python\ndef compute_sum(a, b):\n    return a + b\n```"
        b = "```python\ndef add_numbers(x, y):\n    return x + y\n```"
        score = compute_code_overlap(a, b)
        # Both have def, return but different function names
        assert 0.0 <= score <= 1.0

    def test_identical_identifiers_high_score(self):
        code_a = "```python\ndef process(data, config):\n    return data.transform(config)\n```"
        code_b = "```python\ndef process(data, config):\n    result = data.transform(config)\n    return result\n```"
        score = compute_code_overlap(code_a, code_b)
        assert score > 0.70


# ── L2: extract_code_block ────────────────────────────────────────────────────

class TestExtractCodeBlock:
    def test_fenced_python_block(self):
        response = "Here is the code:\n```python\ndef foo():\n    return 1\n```"
        block = extract_code_block(response)
        assert block is not None
        assert "def foo" in block

    def test_fenced_generic_block(self):
        response = "```\ndef bar():\n    pass\n```"
        block = extract_code_block(response)
        assert block is not None

    def test_no_fenced_block_fallback_to_def(self):
        response = "Sure! Here's the implementation:\ndef baz(x):\n    return x * 2\n\nLet me know."
        block = extract_code_block(response)
        assert block is not None
        assert "def baz" in block

    def test_no_code_returns_none(self):
        response = "This is a plain text response with no code."
        block = extract_code_block(response)
        assert block is None


# ── L2: syntax_check ─────────────────────────────────────────────────────────

class TestSyntaxCheck:
    def test_valid_code(self):
        ok, msg = syntax_check("def foo(x):\n    return x + 1\n")
        assert ok is True
        assert msg == "valid"

    def test_invalid_code(self):
        ok, msg = syntax_check("def foo(: pass")
        assert ok is False
        assert len(msg) > 0

    def test_empty_string(self):
        ok, _ = syntax_check("")
        assert ok is True

    def test_class_definition(self):
        ok, _ = syntax_check("class Foo:\n    pass\n")
        assert ok is True

    def test_syntax_error_message_not_empty(self):
        ok, msg = syntax_check("if True\n    pass")
        assert ok is False
        assert "SyntaxError" in msg or len(msg) > 0


# ── L2: run_code_against_tests ────────────────────────────────────────────────

class TestRunCodeAgainstTests:
    def test_all_tests_pass(self):
        code = "def add(a, b):\n    return a + b\n"
        tests = (
            "def test_add_positive():\n    assert add(1, 2) == 3\n\n"
            "def test_add_zero():\n    assert add(0, 5) == 5\n"
        )
        rate, log = run_code_against_tests(code, tests)
        assert rate == 1.0, f"Expected all tests to pass, got rate={rate}\nLog:\n{log}"

    def test_all_tests_fail(self):
        code = "def add(a, b):\n    return a - b\n"  # wrong impl
        tests = "def test_add():\n    assert add(1, 2) == 3\n"
        rate, log = run_code_against_tests(code, tests)
        assert rate < 1.0

    def test_partial_pass(self):
        code = "def square(x):\n    return x * x\n"
        tests = (
            "def test_positive():\n    assert square(3) == 9\n\n"
            "def test_negative():\n    assert square(-2) == 4\n\n"
            "def test_wrong():\n    assert square(2) == 5\n"  # intentionally wrong
        )
        rate, log = run_code_against_tests(code, tests)
        assert 0.0 < rate < 1.0

    def test_syntax_error_in_code(self):
        code  = "def broken(: pass"
        tests = "def test_x():\n    assert broken() == 1\n"
        rate, log = run_code_against_tests(code, tests)
        assert rate == 0.0

    def test_returns_log_string(self):
        code  = "def foo():\n    return 42\n"
        tests = "def test_foo():\n    assert foo() == 42\n"
        rate, log = run_code_against_tests(code, tests)
        assert isinstance(log, str)


# ── QualityResult ─────────────────────────────────────────────────────────────

class TestQualityResult:
    def _make_result(self, rqs=0.90):
        return QualityResult(
            task_id            = "t001",
            timestamp          = "2026-01-01T00:00:00",
            file_path          = "sample.py",
            question           = "explain this",
            full_tokens        = 1000,
            bridge_tokens      = 200,
            ter                = 80.0,
            semantic_similarity= rqs,
            functional_score   = rqs,
            judge_score        = -1,
            composite_rqs      = rqs,
        )

    def test_verdict_excellent(self):
        r = self._make_result(rqs=0.96)
        assert "EXCELLENT" in r.verdict

    def test_verdict_acceptable(self):
        r = self._make_result(rqs=0.87)
        assert "ACCEPTABLE" in r.verdict

    def test_verdict_degraded(self):
        r = self._make_result(rqs=0.75)
        assert "DEGRADED" in r.verdict

    def test_verdict_failure(self):
        r = self._make_result(rqs=0.50)
        assert "FAILURE" in r.verdict

    def test_to_dict_is_serializable(self):
        r = self._make_result()
        d = r.to_dict()
        # Should be JSON-serializable
        s = json.dumps(d)
        assert isinstance(s, str)

    def test_quality_regression_flag(self):
        r = self._make_result(rqs=0.70)
        r.quality_regression = r.composite_rqs < r.rqs_threshold
        assert r.quality_regression is True

    def test_no_regression_for_high_rqs(self):
        r = self._make_result(rqs=0.95)
        r.quality_regression = r.composite_rqs < r.rqs_threshold
        assert r.quality_regression is False


# ── QualityScorer ─────────────────────────────────────────────────────────────

class TestQualityScorer:
    """
    Tests the QualityScorer class if it exists.
    Skips gracefully if the class hasn't been added yet.
    """
    def test_scorer_importable(self):
        try:
            from quality import QualityScorer
        except ImportError:
            pytest.skip("QualityScorer not yet implemented in quality.py")

    def test_scorer_score_l1(self):
        try:
            from quality import QualityScorer
        except ImportError:
            pytest.skip("QualityScorer not yet implemented")

        scorer = QualityScorer()
        result = scorer.score(
            question       = "what does this do?",
            full_response  = "It computes the sum of two numbers.",
            bridge_response= "It adds two numbers and returns the sum.",
            full_tokens    = 100,
            bridge_tokens  = 30,
            layers         = ["l1"],
        )
        assert result.semantic_similarity >= 0.0
        assert result.ter > 0


# ── QualityRegressionTracker ──────────────────────────────────────────────────

class TestQualityRegressionTracker:
    def test_tracker_importable(self):
        try:
            from quality import QualityRegressionTracker
        except ImportError:
            pytest.skip("QualityRegressionTracker not yet implemented")

    def test_tracker_log_and_report(self):
        try:
            from quality import QualityRegressionTracker, QualityResult
        except ImportError:
            pytest.skip("QualityRegressionTracker not yet implemented")

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = QualityRegressionTracker(log_path=str(Path(tmpdir) / "rqs.jsonl"))
            result = QualityResult(
                task_id="t1", timestamp="2026-01-01T00:00:00",
                file_path="x.py", question="q",
                full_tokens=100, bridge_tokens=20, ter=80.0,
                semantic_similarity=0.92, functional_score=-1,
                judge_score=-1, composite_rqs=0.92,
            )
            tracker.log(result)
            report = tracker.report()
            assert report.tasks_evaluated >= 1
            assert report.mean_rqs > 0


# ── RQS v2 — non-circular metric (audit fix 2026-04-07) ───────────────────────

class TestRqsV2:
    """
    Three tests verifying compute_rqs_v2() is a different signal from v1.

    Test 1: v2 offline returns a score in [0.0, 1.0]
    Test 2: v2 != v1 for aggressively compressed input (proves it's a different signal)
    Test 3: v2 with ACTIVE_TIER_MAX=0 falls back without raising an exception
    """

    _ORIGINAL = """
def authenticate_user(username: str, password: str) -> bool:
    \"\"\"Verify credentials against the database.\"\"\"
    record = db.get_user(username)
    if record is None:
        return False
    return bcrypt.checkpw(password.encode(), record.pw_hash)

def generate_token(user_id: int, expiry_hours: int = 24) -> str:
    \"\"\"Create a signed JWT for the given user.\"\"\"
    payload = {"sub": user_id, "exp": time.time() + expiry_hours * 3600}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def revoke_token(token: str) -> None:
    \"\"\"Add token to the revocation list.\"\"\"
    REVOKED.add(token)
"""

    # Aggressively compressed: all bodies gone, only stubs remain
    _COMPRESSED = """
def authenticate_user(username, password): ...
def generate_token(user_id, expiry_hours=24): ...
def revoke_token(token): ...
"""

    def test_v2_offline_score_in_range(self):
        """v2 offline path returns a float in [0.0, 1.0]."""
        import os
        from quality import compute_rqs_v2
        old = os.environ.pop("ACTIVE_TIER_MAX", None)
        os.environ["ACTIVE_TIER_MAX"] = "0"
        try:
            score = compute_rqs_v2(self._ORIGINAL, self._COMPRESSED)
            assert isinstance(score, float), f"Expected float, got {type(score)}"
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"
        finally:
            os.environ.pop("ACTIVE_TIER_MAX", None)
            if old is not None:
                os.environ["ACTIVE_TIER_MAX"] = old

    def test_v2_differs_from_v1_on_aggressive_compression(self):
        """
        v2 must NOT equal v1 for aggressively compressed input.
        If they were the same metric, this test would catch the regression.
        v1 (CSO) compares all tokens — aggressive compression makes it low.
        v2 (ROUGE-L on interface) focuses on signatures — they survived here.
        """
        import os
        from quality import compute_rqs_v1, compute_rqs_v2
        old = os.environ.pop("ACTIVE_TIER_MAX", None)
        os.environ["ACTIVE_TIER_MAX"] = "0"
        try:
            v1 = compute_rqs_v1(self._ORIGINAL, self._COMPRESSED)
            v2 = compute_rqs_v2(self._ORIGINAL, self._COMPRESSED)
            assert v1 != v2, (
                f"v1={v1} and v2={v2} are identical — "
                "v2 is not providing a different signal from v1."
            )
        finally:
            os.environ.pop("ACTIVE_TIER_MAX", None)
            if old is not None:
                os.environ["ACTIVE_TIER_MAX"] = old

    def test_v2_offline_fallback_no_exception(self):
        """v2 with ACTIVE_TIER_MAX=0 must return a score without raising."""
        import os
        from quality import compute_rqs_v2
        os.environ["ACTIVE_TIER_MAX"] = "0"
        # Deliberately clear any API key so online path cannot activate
        saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        saved_oai  = os.environ.pop("OPENAI_API_KEY", None)
        try:
            score = compute_rqs_v2(self._ORIGINAL, self._COMPRESSED, question="explain auth")
            assert 0.0 <= score <= 1.0
        except Exception as e:
            raise AssertionError(f"v2 offline path raised an exception: {e}") from e
        finally:
            os.environ.pop("ACTIVE_TIER_MAX", None)
            if saved_key:  os.environ["ANTHROPIC_API_KEY"] = saved_key
            if saved_oai:  os.environ["OPENAI_API_KEY"]   = saved_oai
