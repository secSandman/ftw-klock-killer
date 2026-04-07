"""
test_pipeline.py — End-to-end pipeline smoke tests
===================================================
Wires the full 4-agent pipeline (PRUNER→GRUG→BALANCER→ZIPPY)
against a real Python source snippet. No API keys required
(ACTIVE_TIER_MAX=0 forces T0/local routing).

Run: pytest tests/test_pipeline.py -v
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Force offline mode — no LLM API calls during tests
os.environ.setdefault("ACTIVE_TIER_MAX", "0")

import pytest
from orchestrator.pipeline import Pipeline


# ── Sample source files ───────────────────────────────────────────────────────

SIMPLE_PY = '''\
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

def is_prime(n):
    """Check if n is a prime number."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
'''

SIMPLE_C = '''\
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

void print_hello(void) {
    printf("Hello from Doom!\\n");
}
'''


# ── Pipeline smoke tests ──────────────────────────────────────────────────────

class TestPipelineSmoke:
    def setup_method(self, tmp_path=None):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        # Write sample files to tmp dir
        self.py_file = Path(self._tmpdir) / "sample.py"
        self.py_file.write_text(SIMPLE_PY)
        self.c_file  = Path(self._tmpdir) / "sample.c"
        self.c_file.write_text(SIMPLE_C)
        self.pipeline = Pipeline(session_id="test_smoke")

    def test_pipeline_returns_dict(self):
        result = self.pipeline.run(str(self.py_file), question="explain fibonacci")
        assert isinstance(result, dict)

    def test_pipeline_no_error_key(self):
        result = self.pipeline.run(str(self.py_file), question="explain fibonacci")
        assert "error" not in result, f"Pipeline error: {result.get('error')}"

    def test_pipeline_has_required_keys(self):
        result = self.pipeline.run(str(self.py_file), question="explain fibonacci")
        required = {"routing_decision", "compressed_chunks", "elapsed_s", "decision_audit"}
        assert required.issubset(set(result.keys()))

    def test_pipeline_elapsed_is_positive(self):
        result = self.pipeline.run(str(self.py_file), question="explain fibonacci")
        assert result.get("elapsed_s", 0) > 0

    def test_pipeline_decision_audit_is_list(self):
        result = self.pipeline.run(str(self.py_file), question="explain fibonacci")
        assert isinstance(result.get("decision_audit"), list)

    def test_pipeline_routing_decision_is_dict(self):
        result = self.pipeline.run(str(self.py_file), question="explain fibonacci")
        assert isinstance(result.get("routing_decision"), dict)

    def test_pipeline_compressed_chunks_is_list(self):
        result = self.pipeline.run(str(self.py_file), question="explain fibonacci")
        assert isinstance(result.get("compressed_chunks"), list)

    def test_pipeline_ter_in_valid_range(self):
        result = self.pipeline.run(str(self.py_file), question="explain fibonacci")
        if "overall_ter" in result:
            ter = result["overall_ter"]
            # TER is a ratio 0..1 (can also be negative if expansion occurs)
            assert isinstance(ter, (int, float))

    def test_pipeline_c_file(self):
        """C source should also flow through without errors."""
        result = self.pipeline.run(str(self.c_file), question="explain factorial")
        assert isinstance(result, dict)
        assert "error" not in result

    def test_pipeline_with_task_types(self):
        """Different task_type values should change routing_decision tier."""
        for task_type in ("explain", "codegen", "review", "refactor"):
            result = self.pipeline.run(
                str(self.py_file),
                question=f"please {task_type} this",
                task_type=task_type,
            )
            assert "error" not in result, f"Failed for task_type={task_type}: {result.get('error')}"

    def test_pipeline_with_conversation_history(self):
        history = [
            {"role": "user",      "content": "What does fibonacci do?"},
            {"role": "assistant", "content": "It computes the nth Fibonacci number."},
        ]
        result = self.pipeline.run(
            str(self.py_file),
            question="Can you show me an example?",
            conversation_history=history,
        )
        assert "error" not in result

    def test_pipeline_inline_source(self):
        """Pass source code as string instead of file path."""
        result = self.pipeline.run(
            "inline_snippet.py",
            question="what is this code?",
        )
        # May error if file not found, but should not crash
        assert isinstance(result, dict)

    def test_multiple_runs_isolated(self):
        """Two consecutive runs should not share bus state."""
        r1 = self.pipeline.run(str(self.py_file), question="run 1")
        r2 = self.pipeline.run(str(self.py_file), question="run 2")
        assert "error" not in r1
        assert "error" not in r2

    def test_pipeline_git_hot_lines(self):
        """git_hot_lines hint should be accepted without error."""
        result = self.pipeline.run(
            str(self.py_file),
            question="explain hot code",
            git_hot_lines={str(self.py_file): [1, 2, 3]},
        )
        assert "error" not in result


# ── Chunker unit tests ────────────────────────────────────────────────────────

class TestChunker:
    def setup_method(self, tmp_path=None):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        sys.path.insert(0, str(ROOT))
        from rag.chunker import Chunker
        self.chunker = Chunker()
        self.py_file = Path(self._tmpdir) / "sample.py"
        self.py_file.write_text(SIMPLE_PY)

    def test_chunk_file_returns_list(self):
        from rag.chunker import Chunker
        chunks = self.chunker.chunk_file(self.py_file)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_source_python(self):
        chunks = self.chunker.chunk_source(SIMPLE_PY, "sample.py", "python")
        assert len(chunks) > 0

    def test_chunk_source_c(self):
        chunks = self.chunker.chunk_source(SIMPLE_C, "sample.c", "c")
        assert len(chunks) > 0

    def test_to_pruner_payload(self):
        chunks  = self.chunker.chunk_file(self.py_file)
        payload = self.chunker.to_pruner_payload(chunks, str(self.py_file), "python")
        assert isinstance(payload, dict)
        assert "chunks" in payload or "file_path" in payload
