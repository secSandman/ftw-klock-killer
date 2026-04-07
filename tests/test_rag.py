"""
test_rag.py — RAG layer smoke tests
=====================================
Tests hasher, rainbow table lookup, and ETL utilities.
No Qdrant required (mock or skip).

Run: pytest tests/test_rag.py -v
"""

import sys
import json
import hashlib
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from rag.hasher import Hasher
from rainbow.rainbow_table import RainbowTable


# ── Hasher tests ──────────────────────────────────────────────────────────────

class TestHasher:
    def setup_method(self):
        self.hasher = Hasher()

    def test_hash_returns_hex_string(self):
        h = self.hasher.hash("hello world")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_hash_deterministic(self):
        a = self.hasher.hash("test input")
        b = self.hasher.hash("test input")
        assert a == b

    def test_hash_different_inputs_differ(self):
        a = self.hasher.hash("input one")
        b = self.hasher.hash("input two")
        assert a != b

    def test_hash_empty_string(self):
        # SHA-256 of empty string is well-known
        expected = hashlib.sha256(b"").hexdigest()
        h = self.hasher.hash("")
        assert h == expected

    def test_hash_chunk_returns_dict(self):
        from languages.python_parser import PythonParser
        parser = PythonParser()
        chunks = parser.parse_source("def foo(): return 1", "foo.py")
        result = self.hasher.hash_chunk(chunks[0])
        assert isinstance(result, dict)
        assert "sha256" in result
        assert "rainbow_hit" in result

    def test_lookup_miss(self):
        result = self.hasher.lookup("definitely_not_in_rainbow_" + "x" * 30)
        assert result is None

    def test_hash_unicode(self):
        h = self.hasher.hash("日本語テスト")
        assert isinstance(h, str)
        assert len(h) == 64


# ── RainbowTable tests ────────────────────────────────────────────────────────

class TestRainbowTable:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def _make_table_with_entries(self, entries):
        """Create a temporary rainbow JSONL file with given entries."""
        path = Path(self._tmpdir) / "rainbow_test.jsonl"
        with path.open("w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return RainbowTable(table_paths=[str(path)])

    def test_lookup_hit(self):
        sha = hashlib.sha256(b"print(x)").hexdigest()
        table = self._make_table_with_entries([
            {"hash": sha, "original": "print(x)", "compressed": "print x"}
        ])
        result = table.lookup(sha)
        assert result == "print x"

    def test_lookup_miss(self):
        table = self._make_table_with_entries([])
        result = table.lookup("0" * 64)
        assert result is None

    def test_add_and_lookup(self):
        table = self._make_table_with_entries([])
        sha = hashlib.sha256(b"new entry").hexdigest()
        table.add(sha, "new_compressed")
        result = table.lookup(sha)
        assert result == "new_compressed"

    def test_empty_table(self):
        table = self._make_table_with_entries([])
        assert table.lookup("a" * 64) is None

    def test_multiple_entries(self):
        entries = []
        for i in range(10):
            text = f"pattern_{i}"
            sha  = hashlib.sha256(text.encode()).hexdigest()
            entries.append({"hash": sha, "original": text, "compressed": f"p{i}"})

        table = self._make_table_with_entries(entries)
        for i in range(10):
            text = f"pattern_{i}"
            sha  = hashlib.sha256(text.encode()).hexdigest()
            assert table.lookup(sha) == f"p{i}"

    def test_invalid_jsonl_line_skipped(self):
        """Corrupted lines should be skipped, not crash the loader."""
        path = Path(self._tmpdir) / "corrupt.jsonl"
        with path.open("w") as f:
            f.write('{"hash": "abc", "compressed": "ok"}\n')
            f.write("NOT JSON {{{{{\n")
            sha = hashlib.sha256(b"good").hexdigest()
            f.write(json.dumps({"hash": sha, "original": "good", "compressed": "good_c"}) + "\n")

        table = RainbowTable(table_paths=[str(path)])
        assert table.lookup(sha) == "good_c"


# ── Build rainbow smoke test ──────────────────────────────────────────────────

class TestBuildRainbow:
    def test_build_rainbow_produces_jsonl(self):
        """Run build_rainbow.py and check output files exist in data/."""
        import subprocess
        script = ROOT / "rainbow" / "build_rainbow.py"
        if not script.exists():
            pytest.skip("build_rainbow.py not found")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        # Should complete without error
        assert result.returncode == 0, f"build_rainbow.py failed:\n{result.stderr}"

        # Check output files exist in data/
        out_py = ROOT / "data" / "rainbow_stdlib_python.jsonl"
        out_c  = ROOT / "data" / "rainbow_stdlib_c.jsonl"
        assert out_py.exists(), "rainbow_stdlib_python.jsonl not created"
        assert out_c.exists(),  "rainbow_stdlib_c.jsonl not created"

        # Validate structure of first entry in Python rainbow
        first_line = out_py.read_text().splitlines()[0]
        entry = json.loads(first_line)
        assert "hash" in entry
        assert "compressed" in entry
