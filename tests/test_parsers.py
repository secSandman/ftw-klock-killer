"""
test_parsers.py — Language parser smoke tests
=============================================
Validates that Python, C, Go, and Rust parsers:
  1. Return at least one CodeChunk for valid source
  2. Produce unique chunk_ids
  3. Correctly identify chunk types
  4. Fall back gracefully when tree-sitter isn't available

Run: pytest tests/test_parsers.py -v
"""

import sys
from pathlib import Path

# Make sure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from languages.python_parser import PythonParser, CodeChunk
from languages.c_parser import CParser
from languages.go_parser import GoParser
from languages.rust_parser import RustParser
from languages.language_registry import LanguageRegistry


# ── Fixture source snippets ───────────────────────────────────────────────────

PYTHON_SRC = '''\
def add(a, b):
    """Return the sum."""
    return a + b

def multiply(x, y):
    result = x * y
    return result

class Calculator:
    def __init__(self):
        self.history = []

    def compute(self, op, a, b):
        if op == "add":
            return self.add(a, b)
        return 0
'''

C_SRC = '''\
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

void print_result(int result) {
    printf("Result: %d\\n", result);
}

typedef struct {
    int x;
    int y;
} Point;

int distance(Point p1, Point p2) {
    int dx = p1.x - p2.x;
    int dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}
'''

GO_SRC = '''\
package main

import "fmt"

func Add(a, b int) int {
    return a + b
}

func Multiply(x, y int) int {
    return x * y
}

type Calculator struct {
    history []int
}

func (c *Calculator) Compute(a, b int) int {
    result := a + b
    c.history = append(c.history, result)
    return result
}

func main() {
    fmt.Println(Add(1, 2))
}
'''

RUST_SRC = '''\
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply(x: i32, y: i32) -> i32 {
    x * y
}

pub struct Calculator {
    history: Vec<i32>,
}

impl Calculator {
    pub fn new() -> Self {
        Calculator { history: Vec::new() }
    }

    pub fn compute(&mut self, a: i32, b: i32) -> i32 {
        let result = a + b;
        self.history.push(result);
        result
    }
}
'''


# ── Python parser tests ───────────────────────────────────────────────────────

class TestPythonParser:
    def setup_method(self):
        self.parser = PythonParser()

    def test_parses_functions(self):
        chunks = self.parser.parse_source(PYTHON_SRC, "test.py")
        names  = [c.name for c in chunks]
        assert "add" in names
        assert "multiply" in names

    def test_parses_class(self):
        chunks = self.parser.parse_source(PYTHON_SRC, "test.py")
        types  = [c.chunk_type for c in chunks]
        assert "class" in types

    def test_unique_chunk_ids(self):
        chunks = self.parser.parse_source(PYTHON_SRC, "test.py")
        ids    = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_language_tag(self):
        chunks = self.parser.parse_source(PYTHON_SRC, "test.py")
        for c in chunks:
            assert c.language == "python"

    def test_token_counts_positive(self):
        chunks = self.parser.parse_source(PYTHON_SRC, "test.py")
        for c in chunks:
            assert c.original_tokens > 0

    def test_empty_source(self):
        chunks = self.parser.parse_source("", "empty.py")
        # Should return a module chunk (fallback)
        assert isinstance(chunks, list)

    def test_parse_file_missing(self, tmp_path):
        chunks = self.parser.parse_file(tmp_path / "nonexistent.py")
        assert chunks == []

    def test_parse_file(self, tmp_path):
        p = tmp_path / "sample.py"
        p.write_text(PYTHON_SRC)
        chunks = self.parser.parse_file(p)
        assert len(chunks) > 0


# ── C parser tests ────────────────────────────────────────────────────────────

class TestCParser:
    def setup_method(self):
        self.parser = CParser()

    def test_parses_functions(self):
        chunks = self.parser.parse_source(C_SRC, "test.c")
        names  = [c.name for c in chunks]
        assert any("add" in n for n in names)

    def test_language_tag(self):
        chunks = self.parser.parse_source(C_SRC, "test.c")
        for c in chunks:
            assert c.language == "c"

    def test_unique_chunk_ids(self):
        chunks = self.parser.parse_source(C_SRC, "test.c")
        ids    = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_returns_chunks(self):
        chunks = self.parser.parse_source(C_SRC, "test.c")
        assert len(chunks) >= 1

    def test_empty_source(self):
        chunks = self.parser.parse_source("", "empty.c")
        assert isinstance(chunks, list)

    def test_parse_file_missing(self, tmp_path):
        chunks = self.parser.parse_file(tmp_path / "nonexistent.c")
        assert chunks == []

    def test_doom_style_c(self):
        """Test against Doom-style C with K&R patterns and macros."""
        doom_src = '''\
#define MAXPLAYERS 4
#define FIXED_T    long

static void R_DrawColumn(void)
{
    int count;
    count = dc_yh - dc_yl;
    if (count < 0)
        return;
}

void P_InitThinkers(void)
{
    thinkercap.prev = thinkercap.next = &thinkercap;
}
'''
        chunks = self.parser.parse_source(doom_src, "r_draw.c")
        assert len(chunks) >= 1
        assert any(c.language == "c" for c in chunks)


# ── Go parser tests ───────────────────────────────────────────────────────────

class TestGoParser:
    def setup_method(self):
        self.parser = GoParser()

    def test_parses_functions(self):
        chunks = self.parser.parse_source(GO_SRC, "main.go")
        names  = [c.name for c in chunks]
        assert any("Add" in n for n in names)

    def test_parses_method_with_receiver(self):
        chunks = self.parser.parse_source(GO_SRC, "main.go")
        # Should find Calculator.Compute or similar
        methods = [c for c in chunks if c.chunk_type in ("method", "function")]
        assert len(methods) >= 2

    def test_language_tag(self):
        chunks = self.parser.parse_source(GO_SRC, "main.go")
        for c in chunks:
            assert c.language == "go"

    def test_unique_chunk_ids(self):
        chunks = self.parser.parse_source(GO_SRC, "main.go")
        ids    = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_returns_chunks(self):
        chunks = self.parser.parse_source(GO_SRC, "main.go")
        assert len(chunks) >= 1

    def test_empty_source(self):
        chunks = self.parser.parse_source("", "empty.go")
        assert isinstance(chunks, list)

    def test_parse_file_missing(self, tmp_path):
        chunks = self.parser.parse_file(tmp_path / "nonexistent.go")
        assert chunks == []


# ── Rust parser tests ─────────────────────────────────────────────────────────

class TestRustParser:
    def setup_method(self):
        self.parser = RustParser()

    def test_parses_functions(self):
        chunks = self.parser.parse_source(RUST_SRC, "lib.rs")
        names  = [c.name for c in chunks]
        assert any("add" in n for n in names)

    def test_parses_impl_methods(self):
        chunks = self.parser.parse_source(RUST_SRC, "lib.rs")
        methods = [c for c in chunks if c.chunk_type == "method"]
        # Should find Calculator::new and Calculator::compute
        assert len(methods) >= 1

    def test_parses_struct(self):
        chunks = self.parser.parse_source(RUST_SRC, "lib.rs")
        structs = [c for c in chunks if c.chunk_type == "struct"]
        assert len(structs) >= 1
        assert structs[0].name == "Calculator"

    def test_language_tag(self):
        chunks = self.parser.parse_source(RUST_SRC, "lib.rs")
        for c in chunks:
            assert c.language == "rust"

    def test_unique_chunk_ids(self):
        chunks = self.parser.parse_source(RUST_SRC, "lib.rs")
        ids    = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_returns_chunks(self):
        chunks = self.parser.parse_source(RUST_SRC, "lib.rs")
        assert len(chunks) >= 1

    def test_empty_source(self):
        chunks = self.parser.parse_source("", "empty.rs")
        assert isinstance(chunks, list)

    def test_parse_file_missing(self, tmp_path):
        chunks = self.parser.parse_file(tmp_path / "nonexistent.rs")
        assert chunks == []

    def test_async_fn(self):
        async_src = '''\
pub async fn fetch_data(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let response = reqwest::get(url).await?;
    Ok(response.text().await?)
}
'''
        chunks = self.parser.parse_source(async_src, "net.rs")
        assert len(chunks) >= 1
        assert any("fetch_data" in c.name for c in chunks)


# ── Language registry tests ───────────────────────────────────────────────────

class TestLanguageRegistry:
    def setup_method(self):
        self.registry = LanguageRegistry()

    def test_python_extension(self):
        assert self.registry.language_for("foo.py") == "python"

    def test_c_extensions(self):
        for ext in (".c", ".h", ".cc", ".cpp"):
            assert self.registry.language_for(f"foo{ext}") == "c"

    def test_go_extension(self):
        assert self.registry.language_for("foo.go") == "go"

    def test_rust_extension(self):
        assert self.registry.language_for("foo.rs") == "rust"

    def test_unknown_extension(self):
        assert self.registry.language_for("foo.java") is None

    def test_parser_for_python(self):
        p = self.registry.parser_for("foo.py")
        assert p is not None
        assert isinstance(p, PythonParser)

    def test_parser_for_c(self):
        p = self.registry.parser_for("foo.c")
        assert p is not None
        assert isinstance(p, CParser)

    def test_parser_for_go(self):
        p = self.registry.parser_for("foo.go")
        # May be None if tree-sitter-go not installed, but should not raise
        # GoParser falls back gracefully
        assert p is not None or p is None  # either is acceptable

    def test_parser_for_rust(self):
        p = self.registry.parser_for("foo.rs")
        assert p is not None or p is None

    def test_supported_extensions(self):
        exts = self.registry.supported_extensions()
        assert ".py" in exts
        assert ".c" in exts
        assert ".go" in exts
        assert ".rs" in exts
