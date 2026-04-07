"""
python_parser.py — Python AST-based code chunker
Wraps the existing ast module. No tree-sitter needed for Python.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class CodeChunk:
    """A self-contained unit of source code for embedding + compression."""
    chunk_id:       str
    file_path:      str
    language:       str
    chunk_type:     str          # "function" | "class" | "module" | "struct"
    name:           str
    start_line:     int
    end_line:       int
    source:         str
    original_tokens: int = 0
    parent_name:    Optional[str] = None


class PythonParser:
    """Extracts CodeChunks from Python source via ast module."""

    def parse_file(self, file_path: str | Path) -> List[CodeChunk]:
        path = Path(file_path)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return self.parse_source(source, str(path))

    def parse_source(self, source: str, file_path: str = "<string>") -> List[CodeChunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fallback: return the whole file as one module chunk
            return [self._module_chunk(source, file_path)]

        chunks: List[CodeChunk] = []
        lines  = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._fn_chunk(node, lines, file_path)
                chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._class_chunk(node, lines, file_path)
                chunks.append(chunk)

        if not chunks:
            chunks.append(self._module_chunk(source, file_path))

        return chunks

    # ── Chunk builders ────────────────────────────────────────────────

    def _fn_chunk(self, node: ast.FunctionDef, lines: List[str], file_path: str) -> CodeChunk:
        start = node.lineno - 1
        end   = getattr(node, "end_lineno", node.lineno) - 1
        src   = "\n".join(lines[start:end + 1])
        return CodeChunk(
            chunk_id       = f"{Path(file_path).stem}::{node.name}:{node.lineno}",
            file_path      = file_path,
            language       = "python",
            chunk_type     = "function",
            name           = node.name,
            start_line     = node.lineno,
            end_line       = getattr(node, "end_lineno", node.lineno),
            source         = src,
            original_tokens= self._rough_tokens(src),
        )

    def _class_chunk(self, node: ast.ClassDef, lines: List[str], file_path: str) -> CodeChunk:
        start = node.lineno - 1
        end   = getattr(node, "end_lineno", node.lineno) - 1
        src   = "\n".join(lines[start:end + 1])
        return CodeChunk(
            chunk_id       = f"{Path(file_path).stem}::{node.name}:{node.lineno}",
            file_path      = file_path,
            language       = "python",
            chunk_type     = "class",
            name           = node.name,
            start_line     = node.lineno,
            end_line       = getattr(node, "end_lineno", node.lineno),
            source         = src,
            original_tokens= self._rough_tokens(src),
        )

    def _module_chunk(self, source: str, file_path: str) -> CodeChunk:
        n_lines = len(source.splitlines())
        return CodeChunk(
            chunk_id       = f"{Path(file_path).stem}::module:1",
            file_path      = file_path,
            language       = "python",
            chunk_type     = "module",
            name           = Path(file_path).stem,
            start_line     = 1,
            end_line       = n_lines,
            source         = source,
            original_tokens= self._rough_tokens(source),
        )

    @staticmethod
    def _rough_tokens(text: str) -> int:
        import re
        return len(re.findall(r'\S+', text))
