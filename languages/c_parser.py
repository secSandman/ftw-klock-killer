"""
c_parser.py — C/C++ code chunker using tree-sitter
====================================================
Handles: K&R C, C99, C11, GNU extensions, heavy macro code.
Perfect for Doom, Quake, and other GPL video game engines.

tree-sitter grammar: tree-sitter-languages or tree-sitter-c
Install: pip install tree-sitter-languages
         OR: pip install tree-sitter tree-sitter-c

Chunk types extracted:
  - function_definition    → function chunk
  - struct_specifier       → struct chunk
  - preprocessor_function  → macro chunk (large #define blocks)
  - translation_unit       → module fallback
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from languages.python_parser import CodeChunk


# ── Tree-sitter loading (graceful degradation) ─────────────────────────────

_TS_AVAILABLE = False
_C_LANGUAGE   = None

def _load_tree_sitter():
    global _TS_AVAILABLE, _C_LANGUAGE
    try:
        from tree_sitter_languages import get_language
        _C_LANGUAGE = get_language("c")
        _TS_AVAILABLE = True
        return
    except ImportError:
        pass
    try:
        import tree_sitter_c as tsc
        from tree_sitter import Language
        _C_LANGUAGE = Language(tsc.language())
        _TS_AVAILABLE = True
    except ImportError:
        pass

_load_tree_sitter()


class CParser:
    """
    Extracts CodeChunks from C/C++ source.

    Falls back to brace-counting regex if tree-sitter isn't installed.
    The regex fallback handles Doom/Quake-style C reasonably well.
    """

    def parse_file(self, file_path: str | Path) -> List[CodeChunk]:
        path = Path(file_path)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return self.parse_source(source, str(path))

    def parse_source(self, source: str, file_path: str = "<string>") -> List[CodeChunk]:
        if _TS_AVAILABLE and _C_LANGUAGE:
            return self._parse_tree_sitter(source, file_path)
        return self._parse_regex_fallback(source, file_path)

    # ── Tree-sitter path ────────────────────────────────────────────

    def _parse_tree_sitter(self, source: str, file_path: str) -> List[CodeChunk]:
        from tree_sitter import Parser
        parser = Parser()
        parser.set_language(_C_LANGUAGE)
        tree   = parser.parse(source.encode("utf-8", errors="replace"))
        root   = tree.root_node
        chunks: List[CodeChunk] = []
        src_bytes = source.encode("utf-8", errors="replace")

        # Only inspect top-level declarations (direct children of translation_unit).
        # A full BFS would also yield struct_specifiers nested *inside* function bodies
        # (e.g. struct literals, compound initialisers), causing token double-counting.
        for node in self._top_level_nodes(root):
            if node.type == "function_definition":
                chunk = self._fn_from_node(node, src_bytes, source, file_path)
                if chunk:
                    chunks.append(chunk)
            elif node.type in ("struct_specifier", "union_specifier", "enum_specifier"):
                chunk = self._struct_from_node(node, src_bytes, source, file_path)
                if chunk:
                    chunks.append(chunk)
            elif node.type == "declaration":
                # A typedef struct { ... } Name; lives under a declaration node.
                for child in node.children:
                    if child.type in ("struct_specifier", "union_specifier", "enum_specifier"):
                        chunk = self._struct_from_node(child, src_bytes, source, file_path)
                        if chunk:
                            chunks.append(chunk)

        if chunks:
            hdr = self._header_chunk(source, file_path, chunks[0].start_line)
            if hdr:
                chunks.insert(0, hdr)
        return chunks or [self._module_chunk(source, file_path)]

    def _fn_from_node(self, node, src_bytes: bytes, source: str, file_path: str) -> Optional[CodeChunk]:
        # Extract function name from the declarator child
        name = self._extract_fn_name(node, src_bytes)
        if not name:
            return None
        start_line = node.start_point[0] + 1
        end_line   = node.end_point[0] + 1
        fn_src = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        return CodeChunk(
            chunk_id       = f"{Path(file_path).stem}::{name}:{start_line}",
            file_path      = file_path,
            language       = "c",
            chunk_type     = "function",
            name           = name,
            start_line     = start_line,
            end_line       = end_line,
            source         = fn_src,
            original_tokens= self._rough_tokens(fn_src),
        )

    def _struct_from_node(self, node, src_bytes: bytes, source: str, file_path: str) -> Optional[CodeChunk]:
        start_line = node.start_point[0] + 1
        end_line   = node.end_point[0] + 1
        st_src = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        # Try to get struct name
        name = "anonymous_struct"
        for child in node.children:
            if child.type == "type_identifier":
                name = src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                break
        return CodeChunk(
            chunk_id       = f"{Path(file_path).stem}::{name}:{start_line}",
            file_path      = file_path,
            language       = "c",
            chunk_type     = "struct",
            name           = name,
            start_line     = start_line,
            end_line       = end_line,
            source         = st_src,
            original_tokens= self._rough_tokens(st_src),
        )

    def _extract_fn_name(self, node, src_bytes: bytes) -> Optional[str]:
        """Navigate tree-sitter node tree to find the function name identifier."""
        for child in node.children:
            if child.type in ("function_declarator", "pointer_declarator"):
                return self._extract_fn_name(child, src_bytes)
            if child.type == "identifier":
                return src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        return None

    @staticmethod
    def _top_level_nodes(root):
        """Yield only direct children of the translation_unit root.

        Avoids double-counting tokens from nested nodes (e.g. struct specifiers
        that appear inside function bodies as compound initialisers).
        """
        yield from root.children

    @staticmethod
    def _walk(node):
        """BFS walk of tree-sitter node tree (kept for potential future use)."""
        queue = [node]
        while queue:
            current = queue.pop(0)
            yield current
            queue.extend(current.children)

    # ── Regex fallback ───────────────────────────────────────────────

    def _parse_regex_fallback(self, source: str, file_path: str) -> List[CodeChunk]:
        """
        Regex-based C function extractor.
        Handles most Doom/Quake-style C without tree-sitter.
        Finds: return_type function_name(params) { ... }
        """
        chunks: List[CodeChunk] = []
        lines  = source.splitlines()

        # Pattern: optional return type, function name, parentheses, opening brace
        fn_pattern = re.compile(
            r'^(?:static\s+|extern\s+|inline\s+|void\s+|int\s+|char\s+|'
            r'float\s+|double\s+|long\s+|unsigned\s+)?'
            r'(?:\w+\s+\*?\s*)?(\w+)\s*\([^;{]*\)\s*\{',
            re.MULTILINE,
        )

        for m in fn_pattern.finditer(source):
            name       = m.group(1)
            if name in ("if", "while", "for", "switch", "else", "do"):
                continue  # skip control flow keywords
            start_byte = m.start()
            start_line = source[:start_byte].count("\n") + 1
            # Extract body by counting braces
            body_src, end_line = self._extract_brace_body(source, m.start(), lines)
            if body_src is None:
                continue
            chunks.append(CodeChunk(
                chunk_id       = f"{Path(file_path).stem}::{name}:{start_line}",
                file_path      = file_path,
                language       = "c",
                chunk_type     = "function",
                name           = name,
                start_line     = start_line,
                end_line       = end_line,
                source         = body_src,
                original_tokens= self._rough_tokens(body_src),
            ))

        if chunks:
            hdr = self._header_chunk(source, file_path, chunks[0].start_line)
            if hdr:
                chunks.insert(0, hdr)
        return chunks or [self._module_chunk(source, file_path)]

    @staticmethod
    def _extract_brace_body(source: str, start: int, lines: List[str]) -> tuple:
        """Extract balanced brace body starting from `start` position."""
        depth  = 0
        i      = start
        in_str = False
        str_ch = None
        while i < len(source):
            ch = source[i]
            if in_str:
                if ch == str_ch and source[i-1:i] != "\\":
                    in_str = False
            elif ch in ('"', "'"):
                in_str = True
                str_ch = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    body = source[start:i+1]
                    end_line = source[:i+1].count("\n") + 1
                    return body, end_line
            i += 1
        return None, -1

    def _header_chunk(
        self, source: str, file_path: str, first_fn_line: int
    ) -> Optional[CodeChunk]:
        """
        Extract the file preamble (lines 1..first_fn_line-1) as a 'header' chunk.
        Only created when the header contains at least one #include or #define
        and has >= 3 lines.
        """
        if first_fn_line <= 1:
            return None
        lines = source.splitlines(keepends=True)
        hdr_lines = lines[: first_fn_line - 1]
        hdr_src = "".join(hdr_lines).rstrip("\n") + "\n"
        n = len([l for l in hdr_lines if l.strip()])
        if n < 3:
            return None
        if not re.search(r'#\s*(include|define)', hdr_src):
            return None
        return CodeChunk(
            chunk_id       = f"{Path(file_path).stem}::__header__:1",
            file_path      = file_path,
            language       = "c",
            chunk_type     = "header",
            name           = "__header__",
            start_line     = 1,
            end_line       = first_fn_line - 1,
            source         = hdr_src,
            original_tokens= self._rough_tokens(hdr_src),
        )

    def _module_chunk(self, source: str, file_path: str) -> CodeChunk:
        n_lines = len(source.splitlines())
        return CodeChunk(
            chunk_id       = f"{Path(file_path).stem}::module:1",
            file_path      = file_path,
            language       = "c",
            chunk_type     = "module",
            name           = Path(file_path).stem,
            start_line     = 1,
            end_line       = n_lines,
            source         = source,
            original_tokens= self._rough_tokens(source),
        )

    @staticmethod
    def _rough_tokens(text: str) -> int:
        return len(re.findall(r'\S+', text))
