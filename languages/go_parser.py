"""
go_parser.py — Go source code chunker using tree-sitter
========================================================
Handles: Go functions, methods, interfaces, structs.
Grounded in Go's explicit error-handling and interface-based design.

tree-sitter grammar: tree-sitter-languages or tree-sitter-go
Install: pip install tree-sitter-languages
         OR: pip install tree-sitter tree-sitter-go

Chunk types extracted:
  - function_declaration    → function chunk
  - method_declaration      → method chunk (receiver + func)
  - type_declaration        → struct/interface chunk
  - source_file             → module fallback
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from languages.python_parser import CodeChunk


# ── Tree-sitter loading (graceful degradation) ─────────────────────────────

_TS_AVAILABLE = False
_GO_LANGUAGE  = None


def _load_tree_sitter():
    global _TS_AVAILABLE, _GO_LANGUAGE
    try:
        from tree_sitter_languages import get_language
        _GO_LANGUAGE = get_language("go")
        _TS_AVAILABLE = True
        return
    except ImportError:
        pass
    try:
        import tree_sitter_go as tsg
        from tree_sitter import Language
        _GO_LANGUAGE = Language(tsg.language())
        _TS_AVAILABLE = True
    except ImportError:
        pass


_load_tree_sitter()


class GoParser:
    """
    Extracts CodeChunks from Go source files.

    Falls back to brace-counting regex if tree-sitter isn't installed.
    Handles idiomatic Go: receivers, multiple return values, interfaces.
    """

    def parse_file(self, file_path: str | Path) -> List[CodeChunk]:
        path = Path(file_path)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return self.parse_source(source, str(path))

    def parse_source(self, source: str, file_path: str = "<string>") -> List[CodeChunk]:
        if _TS_AVAILABLE and _GO_LANGUAGE:
            return self._parse_tree_sitter(source, file_path)
        return self._parse_regex_fallback(source, file_path)

    # ── Tree-sitter path ────────────────────────────────────────────

    def _parse_tree_sitter(self, source: str, file_path: str) -> List[CodeChunk]:
        from tree_sitter import Parser
        parser = Parser()
        parser.set_language(_GO_LANGUAGE)
        tree      = parser.parse(source.encode("utf-8", errors="replace"))
        root      = tree.root_node
        chunks: List[CodeChunk] = []
        src_bytes = source.encode("utf-8", errors="replace")

        for node in self._walk(root):
            if node.type == "function_declaration":
                chunk = self._fn_from_node(node, src_bytes, file_path, chunk_type="function")
                if chunk:
                    chunks.append(chunk)
            elif node.type == "method_declaration":
                chunk = self._method_from_node(node, src_bytes, file_path)
                if chunk:
                    chunks.append(chunk)
            elif node.type == "type_declaration":
                chunk = self._type_from_node(node, src_bytes, file_path)
                if chunk:
                    chunks.append(chunk)

        return chunks or [self._module_chunk(source, file_path)]

    def _fn_from_node(self, node, src_bytes: bytes, file_path: str,
                       chunk_type: str = "function") -> Optional[CodeChunk]:
        name = None
        for child in node.children:
            if child.type == "identifier":
                name = src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                break
        if not name:
            return None
        fn_src     = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        start_line = node.start_point[0] + 1
        end_line   = node.end_point[0] + 1
        return CodeChunk(
            chunk_id        = f"{Path(file_path).stem}::{name}:{start_line}",
            file_path       = file_path,
            language        = "go",
            chunk_type      = chunk_type,
            name            = name,
            start_line      = start_line,
            end_line        = end_line,
            source          = fn_src,
            original_tokens = self._rough_tokens(fn_src),
        )

    def _method_from_node(self, node, src_bytes: bytes, file_path: str) -> Optional[CodeChunk]:
        """Methods have a receiver + name; emit as 'method' type."""
        name       = None
        receiver   = None
        for child in node.children:
            if child.type == "parameter_list" and receiver is None:
                # First parameter_list is the receiver
                receiver_src = src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                # Extract type name: (*Foo) or (Foo)
                m = re.search(r'\*?(\w+)', receiver_src)
                receiver = m.group(1) if m else "T"
            elif child.type == "field_identifier" and name is None:
                name = src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")

        if not name:
            return None

        qualified  = f"{receiver}.{name}" if receiver else name
        fn_src     = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        start_line = node.start_point[0] + 1
        end_line   = node.end_point[0] + 1
        return CodeChunk(
            chunk_id        = f"{Path(file_path).stem}::{qualified}:{start_line}",
            file_path       = file_path,
            language        = "go",
            chunk_type      = "method",
            name            = qualified,
            start_line      = start_line,
            end_line        = end_line,
            source          = fn_src,
            original_tokens = self._rough_tokens(fn_src),
        )

    def _type_from_node(self, node, src_bytes: bytes, file_path: str) -> Optional[CodeChunk]:
        """type Foo struct { ... } or type Bar interface { ... }"""
        name = None
        for child in node.children:
            if child.type == "type_spec":
                for sub in child.children:
                    if sub.type == "type_identifier":
                        name = src_bytes[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                        break
            if name:
                break
        if not name:
            return None

        ty_src     = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        start_line = node.start_point[0] + 1
        end_line   = node.end_point[0] + 1
        return CodeChunk(
            chunk_id        = f"{Path(file_path).stem}::{name}:{start_line}",
            file_path       = file_path,
            language        = "go",
            chunk_type      = "type",
            name            = name,
            start_line      = start_line,
            end_line        = end_line,
            source          = ty_src,
            original_tokens = self._rough_tokens(ty_src),
        )

    @staticmethod
    def _walk(node):
        queue = [node]
        while queue:
            current = queue.pop(0)
            yield current
            queue.extend(current.children)

    # ── Regex fallback ───────────────────────────────────────────────

    def _parse_regex_fallback(self, source: str, file_path: str) -> List[CodeChunk]:
        """
        Regex-based Go function/method extractor.
        Captures:
          func FuncName(...)           — top-level function
          func (r RecvType) MethodName — method with receiver
        """
        chunks: List[CodeChunk] = []
        lines  = source.splitlines()

        # Match: func [(<receiver>)] <Name>(<params>) [<returns>] {
        fn_pattern = re.compile(
            r'^func\s+'
            r'(?:\(\s*\w+\s+\*?(\w+)\s*\)\s+)?'   # optional receiver group 1
            r'(\w+)\s*\(',                           # function name group 2
            re.MULTILINE,
        )

        for m in fn_pattern.finditer(source):
            receiver  = m.group(1)
            func_name = m.group(2)
            name      = f"{receiver}.{func_name}" if receiver else func_name

            # Find the opening brace of this function
            brace_pos = source.find("{", m.end())
            if brace_pos == -1:
                continue
            start_byte  = m.start()
            start_line  = source[:start_byte].count("\n") + 1
            body_src, end_line = self._extract_brace_body(source, brace_pos, lines)
            if body_src is None:
                continue
            full_src = source[start_byte:source.index("}", brace_pos) + 1] if brace_pos != -1 else body_src

            chunks.append(CodeChunk(
                chunk_id        = f"{Path(file_path).stem}::{name}:{start_line}",
                file_path       = file_path,
                language        = "go",
                chunk_type      = "method" if receiver else "function",
                name            = name,
                start_line      = start_line,
                end_line        = end_line,
                source          = full_src,
                original_tokens = self._rough_tokens(full_src),
            ))

        return chunks or [self._module_chunk(source, file_path)]

    @staticmethod
    def _extract_brace_body(source: str, brace_start: int, lines: list) -> tuple:
        """Extract balanced brace body starting from `brace_start` (the '{')."""
        depth  = 0
        i      = brace_start
        in_str = False
        str_ch = None
        while i < len(source):
            ch = source[i]
            if in_str:
                if ch == str_ch and (i == 0 or source[i-1] != "\\"):
                    in_str = False
            elif ch in ('"', "'", "`"):
                in_str = True
                str_ch = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    body     = source[brace_start:i+1]
                    end_line = source[:i+1].count("\n") + 1
                    return body, end_line
            i += 1
        return None, -1

    def _module_chunk(self, source: str, file_path: str) -> CodeChunk:
        n_lines = len(source.splitlines())
        return CodeChunk(
            chunk_id        = f"{Path(file_path).stem}::module:1",
            file_path       = file_path,
            language        = "go",
            chunk_type      = "module",
            name            = Path(file_path).stem,
            start_line      = 1,
            end_line        = n_lines,
            source          = source,
            original_tokens = self._rough_tokens(source),
        )

    @staticmethod
    def _rough_tokens(text: str) -> int:
        return len(re.findall(r'\S+', text))
