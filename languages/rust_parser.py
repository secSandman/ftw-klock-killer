"""
rust_parser.py — Rust source code chunker using tree-sitter
============================================================
Handles: Rust functions, impl blocks, traits, structs, enums.
Grounded in Rust's ownership model and trait-based polymorphism.

Rust is structurally different: functions live inside impl blocks,
so we emit both the impl block as a "class-like" chunk AND each
fn inside it as an individual method chunk.

tree-sitter grammar: tree-sitter-languages or tree-sitter-rust
Install: pip install tree-sitter-languages
         OR: pip install tree-sitter tree-sitter-rust

Chunk types extracted:
  - function_item       → standalone function chunk
  - impl_item           → impl block (methods extracted individually)
  - trait_item          → trait definition chunk
  - struct_item         → struct chunk
  - enum_item           → enum chunk
  - source_file         → module fallback
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from languages.python_parser import CodeChunk


# ── Tree-sitter loading (graceful degradation) ─────────────────────────────

_TS_AVAILABLE   = False
_RUST_LANGUAGE  = None


def _load_tree_sitter():
    global _TS_AVAILABLE, _RUST_LANGUAGE
    try:
        from tree_sitter_languages import get_language
        _RUST_LANGUAGE = get_language("rust")
        _TS_AVAILABLE  = True
        return
    except ImportError:
        pass
    try:
        import tree_sitter_rust as tsr
        from tree_sitter import Language
        _RUST_LANGUAGE = Language(tsr.language())
        _TS_AVAILABLE  = True
    except ImportError:
        pass


_load_tree_sitter()


class RustParser:
    """
    Extracts CodeChunks from Rust source files.

    Falls back to brace-counting regex if tree-sitter isn't installed.
    Handles generics, lifetimes, derive macros, async fn, pub(crate) etc.
    """

    def parse_file(self, file_path: str | Path) -> List[CodeChunk]:
        path = Path(file_path)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return self.parse_source(source, str(path))

    def parse_source(self, source: str, file_path: str = "<string>") -> List[CodeChunk]:
        if _TS_AVAILABLE and _RUST_LANGUAGE:
            return self._parse_tree_sitter(source, file_path)
        return self._parse_regex_fallback(source, file_path)

    # ── Tree-sitter path ────────────────────────────────────────────

    def _parse_tree_sitter(self, source: str, file_path: str) -> List[CodeChunk]:
        from tree_sitter import Parser
        parser = Parser()
        parser.set_language(_RUST_LANGUAGE)
        tree      = parser.parse(source.encode("utf-8", errors="replace"))
        root      = tree.root_node
        chunks: List[CodeChunk] = []
        src_bytes = source.encode("utf-8", errors="replace")

        for node in root.children:
            if node.type == "function_item":
                chunk = self._fn_chunk(node, src_bytes, file_path, parent_name=None)
                if chunk:
                    chunks.append(chunk)
            elif node.type == "impl_item":
                impl_chunks = self._impl_chunks(node, src_bytes, file_path)
                chunks.extend(impl_chunks)
            elif node.type == "trait_item":
                chunk = self._named_chunk(node, src_bytes, file_path, "trait")
                if chunk:
                    chunks.append(chunk)
            elif node.type == "struct_item":
                chunk = self._named_chunk(node, src_bytes, file_path, "struct")
                if chunk:
                    chunks.append(chunk)
            elif node.type == "enum_item":
                chunk = self._named_chunk(node, src_bytes, file_path, "enum")
                if chunk:
                    chunks.append(chunk)

        return chunks or [self._module_chunk(source, file_path)]

    def _fn_chunk(self, node, src_bytes: bytes, file_path: str,
                  parent_name: Optional[str]) -> Optional[CodeChunk]:
        """Extract a function_item node."""
        name = None
        for child in node.children:
            if child.type == "identifier":
                name = src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                break
        if not name:
            return None

        qualified  = f"{parent_name}::{name}" if parent_name else name
        fn_src     = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        start_line = node.start_point[0] + 1
        end_line   = node.end_point[0] + 1
        return CodeChunk(
            chunk_id        = f"{Path(file_path).stem}::{qualified}:{start_line}",
            file_path       = file_path,
            language        = "rust",
            chunk_type      = "method" if parent_name else "function",
            name            = qualified,
            start_line      = start_line,
            end_line        = end_line,
            source          = fn_src,
            original_tokens = self._rough_tokens(fn_src),
            parent_name     = parent_name,
        )

    def _impl_chunks(self, node, src_bytes: bytes, file_path: str) -> List[CodeChunk]:
        """
        Decompose an impl block into individual method chunks.
        impl Foo { fn bar() {...}  fn baz() {...} }
        → chunks for bar and baz with parent_name=Foo
        """
        chunks: List[CodeChunk] = []

        # Determine the impl type name: `impl Foo` or `impl Bar for Baz`
        impl_name = "UnknownImpl"
        for child in node.children:
            if child.type == "type_identifier":
                impl_name = src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                break

        # Walk the declaration_list for function_items
        for child in node.children:
            if child.type == "declaration_list":
                for item in child.children:
                    if item.type == "function_item":
                        chunk = self._fn_chunk(item, src_bytes, file_path, parent_name=impl_name)
                        if chunk:
                            chunks.append(chunk)

        # If no methods found, emit the whole impl block as one chunk
        if not chunks:
            chunk = self._named_chunk(node, src_bytes, file_path, "impl")
            if chunk:
                chunks.append(chunk)

        return chunks

    def _named_chunk(self, node, src_bytes: bytes, file_path: str, chunk_type: str) -> Optional[CodeChunk]:
        """Generic named item chunk (struct, enum, trait, impl)."""
        name = None
        for child in node.children:
            if child.type == "type_identifier":
                name = src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                break
        if not name:
            name = chunk_type

        item_src   = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        start_line = node.start_point[0] + 1
        end_line   = node.end_point[0] + 1
        return CodeChunk(
            chunk_id        = f"{Path(file_path).stem}::{name}:{start_line}",
            file_path       = file_path,
            language        = "rust",
            chunk_type      = chunk_type,
            name            = name,
            start_line      = start_line,
            end_line        = end_line,
            source          = item_src,
            original_tokens = self._rough_tokens(item_src),
        )

    # ── Regex fallback ───────────────────────────────────────────────

    def _parse_regex_fallback(self, source: str, file_path: str) -> List[CodeChunk]:
        """
        Regex-based Rust function extractor.
        Captures standalone fns and impl methods.

        Pattern matches:
          [pub] [async] fn name<T>(params) [-> ReturnType] {
          pub(crate) async fn name(...) -> Result<X> {
        """
        chunks: List[CodeChunk] = []
        lines  = source.splitlines()

        # First pass: find impl blocks to get parent_name context
        impl_ranges: List[tuple] = []  # (start_line, end_line, type_name)
        impl_pat = re.compile(r'^\s*(?:pub\s+)?impl(?:<[^>]*>)?\s+(?:\w+\s+for\s+)?(\w+)', re.MULTILINE)
        for m in impl_pat.finditer(source):
            impl_name  = m.group(1)
            brace_pos  = source.find("{", m.end())
            if brace_pos == -1:
                continue
            _, end_line = self._extract_brace_body(source, brace_pos, lines)
            start_line  = source[:m.start()].count("\n") + 1
            if end_line != -1:
                impl_ranges.append((start_line, end_line, impl_name))

        # Pattern: optional visibility + optional async + fn + name
        fn_pattern = re.compile(
            r'^(?:[ \t]*)(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(',
            re.MULTILINE,
        )

        for m in fn_pattern.finditer(source):
            name       = m.group(1)
            brace_pos  = source.find("{", m.end())
            if brace_pos == -1:
                continue
            start_byte = m.start()
            start_line = source[:start_byte].count("\n") + 1
            full_src_start = source.rfind("\n", 0, start_byte) + 1  # include leading whitespace

            body_src, end_line = self._extract_brace_body(source, brace_pos, lines)
            if body_src is None:
                continue

            # Determine parent impl context
            parent_name = None
            for (istart, iend, iname) in impl_ranges:
                if istart <= start_line <= iend:
                    parent_name = iname
                    break

            qualified = f"{parent_name}::{name}" if parent_name else name
            full_src  = source[full_src_start:source.find("}", brace_pos) + 1]

            chunks.append(CodeChunk(
                chunk_id        = f"{Path(file_path).stem}::{qualified}:{start_line}",
                file_path       = file_path,
                language        = "rust",
                chunk_type      = "method" if parent_name else "function",
                name            = qualified,
                start_line      = start_line,
                end_line        = end_line,
                source          = full_src,
                original_tokens = self._rough_tokens(full_src),
                parent_name     = parent_name,
            ))

        # Second pass: extract struct / enum definitions
        struct_pattern = re.compile(
            r'^(?:[ \t]*)(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum)\s+(\w+)',
            re.MULTILINE,
        )
        fn_line_set = {c.start_line for c in chunks}
        for m in struct_pattern.finditer(source):
            name       = m.group(1)
            brace_pos  = source.find("{", m.end())
            if brace_pos == -1:
                continue
            start_line = source[:m.start()].count("\n") + 1
            if start_line in fn_line_set:
                continue  # already captured
            body_src, end_line = self._extract_brace_body(source, brace_pos, lines)
            if body_src is None:
                continue
            item_type  = "struct" if "struct" in m.group(0) else "enum"
            full_src   = source[source.rfind("\n", 0, m.start()) + 1:source.find("}", brace_pos) + 1]
            chunks.append(CodeChunk(
                chunk_id        = f"{Path(file_path).stem}::{name}:{start_line}",
                file_path       = file_path,
                language        = "rust",
                chunk_type      = item_type,
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
            elif ch in ('"', "'"):
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
            language        = "rust",
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
