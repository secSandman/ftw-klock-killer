"""
language_registry.py — Maps file extensions to parsers
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from languages.python_parser import PythonParser
    from languages.c_parser import CParser


class LanguageRegistry:
    """Routes file paths to their appropriate language parser."""

    _EXTENSION_MAP: Dict[str, str] = {
        ".py":  "python",
        ".c":   "c",
        ".h":   "c",
        ".cc":  "c",
        ".cpp": "c",  # tree-sitter-c handles basic C++
        ".go":  "go",
        ".rs":  "rust",
    }

    def __init__(self):
        self._parsers: Dict[str, object] = {}

    def parser_for(self, file_path: str | Path) -> Optional[object]:
        """Return the appropriate parser instance for a file path."""
        ext = Path(file_path).suffix.lower()
        lang = self._EXTENSION_MAP.get(ext)
        if lang is None:
            return None
        if lang not in self._parsers:
            self._parsers[lang] = self._load_parser(lang)
        return self._parsers[lang]

    def language_for(self, file_path: str | Path) -> Optional[str]:
        ext = Path(file_path).suffix.lower()
        return self._EXTENSION_MAP.get(ext)

    def supported_extensions(self):
        return list(self._EXTENSION_MAP.keys())

    def _load_parser(self, lang: str) -> object:
        if lang == "python":
            from languages.python_parser import PythonParser
            return PythonParser()
        elif lang == "c":
            from languages.c_parser import CParser
            return CParser()
        elif lang == "go":
            try:
                from languages.go_parser import GoParser
                return GoParser()
            except ImportError:
                return None
        elif lang == "rust":
            try:
                from languages.rust_parser import RustParser
                return RustParser()
            except ImportError:
                return None
        return None
