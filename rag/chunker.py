"""
chunker.py — Multi-language code chunker
Dispatches to the right language parser and returns CodeChunks.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List
sys.path.insert(0, str(Path(__file__).parent.parent))

from languages.language_registry import LanguageRegistry
from languages.python_parser import CodeChunk


class Chunker:
    """Dispatches files to language-specific parsers and returns CodeChunks."""

    def __init__(self):
        self._registry = LanguageRegistry()

    def chunk_file(self, file_path: str | Path) -> List[CodeChunk]:
        """Parse a single file and return all chunks."""
        parser = self._registry.parser_for(file_path)
        if parser is None:
            return []
        return parser.parse_file(file_path)

    def chunk_source(self, source: str, file_path: str, language: str = None) -> List[CodeChunk]:
        """Parse source text directly."""
        lang = language or self._registry.language_for(file_path) or "python"
        parser = self._registry._load_parser(lang)
        if parser is None:
            return []
        return parser.parse_source(source, file_path)

    def chunk_repo(self, repo_path: str | Path, extensions: List[str] = None) -> List[CodeChunk]:
        """Walk a directory and chunk all supported source files."""
        repo = Path(repo_path)
        supported = set(extensions or self._registry.supported_extensions())
        all_chunks: List[CodeChunk] = []
        for p in sorted(repo.rglob("*")):
            if p.suffix.lower() in supported and p.is_file():
                try:
                    chunks = self.chunk_file(p)
                    all_chunks.extend(chunks)
                except Exception:
                    continue
        return all_chunks

    def to_pruner_payload(self, chunks: List[CodeChunk], file_path: str = "", language: str = "unknown") -> dict:
        """Convert chunks to the dict format PRUNER expects on its input slice."""
        return {
            "file_path": file_path,
            "language":  language,
            "chunks": [
                {
                    "chunk_id":      c.chunk_id,
                    "chunk_type":    c.chunk_type,
                    "start_line":    c.start_line,
                    "end_line":      c.end_line,
                    "source":        c.source,
                    "original_tokens": c.original_tokens,
                    "name":          c.name,
                }
                for c in chunks
            ],
        }
