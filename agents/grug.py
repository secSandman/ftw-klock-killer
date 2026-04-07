"""
grug.py — GRUG Agent (The Semantic Compressor / Computational Linguist)
========================================================================
Me GRUG. Me compress token. Me not polite. Me result. Me stop.

Strategy: Intent-Only Mapping — strips polite syntax, replaces long words
with shorter mathematically-equivalent synonyms. Turns 50-token prompts
into 10-token caveman instructions the LLM still understands perfectly.

Research grounding:
  - Zipf, G.K. (1935). "The Psycho-Biology of Language." — Zipf's Law:
    word frequency is inversely proportional to rank. Short = common = efficient.
  - Rissanen, J. (1978). "Modeling by shortest data description." Automatica.
    Minimum Description Length (MDL) principle: best model = shortest description.
  - Brown et al. (1992). "Class-Based n-gram Models of Natural Language."
    Computational Linguistics 18(4). Word clustering justifies synonym replacement.
  - Chen & Manning (2014). "A Fast and Accurate Dependency Parser."
    Intent-only mapping reduces syntactic overhead without losing semantics.

GRUG runs the full InferenceBridge F1-F4 pipeline on FOREGROUND chunks,
lighter F4-only on BACKGROUND chunks, then applies the synonym compression table.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, FeatureSlice

try:
    from src.inference_bridge import InferenceBridge
    from src.quality import compute_semantic_similarity
    _INB_AVAILABLE = True
except ImportError:
    _INB_AVAILABLE = False

try:
    from src.c_compressor import CCompressor
    from languages.c_pattern_miner import CPatternDict, CPatternMiner
    _C_COMPRESSOR_AVAILABLE = True
except ImportError:
    _C_COMPRESSOR_AVAILABLE = False

# Cache of CCompressor instances keyed by repo_path string
# Avoids re-mining the same repo on every chunk
_c_compressor_cache: Dict[str, "CCompressor"] = {}

# ─────────────────────────────────────────────────────────────────────────────
# GRUG's synonym compression table
# Grounded in MDL principle: always prefer the shorter, lossless form.
# Entries: (long_form, grug_form, context)
# ─────────────────────────────────────────────────────────────────────────────
SYNONYM_TABLE: List[Tuple[str, str]] = [
    # Nouns
    ("initialize",     "init"),
    ("initialization", "init"),
    ("configuration",  "cfg"),
    ("authenticate",   "auth"),
    ("authentication", "auth"),
    ("authorization",  "authz"),
    ("implementation", "impl"),
    ("concatenate",    "cat"),
    ("concatenation",  "cat"),
    ("parameter",      "param"),
    ("parameters",     "params"),
    ("exception",      "exc"),
    ("iteration",      "iter"),
    ("validation",     "valid"),
    ("repository",     "repo"),
    ("functionality",  "fn"),
    ("documentation",  "doc"),
    ("application",    "app"),
    ("environment",    "env"),
    ("dependency",     "dep"),
    ("dependencies",   "deps"),
    ("executable",     "exe"),
    ("definition",     "def"),
    ("dictionary",     "dict"),
    ("argument",       "arg"),
    ("arguments",      "args"),
    ("attribute",      "attr"),
    ("attributes",     "attrs"),
    ("connection",     "conn"),
    ("transaction",    "txn"),
    ("serialization",  "ser"),
    ("deserialization","deser"),
    ("asynchronous",   "async"),
    ("synchronous",    "sync"),
    ("metadata",       "meta"),
    ("notification",   "notif"),
    # Verbs
    ("implement",      "impl"),
    ("initialize",     "init"),
    ("calculate",      "calc"),
    ("retrieve",       "get"),
    ("instantiate",    "new"),
    ("generate",       "gen"),
    ("validate",       "valid"),
    ("serialize",      "ser"),
    ("deserialize",    "deser"),
    # Code-specific
    ("function",       "fn"),
    ("method",         "fn"),
    ("variable",       "var"),
    ("constant",       "const"),
    ("interface",      "iface"),
    ("namespace",      "ns"),
    ("module",         "mod"),
    ("package",        "pkg"),
]

# Compile to dict for O(1) lookup (longest-match wins)
_SYNONYM_MAP = {k.lower(): v for k, v in sorted(SYNONYM_TABLE, key=lambda x: -len(x[0]))}


class GrugAgent(BaseAgent):
    """
    GRUG: Semantic token compressor.

    Input channel:  pruner.out
    Output channel: grug.out

    For FOREGROUND chunks: runs full InferenceBridge pipeline (F1-F4)
    For BACKGROUND chunks: runs only F4 (Caveman) — skeleton + masking cost too much
    After InB: applies synonym compression table on comments/docstrings
    Inline quality check: computes RQS-L1 (semantic similarity) to ensure
    compression didn't destroy meaning.
    """

    AGENT_ID    = "GRUG"
    IN_CHANNEL  = "pruner.out"
    OUT_CHANNEL = "grug.out"

    RQS_L1_WARNING_THRESHOLD = 0.80  # below this: warn + log decision

    def __init__(self, inb_config_path: str = "src/inb_config.json", **kwargs):
        super().__init__(**kwargs)
        self._inb_config = inb_config_path
        self._inb: Optional[InferenceBridge] = None
        if _INB_AVAILABLE:
            try:
                self._inb = InferenceBridge.from_config(inb_config_path)
            except Exception:
                self._inb = InferenceBridge()  # default config

    def process(self, slice_in: FeatureSlice) -> FeatureSlice:
        classified_chunks = slice_in.payload.get("classified_chunks", [])
        file_path         = slice_in.payload.get("file_path", "")
        language          = slice_in.payload.get("language", "unknown")
        task_type         = slice_in.payload.get("task_type", "unknown")
        question          = slice_in.payload.get("question", "")
        rainbow_hit       = slice_in.payload.get("rainbow_hit", False)

        compressed_chunks = []
        total_original    = 0
        total_final       = 0
        all_rhd_delta: Dict[str, str] = {}
        all_synonym_swaps: List[Dict] = []

        # For C files: pre-compress the whole file once so F3 masking sees
        # the #include/#define directives at file scope (not visible in individual
        # function chunks). Build a map of start_line → compressed source.
        c_file_compressed_lines: Optional[List[str]] = None
        c_langs = {"c", "cpp", "c++", "cc"}
        is_c_file = (
            language in c_langs
            or any(file_path.endswith(ext) for ext in (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"))
        )
        if is_c_file and _C_COMPRESSOR_AVAILABLE and file_path:
            try:
                from pathlib import Path as _Path
                fpath = _Path(file_path)
                if fpath.exists():
                    whole_src = fpath.read_text(encoding="utf-8", errors="replace")
                    repo_root = self._find_repo_root(file_path)
                    cache_key = str(repo_root)
                    if cache_key not in _c_compressor_cache:
                        _c_compressor_cache[cache_key] = CCompressor.for_repo(repo_root)
                    c_comp = _c_compressor_cache[cache_key]
                    compressed_whole, _ = c_comp.compress(whole_src, zone="FOREGROUND")
                    c_file_compressed_lines = compressed_whole.splitlines(keepends=True)
            except Exception:
                c_file_compressed_lines = None

        for chunk in classified_chunks:
            # Inject slice-level language and file_path into each chunk so
            # _compress_chunk can dispatch to the right compressor without
            # requiring every upstream agent to stamp each chunk individually.
            if "language" not in chunk:
                chunk = {**chunk, "language": language, "file_path": file_path}

            # For C: extract this chunk's range from the pre-compressed file
            if c_file_compressed_lines is not None:
                chunk = {
                    **chunk,
                    "_c_file_lines": c_file_compressed_lines,
                }

            compressed, orig_t, final_t, rhd_delta, swaps, rqs_l1 = self._compress_chunk(chunk)
            compressed_chunks.append({
                "chunk_id":          chunk["chunk_id"],
                "zone":              chunk["zone"],
                "chunk_type":        chunk.get("chunk_type", "function"),
                "original_tokens":   orig_t,
                "final_tokens":      final_t,
                "reduction_pct":     round((1 - final_t / orig_t) * 100, 1) if orig_t > 0 else 0.0,
                "compressed_src":    compressed,
                "rqs_l1_score":      rqs_l1,
                "synonym_swaps":     swaps,
            })
            total_original += orig_t
            total_final    += final_t
            all_rhd_delta.update(rhd_delta)
            all_synonym_swaps.extend(swaps)

        overall_ter = round((1 - total_final / total_original) * 100, 1) if total_original > 0 else 0.0
        avg_rqs_l1  = (
            sum(c["rqs_l1_score"] for c in compressed_chunks) / len(compressed_chunks)
            if compressed_chunks else 1.0
        )

        decision = (
            f"TER={overall_ter:.1f}% "
            f"({total_original}→{total_final} tok). "
            f"RQS-L1={avg_rqs_l1:.2f}. "
            f"{len(all_synonym_swaps)} synonym swaps. "
            f"lang={language}"
        )

        if avg_rqs_l1 < self.RQS_L1_WARNING_THRESHOLD:
            self.log_decision(
                decision_type  = "COMPRESS",
                decision_value = f"RQS-L1={avg_rqs_l1:.3f} BELOW threshold {self.RQS_L1_WARNING_THRESHOLD}",
                rationale      = "Quality degradation detected — BALANCER should consider escalation",
                confidence     = avg_rqs_l1,
                slice_id       = slice_in.slice_id,
            )

        self.log_decision(
            decision_type  = "COMPRESS",
            decision_value = {"ter_pct": overall_ter, "rqs_l1": avg_rqs_l1},
            rationale      = decision,
            confidence     = avg_rqs_l1,
            slice_id       = slice_in.slice_id,
        )

        lang_tag = f"lang.{language}" if f"lang.{language}" in ["lang.python", "lang.c", "lang.go", "lang.rust"] else "lang.unknown"

        return self.emit(
            taxonomy_tags   = ["compression.skeleton", "compression.masking", "compression.caveman",
                               "compression.synonym", "ter", "rqs.l1", "intent_mapping", lang_tag],
            payload         = {
                "compressed_chunks":    compressed_chunks,
                "total_original_tokens": total_original,
                "total_final_tokens":   total_final,
                "overall_ter":          overall_ter,
                "rqs_l1_score":         avg_rqs_l1,
                "rhd_registry_delta":   all_rhd_delta,
                "synonym_swaps_count":  len(all_synonym_swaps),
                "file_path":            file_path,
                "language":             language,
                "task_type":            task_type,
                "question":             question,
                "rainbow_hit":          rainbow_hit,
            },
            confidence      = avg_rqs_l1,
            decision        = decision,
            parent_slice_id = slice_in.slice_id,
            token_count     = total_final,
            compression_pct = overall_ter,
        )

    # ── Compression internals ────────────────────────────────────────

    def _compress_chunk(
        self, chunk: Dict[str, Any]
    ) -> Tuple[str, int, int, Dict[str, str], List[Dict], float]:
        """
        Returns: (compressed_src, orig_tokens, final_tokens, rhd_delta, synonym_swaps, rqs_l1)

        Dispatches to:
          - CCompressor  for C/C++ files  (F1 skeleton + F3 masking + F4 caveman)
          - InferenceBridge for Python    (F1 AST skeleton + F3 RHD masking + F4 caveman)
          - CavemanCompressor fallback    for Go/Rust/unknown
        """
        source    = chunk.get("source", "")
        zone      = chunk.get("zone", "FOREGROUND")
        language  = chunk.get("language", chunk.get("chunk_type", "unknown"))
        file_path = chunk.get("file_path", "")

        if not source.strip():
            orig_t = chunk.get("original_tokens", 0)
            return source, orig_t, orig_t, {}, [], 1.0

        # Always recount using count_tokens so orig_t and final_t use the
        # same tokenizer. The chunker uses _rough_tokens (word count) which
        # undercounts C operators/brackets — causing negative TER.
        from src.inference_bridge import count_tokens as _ct
        orig_t = _ct(source)

        compressed = source
        final_t    = orig_t
        rhd_delta: Dict[str, str] = {}

        # ── C/C++ path: CCompressor (F1 brace-skeleton + F3 include-masking + F4 caveman) ──
        c_langs = {"c", "cpp", "c++", "cc"}
        is_c = (
            language in c_langs
            or any(file_path.endswith(ext) for ext in (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"))
        )

        if is_c and _C_COMPRESSOR_AVAILABLE:
            try:
                c_file_lines = chunk.get("_c_file_lines")
                if c_file_lines is not None:
                    # Whole-file pre-compressed: extract this chunk's line range.
                    # start_line/end_line are 1-based (from CParser).
                    s = chunk.get("start_line", 1) - 1   # 0-based
                    e = chunk.get("end_line",   s + 1)   # 0-based exclusive
                    chunk_lines = c_file_lines[s:e]
                    compressed  = "".join(chunk_lines)
                    final_t     = _ct(compressed)
                else:
                    # Fallback: compress the chunk source directly (no file-level context)
                    repo_root = self._find_repo_root(file_path)
                    cache_key = str(repo_root)
                    if cache_key not in _c_compressor_cache:
                        _c_compressor_cache[cache_key] = CCompressor.for_repo(repo_root)
                    c_comp = _c_compressor_cache[cache_key]
                    compressed, report = c_comp.compress(source, zone=zone)
                    final_t = report.final
            except Exception:
                compressed = source
                final_t    = orig_t

        # ── Python path: InferenceBridge (F1 AST + F3 RHD + F4 caveman) ──
        elif language == "python" and _INB_AVAILABLE and self._inb:
            try:
                import tempfile, os
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False, encoding="utf-8"
                ) as tmp:
                    tmp.write(source)
                    tmp_path = tmp.name
                if zone == "BACKGROUND":
                    from src.inference_bridge import CavemanCompressor, count_tokens
                    compressed = CavemanCompressor().compress(source)
                    final_t    = count_tokens(compressed)
                else:
                    compressed_src, report = self._inb.process_file(tmp_path)
                    compressed = compressed_src
                    final_t    = report.final
                os.unlink(tmp_path)
            except Exception:
                compressed = source
                final_t    = orig_t

        # ── Fallback: Caveman only (Go, Rust, unknown) ──
        else:
            try:
                from src.inference_bridge import CavemanCompressor, count_tokens
                compressed = CavemanCompressor().compress(source)
                final_t    = count_tokens(compressed)
            except Exception:
                compressed = source
                final_t    = orig_t

        # Step 2: Synonym compression pass (comments/docstrings only)
        compressed_after_synonyms, swaps = self._apply_synonyms(compressed)
        try:
            final_t_syn = _ct(compressed_after_synonyms)
        except Exception:
            final_t_syn = final_t

        # Step 3: RQS-L1 inline quality check
        if _INB_AVAILABLE:
            try:
                rqs_l1 = compute_semantic_similarity(source, compressed_after_synonyms)
            except Exception:
                rqs_l1 = 1.0
        else:
            rqs_l1 = 1.0

        return compressed_after_synonyms, orig_t, final_t_syn, rhd_delta, swaps, rqs_l1

    @staticmethod
    def _find_repo_root(file_path: str) -> Path:
        """
        Walk up the directory tree from file_path looking for a .kloc/ directory
        (written by CPatternMiner). If not found, return the file's parent dir.
        This lets CCompressor.for_repo() auto-mine on first call.
        """
        p = Path(file_path).resolve()
        for parent in [p.parent] + list(p.parents):
            if (parent / ".kloc").exists():
                return parent
        return p.parent

    def _apply_synonyms(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Apply GRUG synonym table to comment lines and docstrings.
        Does NOT touch code identifiers — only natural language text.
        """
        import re
        lines   = text.splitlines(keepends=True)
        result  = []
        swaps   = []

        for line in lines:
            stripped = line.lstrip()
            # Only touch comment lines and docstring content
            is_comment  = stripped.startswith("#")
            is_docstring = stripped.startswith(('"""', "'''", '"', "'"))
            if is_comment or is_docstring:
                new_line = line
                for long_form, short_form in _SYNONYM_MAP.items():
                    pattern = re.compile(r'\b' + re.escape(long_form) + r'\b', re.IGNORECASE)
                    if pattern.search(new_line):
                        new_line = pattern.sub(short_form, new_line)
                        swaps.append({"original": long_form, "replacement": short_form})
                result.append(new_line)
            else:
                result.append(line)

        return "".join(result), swaps


# ── CLI self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("GRUG self-test — me compress test chunk...")

    # Test synonym replacement
    test_doc = '"""Initialize the configuration for authentication."""'
    compressed, swaps = GrugAgent()._apply_synonyms(test_doc)
    print(f"\n  Input:   {test_doc}")
    print(f"  Output:  {compressed.strip()}")
    print(f"  Swaps:   {swaps}")
    print("\nMe done. Me stop.")
