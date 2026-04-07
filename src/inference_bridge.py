#!/usr/bin/env python3
"""
inference_bridge.py — Inference-Bridge (InB) Core Pipeline
===========================================================
Semantic token compressor: reduces Python source code token count by >80%
before sending to an LLM. Four biologically-inspired filters run in sequence.

Pipeline order (FIXED — do not reorder):
  F1 Mushroom Body   — AST skeleton: replaces function bodies with ...
  F2 Retinal Delta   — git diff: marks Hot (keep) vs Cold (skeleton) lines
  F3 Chromatophoric  — masks repeated imports/boilerplate as [§:hash] markers
  F4 Caveman         — compresses comments: stop-words + vowel pruning

Usage:
  python inference_bridge.py mask path/to/file.py
  python inference_bridge.py mask path/to/file.py --output compressed.py
  python inference_bridge.py unmask compressed.py
  python inference_bridge.py stats ./my-repo/
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants (monkey-patchable for per-run overrides)
# ─────────────────────────────────────────────────────────────────────────────

SKELETAL_PI_THRESHOLD      = float(os.environ.get("KLOC_PI_THRESHOLD",              "0.70"))
ENTROPY_THRESHOLD          = float(os.environ.get("KLOC_PI_ENTROPY_THRESHOLD",      "4.0"))
KNOWN_QUANTITY_MIN_FREQ    = int(  os.environ.get("KLOC_F3_KQ_THRESHOLD",           "3"))
HZS_COLD_CUTOFF            = float(os.environ.get("KLOC_PI_HZS_COLD_CUTOFF",       "0.05"))
HZS_LAMBDA                 = float(os.environ.get("KLOC_PI_HZS_LAMBDA",            "0.1"))
N_COMMITS                  = int(  os.environ.get("KLOC_PI_N_COMMITS",             "1"))
VOWEL_PRUNE_MIN_LEN        = int(  os.environ.get("KLOC_F4_VOWEL_PRUNE_MIN_LEN",   "5"))

MASK_PREFIX = "[§:"
MASK_SUFFIX = "]"
_MASK_RE    = re.compile(r"\[§:([a-f0-9]{8})\]")

STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "it", "its", "be",
    "was", "are", "were", "been", "has", "have", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "that", "this", "these", "those", "then", "than", "so", "up",
    "out", "no", "not", "nor", "yet", "both", "either", "neither", "each",
    "few", "more", "most", "other", "some", "such", "what", "which", "who",
    "also", "just", "into", "only", "over", "very", "here", "there",
    "when", "where", "how", "all", "any", "since", "while",
})


# ─────────────────────────────────────────────────────────────────────────────
# Token counting (no tiktoken required)
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r'""".*?"""|'          # triple-double-quoted strings
    r"'''.*?'''|"          # triple-single-quoted strings
    r'"[^"\n]*"|'          # double-quoted strings
    r"'[^'\n]*'|"          # single-quoted strings
    r"[A-Za-z_]\w*|"       # identifiers / keywords
    r"[0-9]+(?:\.[0-9]+)?|"# numbers
    r"[+\-*/=<>!&|^~%@]+|" # operators
    r"[(){}\[\],.:;]|"     # delimiters
    r"\S",                  # any other non-whitespace
    re.DOTALL,
)


def count_tokens(text: str) -> int:
    """
    Lightweight token estimator — no external deps.
    Approximates GPT-4 tokenization within ~5% for typical Python source.
    Identifiers >8 chars count as 2 tokens (subword splitting approximation).
    """
    if not text:
        return 0
    count = 0
    for tok in _TOKEN_RE.findall(text):
        count += 2 if (len(tok) > 8 and tok.isalnum()) else 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Math primitives
# ─────────────────────────────────────────────────────────────────────────────

def shannon_entropy(text: str) -> float:
    """
    Shannon entropy in bits per character.
    Low entropy → repetitive/boilerplate → good masking candidate.
    High entropy → information-dense → keep at full fidelity.
    """
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    h = -sum((c / n) * math.log2(c / n) for c in freq.values())
    return round(h, 4)


def predictability_index(source: str) -> float:
    """
    Predictability Index (PI) for a function body. Range [0, 1].
    High PI → body is predictable from its signature → safe to replace with `...`.

    Strategy: start from a high base and apply penalties for complexity.
    Python source entropy is nearly constant, so entropy is not used here —
    instead we measure structural complexity and length.

    Base score by line count:
      1-2 lines  → 0.90
      3-5 lines  → 0.80
      6-10 lines → 0.70
      11-20 lines → 0.55
      >20 lines  → decays toward 0

    Penalties: -0.06 per complex construct (loop/try/yield/with)
    Bonuses:   +0.15 if most lines are simple assignments/returns
    """
    if not source.strip():
        return 1.0

    lines = [l for l in source.strip().splitlines() if l.strip()]
    n = len(lines)

    # Base score by length: most Python functions are predictable from context
    if n <= 2:
        base = 0.95
    elif n <= 5:
        base = 0.90
    elif n <= 10:
        base = 0.85
    elif n <= 20:
        base = 0.80
    elif n <= 35:
        base = 0.70
    else:
        base = max(0.0, 0.70 - (n - 35) * 0.02)

    # Complexity penalty: for-loops, while, try/except, generators, async-for
    _COMPLEX = (
        re.compile(r"^\s*for\b.*\bin\b"),
        re.compile(r"^\s*while\b"),
        re.compile(r"^\s*try\s*:"),
        re.compile(r"^\s*except\b"),
        re.compile(r"\byield\b"),
        re.compile(r"\basync\s+for\b"),
        re.compile(r"\basync\s+with\b"),
    )
    complex_count = sum(
        1 for line in lines if any(p.search(line) for p in _COMPLEX)
    )
    _PI_LOOP_PENALTY     = float(os.environ.get("KLOC_PI_LOOP_PENALTY",            "0.06"))
    _PI_COMPLEXITY_CAP   = float(os.environ.get("KLOC_PI_COMPLEXITY_PENALTY_CAP",  "0.35"))
    _PI_BOILERPLATE_BON  = float(os.environ.get("KLOC_PI_BOILERPLATE_BONUS",       "0.15"))
    _PI_SIMPLE_RATIO_THR = float(os.environ.get("KLOC_PI_SIMPLE_RATIO_THRESHOLD",  "0.50"))
    _PI_LOOP_FREE_BON    = float(os.environ.get("KLOC_PI_LOOP_FREE_BONUS",         "0.10"))

    complexity_penalty = min(_PI_COMPLEXITY_CAP, complex_count * _PI_LOOP_PENALTY)

    # Boilerplate bonus: simple assign/return/raise/pass lines
    # Only awarded when complex_count == 0 — a loop-containing function must
    # never receive this bonus (the critical constraint: loop + >20 lines → PI < 0.70).
    _SIMPLE = (
        re.compile(r"^\s*return\b"),
        re.compile(r"^\s*self\.\w+\s*="),
        re.compile(r"^\s*raise\b"),
        re.compile(r"^\s*pass\s*$"),
        re.compile(r"^\s*\.\.\.\s*$"),
        re.compile(r"^\s*super\("),
        re.compile(r"^\s*logger\."),
        # dict-item and local-variable assignment boilerplate (e.g. CRUD builders)
        re.compile(r"^\s*\w[\w\[\]\"']*\s*\[[\w\"'\.]+\]\s*="),
        re.compile(r"^\s*[a-z_]\w*\s*=\s*\w"),
    )
    simple_count = sum(
        1 for line in lines if any(p.match(line) for p in _SIMPLE)
    )
    simple_ratio = simple_count / n if n > 0 else 0
    boilerplate_bonus = _PI_BOILERPLATE_BON if (simple_ratio >= _PI_SIMPLE_RATIO_THR and complex_count == 0) else 0.0

    # Loop-free long-function recovery: a body >20 lines with zero complex
    # constructs is an assignment cascade — more predictable than length implies.
    loop_free_bonus = _PI_LOOP_FREE_BON if (n > 20 and complex_count == 0) else 0.0

    pi = base - complexity_penalty + boilerplate_bonus + loop_free_bonus
    return round(min(1.0, max(0.0, pi)), 4)


def hot_zone_score(line_idx: int, changed_lines: set[int]) -> float:
    """
    Hot Zone Score (HZS) for a line given a set of recently-changed 1-indexed lines.
    Uses exponential decay from the nearest changed line.

    Returns [0, 1]:
      1.0 = this line was changed
      ~0.37 at distance=10 with default lambda=0.1
      < HZS_COLD_CUTOFF → Cold zone
    """
    if not changed_lines:
        return 1.0  # no diff info → treat all lines as Hot
    min_dist = min(abs(line_idx - cl) for cl in changed_lines)
    return round(math.exp(-HZS_LAMBDA * min_dist), 4)


_SKELETON_SKIP_IDS = frozenset({
    # Python keywords
    "self", "cls", "return", "raise", "pass", "yield", "break", "continue",
    "and", "or", "not", "in", "is", "if", "else", "elif", "for", "while",
    "try", "except", "finally", "with", "as", "import", "from", "class",
    "def", "lambda", "del", "global", "nonlocal", "assert", "async", "await",
    # Common builtins
    "None", "True", "False", "len", "str", "int", "float", "bool", "list",
    "dict", "set", "tuple", "type", "range", "print", "open", "super",
    "isinstance", "hasattr", "getattr", "setattr", "enumerate", "zip",
    "map", "filter", "sorted", "reversed", "any", "all", "min", "max",
    # Common stdlib/boilerplate
    "logger", "log", "logging", "os", "sys", "re", "json", "path",
    "append", "extend", "update", "get", "items", "keys", "values",
    "strip", "split", "join", "format", "upper", "lower", "replace",
    "None", "true", "false", "none",
})


def _extract_skeleton_identifiers(body_src: str, max_ids: int = 15) -> list[str]:
    """
    Extract rare domain-specific identifiers from a function body.

    Used to annotate `...` skeleton lines so TF vectors retain domain signal.
    Strategy: collect all identifiers, exclude common words, sort by rarity
    (fewest occurrences = most domain-specific), return top N.
    """
    words = re.findall(r"[a-zA-Z_]\w*", body_src)
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    candidates = [
        w for w in freq
        if w not in _SKELETON_SKIP_IDS
        and not w.startswith("__")
        and len(w) >= 3
        and not w.isupper()          # exclude ALL_CAPS constants
    ]
    # Sort by rarity first, then alphabetically for determinism
    candidates.sort(key=lambda w: (freq[w], w))
    return candidates[:max_ids]


# ─────────────────────────────────────────────────────────────────────────────
# F1: Mushroom Body Skeletonizer
# ─────────────────────────────────────────────────────────────────────────────

class MushroomBodySkeletonizer:
    """
    F1: Replaces function/method bodies with `...` when PI >= pi_threshold.

    Uses AST parsing to locate function definitions. Processes only
    non-nested functions (module-level and class methods) to avoid
    double-skeletonizing nested closures. Hot-zone lines are always
    preserved at full fidelity regardless of PI.
    """

    def __init__(self, pi_threshold: float = SKELETAL_PI_THRESHOLD):
        self.pi_threshold = pi_threshold

    def _collect_functions(self, tree: ast.AST) -> list:
        """
        Collect non-nested FunctionDef / AsyncFunctionDef nodes.
        Recurses into ClassDef bodies to find methods, but does NOT
        recurse into function bodies (skipping nested closures).
        """
        funcs = []

        class _Collector(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                funcs.append(node)
                # Do NOT call generic_visit — skip nested functions

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node):
                self.generic_visit(node)  # recurse into class body

        _Collector().visit(tree)
        return funcs

    def skeletonize(self, source: str, hot_lines: Optional[set[int]] = None) -> str:
        """
        Replace qualifying function bodies with `...`.

        hot_lines: set of 1-indexed line numbers that are "hot" (recently
                   changed). Bodies containing any hot line are preserved.
        Returns the modified source, or the original source on parse error.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        lines = source.splitlines(keepends=True)
        hot = hot_lines or set()

        # Collect (start_idx_0, end_idx_0_exclusive, replacement_str) tuples
        replacements: list[tuple[int, int, str]] = []

        for node in self._collect_functions(tree):
            if not node.body:
                continue

            body_first_line = node.body[0].lineno      # 1-indexed
            body_last_line  = node.end_lineno           # 1-indexed

            # Skip if any hot line falls within the body
            if hot and hot & set(range(body_first_line, body_last_line + 1)):
                continue

            body_src = "".join(lines[body_first_line - 1 : body_last_line])
            pi = predictability_index(body_src)
            if pi < self.pi_threshold:
                continue

            # Determine body indent from the first body line
            first_body_line = lines[body_first_line - 1]
            body_indent = " " * (len(first_body_line) - len(first_body_line.lstrip()))

            # Preserve a leading docstring if present
            first_stmt = node.body[0]
            has_docstring = (
                isinstance(first_stmt, ast.Expr)
                and isinstance(getattr(first_stmt, "value", None), (ast.Constant, ast.Str))
            )

            _sig_retain  = os.environ.get("KLOC_SKELETON_SIG_RETAIN", "0") == "1"
            _sig_max_ids = int(os.environ.get("KLOC_SKELETON_SIG_MAX_IDS", "15"))
            if _sig_retain:
                sig_ids  = _extract_skeleton_identifiers(body_src, _sig_max_ids)
                sig_comment = ("  # " + " ".join(sig_ids)) if sig_ids else ""
            else:
                sig_comment = ""

            if has_docstring and len(node.body) > 1:
                # Keep the docstring; replace everything after it
                doc_end  = first_stmt.end_lineno          # 1-indexed
                rest_start = doc_end                       # 0-indexed: doc_end - 1 + 1
                rest_end   = body_last_line                # 0-indexed exclusive: body_last_line
                if rest_start < rest_end:
                    replacements.append((rest_start, rest_end,
                                         body_indent + f"...{sig_comment}\n"))
            else:
                # Replace entire body
                replacements.append(
                    (body_first_line - 1, body_last_line,
                     body_indent + f"...{sig_comment}\n")
                )

        # Apply from bottom to top to preserve line indices
        replacements.sort(key=lambda r: r[0], reverse=True)
        for start, end, repl in replacements:
            lines[start:end] = [repl]

        return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# F2: Retinal Delta
# ─────────────────────────────────────────────────────────────────────────────

class RetinalDelta:
    """
    F2: Git diff analysis — identifies Hot (recently changed) vs Cold lines.

    Hot lines are always kept at full fidelity during skeletonization.
    All lines are treated as Hot when git is unavailable or history is absent.
    """

    def __init__(self, repo_path: str, n_commits: int = N_COMMITS):
        self.repo_path = Path(repo_path)
        self.n_commits = n_commits
        self._cache: Optional[dict[str, set[int]]] = None

    def extract_diffs(self) -> dict[str, set[int]]:
        """
        Returns {relative_file_path: set_of_changed_1indexed_line_numbers}
        for Python files changed in the last n_commits commits.
        Falls back to {} on any error.
        """
        if self._cache is not None:
            return self._cache

        try:
            result = subprocess.run(
                ["git", "diff", f"HEAD~{self.n_commits}", "HEAD",
                 "--unified=0", "--name-only"],
                capture_output=True, text=True,
                cwd=str(self.repo_path), timeout=30,
            )
            if result.returncode != 0:
                self._cache = {}
                return {}

            diffs: dict[str, set[int]] = {}
            for rel_path in result.stdout.strip().splitlines():
                if not rel_path.endswith(".py"):
                    continue
                lines = self._changed_lines_for(rel_path)
                if lines:
                    diffs[rel_path] = lines

            self._cache = diffs
            return diffs

        except Exception:
            self._cache = {}
            return {}

    def _changed_lines_for(self, rel_path: str) -> set[int]:
        """Parse unified diff for a single file → set of new-side line numbers."""
        try:
            result = subprocess.run(
                ["git", "diff", f"HEAD~{self.n_commits}", "HEAD",
                 "--unified=0", "--", rel_path],
                capture_output=True, text=True,
                cwd=str(self.repo_path), timeout=30,
            )
            if result.returncode != 0:
                return set()

            changed: set[int] = set()
            hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
            for line in result.stdout.splitlines():
                m = hunk_re.match(line)
                if m:
                    start = int(m.group(1))
                    count = int(m.group(2)) if m.group(2) is not None else 1
                    changed.update(range(start, start + count))
            return changed

        except Exception:
            return set()

    def get_hot_lines(self, file_path: str) -> set[int]:
        """
        Return the set of hot 1-indexed line numbers for the given file.
        Returns empty set when file has no recent changes (all Cold).
        """
        diffs = self.extract_diffs()
        abs_target = Path(file_path).resolve()

        for rel_path, changed in diffs.items():
            candidate = (self.repo_path / rel_path).resolve()
            if candidate == abs_target or Path(rel_path).name == abs_target.name:
                return changed

        return set()


# ─────────────────────────────────────────────────────────────────────────────
# F3: Chromatophoric Masker
# ─────────────────────────────────────────────────────────────────────────────

class ChromatophoricMasker:
    """
    F3: Masks repeated imports and boilerplate with [§:hash] markers.

    Maintains a Re-Hydration Dictionary (RHD) registry so unmask() can
    losslessly restore any marker back to its original block. The registry
    is a bijection: each hash maps to exactly one unique block.
    """

    def __init__(
        self,
        registry_path: str = "inb_registry.json",
        min_freq: int = KNOWN_QUANTITY_MIN_FREQ,
        entropy_threshold: float = ENTROPY_THRESHOLD,
    ):
        self.registry_path = Path(registry_path)
        self.min_freq = min_freq
        self.entropy_threshold = entropy_threshold
        self._registry: dict[str, str] = {}
        self._load_registry()

    def _load_registry(self):
        if self.registry_path.exists():
            try:
                with open(self.registry_path, encoding="utf-8") as f:
                    self._registry = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._registry = {}

    def _save_registry(self):
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2)
        except OSError:
            pass

    def make_rhd_hash(self, block: str, salt: int = 0) -> str:
        """8-char SHA-256 hex digest for a block (with optional salt for collision resolution)."""
        return hashlib.sha256(f"{block}|{salt}".encode()).hexdigest()[:8]

    def _register(self, block: str) -> str:
        """Register block in the RHD and return its unique hash ID."""
        h = self.make_rhd_hash(block)
        salt = 0
        while h in self._registry and self._registry[h] != block:
            salt += 1
            h = self.make_rhd_hash(block, salt)
        self._registry[h] = block
        return h

    # Module-level boilerplate patterns that are always maskable regardless of
    # import prefix (low-entropy, appears in virtually every Python module).
    _BOILERPLATE_RE = re.compile(
        r"^(?:logger|log|_logger)\s*=\s*logging\.getLogger\("
        r"|^__all__\s*=\s*\["
        r"|^__version__\s*=\s*[\"']"
        r"|^__author__\s*=\s*[\"']"
    )

    def _is_maskable(self, line: str) -> bool:
        """
        A line is maskable when it is an import statement with low entropy,
        OR a common module-level boilerplate assignment (logger, __all__, etc.).
        Low entropy = repetitive/predictable = "Known Quantity" boilerplate.
        """
        s = line.strip()
        if not s:
            return False
        is_import = s.startswith("import ") or s.startswith("from ")
        if is_import and shannon_entropy(s) < self.entropy_threshold:
            return True
        return bool(self._BOILERPLATE_RE.match(s))

    def mask(self, source: str, file_path: str = "") -> str:
        """Replace maskable lines with [§:hash] markers and persist registry.

        Token-aware: only substitutes when the placeholder is shorter than the
        original line (avoids negative TER on short imports like `import os`).
        """
        out = []
        for line in source.splitlines(keepends=True):
            bare = line.rstrip("\r\n")
            if self._is_maskable(bare):
                h = self._register(bare)
                placeholder = f"{MASK_PREFIX}{h}{MASK_SUFFIX}"
                if count_tokens(placeholder) < count_tokens(bare):
                    eol = line[len(bare):]
                    out.append(f"{placeholder}{eol}")
                else:
                    out.append(line)
            else:
                out.append(line)
        self._save_registry()
        return "".join(out)

    def unmask(self, text: str) -> str:
        """
        Restore all [§:hash] markers back to their original blocks.
        Unrecognised markers (not in registry) are left in place.
        """
        self._load_registry()

        def _replace(m: re.Match) -> str:
            h = m.group(1)
            return self._registry.get(h, m.group(0))

        return _MASK_RE.sub(_replace, text)

    def scan_repo_frequencies(
        self, repo_path: str, extensions: tuple[str, ...] = (".py",)
    ) -> dict[str, int]:
        """
        Count how many files in the repo contain each import line.
        Useful for identifying Known Quantities (imports in N+ files).
        """
        freq: dict[str, int] = {}
        for fpath in Path(repo_path).rglob("*"):
            if fpath.suffix not in extensions:
                continue
            try:
                seen: set[str] = set()
                for line in fpath.read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = line.strip()
                    if s.startswith(("import ", "from ")) and s not in seen:
                        freq[s] = freq.get(s, 0) + 1
                        seen.add(s)
            except OSError:
                continue
        return freq


# ─────────────────────────────────────────────────────────────────────────────
# F4: Caveman Compressor
# ─────────────────────────────────────────────────────────────────────────────

class CavemanCompressor:
    """
    F4: Compresses comments and docstrings by:
      1. Removing English stop-words
      2. Pruning interior vowels from words longer than vowel_prune_min_len

    Only comment (#) lines and content lines inside triple-quoted strings
    are modified. All code lines are passed through unchanged.
    """

    def __init__(
        self,
        stop_words: frozenset = STOP_WORDS,
        vowel_prune_min_len: int = VOWEL_PRUNE_MIN_LEN,
    ):
        self.stop_words = stop_words
        self.vowel_prune_min_len = vowel_prune_min_len
        # Matches interior vowels after a consonant (preserves first/last char)
        self._vowel_re = re.compile(
            r"(?<=[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])[aeiouAEIOU]+",
        )

    def _prune(self, word: str) -> str:
        """Remove interior vowels from a word if it meets the length threshold."""
        if len(word) < self.vowel_prune_min_len or not word[0].isalpha():
            return word
        # Apply only to the interior (keep first and last character)
        inner = self._vowel_re.sub("", word[1:-1])
        return word[0] + inner + word[-1]

    def _compress_text(self, text: str) -> str:
        """Remove stop-words and prune vowels from a plain-text snippet."""
        result = []
        for word in text.split():
            clean = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", word).lower()
            if clean not in self.stop_words:
                result.append(self._prune(word))
        return " ".join(result)

    def _compress_comment_line(self, line: str) -> str:
        """Compress a single `# comment` line."""
        stripped = line.lstrip()
        indent   = line[: len(line) - len(stripped)]
        eol      = "\n" if line.endswith("\n") else ""
        # Separate the `#` marker from the text
        m = re.match(r"(#+\s*)(.*)", stripped.rstrip())
        if not m:
            return line
        marker, text = m.group(1), m.group(2)
        return indent + marker + self._compress_text(text) + eol

    def compress(self, source: str) -> str:
        """
        Compress all comment lines and docstring content lines.
        Code lines are passed through completely unchanged.
        """
        lines  = source.splitlines(keepends=True)
        result = []
        in_triple = False
        tq        = ""          # active triple-quote delimiter

        for line in lines:
            stripped = line.strip()

            if in_triple:
                if tq in stripped:
                    # This line closes the triple string
                    in_triple = False
                    result.append(line)  # keep closing delimiter as-is
                else:
                    # Interior docstring content line — compress
                    lead  = line[: len(line) - len(line.lstrip())]
                    eol   = "\n" if line.endswith("\n") else ""
                    result.append(lead + self._compress_text(stripped) + eol)
                continue

            # Detect opening of a triple-quoted string
            opened = False
            for q in ('"""', "'''"):
                idx = line.find(q)
                if idx == -1:
                    continue
                # Count occurrences: ≥2 means it opens and closes on same line
                if line.count(q) >= 2:
                    # Inline triple string — compress its content
                    def _inline_compress(m: re.Match) -> str:
                        return m.group(1) + self._compress_text(m.group(2)) + m.group(1)
                    compressed = re.sub(
                        r'("""|' + r"'''" + r")(.*?)\1",
                        _inline_compress, line, flags=re.DOTALL
                    )
                    result.append(compressed)
                else:
                    # Opening of a multiline triple string
                    in_triple = True
                    tq        = q
                    result.append(line)  # keep opening line as-is
                opened = True
                break

            if opened:
                continue

            # Plain comment line
            if stripped.startswith("#"):
                result.append(self._compress_comment_line(line))
                continue

            # Everything else: code — pass through unchanged
            result.append(line)

        return "".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# Token Report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenReport:
    """Token counts at each stage of the InB pipeline."""
    file_path: str
    original: int
    after_skeleton: int
    after_masking: int
    after_caveman: int

    @property
    def final(self) -> int:
        return self.after_caveman

    @property
    def reduction_pct(self) -> float:
        if self.original == 0:
            return 0.0
        return round((1.0 - self.final / self.original) * 100.0, 2)

    def __str__(self) -> str:
        def _pct(n: int) -> str:
            if self.original == 0:
                return "0.0"
            return f"{(1 - n / self.original) * 100:.1f}"

        return (
            f"File: {self.file_path}\n"
            f"  Original:       {self.original:>6,} tokens\n"
            f"  After skeleton: {self.after_skeleton:>6,} tokens  ({_pct(self.after_skeleton)}%↓)\n"
            f"  After masking:  {self.after_masking:>6,} tokens  ({_pct(self.after_masking)}%↓)\n"
            f"  After caveman:  {self.after_caveman:>6,} tokens  ({_pct(self.after_caveman)}%↓)\n"
            f"  Total saved:    {self.reduction_pct:.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# InferenceBridge — Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class InferenceBridge:
    """
    Main orchestrator for the Inference-Bridge token compression pipeline.

    Pipeline (fixed order):
      1. F2 Retinal Delta   — fetch hot lines from git diff
      2. F1 Mushroom Body   — skeletonize cold function bodies (needs hot lines)
      3. F3 Chromatophoric  — mask repeated imports (after skeleton: AST no longer needed)
      4. F4 Caveman         — compress comments and docstrings

    Usage:
        inb = InferenceBridge(repo_path="./my-project")
        compressed, report = inb.process_file("./my-project/app/views.py")
        # Send `compressed` to your LLM
        restored = inb.unmask(llm_response)
    """

    def __init__(
        self,
        repo_path: str = ".",
        registry_path: str = "inb_registry.json",
        pi_threshold: float = SKELETAL_PI_THRESHOLD,
        entropy_threshold: float = ENTROPY_THRESHOLD,
        known_quantity_min_freq: int = KNOWN_QUANTITY_MIN_FREQ,
        n_commits: int = N_COMMITS,
        enable_skeleton: bool = True,
        enable_delta: bool = True,
        enable_masking: bool = True,
        enable_caveman: bool = True,
    ):
        self.repo_path       = repo_path
        self.enable_skeleton = enable_skeleton
        self.enable_delta    = enable_delta
        self.enable_masking  = enable_masking
        self.enable_caveman  = enable_caveman

        self._skeletonizer = MushroomBodySkeletonizer(pi_threshold=pi_threshold)
        self._delta        = RetinalDelta(repo_path=repo_path, n_commits=n_commits)
        self._masker       = ChromatophoricMasker(
            registry_path=registry_path,
            min_freq=known_quantity_min_freq,
            entropy_threshold=entropy_threshold,
        )
        self._caveman = CavemanCompressor()

    @classmethod
    def from_config(
        cls, config_path: str = "inb_config.json", **overrides
    ) -> "InferenceBridge":
        """Construct an InferenceBridge from an inb_config.json file."""
        cfg: dict = {}
        p = Path(config_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)

        feats = cfg.get("features", {})
        kwargs: dict = dict(
            pi_threshold          = cfg.get("pi_threshold", SKELETAL_PI_THRESHOLD),
            entropy_threshold     = cfg.get("entropy_threshold", ENTROPY_THRESHOLD),
            known_quantity_min_freq = cfg.get("known_quantity_min_freq", KNOWN_QUANTITY_MIN_FREQ),
            n_commits             = cfg.get("n_commits", N_COMMITS),
            registry_path         = cfg.get("registry_path", "inb_registry.json"),
            enable_skeleton       = feats.get("skeleton", True),
            enable_masking        = feats.get("masking",  True),
            enable_delta          = feats.get("delta",    True),
            enable_caveman        = feats.get("caveman",  True),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def process_file(self, file_path: str) -> tuple[str, TokenReport]:
        """
        Compress a single file through the full InB pipeline.

        Returns:
            (compressed_source, TokenReport)

        The compressed source is safe to pass to your LLM as context.
        Call unmask() on the LLM response to restore any [§:X] markers.
        """
        path = Path(file_path)
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            empty = TokenReport(str(file_path), 0, 0, 0, 0)
            return "", empty

        original_tokens = count_tokens(source)
        current = source

        # F2: Get hot lines (needed before F1)
        hot_lines: Optional[set[int]] = None
        if self.enable_delta:
            hot_lines = self._delta.get_hot_lines(str(file_path)) or None

        # F1: Skeleton
        if self.enable_skeleton:
            current = self._skeletonizer.skeletonize(current, hot_lines=hot_lines)
        after_skeleton = count_tokens(current)

        # F3: Masking (runs after skeleton so AST is still valid before this step)
        if self.enable_masking:
            current = self._masker.mask(current, file_path=str(file_path))
        after_masking = count_tokens(current)

        # F4: Caveman
        if self.enable_caveman:
            current = self._caveman.compress(current)
        after_caveman = count_tokens(current)

        report = TokenReport(
            file_path      = str(file_path),
            original       = original_tokens,
            after_skeleton = after_skeleton,
            after_masking  = after_masking,
            after_caveman  = after_caveman,
        )
        return current, report

    def process_repo(
        self,
        target_extensions: tuple[str, ...] = (".py",),
    ) -> dict:
        """
        Compress every matching file in the repo.

        Returns a dict with keys:
          files_processed, total_original_tokens, total_final_tokens,
          overall_reduction_pct, target_met, files (list of per-file dicts)
        """
        root = Path(self.repo_path)
        py_files = [
            f for f in root.rglob("*")
            if f.suffix in target_extensions
            and "__pycache__" not in str(f)
            and not any(part.startswith(".") for part in f.parts)
        ]

        total_orig = total_final = 0
        file_results: list[dict] = []

        for fpath in py_files:
            try:
                _, report = self.process_file(str(fpath))
            except Exception:
                continue
            total_orig  += report.original
            total_final += report.final
            file_results.append({
                "file":           str(fpath.relative_to(root)),
                "original":       report.original,
                "after_skeleton": report.after_skeleton,
                "after_masking":  report.after_masking,
                "final":          report.final,
                "reduction_pct":  report.reduction_pct,
            })

        overall_pct = (
            round((1.0 - total_final / total_orig) * 100.0, 2)
            if total_orig > 0 else 0.0
        )

        return {
            "files_processed":       len(file_results),
            "total_original_tokens": total_orig,
            "total_final_tokens":    total_final,
            "overall_reduction_pct": overall_pct,
            "target_met":            overall_pct >= 80.0,
            "files":                 file_results,
        }

    def unmask(self, text: str) -> str:
        """Restore any [§:hash] markers in text back to original code."""
        return self._masker.unmask(text)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_mask(args):
    path = Path(args.path)
    repo = str(path.parent if path.is_file() else path)
    inb  = InferenceBridge(repo_path=repo, enable_delta=not args.no_delta)

    if path.is_file():
        compressed, report = inb.process_file(str(path))
        if args.output:
            Path(args.output).write_text(compressed, encoding="utf-8")
            print(f"Saved to {args.output}  ({report.reduction_pct:.1f}% reduction)")
        else:
            print(compressed)
    elif path.is_dir():
        results = inb.process_repo()
        print(json.dumps(results, indent=2))
    else:
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)


def _cmd_unmask(args):
    path = Path(args.path)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    text     = path.read_text(encoding="utf-8")
    masker   = ChromatophoricMasker()
    restored = masker.unmask(text)
    if args.output:
        Path(args.output).write_text(restored, encoding="utf-8")
        print(f"Restored → {args.output}")
    else:
        print(restored)


def _cmd_stats(args):
    path = Path(args.path)
    inb  = InferenceBridge(repo_path=str(path), enable_delta=False)
    res  = inb.process_repo()

    print(f"\nToken Reduction Report: {path}")
    print("─" * 60)
    print(f"  Files processed:    {res['files_processed']}")
    print(f"  Original tokens:    {res['total_original_tokens']:,}")
    print(f"  Final tokens:       {res['total_final_tokens']:,}")
    print(f"  Reduction:          {res['overall_reduction_pct']:.1f}%")
    print(f"  Target (>80%) met:  {'YES ✓' if res['target_met'] else 'NO ✗'}")
    print("─" * 60)

    if getattr(args, "verbose", False):
        for f in sorted(res["files"], key=lambda x: x["reduction_pct"], reverse=True):
            print(f"  {f['reduction_pct']:>5.1f}%  {f['file']}")


def main():
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Inference-Bridge (InB) — Semantic Token Compressor",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_mask = sub.add_parser("mask", help="Compress a file or directory")
    p_mask.add_argument("path")
    p_mask.add_argument("--output", "-o")
    p_mask.add_argument("--no-delta", action="store_true")

    p_unmask = sub.add_parser("unmask", help="Restore [§:X] markers")
    p_unmask.add_argument("path")
    p_unmask.add_argument("--output", "-o")

    p_stats = sub.add_parser("stats", help="Token reduction report for a repo")
    p_stats.add_argument("path")
    p_stats.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    {"mask": _cmd_mask, "unmask": _cmd_unmask, "stats": _cmd_stats}[args.command](args)


if __name__ == "__main__":
    main()
