"""
c_compressor.py — C/C++ compression pipeline
=============================================
Mirrors InferenceBridge F1-F4 for C source without requiring Python AST.

Pipeline:
  F1  CSkeletonizer      — strip predictable function bodies → /* skeletonized */
  F3  C_IncludeMasker    — replace KQ #includes/#defines with [§:HASH] tokens
  F4  CavemanCompressor  — strip stop-words from comments (language-agnostic, reused)

Uses the per-repo CPatternDict built by CPatternMiner.

Design constraints:
  - No libclang, no tree-sitter (runtime optional dependency)
  - Brace-counting parser for skeleton — handles Doom/Quake K&R style fine
  - All passes are idempotent: running twice gives same result
  - FOREGROUND chunks: full F1+F3+F4 pipeline
  - BACKGROUND chunks: F3+F4 only (cheaper, skeleton overhead skipped)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference_bridge import CavemanCompressor, count_tokens, STOP_WORDS, VOWEL_PRUNE_MIN_LEN
from languages.c_pattern_miner import CPatternDict, CPatternMiner, DICT_PATH_RELATIVE


# ── F4 (C-specific): C_CavemanCompressor ───────────────────────────────────────

class C_CavemanCompressor(CavemanCompressor):
    """
    F4 for C/C++: compresses comments while leaving preprocessor directives intact.

    Key difference from the base CavemanCompressor (which is Python-specific):
      - Python `#` = comment marker → base class compresses those lines
      - C `#`     = preprocessor directive (#include, #define, #pragma …) → SKIP entirely
      - C `//`    = single-line comment → compress the text after //
      - C `/* */` = block comment → compress content lines inside the block

    Without this fix the base class was vowel-pruning `#include <stdio.h>` into
    `#inclde <stdio.h>` — silently corrupting every include directive.
    """

    _RE_BLOCK_OPEN  = re.compile(r'/\*')
    _RE_BLOCK_CLOSE = re.compile(r'\*/')
    _RE_LINE_CMT    = re.compile(r'^(\s*)(//+\s?)(.*)')

    def compress(self, source: str) -> str:
        lines  = source.splitlines(keepends=True)
        result = []
        in_block = False

        for line in lines:
            stripped = line.strip()

            # ── inside a /* ... */ block comment ─────────────────────────────
            if in_block:
                if self._RE_BLOCK_CLOSE.search(line):
                    in_block = False
                    result.append(line)   # keep closing */ line as-is
                else:
                    # interior block-comment content — compress
                    lead = line[: len(line) - len(line.lstrip())]
                    eol  = "\n" if line.endswith("\n") else ""
                    # strip leading * decorator if present
                    inner = stripped.lstrip("*").strip()
                    result.append(lead + "* " + self._compress_text(inner) + eol)
                continue

            # ── opening of /* block comment ───────────────────────────────────
            if self._RE_BLOCK_OPEN.search(line):
                # single-line /* ... */ — compress content between markers
                def _compress_block(m: re.Match) -> str:
                    return "/* " + self._compress_text(m.group(1).strip()) + " */"
                compressed = re.sub(r'/\*(.*?)\*/', _compress_block, line)
                result.append(compressed)
                # if /* without */, enter block mode
                if not self._RE_BLOCK_CLOSE.search(line):
                    in_block = True
                continue

            # ── preprocessor directive — SKIP (never compress) ───────────────
            if stripped.startswith("#"):
                result.append(line)
                continue

            # ── // single-line comment ────────────────────────────────────────
            m = self._RE_LINE_CMT.match(line)
            if m:
                lead, marker, text = m.group(1), m.group(2), m.group(3)
                eol = "\n" if line.endswith("\n") else ""
                result.append(lead + marker + self._compress_text(text) + eol)
                continue

            # ── code line — pass through unchanged ───────────────────────────
            result.append(line)

        return "".join(result)

# ── Thresholds — mirror InferenceBridge ────────────────────────────────────────

import os as _os
C_PI_SKELETON_THRESHOLD = float(_os.environ.get("KLOC_CPI_THRESHOLD",   "0.70"))
C_MIN_BODY_LINES        = int(  _os.environ.get("KLOC_CPI_MIN_BODY_LINES", "3"))

_C_SKIP_IDS: frozenset = frozenset({
    # C keywords
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
    "return", "goto", "typedef", "struct", "union", "enum", "void", "int",
    "char", "float", "double", "long", "short", "unsigned", "signed",
    "static", "extern", "const", "volatile", "register", "auto", "inline",
    "sizeof", "NULL", "true", "false",
    # Common C boilerplate
    "ptr", "buf", "len", "idx", "num", "val", "tmp", "ret", "err",
    "printf", "malloc", "free", "memset", "memcpy", "strcmp", "strcpy",
})


def _c_extract_skeleton_identifiers(body_src: str, max_ids: int = 15) -> list:
    """Extract rare domain identifiers from a C function body for skeleton annotation."""
    words = re.findall(r"[a-zA-Z_]\w*", body_src)
    freq: dict = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    candidates = [
        w for w in freq
        if w not in _C_SKIP_IDS
        and not w.startswith("_")
        and len(w) >= 4
        and not w.isupper()
    ]
    candidates.sort(key=lambda w: (freq[w], w))
    return candidates[:max_ids]


# C keywords that raise complexity and lower PI
_C_COMPLEX: frozenset = frozenset([
    "for", "while", "do", "switch", "goto",
    "setjmp", "longjmp", "__asm", "asm", "volatile",
])

# C patterns that count as "simple" (boost PI boilerplate bonus)
_C_SIMPLE_RE = re.compile(
    r"""^\s*(?:
        return\s              |   # return expr;
        [a-zA-Z_]\w*\s*=\s*   |   # assignment
        [a-zA-Z_]\w*\s*\+=    |   # +=
        [a-zA-Z_]\w*\s*-=     |   # -=
        [a-zA-Z_]\w*\s*\+\+   |   # post-increment
        \+\+[a-zA-Z_]\w*      |   # pre-increment
        [a-zA-Z_]\w*\([^)]*\)\s*; |  # simple function call
        if\s*\([^)]+\)\s*return  # guard clause
    )""",
    re.VERBOSE,
)


# ── CTokenReport ───────────────────────────────────────────────────────────────

@dataclass
class CTokenReport:
    original:       int
    after_skeleton: int
    after_masking:  int
    after_caveman:  int

    @property
    def final(self) -> int:
        return self.after_caveman

    @property
    def ter_pct(self) -> float:
        if self.original == 0:
            return 0.0
        return round((1 - self.final / self.original) * 100, 1)

    def __str__(self) -> str:
        return (
            f"TER={self.ter_pct:.1f}%  "
            f"({self.original}→{self.after_skeleton}→{self.after_masking}→{self.final} tok)"
        )


# ── C Predictability Index ─────────────────────────────────────────────────────

class C_PI:
    """
    Predictability Index for C functions.
    Range [0, 1]. Higher = more predictable = safer to skeletonize.

    Formula mirrors InferenceBridge predictability_index() but for C:
      base           = f(line count)
      complexity_pen = 0.06 per complex keyword occurrence (capped at 0.30)
      boilerplate_bon= +0.15 if ≥50% of lines are simple
      loop_free_bon  = +0.10 if n>20 and zero complex keywords
    """

    @staticmethod
    def score(source: str) -> float:
        lines = [l for l in source.splitlines() if l.strip()]
        n = len(lines)

        if   n <= 2:   base = 0.95
        elif n <= 5:   base = 0.90
        elif n <= 10:  base = 0.85
        elif n <= 20:  base = 0.80
        elif n <= 35:  base = 0.70
        else:          base = 0.60

        complex_count = sum(
            1 for l in lines
            if any(re.search(r'\b' + kw + r'\b', l) for kw in _C_COMPLEX)
        )
        _cpi_pen  = float(_os.environ.get("KLOC_CPI_COMPLEXITY_PENALTY", "0.06"))
        _cpi_cap  = float(_os.environ.get("KLOC_CPI_COMPLEXITY_CAP",     "0.30"))
        _cpi_bbon = float(_os.environ.get("KLOC_CPI_BOILERPLATE_BONUS",  "0.15"))
        _cpi_lfb  = float(_os.environ.get("KLOC_CPI_LOOP_FREE_BONUS",    "0.10"))

        complexity_penalty = min(complex_count * _cpi_pen, _cpi_cap)

        simple_count    = sum(1 for l in lines if _C_SIMPLE_RE.match(l))
        # Boilerplate bonus ONLY when complex_count == 0 — mirrors the Python PI formula fix.
        # A loop-containing function must never receive this bonus (critical constraint).
        boilerplate_bon = _cpi_bbon if (n > 0 and (simple_count / n) >= 0.5 and complex_count == 0) else 0.0
        loop_free_bon   = _cpi_lfb  if n > 20 and complex_count == 0 else 0.0

        return round(max(0.0, min(1.0,
            base - complexity_penalty + boilerplate_bon + loop_free_bon
        )), 4)


# ── F1: CSkeletonizer ──────────────────────────────────────────────────────────

class CSkeletonizer:
    """
    F1 for C: replaces predictable (high-PI, cold) function bodies
    with a single-line skeleton marker.

    Uses brace-counting (not AST) — handles K&R, C99, C11 and macro-heavy
    code as found in Doom, Quake, etc.

    Does NOT touch:
      - Control flow blocks (if/for/while/switch/do)
      - Struct/union/enum bodies
      - Functions with any hot lines inside them
      - Functions with PI < threshold
      - Bodies shorter than C_MIN_BODY_LINES
    """

    _CONTROL_FLOW_RE = re.compile(
        r'\b(if|else|for|while|do|switch|struct|union|enum|typedef)\b\s*[({]'
    )
    _FN_RETURN_TYPES = re.compile(
        r'\b(void|int|char|float|double|unsigned|signed|long|short|'
        r'static|inline|extern|bool|size_t|uint\d+_t|int\d+_t|'
        r'byte|word|dword|fixed_t|angle_t|mobjtype_t)\b'
    )

    def __init__(self, threshold: float = C_PI_SKELETON_THRESHOLD):
        self.threshold = threshold

    def skeletonize(
        self,
        source:      str,
        hot_lines:   Optional[set] = None,
        file_offset: int = 0,
    ) -> Tuple[str, int]:
        """
        Returns (skeletonized_source, count_of_skeletonized_functions).

        hot_lines: set of 1-based absolute line numbers that must not be touched.
        file_offset: add to line numbers when checking hot_lines (for chunk subsets).
        """
        hot_lines = hot_lines or set()
        lines  = source.splitlines(keepends=True)
        result = list(lines)
        skeletonized = 0
        i = 0

        while i < len(result):
            stripped = result[i].rstrip()

            if stripped.endswith("{") and self._is_fn_opener(result, i):
                body_start = i + 1
                body_end   = self._find_close_brace(result, i)

                if body_end is None or (body_end - body_start) < C_MIN_BODY_LINES:
                    i += 1
                    continue

                # Check for hot lines in this body range
                abs_start = file_offset + body_start + 1
                abs_end   = file_offset + body_end
                is_hot = any(abs_start <= hl <= abs_end for hl in hot_lines)
                if is_hot:
                    i += 1
                    continue

                body_src = "".join(result[body_start:body_end])
                pi       = C_PI.score(body_src)
                n_lines  = body_end - body_start

                if pi >= self.threshold:
                    indent = re.match(r'(\s*)', result[body_start]).group(1) if result[body_start:] else "    "
                    _c_sig_retain  = _os.environ.get("KLOC_SKELETON_SIG_RETAIN", "0") == "1"
                    _c_sig_max_ids = int(_os.environ.get("KLOC_SKELETON_SIG_MAX_IDS", "15"))
                    if _c_sig_retain:
                        sig_ids = _c_extract_skeleton_identifiers(body_src, _c_sig_max_ids)
                        sig_comment = (" " + " ".join(sig_ids)) if sig_ids else ""
                    else:
                        sig_comment = ""
                    result[body_start:body_end] = [
                        f"{indent}/* ...{n_lines}L PI={pi:.2f}{sig_comment}... */\n"
                    ]
                    skeletonized += 1
                    i = body_start + 1
                    continue

            i += 1

        return "".join(result), skeletonized

    def _is_fn_opener(self, lines: List[str], idx: int) -> bool:
        """
        Heuristic: is the { on lines[idx] opening a function body?
        Look back up to 5 lines for a signature pattern.
        """
        window_start = max(0, idx - 4)
        sig = " ".join(l.strip() for l in lines[window_start : idx + 1])

        # Reject pure control flow openers (not if — allow else-if chains)
        if re.search(r'\b(for|while|do|switch)\b\s*\(', sig):
            return False
        # Reject struct/union/enum/typedef bodies
        if re.search(r'\b(struct|union|enum|typedef)\b', sig):
            return False

        # Accept known return types
        if self._FN_RETURN_TYPES.search(sig):
            return True
        # Accept pointer return:  Entity *enemy_spawn(...) {
        if re.search(r'\w+\s*\*+\s*\w+\s*\(', sig):
            return True
        # Accept typed-param signature:  foo(Type *bar, int n) {
        if re.search(r'\w+\s*\([^)]*\w+\s+\*?\w+[^)]*\)\s*\{', sig):
            return True
        # Accept no-param or void-param:  foo(void) {  or  foo() {
        if re.search(r'\w+\s*\(\s*(?:void\s*)?\)\s*\{', sig):
            return True

        return False

    @staticmethod
    def _find_close_brace(lines: List[str], open_line: int) -> Optional[int]:
        """
        Find the 0-based index of the line containing the matching } for the
        { on lines[open_line].  Returns None if not found.
        """
        depth = 0
        for i in range(open_line, len(lines)):
            depth += lines[i].count("{") - lines[i].count("}")
            if i > open_line and depth == 0:
                return i
        return None


# ── F3: C_IncludeMasker ────────────────────────────────────────────────────────

class C_IncludeMasker:
    """
    F3 for C: replaces known-quantity #include, #define, and #pragma
    directives with short [§:HASH] placeholder tokens.

    Only patterns in the per-repo CPatternDict (frequency >= KQ_THRESHOLD)
    are masked — everything else is left intact.
    """

    _RE_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"][^>"]+[>"]', re.MULTILINE)
    _RE_DEFINE  = re.compile(r'^\s*#\s*define\s+\w+(?:\([^)]*\))?\s+\S[^\n]*', re.MULTILINE)
    _RE_PRAGMA  = re.compile(r'^\s*#\s*pragma\s+\S[^\n]*', re.MULTILINE)

    def __init__(self, pattern_dict: CPatternDict):
        self._pdict = pattern_dict
        # Pre-filter: only keep entries where the compressed form is actually cheaper.
        # count_tokens is O(n) so we do it once at construction, not per-match.
        self._savings: Dict[str, str] = {}   # original -> compressed (token-cheaper only)
        for e in pattern_dict.patterns:
            orig_t  = count_tokens(e.original)
            comp_t  = count_tokens(e.compressed)
            if comp_t < orig_t:
                self._savings[e.original] = e.compressed

    def mask(self, source: str) -> Tuple[str, int]:
        """Returns (masked_source, mask_count).

        Only replaces patterns where the placeholder is actually fewer tokens
        than the original text — prevents negative TER on short patterns.
        """
        count  = [0]
        result = source

        def _replace(m: re.Match) -> str:
            repl = self._savings.get(m.group(0).strip())
            if repl:
                count[0] += 1
                return repl
            return m.group(0)

        result = self._RE_INCLUDE.sub(_replace, result)
        result = self._RE_DEFINE.sub(_replace,  result)
        result = self._RE_PRAGMA.sub(_replace,  result)

        # Mask repeated identifiers (ALL_CAPS macros, long_underscore names)
        id_savings = {
            orig: comp
            for orig, comp in self._savings.items()
            if self._pdict.lookup_original(orig) is not None
            and self._pdict.lookup_original(orig).pattern_type == "identifier"
        }
        if id_savings:
            sorted_ids = sorted(id_savings.keys(), key=len, reverse=True)
            id_re = re.compile(
                r'\b(' + '|'.join(re.escape(k) for k in sorted_ids[:60]) + r')\b'
            )
            def _replace_id(m: re.Match) -> str:
                repl = id_savings.get(m.group(0))
                if repl:
                    count[0] += 1
                    return repl
                return m.group(0)
            result = id_re.sub(_replace_id, result)

        return result, count[0]

    def unmask(self, text: str) -> str:
        """
        Reverse all [§:HASH] substitutions in *text*, restoring the original
        #include / #define / #pragma lines.

        Iterates over self._savings (original → compressed) and replaces every
        occurrence of the compressed token with its original text.  Uses regex
        word-boundaries so a token embedded in a larger string is still matched.
        Unknown / non-C tokens are left untouched.
        """
        import re as _re
        result = text
        # Build reverse map: compressed → original
        # Sort by compressed length descending so longer tokens match first
        # (avoids partial-match issues if two tokens share a prefix).
        reverse = sorted(
            ((comp, orig) for orig, comp in self._savings.items()),
            key=lambda kv: -len(kv[0]),
        )
        for compressed_token, original in reverse:
            # Escape the token for use in a regex; it contains [§: ] characters
            pattern = _re.escape(compressed_token)
            result  = _re.sub(pattern, lambda m, o=original: o, result)
        return result


# ── Main CCompressor ───────────────────────────────────────────────────────────

class CCompressor:
    """
    Full C/C++ compression pipeline.

    FOREGROUND: F1 (skeleton) + F3 (masking) + F4 (caveman)
    BACKGROUND: F3 (masking) + F4 (caveman)  [skip skeleton — too cheap to matter]

    Construct via CCompressor.for_repo(path) which auto-mines or loads
    the per-repo pattern dict.
    """

    def __init__(self, pattern_dict: CPatternDict):
        self.pattern_dict = pattern_dict
        self.skeletonizer = CSkeletonizer()
        self.masker       = C_IncludeMasker(pattern_dict)
        self.caveman      = C_CavemanCompressor()   # C-aware: skips #preprocessor lines

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def for_repo(
        cls,
        repo_path:    Path,
        force_remine: bool = False,
    ) -> "CCompressor":
        """
        Load existing pattern dict if present, else mine and save it.
        """
        repo_path = Path(repo_path)
        dict_path = repo_path / DICT_PATH_RELATIVE

        if not force_remine and dict_path.exists():
            pdict = CPatternDict.load(dict_path)
        else:
            miner = CPatternMiner()
            pdict = miner.mine(repo_path)
            pdict.save(dict_path)

        return cls(pdict)

    # ── Core compression ───────────────────────────────────────────────

    def compress(
        self,
        source:    str,
        hot_lines: Optional[set] = None,
        zone:      str = "FOREGROUND",
    ) -> Tuple[str, CTokenReport]:
        """
        Compress a C source string.

        Returns (compressed_source, CTokenReport).
        """
        orig_t = count_tokens(source)

        # F1: Skeleton — FOREGROUND only
        if zone == "FOREGROUND":
            after_skel, _skel_n = self.skeletonizer.skeletonize(source, hot_lines)
        else:
            after_skel = source
        skel_t = count_tokens(after_skel)

        # F3: Include / define masking
        after_mask, _mask_n = self.masker.mask(after_skel)
        mask_t = count_tokens(after_mask)

        # F4: Caveman (comment compression — already language-agnostic)
        after_cav = self.caveman.compress(after_mask)
        cav_t     = count_tokens(after_cav)

        report = CTokenReport(
            original       = orig_t,
            after_skeleton = skel_t,
            after_masking  = mask_t,
            after_caveman  = cav_t,
        )
        return after_cav, report


# ── CLI self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import argparse
    parser = argparse.ArgumentParser(description="C compressor self-test")
    parser.add_argument("--repo",  required=True, help="Repo root (must contain C files)")
    parser.add_argument("--file",  required=True, help="A .c file inside the repo to compress")
    parser.add_argument("--zone",  default="FOREGROUND", choices=["FOREGROUND", "BACKGROUND"])
    parser.add_argument("--force-remine", action="store_true")
    args = parser.parse_args()

    repo_path = Path(args.repo)
    file_path = Path(args.file)

    print(f"\n  Repo: {repo_path}")
    compressor = CCompressor.for_repo(repo_path, force_remine=args.force_remine)
    print(f"  Pattern dict: {compressor.pattern_dict.summary()}")

    source = file_path.read_text(encoding="utf-8", errors="replace")
    compressed, report = compressor.compress(source, zone=args.zone)

    print(f"\n  File:  {file_path.name}")
    print(f"  Zone:  {args.zone}")
    print(f"  {report}")
    print(f"\n  --- First 30 lines of compressed output ---")
    for ln in compressed.splitlines()[:30]:
        print(f"  {ln}")
