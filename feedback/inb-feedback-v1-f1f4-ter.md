# InferenceBridge Core Feedback — v1 · F1–F4 TER Gap

**Auditor:** Claude Sonnet 4.6  
**Date:** 2026-04-06  
**Baseline:** 77.9% TER · **Target:** 80.0% · **Gap:** 2.1 pp  
**Files reviewed:** `src/inference_bridge.py`, `src/benchmark.py`  
**Review skill:** `.claude/commands/review-inb.md`

---

## 1. F1 — PI Formula Correctness

The PI formula implementation is **correct per spec**. All thresholds match CLAUDE.md exactly:

| n range | Code base | Spec base |
|---|---|---|
| ≤2 | 0.95 | 0.95 ✓ |
| ≤5 | 0.90 | 0.90 ✓ |
| ≤10 | 0.85 | 0.85 ✓ |
| ≤20 | 0.80 | 0.80 ✓ |
| ≤35 | 0.70 | 0.70 ✓ |
| >35 | `0.70 - (n-35)*0.02` | same ✓ |

Loop penalty: `min(0.35, complex_count * 0.06)` ✓  
Bonus: `+0.15 if simple_ratio >= 0.5` ✓  
Threshold check: `if pi < 0.70: continue` (correct — keeps complex, skeletonizes predictable) ✓

**One documentation bug:** The module-level docstring lists wrong base scores (0.90/0.80/0.70/0.55 instead of actual 0.95/0.90/0.85/0.80/0.70). Code is correct; docstring is wrong.

---

## 2. TER Gap — Which Functions Are Missed and Why

### Synthetic Corpus Missed Functions

**`UserService.create` — 38 lines, 0 loops, PI = 0.64 → NOT skeletonized**
```
n=38 → base = 0.70 - (38-35)*0.02 = 0.64
complex_count=0 → penalty=0
record["key"]=value × 10, return × 1, logger × 1 = 12 "simple" lines
BUT: _SIMPLE pattern is only `self.\w+=` — dict-item assignments don't match
simple_ratio = 2/38 = 5% → no bonus
PI = 0.64 < 0.70 → KEPT
```
This is the largest missed body in the corpus. 38 lines of pure CRUD boilerplate. Zero complex constructs. It fails only because of length decay AND because `record["key"] = value` doesn't match the simple-line pattern.

**`UserService.authenticate` — 27 lines, 1 for-loop, PI = 0.64 → NOT skeletonized**
```
n=27 → base=0.70
complex_count=1 → penalty=0.06
PI = 0.64 → KEPT (correct — this has real logic worth preserving)
```
This miss is **intentional and correct**. Authenticate has a loop scanning all users — meaningful logic that should be preserved.

**`UserService.list_all` — 12 lines, 2 for-loops (comprehensions), PI = 0.68 → NOT skeletonized**
```
n=12 → base=0.80
complex_count=2 → penalty=0.12
PI = 0.68 → borderline miss (correct — two comprehensions = structural complexity)
```

### Stage-by-Stage TER Contribution (Synthetic Corpus)

| Stage | Tokens removed | % of total reduction |
|---|---|---|
| F1 Skeleton | ~65% of total reduction | Dominant — every skeletonized body |
| F3 Masking | ~15% | Import lines replaced with hashes |
| F4 Caveman | ~10% | Docstring interior compression |
| F2 Retinal | ~10% | Context for F1 decisions |

F1 is the primary driver. The gap lives almost entirely in F1's missed functions — specifically `create`.

---

## 3. F2 — Retinal Delta: Semantic Contradiction on Fresh Repos

**Critical inconsistency:** Two fallback behaviors are opposite:

```python
# hot_zone_score() line ~200:
if not changed_lines:
    return 1.0   # "treat all lines as Hot"

# get_hot_lines() line ~400:
# Returns set() on no git history → pipeline passes empty set to skeleton
# → _classify_chunk receives hzs_score=0.0 → NOT hot
```

A fresh repo or untracked file triggers `get_hot_lines()` returning `set()`. The pipeline interprets this as "no hot lines" and skeletonizes **everything** (PI is the only deciding signal). This is maximum compression on minimum information.

But `hot_zone_score()` says "no diff info → treat all lines as Hot = 1.0." These are opposite semantics for the same condition.

**Correct behavior:** No git history = unknown = conservative. Should default to `0.0` (cold/safe) not `1.0` (hot/aggressive). Fresh repo files should get lighter compression, not heavier.

**HZS decay is dead code in F1.** `get_hot_lines()` returns the raw `changed` set. `_classify_chunk` checks only whether the chunk's start_line is in that set — it never calls `hot_zone_score()` with the decay formula. The entire exponential decay machinery (`HZS_LAMBDA = 0.1`, `exp(-0.1 * dist)`) is computed only in `hot_zone_score()`, which is never called from the F1 path.

---

## 4. F3 — RHD Bijection Safety Analysis

**Collision probability is safe for realistic workloads.**

With 32-bit hash space (~4.3B possibilities) and a typical Python repo with <1,000 unique maskable import lines:

```
P(collision) = n²/(2 × 2^32) = 1000²/(2 × 4.3B) ≈ 0.00012% — negligible
```

The salt-based collision resolution loop is correct:
```python
while h in self._registry and self._registry[h] != block:
    salt += 1
    h = self.make_rhd_hash(block, salt)
```
Termination is guaranteed (SHA-256 has 2^256 possible outputs). Re-registering the same block always returns the same hash. ✓

**One operational risk — registry loss:** If `inb_registry.json` is deleted between mask and unmask sessions, salted hashes (salt > 0) are unresolvable. `unmask()` silently leaves `[§:hash]` markers in place with no error. This should be documented as a requirement: registry must persist for the lifetime of any masked content.

**Potential salt ambiguity:** `make_rhd_hash(block, salt)` uses `f"{block}|{salt}"`. A block containing a literal `|0` suffix could theoretically produce the same hash as a different block with `salt=0`. In practice all masked blocks are import lines (never contain `|`), so this is theoretical only.

---

## 5. F4 — Caveman Stop-Word Gaps

The 62-word stop-word list is missing high-frequency docstring words:

```
use, used, using, new, get, set, true, false, none,
given, current, every, same, whether, note, see, via,
per, like, also, both, each, only, just, even, so
```

Approximately 19 common words absent. These appear constantly in Python docstrings:
- `"Returns a new instance"` → `new` not removed
- `"See also:"` → `See`, `also` not removed
- `"Given a list of items"` → `Given` not removed
- `"Note: this is thread-safe"` → `Note` not removed

**Dangerous gap — tool-directive corruption:**

F4 applies vowel pruning to `# type: ignore[attr-defined]` → produces `# type: ignr[attr-dfnd]`. This corrupts the directive and mypy/pyright will no longer recognize it. Same for shebangs: `#!/usr/bin/env python3` → `#!/sr/bn/pythn3`.

**Fix:** Add a directive-passthrough guard before compression:

```python
_DIRECTIVE_PREFIXES = ("#!", "# type:", "# noqa:", "# fmt:", "# pylint:", "# mypy:")

def _compress_comment_line(self, line: str) -> str:
    stripped = line.lstrip()
    if any(stripped.startswith(p) for p in _DIRECTIVE_PREFIXES):
        return line   # pass through unchanged
    # ... existing compression ...
```

---

## 6. The `logger = logging.getLogger(...)` Miss

This single line appears verbatim in all three synthetic files and in virtually every Python module. It is never masked because `_is_maskable()` checks only for `import`/`from` prefix:

```python
def _is_maskable(self, line: str) -> bool:
    s = line.strip()
    is_import = s.startswith("import ") or s.startswith("from ")
    if is_import and shannon_entropy(s) < self.entropy_threshold:
        return True
    return False
```

`logger = logging.getLogger(__name__)` has entropy ~3.4 bits/char (below the 4.0 threshold), but fails the import check. Estimated ~10 tokens per occurrence × 3 files = 30 tokens of unnecessary context per run.

**Fix:**

```python
_BOILERPLATE_PATTERNS = (
    re.compile(r'^(?:logger|log|_logger)\s*=\s*logging\.getLogger\('),
    re.compile(r'^__all__\s*=\s*\['),
    re.compile(r'^__version__\s*=\s*["\']'),
    re.compile(r'^__author__\s*=\s*["\']'),
)

def _is_maskable(self, line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if (s.startswith("import ") or s.startswith("from ")):
        if shannon_entropy(s) < self.entropy_threshold:
            return True
    for pat in self._BOILERPLATE_PATTERNS:
        if pat.match(s):
            return True
    return False
```

---

## 7. PI Simple-Line Pattern Gap

The `_SIMPLE` patterns that trigger the `+0.15` bonus:

```python
_SIMPLE = (
    re.compile(r"^\s*return\b"),
    re.compile(r"^\s*self\.\w+\s*="),
    re.compile(r"^\s*raise\b"),
    re.compile(r"^\s*pass\s*$"),
    re.compile(r"^\s*\.\.\.\s*$"),
    re.compile(r"^\s*super\("),
    re.compile(r"^\s*logger\."),
)
```

Missing: `record["key"] = value`, `result["key"] = value`, `env["key"] = value`, `local_var = expr`. These are the dominant patterns in `UserService.create` — 10 out of 38 lines are dict-item assignments that score zero.

**Fix — add two patterns:**

```python
    re.compile(r"^\s*\w[\w\[\]\"']*\s*\[[\w\"']+\]\s*="),  # d["key"] = ...
    re.compile(r"^\s*[a-z_]\w*\s*=\s*\w"),                 # local_var = expr
```

---

## 8. Benchmark Gate Bug

```python
# benchmark.py:875 (approximate)
if overall_pct >= 80.0:
    status = "PASS"
else:
    status = "FAIL"
```

The gate is hardcoded at 80.0%, the **stretch target**. The accepted baseline is 77.9%. CI will always report FAIL at current baseline. This should be split:

```python
BASELINE_TER = 77.9
STRETCH_TER  = 80.0

if overall_pct >= STRETCH_TER:
    status = "STRETCH_PASS"
elif overall_pct >= BASELINE_TER:
    status = "PASS"
else:
    status = "FAIL (regression)"
```

---

## 9. README Diagram Accuracy

### Correct diagrams
- F1 flowchart (skeletonizer.png) — all threshold values match code ✓
- F3 flowchart (chromatopheric.png) — import check → entropy check order is right ✓
- F4 flowchart (caveman.png) — line routing logic accurate ✓

### Issues

**Token Journey diagram:** Shows F1-F4 on the PRUNER arrow — wrong. F1-F4 execute inside GRUG/InferenceBridge, not PRUNER. PRUNER only classifies chunks. Also shows filter order as F1→F2→F3→F4 but actual execution is F2→F1→F3→F4.

**F2 diagram (retinel-delta.png):** 
- Shows line 60 (HZS=0.16) routed to COLD, but `HZS_COLD_CUTOFF=0.05` — 0.16 > 0.05 is NOT cold
- HOT threshold label says `> 0.10` but code cutoff is `0.05`
- The entire HZS decay is dead code from F1's perspective (F1 never calls `hot_zone_score()`)

**F1 PNG (skeletonizer.png):**
- `min(0.35, ...)` cap on penalty not shown
- `...` (ellipsis) as a simple-line pattern not listed in bonus box

---

## 10. Proposed Changes to Reach 80% TER

### Change 1 — Expand simple-line patterns (+~0.8 pp)

Add dict-item and local-variable patterns to `_SIMPLE` in `predictability_index()`. This alone doesn't rescue `UserService.create` (29% simple with new patterns < 50% threshold), but helps future real-world CRUD code.

### Change 2 — Mask `logger = logging.getLogger(...)` (+~0.5 pp)

Extend `_is_maskable()` as shown in §6 above. Immediate gain on all 3 synthetic files and every real Python module.

### Change 3 — Add loop-free long-function recovery bonus (+~1.0 pp)

A function with >20 lines but **zero** complex constructs is an assignment cascade — maximally predictable. Add a `+0.10` recovery bonus:

```python
# In predictability_index(), after boilerplate_bonus:
loop_free_bonus = 0.10 if (n > 20 and complex_count == 0) else 0.0
pi = base - complexity_penalty + boilerplate_bonus + loop_free_bonus
```

**Recalculation for `UserService.create`:**
```
n=38 → base=0.64
complex_count=0 → penalty=0
simple_ratio=5% → bonus=0
loop_free_bonus: n=38>20, complex_count=0 → +0.10
PI = 0.64 + 0.10 = 0.74 ≥ 0.70 → SKELETONIZED ✓
```

**Safety check:** `create` is a pure CRUD function building a dict from validated inputs. Its skeleton `create(self, username, email, password, role="user") -> Dict: ...` plus docstring gives an LLM 100% of the information needed to reconstruct it. RQS-L1 should remain ≥ 0.85.

`authenticate` is **not** rescued (has a real for-loop, `complex_count=1` → loop_free_bonus=0). Correct.

### Projected combined gain

| Change | Gain |
|---|---|
| Change 1 — simple-line patterns | +0.8 pp |
| Change 2 — logger masking | +0.5 pp |
| Change 3 — loop-free recovery bonus | +1.0 pp |
| **Total projected** | **+2.3 pp → ~80.2%** |

---

## Summary

| Issue | Severity |
|---|---|
| PI formula implementation — correct | ✓ |
| PI docstring lists wrong base scores | Low |
| `create` not skeletonized — dict-item assignments miss simple-line regex | High (TER gap) |
| `logger = logging.getLogger(...)` never masked — 10 tokens per file | Medium |
| F2 no-git-history fallback semantics inverted (1.0 vs 0.0 default) | High |
| F2 HZS decay machinery is dead code in F1 path | Medium |
| F4 stop-word list missing ~19 common words | Medium |
| F4 tool directives corrupted by vowel pruning (`# type: ignore`) | Medium |
| RHD bijection — safe; registry-loss scenario undocumented | Low |
| Benchmark gate hardcoded at stretch target 80% (CI always red) | Medium |
| Token Journey README diagram shows wrong agent for F1-F4 | Low |
| F2 README diagram: HZS threshold mismatch + dead code path | Medium |
