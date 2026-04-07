# PRUNER Agent Feedback — v1 · SDR Classification

**Auditor:** Claude Sonnet 4.6  
**Date:** 2026-04-06  
**File reviewed:** `agents/pruner.py`  
**Review skill:** `.claude/commands/review-pruner.md`

---

## 1. SDR Fidelity — Verdict: Superficial Metaphor

In Hawkins & Ahmad (2016), SDR overlap scoring operates on **binary vectors**: two patterns overlap when they share active bits (columns), with overlap count used as a similarity measure. Key properties: binary representation, `overlap = |A ∩ B|`, sparsity enforced by winner-take-all inhibition.

What PRUNER actually does:

```python
sdr_overlap = pi_score * (1.0 - hzs_score)   # line 153
```

This is a scalar product of two continuous floats — no binary structure, no vector dimensionality, no overlap threshold comparison. Worse: `sdr_overlap` is **computed but never used in the classification decision** (lines 156–160). The actual decision uses only `is_hot` and `is_complex`. `sdr_overlap` is purely decorative in the output dict.

---

## 2. Sparsity Target — Verdict: 10× Too Permissive, Zero Enforcement

Real HTM cortical columns maintain **~2% active columns** (hard constraint via k-WTA inhibition). PRUNER sets `SDR_SPARSITY_TARGET = 0.20` with comment `# these mirror Numenta's ~2% sparsity target` — factually incorrect: 20% ≠ ~2%.

The target is never enforced. The code computes `fg_ratio` post-hoc and uses it only in a confidence calculation:

```python
confidence = 1.0 - abs(fg_ratio - self.SDR_SPARSITY_TARGET)   # line 105
```

`SDR_SPARSITY_TARGET` has zero causal effect on which chunks become FOREGROUND.

Confidence range issue: with `abs(fg_ratio - 0.20)` bounded to [0, 0.80], confidence ranges 0.20–1.0. Pure boilerplate input reports confidence 0.80 — misleadingly high.

---

## 3. PI → Column Overlap Mapping — Three Specific Divergences

**Divergence 1 — Direction inverted from HTM semantics.** In HTM, high overlap = familiarity = recognition. But low PI (complex, novel) → FOREGROUND maps to "low overlap = unrecognized," which in HTM triggers a burst. The metaphor collapses here.

**Divergence 2 — PI is Python-only.** `_COMPLEX` and `_SIMPLE` regex patterns reference Python-specific tokens (`self.`, `super()`, `logger.`, `yield`). PRUNER accepts C, Go, Rust but feeds them to a Python-tuned scoring function. A 200-line C function with `for` loops will be mis-scored because `self.\w+\s*=` never matches. **This is a correctness bug for non-Python input.**

**Divergence 3 — Base scores are empirically calibrated, not derived.** PI assigns `base = 0.95` for 1–2 line functions with no mathematical relationship to HTM's overlap threshold.

---

## 4. HZS → Dendrite Mapping — Wrong Type, Critical Default Bug

**Wrong dendrite type:** `hot_zone_score` measures spatial proximity to recently changed lines — a saliency/attention signal. HTM's proximal/distal distinction is about *feedforward vs contextual* information streams — entirely different axes.

**Critical bug — `hot_zone_score` default (inference_bridge.py line ~200):**

```python
if not changed_lines:
    return 1.0   # no diff info → treat all lines as Hot
```

When no git diff is available, HZS returns 1.0 for every line. Every chunk passes the `is_hot` check → becomes FOREGROUND. The `_classify_chunk` fallback (passes `hzs_score=0.0` when `hot_lines` is empty) dodges this by accident. The function itself is a footgun for any future caller.

**Chunk-range bug:** HZS evaluates `start_line` only. For a 40-line function starting at line 50, a hot change at line 89 scores `exp(-0.1×39) ≈ 0.02` — COLD — even though the change is inside the function body. Correct implementation should use minimum distance from any chunk line to any hot line.

---

## 5. Five Missing HTM Features

**1. Winner-Take-All Inhibition Across Chunks.** No cross-chunk inhibition. A k-WTA pass after classification — selecting only top-k chunks by `sdr_overlap` — would enforce the sparsity target and rank chunks.

**2. Temporal Sequence Memory (Prediction State).** No memory of prior classifications. Re-running PRUNER on the same file twice produces identical results regardless of whether predictions were correct. HTM distinguishes stable-complex (predicted active) from newly-complex (burst).

**3. Synapse Permanence / Learning.** Thresholds (`FOREGROUND_PI_THRESHOLD = 0.70`, `FOREGROUND_HZS_THRESHOLD = 0.10`) are static constants. A per-repository adaptive threshold would make the sparsity target actually binding.

**4. Union Property for Robustness.** PRUNER uses simple OR (`is_hot or is_complex`). Boolean OR false-positive rate = sum of each individual rate. Weighted combination with learned weights would better exploit the SDR union property.

**5. Column Boosting for Underrepresented Patterns.** HTM boosts rarely-active columns so their threshold lowers over time. Chunk types that are always BACKGROUND (e.g., `__init__` methods) would benefit from a score nudge, preventing systematic blind spots.

---

## 6. Line-Level Code Bugs

**Bug 1 — `sdr_overlap` unused in classification (lines 153–160)**

```python
sdr_overlap = pi_score * (1.0 - hzs_score)   # computed
is_foreground = is_hot or is_complex           # sdr_overlap NOT used
```

If this is the primary HTM-inspired signal, the logic should use it. If it's metadata, the comment claiming it "simulate[s] % active columns" is misleading.

**Bug 2 — `start_line`-only HZS evaluation**

```python
hzs_score = hot_zone_score(start_line, hot_lines)   # should use range(start, end)
```

**Bug 3 — Hardcoded output confidence ignores computed confidence**

```python
return self.emit(..., confidence=0.90)   # line 121 — ignores fg_ratio confidence computed above
```

**Bug 4 — `_lang_tags()` called as instance method, allocates new list per call.** Should be a module-level `frozenset`. Also silently falls back to `lang.unknown` with no warning logged.

**Bug 5 — Empty `file_path` silently discards HZS signal:**

```python
if git_hot and file_path:   # empty string → HZS disabled with no log
    hot_lines = set(...)
```

---

## 7. Proposed Fixes (Prioritized)

### Fix 1 — k-WTA Sparsity Enforcement (makes SDR metaphor structurally honest)

```python
# In process(), replace the classification loop:
scored = [self._classify_chunk(chunk, hot_lines) for chunk in chunks]
max_fg = max(1, int(len(scored) * self.SDR_SPARSITY_TARGET))
ranked = sorted(enumerate(scored), key=lambda x: x[1]["sdr_overlap"])
fg_indices = {i for i, _ in ranked[:max_fg]}

classified = []
for i, result in enumerate(scored):
    if result["zone"] == "FOREGROUND" and i not in fg_indices:
        result = {**result, "zone": "BACKGROUND", "demoted_by_sparsity": True}
    classified.append(result)
```

### Fix 2 — Chunk-Range HZS + No-Diff Default

```python
# In _classify_chunk:
end_line = chunk.get("end_line", start_line)
if _INB_AVAILABLE and hot_lines:
    hzs_score = max(hot_zone_score(line, hot_lines) for line in range(start_line, end_line + 1))
else:
    hzs_score = 0.0

# In inference_bridge.py, hot_zone_score:
if not changed_lines:
    return 0.0  # unknown ≠ hot
```

### Fix 3 — Language-Aware PI via Dispatch Table

Add `_pi_for_language(source, language)` in `pruner.py` with C/Go/Rust-appropriate regex patterns. Fall through to InB's `predictability_index` for Python. This requires chunker to populate `chunk_language` on each chunk dict.

---

## Summary

| Issue | Severity |
|---|---|
| `sdr_overlap` unused in classification | High |
| Sparsity target 0.20 ≠ 2%; no enforcement | Medium |
| Python-only PI for multi-language input | High (correctness bug) |
| `start_line`-only HZS misses intra-chunk hot changes | High |
| No-diff HZS default returns 1.0 (footgun) | High |
| Hardcoded output confidence 0.90 ignores computed value | Medium |
| 5 missing HTM features | Medium |

The core FOREGROUND/BACKGROUND heuristic is functional. The SDR framing is a metaphor stretched past its breaking point. The two highest-impact bugs are the `start_line`-only HZS evaluation and `sdr_overlap` having zero influence on the classification outcome.
