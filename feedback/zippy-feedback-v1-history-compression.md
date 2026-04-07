# ZIPPY Agent Feedback — v1 · History Compression

**Auditor:** Claude Sonnet 4.6  
**Date:** 2026-04-06  
**File reviewed:** `agents/zippy.py`, `orchestrator/state_store.py`  
**Review skill:** `.claude/commands/review-zippy.md`

---

## 1. 50-Token Budget — Wrong Floor for Code Tasks

The 50-token ceiling is stated everywhere as if it were a research result. LLMLingua (Wu et al. 2024, arXiv:2310.05736) does **not** endorse a single universal floor — the minimum viable context is task-type-dependent:

| Task type | LLMLingua finding | Min viable budget (1K input) |
|---|---|---|
| Summarisation / factoid QA | 20× compression safe | ~50 tokens ✓ |
| Multi-step reasoning (GSM8K) | Quality degrades below 4× | ~250 tokens |
| Code generation (HumanEval) | Brittle even at 6× | ~85 tokens |

ZIPPY has **no task-type branching** in `_truncate_to_budget`. `task_type` is received in the payload and passed straight through, but never consulted when setting the budget. A 50-token ZIP on a `codegen` or `refactor` task is almost certainly below minimum viable context.

**Additionally:** the token estimator uses `max_tokens / 1.3` → word count with a fixed ratio. This is wrong for code identifiers (1 word = 1 token) and for caveman-compressed output (subword fragments tokenize unpredictably). Estimation error can easily be ±20%.

**Fix:**

```python
_ZIP_BUDGET = {
    "summarise": 50, "explain": 80, "codegen": 120,
    "refactor": 150, "review": 100, "unknown": 80,
}

def _compress_history(self, history, hardcoded, task_type="unknown"):
    budget = _ZIP_BUDGET.get(task_type, _ZIP_BUDGET["unknown"])
    zipped = self._truncate_to_budget(zipped, budget)
```

---

## 2. `_compress_history()` — Dead Work and Ordering Flaw

**Step 1 (substitution) is applied to `all_text` which is then discarded.** The actual compression pipeline reassembles `kept_text` from `kept_history` (post-prune) and applies substitution a second time. The first substitution call on `all_text` produces output that goes nowhere — wasted regex evaluation over the full history.

**Optimal order:**

```
1. Prune low-information turns
2. Extract facts from kept turns (move _extract_facts earlier)
3. Substitute known + newly-extracted facts → short keys
4. CavemanCompressor
5. Truncate to budget
```

Currently `_extract_facts` runs *after* `_compress_history` — newly extracted facts only benefit the next turn.

---

## 3. `_extract_facts()` — Not LZ77

LZ77 builds a dictionary by sliding window + longest match → `(offset, length, next_char)` triples on **substrings**. ZIPPY's `_extract_facts` does:

```python
tokens = re.findall(r'[a-zA-Z_]\w{4,}', all_text)
counts = Counter(tokens)
for token, count in counts.most_common(10):
    if count >= 3 and token not in existing and len(token) > 6:
        short_key = token[:3] + str(len(token))
```

Problems:

1. **Threshold inconsistency:** `\w{4,}` requires ≥5 total chars; `len(token) > 6` then filters to ≥7. LZ77 replaces substrings of length ≥3.

2. **Count threshold of 3 skips pairs.** LZ77 back-references after the *second* occurrence. Threshold of 3 misses all terms appearing exactly twice — the majority of repeated terms in normal conversation.

3. **Key collision possible.** `short_key = token[:3] + str(len(token))` — different tokens sharing the same first 3 chars AND same length produce the same key. No collision detection.

4. **Keys may not be shorter in tokens.** `aut12` is 5 chars. `authenticate` is 2–3 BPE tokens. Savings are marginal.

5. **No phrase matching.** Repeated multi-word phrases like "the UserService.authenticate method" are not captured. True LZ77 handles these.

**What ZIPPY actually does:** unigram frequency dictionary filter — valid token reduction technique, but should be described as "frequent-token abbreviation," not "LZ77 dictionary-based compression."

---

## 4. KR Ratio — Confidence Formula Is Inverted

```python
confidence = 1.0 - kr_ratio
```

A KR ratio of 0.03 (97% compression) → confidence = 0.97. This is backwards-dangerous. Extreme compression should **reduce** confidence, not increase it — there is no quality guard. A ZIP that discards everything except "user: ok" scores confidence = 0.99.

The README defines `True Value = TER × RQS`. ZIPPY should use:

```python
retention_score = compute_rqs_l1(original_sample, zipped)
confidence = (1.0 - kr_ratio) * retention_score
```

This reuses existing RQS-L1 infrastructure and fixes the inversion. The `compute_rqs_l1` call is already available from `src/quality.py`.

---

## 5. Kolmogorov Framing — Honest Assessment

True Kolmogorov complexity K(x) is **uncomputable** (Rice's theorem). ZIPPY's actual approximation chain:

| Layer | Method | Honest description |
|---|---|---|
| Fact hardcoding | Frequent-token abbreviation | Approximate LZ77 dictionary |
| Turn pruning | Identifier-density heuristic | Approximate n-gram pruning |
| Vowel/stopword compression | CavemanCompressor | Entropy-reducing stop-word removal |
| Hard truncation | Byte-budget ceiling | Lossy truncation — NOT compression |

Hard truncation is antithetical to Kolmogorov compression: it does not find a shorter *lossless encoding* — it deletes information. The ZIP cannot reconstruct the original.

The README mermaid token journey diagram's `47-tok history` framing is correct. The Kolmogorov branding in code comments overstates theoretical rigor.

---

## 6. State Store — Race Condition (Critical)

Both `ZippyAgent._save_state()` and `StateStore.save()` use an identical read-modify-write pattern:

```python
all_state = json.loads(self.path.read_text())   # read
all_state[session_id] = state                    # modify
self.path.write_text(json.dumps(all_state))      # write
```

This is a **TOCTOU race condition**. Two concurrent sessions will overwrite each other's state. No file locking, no atomic rename.

**Additional issues:**
- ZIPPY inlines state-store logic instead of using `StateStore` — two diverging codepaths
- All sessions share one file — O(n_sessions) parse on every save
- No TTL or eviction — stale sessions accumulate forever
- `except Exception: return {}` silently swallows corrupt state

**Fix — atomic write + file lock:**

```python
# In StateStore.save():
import os, tempfile
from filelock import FileLock  # or msvcrt/fcntl

with FileLock(str(self._lock_path), timeout=5):
    all_state = self._read_safe()
    all_state[session_id] = state
    tmp = self.path.with_suffix(".tmp")
    tmp.write_text(json.dumps(all_state, indent=2), encoding="utf-8")
    os.replace(tmp, self.path)   # atomic on POSIX; near-atomic on Windows NTFS
```

Also delete `ZippyAgent._load_state` / `_save_state` entirely and use `StateStore` directly.

---

## 7. Turn Pruning — Logic Error and Role-Blindness

```python
if density < _PRUNE_THRESHOLD and len(words) < 10:   # AND condition
    pruned.append(turn_id)
```

**Logic error:** The AND condition means a 200-word all-prose user turn (density ~0.05) is **NOT** pruned because `len(words) < 10` is false — exactly backwards from the intent. A 200-word turn with low identifier density is more wasteful than a 9-word turn with medium density.

**Fix — use OR for user turns, protect long assistant turns:**

```python
threshold = _PRUNE_THRESHOLD * 0.5 if role == "assistant" else _PRUNE_THRESHOLD

if density < threshold or len(words) < 10:
    if role == "assistant" and len(words) >= 20:
        continue   # protect long assistant prose context
    pruned.append(turn_id)
```

**Role-blindness:** User turns like "ok" are legitimately prunable. Assistant turns containing caveats/warnings expressed entirely in prose would also be pruned under current logic even though they carry crucial context.

**Threshold justification:** `0.15` and `10` are round numbers with no ablation study. LLMLingua uses a fine-tuned small LLM for token perplexity scoring — this heuristic is orders of magnitude simpler (acceptable for a v1 prototype, but should be noted).

---

## Summary

| Issue | Severity |
|---|---|
| 50-token budget ignores task type; wrong for codegen/refactor | High |
| KR ratio confidence formula inverted — high compression = max confidence | High |
| State store TOCTOU race condition (both in zippy.py and state_store.py) | High |
| Turn pruning AND condition is logically backward | Medium |
| First substitution call in `_compress_history` is dead work | Medium |
| `_extract_facts` count threshold 3 > LZ77's effective threshold of 2 | Medium |
| No phrase matching in fact extraction | Medium |
| Role-blind pruning can discard valuable assistant context | Medium |
| Kolmogorov branding overstates theoretical rigor | Low |
