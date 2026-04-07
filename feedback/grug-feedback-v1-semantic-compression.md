# GRUG Agent Feedback — v1 · Semantic Compression

**Auditor:** Claude Sonnet 4.6  
**Date:** 2026-04-06  
**File reviewed:** `agents/grug.py`, `src/quality.py` (L1), `src/inference_bridge.py` (F4)  
**Review skill:** `.claude/commands/review-grug.md`

---

## 1. Synonym Table — Collisions and Missing Terms

### Ambiguous / Colliding Entries

| Long form | Grug form | Problem |
|---|---|---|
| `functionality` | `fn` | Collides with `function → fn`. Two different nouns → same token. |
| `method` | `fn` | Same collision — `function` and `method` are architecturally distinct in OOP. |
| `validate` / `validation` | `valid` | Adjective replacing noun/verb. "valid the input" is not English. |
| `implement` | `impl` | Also the target for `implementation`. Verb/noun collision. |
| `definition` | `def` | `def` is a Python keyword — confusing in Python docstrings. |
| `concatenate` | `cat` | `cat` means the Unix utility in shell-adjacent codebases. |
| `instantiate` | `new` | `new` is a keyword in Java/JS/C++. |

### Missing High-Frequency Terms

These appear constantly in software docstrings and are absent from the table:

```
integer → int,  boolean → bool,  object → obj,  default → dflt,
message → msg,  response → resp,  request → req,  error → err,
callback → cb,  database → db,  service → svc,  identifier → id,
temporary → tmp,  timeout → tmo
```

Average character saving per substitution across current 51 entries: ~6.8 chars, ~1 BPE token per swap. The README claim of Zipf alignment is overstated — actual token savings are lower than character savings suggest.

### Duplicate Key Bug

`"initialize"` appears at two positions in `SYNONYM_TABLE`. Python dict construction silently takes the last value. The "Verbs" section re-adds `("initialize", "init")`. Add an assertion at module load:

```python
assert len(_SYNONYM_MAP) == len(set(k for k, _ in SYNONYM_TABLE)), \
    "SYNONYM_TABLE has duplicate source keys"
```

---

## 2. Zipf Alignment — Verdict: Inspired But Not Implemented

**True Zipf-optimal** behavior would prioritize replacements by `chars_saved × occurrences_in_input`, not by source-word length. The current sort `sorted(SYNONYM_TABLE, key=lambda x: -len(x[0]))` is a **substring-collision prevention technique**, not Zipf alignment.

What is missing:
- No frequency weighting. A word appearing 12 times saves 12× more — not exploited.
- Sort is on source word length, not compression gain per token.
- Short-word entries (`method → fn`, 6→2) may not even reduce BPE token count if the original is already a single token.

**Verdict:** Zipf-inspired in spirit but not Zipf-optimal. Lacks frequency weighting and token-level savings estimation.

---

## 3. MDL — Remaining Redundancy After F4

A true MDL coder would additionally eliminate:

**a) Repeated structural patterns after F1.** Multiple `def foo(self, x):\n    ...` skeletons are identical. A dictionary-coding scheme (assign short codes to common patterns) would compress further. The RHD registry already exists for imports — it could be extended.

**b) Repeated compressed comment stems.** After Caveman, `# Rtrns lst f itms` appears across similar utility functions. A second RHD pass would intern these.

**c) Skeletonized bodies are already near-optimal.** Every `    ...\n` is a single token — this is genuinely MDL-aligned.

**d) `_SYNONYM_MAP` is built from a list with duplicate source keys** (see above). The dict build collapses them with undefined ordering.

---

## 4. Pipeline F1→F4 Order — Code vs Documentation Inconsistency

**Actual execution order** (`InferenceBridge.process_file`): F2 → F1 → F3 → F4  
**Module docstring says:** F1 → F2 → F3 → F4  
**README sequence diagram:** shows F2 → F1 (matches code, but contradicts module docstring)

This is a documentation inconsistency, not a logic bug. The actual order is correct:
- F2 must precede F1 (skeleton needs hot lines from git)
- F1 must precede F3 (AST parse would fail if imports are already masked)
- F3 before F4 is a minor efficiency win (reduces lines before Caveman iterates)

**Missed optimization:** Running F3 a **second time after F1** would catch maskable patterns introduced by skeletonization (repeated `    ...\n` bodies). Currently only one masking pass.

---

## 5. RQS-L1 Gate — Critical: Warning-Only Logger

```python
# grug.py lines 182-189
if avg_rqs_l1 < self.RQS_L1_WARNING_THRESHOLD:
    self.log_decision(
        decision_value = f"RQS-L1={avg_rqs_l1:.3f} BELOW threshold ...",
        rationale      = "Quality degradation detected — BALANCER should consider escalation",
        ...
    )
```

**The check only logs. It does not gate any decision.** A chunk with RQS-L1 = 0.40 (severe semantic drift) is forwarded to the LLM identically to one with 0.99. There is no retry, no synonym rollback, no bypass flag.

**Additional threshold mismatch:** `RQS_L1_WARNING_THRESHOLD = 0.80` in GRUG but `quality.py` defines "ACCEPTABLE" at ≥ 0.85. These should be aligned.

**What a real control loop looks like:**

```python
if rqs_l1 < self.RQS_L1_WARNING_THRESHOLD:
    # Tier 1: strip synonyms, keep InB compression
    compressed_synonym_stripped = compressed  # no synonym pass
    rqs_retry = compute_semantic_similarity(source, compressed_synonym_stripped)
    if rqs_retry > rqs_l1:
        compressed_after_synonyms = compressed_synonym_stripped

RQS_HARD_FLOOR = 0.65
if rqs_l1 < RQS_HARD_FLOOR:
    compressed_after_synonyms = source   # full bypass, flag for BALANCER escalation
    final_t_syn = orig_t
```

---

## 6. BACKGROUND Chunk Handling — F3 Masking Skipped Unnecessarily

BACKGROUND chunks receive F4 Caveman only. F3 masking is skipped.

**The rationale holds for F1** (AST parse is expensive) but **not for F3**. F3 masking is a per-line string comparison against an already-populated registry — O(n) with no AST parsing, no subprocess, no file I/O. Skipping it means repeated import lines in BACKGROUND chunks are sent unmasked to the LLM even though identical imports in FOREGROUND chunks were hashed to `[§:hash]`. The RHD registry is populated but not consulted for BACKGROUND.

**F4-only is also semantically backward:** if a BACKGROUND chunk's value is its structure (class definitions, signatures), Caveman (which strips comment vowels) is the least useful pass. F1 skeleton with a relaxed PI threshold (e.g., `pi_threshold=0.60`) would be more impactful for BACKGROUND than Caveman alone.

**Proposed fix:**

```python
if zone == "BACKGROUND":
    masked = ChromatophoricMasker().mask(source)   # add this — cheap
    compressed = CavemanCompressor().compress(masked)
```

---

## 7. LLMLingua Gap — Five Missing Features

**a) No token importance ranking.** GRUG treats all comment tokens as equally compressible. LLMLingua uses perplexity increase per token to rank importance.

**b) No adaptive compression ratio.** Same Caveman pass for `# DANGER: do not modify lock order` and `# helper method`. LLMLingua adjusts compression ratio per segment by information density.

**c) No coarse-to-fine compression.** LLMLingua: sentence-level removal first, then token-level pruning within kept sentences. GRUG is single-pass global.

**d) No budget-aware truncation.** GRUG reports the result but does not target a specific token budget. If over budget, the problem is handed to BALANCER.

**e) Bag-of-words cosine similarity is a weak quality proxy.** `compute_semantic_similarity` uses TF-IDF cosine. After synonym compression (`initialization → init`), the original word is gone — vocabulary overlap drops, but semantic meaning is preserved. The similarity score will incorrectly penalize good compressions. `sentence-transformers` embeddings are the correct tool here (already mentioned in the code comments as a TODO).

---

## 8. Performance Issue — Regex Recompiled Per Line

`_apply_synonyms` calls `re.compile` inside a loop on every invocation. `_SYNONYM_MAP` has ~51 entries — that's 51 `re.compile` calls per line of input. Pre-compile at module load:

```python
_COMPILED_SYNONYMS = [
    (re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE), v)
    for k, v in sorted(SYNONYM_TABLE, key=lambda x: -len(x[0]))
]
```

---

## Summary

| Issue | Severity |
|---|---|
| RQS-L1 is warning-only; does not gate or retry | High |
| Duplicate synonym keys (`initialize` × 2, `implement`/`implementation` both → `impl`) | Medium |
| 13 high-frequency terms missing from synonym table | Medium |
| F3 masking skipped for BACKGROUND at no real performance saving | Medium |
| RQS_L1_WARNING_THRESHOLD (0.80) mismatches quality.py floor (0.85) | Medium |
| Bag-of-words cosine sim produces false positives after synonym compression | Medium |
| Regex recompiled per line (~51 compiles per line of input) | Low |
| Zipf-optimal sorting absent (sorted by source length, not compression gain × frequency) | Low |
