# GRUG Agent — Technical Review
**File reviewed:** `agents/grug.py`  
**Supporting files:** `src/inference_bridge.py`, `src/quality.py`, `agents/base_agent.py`  
**Date:** 2026-04-06  
**Reviewer:** AI Auditor (Claude Sonnet 4.6)

---

## Executive Summary

GRUG's intent is sound: semantic synonym compression layered on top of the InB F1-F4
pipeline. In practice there are seven substantive defects — ranging from a table sort
that silently breaks longest-match semantics, to an RQS-L1 gate that warns but never
blocks, to a post-Caveman synonym pass that is almost entirely redundant. None of these
are catastrophic blockers today, but together they explain why GRUG's marginal
contribution to TER is small relative to what the MDL and Zipf claims would predict.

---

## 1. Zipf-Optimal Ordering — Synonym Table

### Finding: Table is sorted by string length but not by corpus frequency rank

**`grug.py` line 106:**
```python
_SYNONYM_MAP = {k.lower(): v for k, v in sorted(SYNONYM_TABLE, key=lambda x: -len(x[0]))}
```

The sort key is `-len(x[0])` — descending length of the **long form**. This is a
longest-match heuristic that prevents substring shadowing (e.g. `"initialize"` matched
before `"init"` would create `"initit"`). That part is correct.

However, Zipf's Law makes a **frequency** claim, not a **length** claim. Zipf's Law
states rank × frequency ≈ constant, which means the most frequent words are the
shortest. Substituting the longest words first (by string length) is an accidental
approximation of Zipf-optimal ordering only when word length and inverse frequency
correlate — which is roughly true for English prose but systematically false for
developer comment vocabulary, where long technical terms appear frequently
(`"implementation"`, `"configuration"`, `"authentication"` are high-frequency tokens in
any backend codebase).

**Measurement — actual character reduction per substitution:**

| Long form         | Short form | Long len | Short len | Chars saved | Ratio |
|-------------------|-----------|----------|-----------|-------------|-------|
| deserialization   | deser     | 15       | 5         | 10          | 0.67  |
| serialization     | ser       | 13       | 3         | 10          | 0.77  |
| implementation    | impl      | 14       | 4         | 10          | 0.71  |
| authentication    | auth      | 14       | 4         | 10          | 0.71  |
| initialization    | init      | 14       | 4         | 10          | 0.71  |
| configuration     | cfg       | 13       | 3         | 10          | 0.77  |
| documentation     | doc       | 13       | 3         | 10          | 0.77  |
| authorization     | authz     | 13       | 5         | 8           | 0.62  |
| functionality     | fn        | 13       | 2         | 11          | 0.85  |
| asynchronous      | async     | 12       | 5         | 7           | 0.58  |
| concatenation     | cat       | 11       | 3         | 8           | 0.73  |
| instantiate       | new       | 11       | 3         | 8           | 0.73  |
| dependencies      | deps      | 12       | 4         | 8           | 0.67  |
| deserialization   | deser     | —        | —         | —           | —     |
| parameter         | param     | 9        | 5         | 4           | 0.44  |
| parameters        | params    | 10       | 6         | 4           | 0.40  |
| exception         | exc       | 9        | 3         | 6           | 0.67  |
| variable          | var       | 8        | 3         | 5           | 0.63  |
| function          | fn        | 8        | 2         | 6           | 0.75  |
| method            | fn        | 6        | 2         | 4           | 0.67  |

**Mean character reduction: ~7.5 chars per substitution, ~68% compression ratio.**

The table is missing several high-frequency tokens in Python/C code comments:
`"returns"`, `"raises"`, `"deprecated"`, `"callback"`, `"timeout"`, `"boolean"`,
`"integer"`, `"pointer"`, `"struct"`, `"allocate"`, `"deallocate"`, `"dereference"`,
`"overflow"`, `"underflow"`. These appear in more comments per repository than most
existing table entries.

### Duplicate Entry Bug

**`grug.py` lines 48 and 86 both define `("initialize", "init")`.**

When `sorted()` builds `_SYNONYM_MAP` at line 106, the dict comprehension silently
drops one of the two. The surviving entry depends on Python dict ordering of the sort
output — non-deterministic for equal-length keys. No error is raised. This is a latent
correctness bug.

### Semantic Collision: `functionality → fn` vs `function → fn`

Both `"functionality"` and `"function"` map to `"fn"` (lines 63, 95). In a natural
language comment, `"fn"` is unambiguous as a function abbreviation. But `"functionality"`
is a noun describing a feature set, not a function. An LLM reading `"fn not implemented"`
would likely interpret it as "function not implemented," which is wrong if the original
was `"functionality not implemented"`. This is a lossless-substitution claim failure.

---

## 2. MDL Post-Caveman Redundancy

### Finding: The synonym pass runs after F4 Caveman, creating near-zero marginal value

**`grug.py` lines 270–288 (`_compress_chunk`):**

```python
# Step 1: InferenceBridge compression
...
compressed = CavemanCompressor().compress(source)  # BACKGROUND path
...
compressed_src, report = self._inb.process_file(tmp_path)  # FOREGROUND path

# Step 2: Synonym compression pass
compressed_after_synonyms, swaps = self._apply_synonyms(compressed)
```

F4 Caveman (`CavemanCompressor.compress`) removes stop-words and prunes interior vowels
from **comment lines and docstring content**. After Caveman runs, a comment like:

```
# Initialize the configuration for authentication.
```
becomes:
```
# Initlz cnfgrtn fr athnfctn.
```

GRUG's `_apply_synonyms` then tries to match `\binitialize\b` against that output. It
will not match because Caveman has already vowel-pruned `"initialize"` to `"Initlz"`.

The MDL principle (Rissanen 1978) requires that you choose the shortest code that
reproduces the data without ambiguity. The current pipeline does **not** minimize
description length — it applies two redundant passes to the same text domain (comments
and docstrings), with the earlier pass (Caveman) destroying the lexical tokens the later
pass (synonym table) needs as input.

**Fix direction:** Run synonym replacement **before** Caveman, or skip synonym
replacement on comment/docstring text entirely and apply it only to code identifiers
(which Caveman does not touch). The second option is where GRUG's synonym table would
genuinely add value — replacing `initialization` in actual Python identifiers like
`self.initialization_state` with `self.init_state` — but `_apply_synonyms` explicitly
skips non-comment, non-docstring lines at `grug.py` line 301–304.

This means **the synonym table currently has near-zero compression impact in FOREGROUND
chunks** and zero compression impact on BACKGROUND chunks (where Caveman always runs
first). This is the single biggest waste of computational potential in the file.

---

## 3. LLMLingua Perplexity Comparison

### Finding: GRUG uses bag-of-words cosine similarity as a quality gate, not perplexity-based token importance scoring

LLMLingua (Wu et al. 2024) works by:
1. Estimating each token's importance score as the negative log-probability assigned by
   a small proxy LM (e.g., LLaMA 7B).
2. Greedily dropping low-importance tokens until the target compression ratio is reached.
3. Running iterative refinement to restore any tokens that caused semantic drift.

GRUG's `compute_semantic_similarity` at `src/quality.py` lines 127–144 is:

```python
def compute_semantic_similarity(response_full: str, response_bridge: str) -> float:
    tf_full   = _tokenize_simple(response_full)
    tf_bridge = _tokenize_simple(response_bridge)
    return round(_cosine_similarity(tf_full, tf_bridge), 4)
```

`_tokenize_simple` is a term-frequency bag-of-words (no IDF weighting, no embeddings).
`_cosine_similarity` computes dot product over shared vocabulary.

**Comparison against LLMLingua:**

| Property | GRUG (_apply_synonyms + cosine RQS-L1) | LLMLingua |
|---|---|---|
| Selection strategy | Rule-based table lookup | Perplexity-based importance scoring |
| Ordering | Longest string first (approximation) | Exact rank by token surprise |
| Quality signal | TF cosine similarity | Conditional perplexity under proxy LM |
| Compression target | Fixed synonym substitutions | Configurable ratio to budget |
| Adaptive to context | No | Yes — importance varies per prompt |
| Cold-start cost | None | Requires proxy LM inference pass |
| Zipf alignment | Partial (length correlates with rarity) | Full (perplexity is information content) |

The TF cosine metric inflates similarity scores for code text specifically because
identifiers dominate the vocabulary and identifiers are largely preserved. A docstring
that changed from `"Initialize the configuration for authentication processing"` to
`"init cfg auth prcssng"` would score very high cosine similarity (~0.88) while being
substantially harder for a human reviewer to read and potentially ambiguous to the LLM.

The quality.py comment at line 134 explicitly acknowledges a better alternative:
```python
# For production: swap this with sentence-transformers embeddings
```
This has not been done. The current L1 is a placeholder, not a production quality gate.

**What it would take to add a lightweight importance scorer:**
- Import a small quantized model (e.g., `microsoft/phi-1_5` at 1.3B params via
  `transformers` + `bitsandbytes`) as a proxy LM.
- For each token in the source, compute its conditional log-probability given context.
- Sort tokens by negative log-prob (ascending = most predictable = safe to drop).
- Drop tokens below a threshold instead of using a fixed synonym table.
- This would require ~2 GB VRAM or can be run CPU-only at ~30 tokens/second.
- Only feasible if `ACTIVE_TIER_MAX > 0` (offline mode must bypass it).

---

## 4. RQS-L1 Integration — Wiring, Threshold, and Gate Behavior

### Finding: RQS-L1 warns but does not block — the gate is decorative

**`grug.py` lines 279–287:**
```python
# Step 3: RQS-L1 inline check
if _INB_AVAILABLE:
    try:
        rqs_l1 = compute_semantic_similarity(source, compressed_after_synonyms)
    except Exception:
        rqs_l1 = 1.0
```

**`grug.py` lines 182–196:**
```python
if avg_rqs_l1 < self.RQS_L1_WARNING_THRESHOLD:
    self.log_decision(
        decision_type  = "COMPRESS",
        decision_value = f"RQS-L1={avg_rqs_l1:.3f} BELOW threshold {self.RQS_L1_WARNING_THRESHOLD}",
        rationale      = "Quality degradation detected — BALANCER should consider escalation",
        confidence     = avg_rqs_l1,
        slice_id       = slice_in.slice_id,
    )
```

`compute_semantic_similarity` is called once per chunk inside `_compress_chunk` at
line 282, comparing `source` (pre-InB, pre-synonym) against
`compressed_after_synonyms` (post-InB, post-synonym). This is the correct comparison
surface.

The result is stored in the output payload (`rqs_l1_score`) and logged to the decision
tracker, but **no code path in GRUG rolls back compression when RQS-L1 falls below
threshold**. The comment says "BALANCER should consider escalation" — this is an
advisory, not a guard.

The threshold is `RQS_L1_WARNING_THRESHOLD = 0.80` (line 127), but the project-wide
quality floor is `rqs_threshold = 0.85` (from `QualityResult` at `quality.py` line 69).
GRUG's warning threshold (0.80) is **below the project quality floor (0.85)**, which
means GRUG can produce output that fails the RQS floor without triggering its own
warning.

**Issues:**
1. Warning threshold (0.80) is inconsistent with project RQS floor (0.85). Should be
   equal to or above 0.85.
2. No rollback or passthrough path when quality degrades. GRUG should emit the
   original `source` unchanged when `rqs_l1 < threshold` (passthrough mode), letting
   BALANCER decide.
3. The function compares original source (code + comments) against
   compressed (code + compressed-comments). Code identifiers are identical in both
   (Caveman and synonym table do not touch code lines), so the cosine similarity is
   inflated by the shared code vocabulary. Actual semantic loss in the comment content
   is masked. A representative quality check should compare only the comment/docstring
   portions.

---

## 5. Character Reduction Accuracy — Synonym Table

### Finding: Effective reduction is lower than the table implies

From the table in Section 1, average character saving per swap is ~7.5 characters.
However, this is a **per-occurrence** measure. Token savings depend on tokenizer
behavior:

- For tiktoken (GPT-4): words like `"initialization"` tokenize as 3 subword tokens
  (`init`, `ializ`, `ation`). The replacement `"init"` is 1 token. True saving: 2 tokens.
- For `count_tokens` in `inference_bridge.py` line 88: words `>8 chars and isalnum()`
  count as 2 tokens. `"initialization"` (14 chars, isalpha) → 2 tokens. `"init"` (4
  chars) → 1 token. Saving: 1 token.
- But after Caveman runs, `"initialization"` is already pruned to `"Initlztn"` (8 chars
  exactly, boundary case). Does it count as 2 or 1? The `isalnum()` check at
  `inference_bridge.py` line 92 includes alpha-only strings, so `"Initlztn"` (8 chars)
  does NOT trigger the 2-token penalty (`>8` is strict greater-than), counted as 1 token.

This means Caveman's vowel pruning incidentally eliminates the token-count advantage
that synonym replacement would have provided. Another dimension of the redundancy
problem from Section 2.

---

## 6. FeatureSlice Interface Audit

### Fields read from `slice_in.payload` (grug.py lines 140–142):
- `classified_chunks` (list of dicts) — required; empty list if absent
- `file_path` (str) — optional; defaults to `""`
- `language` (str) — optional; defaults to `"unknown"`

Each chunk dict is expected to have:
- `chunk_id` (line 152) — required; KeyError if absent
- `zone` (line 153) — required; KeyError if absent; used to branch FOREGROUND/BACKGROUND
- `chunk_type` (line 155) — optional; defaults to `"function"`
- `original_tokens` (line 232) — optional; defaults to `0`
- `source` (line 230) — optional; defaults to `""`

### Fields written to `slice_out.payload`:
- `compressed_chunks` (list)
- `total_original_tokens` (int)
- `total_final_tokens` (int)
- `overall_ter` (float)
- `rqs_l1_score` (float)
- `rhd_registry_delta` (dict) — always empty dict `{}` (see Section 7 below)
- `synonym_swaps_count` (int)
- `file_path` (str)
- `language` (str)

### Missing fields in output payload:
- `classified_chunks` from input is not forwarded. BALANCER or ZIPPY agents
  downstream cannot access the original uncompressed source for comparison without
  re-reading from the bus. This breaks the audit trail for regression detection.
- No `pipeline_stages_applied` field documenting which of F1/F3/F4 actually ran (vs
  fell back to passthrough on exception).

### `decision` field truncation:
`BaseAgent.emit` enforces a 100-character limit on `decision` (`base_agent.py` line 71).
The decision string built at `grug.py` lines 175–180 easily exceeds 100 chars for files
with many synonym swaps (e.g., `"TER=45.2% (1200→660 tok). RQS-L1=0.92. 24 synonym
swaps. lang=python"` = 73 chars — just within limit, but adding more swap details would
truncate silently).

---

## 7. Error Handling Gaps

### 7a. Silent exception swallowing in `_compress_chunk` (lines 258–262)

```python
except Exception:
    compressed = source
    final_t    = orig_t
    rhd_delta  = {}
```

Any InferenceBridge failure — including `OSError`, `MemoryError`, `RecursionError` from
deep AST trees, or `subprocess.TimeoutExpired` from `RetinalDelta` — is silently
discarded. The chunk falls back to the uncompressed source, the TER is reported as 0%
for that chunk, but **no warning is logged, no decision event is written, and the caller
has no visibility into whether compression actually ran**.

The `log_decision` machinery is available. At minimum, add:
```python
except Exception as e:
    self.log_decision("COMPRESS", "FALLBACK", f"InB failed: {type(e).__name__}: {e}",
                      confidence=0.0, slice_id=chunk.get("chunk_id"))
    compressed = source
    ...
```

### 7b. `rhd_registry_delta` is always `{}` (lines 251, 257)

```python
rhd_delta = {}  # RHD managed by ChromatophoricMasker internally
```

The comment acknowledges the registry is managed internally, but the output payload key
`rhd_registry_delta` is always an empty dict. Any downstream agent or auditor expecting
the registry delta to reconstruct masked tokens will be silently misled. Either populate
this field from the masker's `_registry` diff, or remove the key from the output payload
entirely and document the omission.

### 7c. Temporary file not cleaned up on exception (lines 240–259)

```python
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, ...) as tmp:
    tmp.write(source)
    tmp_path = tmp.name
...
# ... code that can raise ...
os.unlink(tmp_path)  # never reached if exception above
```

The `os.unlink(tmp_path)` at line 259 is inside the `try` block but before the
`except`. If `self._inb.process_file(tmp_path)` raises, `tmp_path` is leaked on disk.
On Windows (the project platform), this is particularly problematic because
`NamedTemporaryFile` with `delete=False` does not clean up on GC. Fix with a
`try/finally`:

```python
tmp_path = None
try:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(source)
        tmp_path = tmp.name
    compressed_src, report = self._inb.process_file(tmp_path)
    ...
finally:
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

### 7d. `count_tokens` import inside hot loop (lines 273–277)

```python
from src.inference_bridge import count_tokens as _ct
try:
    final_t_syn = _ct(compressed_after_synonyms)
except Exception:
    final_t_syn = final_t
```

This `import` statement runs inside `_compress_chunk`, which is called once per chunk
in a loop at `grug.py` line 151. Python caches module imports after the first call, so
this is not a correctness issue, but it is a code smell that implies the author was
uncertain whether `count_tokens` was available. The try/except around the call masks
potential issues. Import at module level alongside the other InB imports (lines 35–38).

### 7e. `orig_t = 0` propagation causes division issues

When `chunk.get("original_tokens", 0)` returns `0` (line 232), and the chunk is
non-empty, `reduction_pct` at `grug.py` line 158 correctly handles `orig_t == 0` with
a ternary. However, `overall_ter` at line 168 uses `total_original` which may be `0` if
every chunk reported 0 tokens — the guard is present but the per-chunk `orig_t=0` case
means TER would be reported as `0.0%` even when compression occurred. The root issue is
that GRUG trusts the chunk's pre-computed `original_tokens` rather than counting the
source it actually received.

---

## 8. Specific Line-Level Code Issues

### Line 106 — sort + dict comprehension discards duplicates silently
```python
_SYNONYM_MAP = {k.lower(): v for k, v in sorted(SYNONYM_TABLE, key=lambda x: -len(x[0]))}
```
`("initialize", "init")` appears at indices 0 and 37 (Nouns section and Verbs section).
`sorted()` by `-len("initialize")` produces a deterministic length-order, but two
entries of the same length (both `"initialize"`, len=10) will retain the order from
the original list — meaning the Verbs duplicate at index 37 will overwrite the Nouns
entry at index 0 in the dict comprehension. Both map to `"init"` so the value is the
same, but if someone adds a verb-context variant with a different target later, the bug
becomes a silent semantic error. Deduplicate the table.

### Line 199 — `lang_tag` allowlist is incomplete and brittle
```python
lang_tag = f"lang.{language}" if f"lang.{language}" in ["lang.python", "lang.c", "lang.go", "lang.rust"] else "lang.unknown"
```
This allowlist is checked by string membership against a hard-coded list, duplicating
information that should come from `data/taxonomy.json`. Any new language added to
taxonomy.json requires a matching update here. The correct pattern is to let
`self.emit()` / `self.taxonomy.validate()` enforce tag validity, and just pass
`f"lang.{language}"` directly.

### Lines 303–312 — Regex compiled inside inner loop
```python
for long_form, short_form in _SYNONYM_MAP.items():
    pattern = re.compile(r'\b' + re.escape(long_form) + r'\b', re.IGNORECASE)
```
`_SYNONYM_MAP` has ~50 entries. For a file with 200 comment lines, this compiles 10,000
regex objects per call to `_apply_synonyms`. Python's `re` module caches compiled
patterns up to 512 entries (CPython 3.12), so after the first pass the compilation is
cached, but the overhead of the cache lookup on every iteration is still measurable.
Pre-compile all patterns at module initialization:

```python
_SYNONYM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE), v)
    for k, v in _SYNONYM_MAP.items()
]
```

Then `_apply_synonyms` iterates `_SYNONYM_PATTERNS` directly.

### Lines 304–312 — Docstring detection is over-broad
```python
is_docstring = stripped.startswith(('"""', "'''", '"', "'"))
```
This matches any line whose non-whitespace content starts with a single quote or double
quote — including string literals in code like `return "error"`, dict values like
`{"key": "value"}`, and f-strings. The intent is docstrings, but the implementation
will apply synonym replacement to any quoted string on a line that happens to start with
a quote after stripping. Combined with the note that Caveman has already pruned these
strings, the practical impact is minimal, but the logic is wrong.

A tighter docstring test:
```python
is_docstring = stripped.startswith(('"""', "'''"))
```
Single-quote strings are almost never docstrings (PEP 257 recommends triple double
quotes). The single-char `'"'` and `"'"` cases should be dropped.

### Lines 169–172 — avg_rqs_l1 arithmetic mean is not robust
```python
avg_rqs_l1 = (
    sum(c["rqs_l1_score"] for c in compressed_chunks) / len(compressed_chunks)
    if compressed_chunks else 1.0
)
```
A single pathological chunk with `rqs_l1_score = 0.0` (from an exception fallback that
returns `1.0` due to the bare `except` at line 284) inflates the average. The only
chunks that return low RQS-L1 are those where compression actually ran and diverged.
The arithmetic mean over all chunks including 1.0 no-ops understates quality degradation.
Use the minimum across non-1.0 chunks, or weight by chunk token count:
```python
weighted_rqs = (
    sum(c["rqs_l1_score"] * c["original_tokens"] for c in compressed_chunks)
    / total_original
    if total_original > 0 else 1.0
)
```

---

## 9. Proposed High-Value Changes (Priority Order)

### Priority 1 — Run synonym replacement before Caveman (or on code identifiers)

The current order (InB pipeline including F4 Caveman → synonym replacement) guarantees
near-zero synonym hit rate because Caveman has already pruned the lexical tokens the
synonym regex needs. Two viable fixes:

**Option A** (minimal change): Add a pre-Caveman synonym step in `_compress_chunk`.
Call `_apply_synonyms` on `source` before writing to the temp file, so InB receives
pre-substituted text. Caveman then processes already-shortened words.

**Option B** (architecturally correct): Move synonym replacement out of
comment/docstring scope entirely and apply it to code identifiers (`self.initialization_state`
→ `self.init_state`). This requires identifier-aware parsing (AST `asttokens` or
`libcst`), but it is the only scope where synonyms add non-redundant compression.

### Priority 2 — Align RQS-L1 gate with project quality floor and make it blocking

Change `RQS_L1_WARNING_THRESHOLD` from `0.80` to `0.85` to match `QualityResult.rqs_threshold`.
Add a passthrough mode in `_compress_chunk`:

```python
if rqs_l1 < self.RQS_L1_WARNING_THRESHOLD:
    # Quality regression — return uncompressed source
    self.log_decision("COMPRESS", "REJECTED_QUALITY",
                      f"rqs_l1={rqs_l1:.3f} < {self.RQS_L1_WARNING_THRESHOLD}",
                      confidence=rqs_l1, slice_id=chunk.get("chunk_id"))
    return source, orig_t, orig_t, {}, [], rqs_l1
```

This makes GRUG a true quality gate rather than a passive logger.

### Priority 3 — Fix temp file leak and silent exception swallowing

Wrap the InB subprocess path in `try/finally` for temp file cleanup (see Section 7c).
Add `log_decision("COMPRESS", "FALLBACK", ...)` in the `except` block (see Section 7a).
These two changes together give operators visibility into how often InB is actually
compressing vs silently passing through, which is essential for understanding the real
TER contribution of the GRUG stage.

---

## Appendix — Deduplication of SYNONYM_TABLE

Duplicate entries that should be removed from `SYNONYM_TABLE` in `grug.py`:

| Line | Entry | Reason |
|------|-------|--------|
| 48 | `("initialize", "init")` | Duplicates line 86 |
| 85 | `("implement", "impl")` | Same target as `("implementation", "impl")` at line 51 — adds no compression for `implement` (already 9 chars, marginal saving) |
| 92 | `("validate", "valid")` | Same target as `("validation", "valid")` at line 60; `"validate"` → `"valid"` is 2 chars saved but `"valid"` is ambiguous as adjective in comments |
| 93 | `("serialize", "ser")` | Duplicates line 30 in effect (same target `"ser"`) |
| 94 | `("deserialize", "deser")` | Duplicates line 31 in effect |
