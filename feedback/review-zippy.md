# ZIPPY Agent — Technical Review
**Reviewer:** Claude Sonnet 4.6 (AI Auditor)
**Date:** 2026-04-06
**Files reviewed:**
- `agents/zippy.py`
- `orchestrator/state_store.py`
- `src/inference_bridge.py` lines 519–634 (`CavemanCompressor`)
- `agents/base_agent.py`

---

## Executive Summary

ZIPPY's architecture is sound in intent: compress conversation history into a ≤50-token ZIP before each LLM turn, persist stable facts as a key-value dictionary, and prune low-information turns. However, the implementation has five significant gaps that can cause silent failures: the 50-token budget is enforced post-hoc via word-split truncation rather than token-accurate measurement; the LZ77 dictionary analogy breaks down because keys are token strings, not byte offsets; the Kolmogorov approximation is just stop-word removal and vowel pruning, which is defensible but substantially weaker than claimed; the prune thresholds are not empirically calibrated; and the state store has a read-modify-write race condition that will corrupt state under concurrent pipeline runs.

---

## 1. 50-Token Budget Enforcement

### How it is tracked and enforced

The budget constant is defined at module level (`zippy.py:47`):
```python
_MAX_ZIP_TOKENS = 50
```

It is enforced exclusively in `_truncate_to_budget()` (`zippy.py:266-273`):
```python
def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
    words = text.split()
    max_words = int(max_tokens / 1.3)   # ≈ 38 words
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"
```

**Critical flaw:** The "1 word ≈ 1.3 tokens" approximation is used for compression without invoking `count_tokens()`. The token count reported in the outgoing `FeatureSlice` (`token_count=tokens_after`, line 142) uses `count_tokens()` on the already-truncated string, but truncation itself is done on word count. For code-heavy text (identifiers, symbols), actual token-per-word ratios can reach 2.0–3.0, so the budget is routinely violated. The `tokens_after` value logged is accurate (because `count_tokens` is called at line 194), but the truncation boundary is wrong — the ZIP handed to the orchestrator can exceed 50 tokens.

**Fix:** Replace the word-split heuristic with a binary search over `count_tokens()`:
```python
def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
    if not _INB_AVAILABLE:
        words = text.split()
        return " ".join(words[:int(max_tokens / 1.3)]) + " [...]"
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    result = " ".join(words[:lo])
    return result if lo == len(words) else result + " [...]"
```
This is O(log n) in word count, well within budget for history-length inputs.

**Token floor question:** LLMLingua (Wu et al. 2024) reports that minimum viable context for comprehension tasks is roughly 20–30% of original length, but never below an empirically-measured floor that varies by task:
- Summarization / explanation: 20–30 tokens can suffice.
- Code generation / refactoring: 50 tokens is marginal; 80–120 is safer because function signatures and type annotations are high-density.
- Multi-turn QA with state: 50 tokens for history alone is aggressive if `hardcoded_state` is not sent back to the LLM along with it.

ZIPPY partially addresses this by passing `hardcoded_state` in the payload (line 127), but the orchestrator must actually inject both `zipped_history` and `hardcoded_state` into the LLM prompt for the ZIP to be semantically complete. If it only forwards `zipped_history`, 50 tokens is insufficient for code tasks.

---

## 2. LZ77 Dictionary Approximation

### How faithful is ZIPPY's deduplication to LZ77?

LZ77 works by replacing repeated byte sequences with `(offset, length, next_char)` back-references into a sliding window over the preceding bytes. The key property is that the window is positionally ordered and operates at the byte level — it exploits locality of reference in adjacent content.

ZIPPY's `_extract_facts()` (`zippy.py:226-257`) does the following:
1. Tokenizes the entire history with `re.findall(r'[a-zA-Z_]\w{4,}', all_text)` — only identifier-like tokens.
2. Counts frequency with `Counter`.
3. Promotes tokens appearing ≥3 times with length >6 to a dictionary entry.
4. Assigns a short key: `token[:3] + str(len(token))` (e.g., `UserService` → `Use11`).

**Faithfulness assessment:**

| LZ77 property | ZIPPY implementation | Gap |
|---|---|---|
| Repeated sequences | Repeated single tokens only | LZ77 handles n-gram phrases; ZIPPY misses multi-word idioms like `UserService.authenticate` |
| Sliding window (locality) | Global frequency over all turns | No locality; a token repeated 3 times in turn 1 and never again still gets hardcoded |
| Back-reference = offset+length | Short key = prefix+length | Keys are human-readable but not minimal; `Use11` is 5 chars, a 2-byte index would be 1 token |
| Decompressibility | Dictionary stored in `hardcoded_state` | Downstream must hold the dictionary to reconstruct; this is correct in principle but not enforced |
| Byte-level matching | Word-boundary regex only | Punctuation-attached tokens (e.g., `UserService,` or `authenticate(`) are not matched by `\b` word boundaries in `_substitute_hardcoded` |

The word-boundary issue at `zippy.py:263` is the most concrete flaw:
```python
text = re.sub(r'\b' + re.escape(long_form) + r'\b', short_key, text)
```
`\b` does not match across a word boundary followed by `(`, `.`, or `,`, so `UserService.authenticate` will not be replaced even if both tokens are individually hardcoded. LZ77 would compress the entire span as a single back-reference.

**Minimum viable dictionary size:** For typical 10–20 turn coding conversations, 5–10 high-frequency identifiers dominate (class names, method names, file paths). The current cap of `counts.most_common(10)` with the ≥3 / >6 length filter is reasonable for this domain, but the filter rejects tokens of length 7 (e.g., `service`, `handler`) despite high frequency due to `len(token) > 6` being strict greater-than. Change to `>= 6` to capture 6-character identifiers.

---

## 3. Kolmogorov Gap

### What computable approximation does ZIPPY actually use?

True Kolmogorov complexity K(x) = length of the shortest program on a universal Turing machine that outputs x. It is formally uncomputable (Rice's theorem). ZIPPY acknowledges this implicitly but does not say what its computable proxy is.

In practice, ZIPPY's compression chain is:
1. **_substitute_hardcoded**: dictionary substitution — approximates the "program" as a lookup table (analogous to the dictionary portion of a compressed file).
2. **_prune_turns**: identifier-density heuristic — approximates information content as proportion of semantic tokens.
3. **CavemanCompressor.compress()**: stop-word removal + vowel pruning — a deterministic, lossless-for-structure but lossy-for-readability transform. This is closer to a static Huffman table (common English words = high probability = short code) than to Kolmogorov compression.
4. **_truncate_to_budget**: hard truncation — not a compression operation; it discards information entirely.

**Is this theoretically justified?**

Partially. The closest rigorous connection is to **Minimum Description Length (MDL)** (Rissanen 1978), where the "model" is the `hardcoded` dictionary and the "data" is the substituted+pruned residual. MDL says the optimal encoding is: length(model) + length(data | model). ZIPPY minimizes length(data | model) but does not account for the cost of transmitting the model itself. If `hardcoded_state` grows large (many session turns), it erodes the compression gain.

The `kr_ratio` metric (`zippy.py:198`) — `tokens_after / tokens_before` — measures only the compression of the residual, not the total encoding cost. A more principled metric would be:

```
total_encoding_cost = count_tokens(zip_text) + count_tokens(str(hardcoded_state))
kr_true = total_encoding_cost / tokens_before
```

**Better approximation available:** LLMLingua uses a small language model (GPT-2 level) to score each token's conditional perplexity given its context — tokens with low perplexity (predictable from context) are dropped. This is a computable approximation of Kolmogorov complexity because predictable tokens carry less information (Shannon entropy connection). ZIPPY could approximate this at zero cost by using the existing `count_tokens()` function as a proxy: compress iteratively and check if each removed token degrades a downstream quality score. This is expensive at runtime but could be run offline to calibrate `_PRUNE_THRESHOLD`.

---

## 4. Turn Pruning Thresholds

### Are the thresholds empirically calibrated?

The thresholds are defined at `zippy.py:48`:
```python
_PRUNE_THRESHOLD = 0.15  # turns with information density below this are dropped
```

Applied in `_prune_turns()` (`zippy.py:201-224`):
```python
if len(words) < 3:
    pruned.append(turn_id)   # line 216 — length-only prune
    continue
identifiers = re.findall(r'[a-zA-Z_]\w{3,}', content)
density = len(identifiers) / len(words)
if density < _PRUNE_THRESHOLD and len(words) < 10:  # line 222
    pruned.append(turn_id)
```

**Issues:**

1. **Double condition with asymmetric risk:** A turn is pruned only when `density < 0.15 AND words < 10`. A 9-word turn with density 0.14 is pruned; a 9-word turn with density 0.0 (pure filler: "Yes that is correct I see") and 11 words is **not pruned**. The `words < 10` guard defeats the density filter for medium-length filler turns. These are exactly the turns most worth pruning.

2. **No empirical calibration:** 0.15 and 10 words appear to be round numbers, not derived from measuring RQS degradation as a function of pruning aggressiveness. LLMLingua's paper characterizes compression quality through a perplexity-preservation metric; no equivalent exists here.

3. **Short factual turns are incorrectly pruned:** A turn like `{"role": "user", "content": "ok", "turn_id": 5}` (2 words) is correctly pruned by the `len(words) < 3` check. But a turn like `{"content": "Yes, use auth.py"}` has 4 words, density = 1/4 = 0.25 (above threshold), and will be kept. However, `{"content": "Yes, use it"}` has 4 words, density = 0/4 = 0.0, and will be pruned — correct. These edge cases behave reasonably, but the regime is not documented.

4. **Role-agnostic pruning:** Assistant turns typically carry higher information (they contain generated code/explanations) than user turns (which may be short acknowledgements). A role-weighted threshold would be more principled.

5. **`import re` inside a loop:** `zippy.py:219` calls `import re` inside the per-turn loop body. Python caches module imports, so this is not a correctness bug, but it is an unnecessary symbol lookup on each iteration. Move to module-level imports.

---

## 5. History Compression Quality — Semantic Coherence

### Does pruning preserve semantic coherence for downstream agents?

**Structural concern:** The compression chain operates on a flat string representation:
```python
all_text = " ".join(
    f"{turn.get('role','?')}: {turn.get('content','')}"
    for turn in history
)  # zippy.py:159-162
```

After `_substitute_hardcoded()` runs on this flat string, the resulting text is passed to `CavemanCompressor.compress()`. However, `CavemanCompressor` is designed for source code: it only compresses `#` comment lines and triple-quoted docstring content. All other lines are passed through unchanged (`src/inference_bridge.py:631`). When given a flat conversation string (no `#` prefixes, no triple-quotes), `CavemanCompressor.compress()` does nothing — it passes every line through unchanged.

This is the biggest semantic gap in ZIPPY: **the CavemanCompressor is applied to a data type it was not designed for.** The fallback `_naive_compress()` (stop-word removal) is actually more appropriate for conversation text than `CavemanCompressor`, yet it is only used when InB is unavailable.

**Consequence:** On the primary code path (InB available), the compression chain is:
1. Substitute hardcoded keys — effective for repeated identifiers
2. Prune low-info turns — partially effective
3. `CavemanCompressor.compress()` — **does nothing** on conversation text
4. Truncate to budget — hard cut, information lost

The compression between steps 1 and 4 is entirely pruning-based, not compressor-based. The KR ratio will be higher (worse) than expected, and the KR-based confidence score (`1.0 - kr_ratio`, line 116) will be incorrectly low.

**Semantic coherence for downstream agents:** Because CavemanCompressor does not alter the surviving turns, their content is intact. Semantic coherence is preserved for kept turns. The risk is entirely from incorrectly pruned turns — factual turns that fail the density/length filter or lack `turn_id` (see below).

---

## 6. FeatureSlice Interface Audit

**Taxonomy tags (`zippy.py:121`):**
```python
taxonomy_tags = ["state.hardcoded", "history.compressed", "kolmogorov", "session.zip", "turn.pruned"]
```
These are hardcoded unconditionally. If zero turns were pruned, the `"turn.pruned"` tag is still emitted. Tags should reflect the actual operation:
```python
tags = ["state.hardcoded", "history.compressed", "kolmogorov", "session.zip"]
if pruned_turns:
    tags.append("turn.pruned")
if delta_state:
    tags.append("state.delta")
```

**`compression_pct` not set:** `BaseAgent.emit()` accepts `compression_pct` (`base_agent.py:197`) but ZIPPY never passes it. The outbound slice always has `compression_pct=None`. This field should be `round((1.0 - kr_ratio) * 100, 1)`.

**`token_count` semantic ambiguity:** `token_count=tokens_after` (line 142) correctly reflects the ZIP size, but the orchestrator may interpret this as the full payload token count. The payload also contains `hardcoded_state`, `routing_decision`, `compressed_chunks`, etc. A comment or separate field (`zip_token_count`) would prevent misinterpretation.

**`decision` field truncation:** `BaseAgent.__init__` applies `decision[:100]` at `base_agent.py:71`. The `decision` string built at `zippy.py:99-103` can exceed 100 characters for large sessions (many facts hardcoded, many turns pruned). It will be silently truncated in the audit log, losing the `pruned_turns` count from the decision string.

**`confidence` computation:** `confidence = 1.0 - kr_ratio` (`zippy.py:116, 139`). When `tokens_before == 0` (empty history), `kr_ratio` is 0.0 (line 198 guards with `max(tokens_before, 1)`), so `confidence = 1.0`. This is reasonable. However, when the ZIPpe is very effective (kr_ratio → 0.05), confidence → 0.95 and the signal is meaningful. When CavemanCompressor does nothing (as shown above), kr_ratio will be high (0.6–0.9) and confidence will be low (0.1–0.4), which will suppress downstream orchestrator trust even though the ZIP is semantically correct. The confidence formula misattributes compressor failure as low-confidence output.

---

## 7. Error Handling Gaps

### State store: silent `except Exception: pass`

Both `_load_state()` (`zippy.py:286-294`) and `_save_state()` (`zippy.py:296-305`) catch all exceptions silently. A corrupt `state.json` (partial write, encoding error) causes `_load_state` to return `{}`, wiping all hardcoded facts for the session. The next call will rebuild from scratch, but the session loses compression efficiency and the operator sees no warning.

**Minimum fix:** Log the exception via `self.log_decision` before returning the empty dict:
```python
except Exception as exc:
    self.log_decision("STATE_LOAD_ERROR", str(exc), "state.json corrupt or unreadable", 0.0)
    return {}
```

### State store: read-modify-write race condition

`StateStore.save()` (`orchestrator/state_store.py:27-35`) and `ZippyAgent._save_state()` (`zippy.py:296-305`) share the same pattern:
1. Read all sessions from `state.json`
2. Overwrite the target session
3. Write all sessions back

If two pipeline runs share the same `session_id` and execute concurrently (e.g., two `kloc.py pipeline` calls with `--session default`), both reads will see the same stale data and one write will overwrite the other's changes. There is no file lock, no atomic rename, and no version field.

On Windows (the target platform), the file write at `zippy.py:305` is:
```python
self.state_path.write_text(json.dumps(...), encoding="utf-8")
```
`write_text` on Windows is not atomic; it truncates then writes. A crash mid-write produces a zero-byte file, which will be silently treated as `{}` on the next load.

**Fix options (in order of simplicity):**
1. Use `tempfile.NamedTemporaryFile` + `os.replace()` for atomic writes.
2. Add a per-session file (`state_{session_id}.json`) to eliminate cross-session interference.
3. Add an `fcntl`-style lock (already conditionally guarded in `message_bus.py` for Windows compatibility — the same pattern should be applied here).

Note: `ZippyAgent` duplicates the entire state-store logic from `orchestrator/state_store.py` without using the `StateStore` class. This is dead code duplication. ZIPPY should instantiate `StateStore` and delegate to it.

### CavemanCompressor exception suppressed

`zippy.py:183-186`:
```python
try:
    zipped = self._caveman.compress(kept_text)
except Exception:
    zipped = kept_text
```
`kept_text` is the uncompressed conversation string. If CavemanCompressor raises, ZIPPY silently falls back to the full (uncompressed) kept text, then truncates to budget. The exception is not logged. A repeated compressor failure would produce oversized prompts with no alert.

### Missing `turn_id` handling

`_prune_turns()` skips turns without a `turn_id` field (`zippy.py:211-212`):
```python
if turn_id < 0:
    continue
```
If `turn_id` is absent (`turn.get("turn_id", -1)`), the turn is never pruned regardless of how low its density is. This is a safe default (preserve unknown turns), but it means that callers who omit `turn_id` entirely will get zero pruning. This should be documented, or turns without `turn_id` should be assigned sequential IDs during `process()`.

---

## 8. Specific Line-Level Issues

| Line | Issue | Severity |
|------|-------|----------|
| `zippy.py:47` | `_MAX_ZIP_TOKENS = 50` is hardcoded constant; not read from env like `ACTIVE_TIER_MAX`. Should be `int(os.getenv("MAX_ZIP_TOKENS", "50"))`. | Medium |
| `zippy.py:116` | `confidence = 1.0 - kr_ratio` conflates compression efficiency with output quality. A KR of 0.9 (poor compression) does not mean the ZIP is unreliable — it means the input was already short. | Medium |
| `zippy.py:159-162` | Flattens conversation history to a single string before pruning. Turn boundaries are lost; CavemanCompressor cannot distinguish `user:` prefixes from content. | Medium |
| `zippy.py:191` | `_truncate_to_budget(zipped, _MAX_ZIP_TOKENS)` uses word-split heuristic, not `count_tokens()`. Budget enforcement is inaccurate for code-heavy text. | High |
| `zippy.py:219` | `import re` inside a loop body. Should be at module top. | Low |
| `zippy.py:222` | `density < _PRUNE_THRESHOLD and len(words) < 10` — the word-count guard contradicts the density filter for medium-length filler turns. | Medium |
| `zippy.py:245` | `len(token) > 6` rejects 6-character identifiers (e.g., `render`, `update`). Should be `>= 6`. | Low |
| `zippy.py:247` | Short key `token[:3] + str(len(token))` is not guaranteed unique. `UserService` (11 chars) → `Use11` and `UserState` (also 9 chars) → `Use9`, but `UserSession` (11 chars) → `Use11` collides with `UserService`. No collision check. | High |
| `zippy.py:263` | `re.sub(r'\b' + re.escape(long_form) + r'\b', ...)` — `\b` boundary fails for identifiers followed by `(`, `.`, `,`. Compound expressions like `UserService.authenticate` are not replaced even when both components are hardcoded. | Medium |
| `zippy.py:269` | Comment "1 word ≈ 1.3 tokens for compressed caveman-speak" is inaccurate for code identifiers (typically 1.5–2.5 tokens/word). | Low |
| `zippy.py:286-294` | Duplicates `StateStore.load()` logic. Should use `orchestrator/state_store.py:StateStore`. | Low |
| `zippy.py:296-305` | Duplicates `StateStore.save()` logic + read-modify-write race condition. | High |
| `zippy.py:305` | Non-atomic file write. Crash mid-write produces zero-byte state file silently treated as `{}`. | High |
| `zippy.py:184` | CavemanCompressor exception caught silently with no log. | Medium |
| `zippy.py:121` | `"turn.pruned"` tag emitted unconditionally even when `pruned_turns == []`. | Low |
| `zippy.py:73` | `self._caveman = CavemanCompressor() if _INB_AVAILABLE else None` — CavemanCompressor.compress() does not compress plain conversation text (only source code comments/docstrings). The code path where `_INB_AVAILABLE=True` is the broken path for this use case. | High |

---

## 9. Proposed High-Value Improvements

### Improvement 1: Fix token-accurate budget truncation (addresses lines 191, 269)

Replace `_truncate_to_budget` with a binary search over `count_tokens()`. See the fix in Section 1 above. Impact: the 50-token budget becomes a hard guarantee instead of an approximation. Estimated effort: 10 lines.

### Improvement 2: Fix short-key collision in `_extract_facts` (addresses line 247)

```python
# Current (collision-prone):
short_key = token[:3] + str(len(token))

# Fixed:
existing_shorts = set(existing_hardcoded.values()) | set(delta.values())
short_key = token[:3] + str(len(token))
suffix = 0
while short_key in existing_shorts:
    suffix += 1
    short_key = token[:3] + str(len(token)) + str(suffix)
```
Impact: prevents two identifiers from mapping to the same key, which would silently corrupt the decompressed ZIP. Estimated effort: 8 lines.

### Improvement 3: Apply correct compressor to conversation text (addresses line 73, Section 5)

`CavemanCompressor` is a source-code compressor. For conversation history, use `_naive_compress()` (which is the more appropriate path) regardless of InB availability, or add a dedicated `compress_conversation()` method to `CavemanCompressor` that applies stop-word + vowel removal to plain prose:

```python
# In _compress_history(), replace:
if self._caveman:
    try:
        zipped = self._caveman.compress(kept_text)
    except Exception:
        zipped = kept_text
else:
    zipped = self._naive_compress(kept_text)

# With:
# CavemanCompressor.compress() only processes # lines and docstrings.
# Conversation text is prose — use naive compress unconditionally.
zipped = self._naive_compress(kept_text)
# Then apply hardcoded substitution again on compressed text.
```
Impact: the compression chain actually compresses conversation text. Current code silently does nothing in the CavemanCompressor step for all normal inputs. This is the highest-value change — it unlocks the full compression pipeline. Estimated effort: 5 lines.

### Improvement 4: Eliminate state-store duplication and add atomic writes (addresses lines 286-305)

Delete `_load_state` and `_save_state` from `ZippyAgent` entirely. Instantiate `StateStore` in `__init__`:
```python
from orchestrator.state_store import StateStore

def __init__(self, state_path=_DEFAULT_STATE_PATH, **kwargs):
    super().__init__(**kwargs)
    self._store = StateStore(state_path)
    self._caveman = CavemanCompressor() if _INB_AVAILABLE else None
```
Then update `orchestrator/state_store.py:StateStore.save()` to use atomic writes:
```python
import tempfile, os
def save(self, session_id: str, state: Dict[str, Any]) -> None:
    all_state = {}
    if self.path.exists():
        try:
            all_state = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            pass
    all_state[session_id] = state
    tmp = self.path.with_suffix(".tmp")
    tmp.write_text(json.dumps(all_state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, self.path)  # atomic on POSIX; near-atomic on Windows (same volume)
```
Impact: eliminates race condition, eliminates code duplication, prevents corrupt-state data loss.

---

## Summary Table

| Area | Severity | Root Cause |
|------|----------|------------|
| Budget truncation uses word-count, not `count_tokens()` | High | `_truncate_to_budget` line 191 |
| Short-key collision in `_extract_facts` | High | `zippy.py:247` |
| CavemanCompressor does nothing on conversation text | High | `zippy.py:73, 182-188` |
| Read-modify-write race condition in state store | High | `zippy.py:296-305`, `state_store.py:27-35` |
| Non-atomic file write corrupts state on crash | High | `zippy.py:305` |
| `confidence = 1.0 - kr_ratio` conflates efficiency with quality | Medium | `zippy.py:116, 139` |
| Prune threshold double condition creates dead zone | Medium | `zippy.py:222` |
| `\b` boundary fails for `identifier.method` patterns | Medium | `zippy.py:263` |
| Silent exception suppression in state load/save | Medium | `zippy.py:287-294` |
| `"turn.pruned"` tag emitted unconditionally | Low | `zippy.py:121` |
| `import re` inside loop body | Low | `zippy.py:219` |
| `len(token) > 6` excludes 6-char identifiers | Low | `zippy.py:245` |
| `compression_pct` never set in outbound slice | Low | `zippy.py:120-143` |
| State store logic duplicated from `StateStore` class | Low | `zippy.py:284-305` |
