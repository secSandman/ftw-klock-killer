# PRUNER Agent — Technical Feedback
**Date:** 2026-04-06  
**Reviewer:** Claude Code (claude-sonnet-4-6)  
**Files audited:**
- `agents/pruner.py`
- `agents/base_agent.py`
- `src/inference_bridge.py` (lines 1–235, focusing on `predictability_index` and `hot_zone_score`)
- `data/taxonomy.json`

---

## 1. SDR Fidelity — Sparse Distributed Representations

### What the implementation actually does

The agent computes a scalar `sdr_overlap = pi_score * (1.0 - hzs_score)` (line 153) and applies two binary thresholds to classify each chunk as FOREGROUND or BACKGROUND (lines 156–160). This is a two-feature linear product, not an SDR.

### What HTM SDR actually requires

In Numenta's Spatial Pooler (Ahmad & Hawkins 2016), an SDR is a **binary vector** of length N (typically 2048 columns) where exactly ~2% of bits are active. Key properties:

| HTM SDR property | PRUNER implementation | Gap |
|---|---|---|
| Binary activation vector of N columns | Single scalar float | Missing entirely |
| Fixed sparsity enforced by inhibition | Soft ratio target not enforced | Missing |
| Winner-Take-All (WTA) inhibition within inhibition radius | No WTA, no radius concept | Missing |
| Permanence-weighted synaptic overlap | No permanence; direct multiplication | Conceptually replaced |
| Boosting of under-represented columns | No boosting | Missing |
| Overlap score = dot product of input bits × permanence | `pi * (1 - hzs)` is not a dot product | Incorrect analog |

The `sdr_overlap` value (line 153) is a **product of two independent scores**, not an overlap count between an input pattern and a learned permanence matrix. In HTM, overlap is the number of active bits in common between the current input and a minicolumn's connected synapses. That mechanism is nowhere present.

### Actual sparsity achieved

`SDR_SPARSITY_TARGET = 0.20` (line 63). This is documented in a comment as mirroring "Numenta's ~2% sparsity target" — that comment is factually wrong. Numenta uses **~2%** for neocortical fidelity; this implementation targets **20%**, a 10× discrepancy. Furthermore, the target is used only in the confidence calculation (line 105):

```python
confidence = 1.0 - abs(fg_ratio - self.SDR_SPARSITY_TARGET)
```

It is **never enforced**. If 80% of chunks are classified FOREGROUND, the `fg_ratio` deviates from 0.20 by 0.60, reducing confidence to 0.40, but the classifications are not revised. There is no feedback loop that caps FOREGROUND count. On a file with many complex functions (e.g., Doom's `r_draw.c`), FOREGROUND could reach 90% and the agent would still emit all of them, defeating compression.

**Verdict:** SDR fidelity is surface-level. The naming borrows HTM vocabulary but the mechanism is a scalar threshold pair, not a distributed representation.

---

## 2. PI → Column Overlap Mapping

### The docstring claim (line 15–16)
```
# Mapping: PI score ≈ column overlap score in HTM.
```

### What `predictability_index` actually computes

`predictability_index` (inference_bridge.py:115–186) is a **rule-based heuristic**:
1. Starts with a base score keyed on non-blank line count (step function with 5 thresholds).
2. Subtracts 0.06 per structural keyword (`for`, `while`, `try`, `except`, `yield`, `async for`, `async with`), capped at −0.35.
3. Adds 0.15 if ≥50% of lines are "boilerplate" (`return`, `self.x =`, `raise`, `pass`, etc.).

### Where the mapping diverges from biology

| HTM column overlap | PI score | Divergence |
|---|---|---|
| Increases when more input bits match the column's synaptic connections | Decreases when more structural keywords appear | Inverted polarity — low PI = "complex" = FOREGROUND, but high HTM overlap = familiar = background. The semantics are accidentally aligned only because both are used inversely. |
| Continuous learning: permanences updated after each pattern | Static: same PI for the same source regardless of history | No learning |
| Spatial: a column's overlap depends on which *other* columns are active | PI is per-function, no cross-chunk context | Spatially independent |
| Operates on binary feature vectors | Operates on raw source text via regex | Different representation space |

The mapping holds as a rough intuition (high PI ≈ predictable ≈ low information content ≈ low overlap with "complex" pattern) but breaks on any file where the majority of functions are long but simple (e.g., large `__init__` methods with many `self.x = y` lines). Such functions receive a low base score (n > 20 lines → base = 0.70) but the boilerplate bonus pushes PI to 0.85, correctly marking them as background. That edge case is handled, but only because of the bonus heuristic, not because the underlying column overlap model is sound.

### The CLAUDE.md critical constraint interaction

The CLAUDE.md PI formula note states: loop-containing functions with >20 body lines get `PI = 0.70 - 0.06 = 0.64 < 0.70 threshold → NOT skeletonized`. PRUNER inherits this constraint because it calls `predictability_index` directly. A 40-line function with a single `for` loop will score `base=0.70, penalty=0.06, PI=0.64`, which is below `FOREGROUND_PI_THRESHOLD=0.70`, correctly routing it to FOREGROUND. However, a 40-line function with **two** `for` loops scores `PI=0.70-0.12=0.58`, also FOREGROUND — correct. But a 40-line function with no loops scores `PI=0.70`, exactly at threshold, classified as BACKGROUND. The off-by-one boundary (strict `<` on line 157: `pi_score < self.FOREGROUND_PI_THRESHOLD`) means PI = 0.70 → `is_complex = False` → BACKGROUND. This is consistent with the InB skeletonizer's `>= pi_threshold` logic but should be made explicit in a comment since the boundary behaviour is load-bearing.

---

## 3. HZS → Dendrite Analogy

### The docstring claim (line 15–16)
```
# HZS score ≈ proximal vs distal dendrite activation (recent vs predicted).
```

### What `hot_zone_score` actually computes

```python
def hot_zone_score(line_idx: int, changed_lines: set[int]) -> float:
    min_dist = min(abs(line_idx - cl) for cl in changed_lines)
    return round(math.exp(-HZS_LAMBDA * min_dist), 4)
```

`HZS_LAMBDA = 0.1`. This is a **1-D spatial exponential decay** from the nearest git-changed line. At distance 0 → HZS = 1.0. At distance 10 → HZS ≈ 0.37. At distance 23 → HZS ≈ 0.10 (the FOREGROUND threshold).

### Biological grounding assessment

HTM dendrites have two distinct roles:
- **Proximal dendrites**: receive feedforward input (what is happening *now*); directly cause column activation.
- **Distal/basal dendrites**: receive lateral/contextual input (what the system *predicts* will happen); place the cell in a "predicted" depolarised state.

The claimed analogy is that HZS ≈ "proximal activation by recent change signal." This is a stretch:

1. **Proximal activation is binary in HTM** (it fires or doesn't once overlap exceeds a threshold). HZS is continuous. The exponential decay has no HTM biological basis — it is borrowed from spatial statistics (e.g., kriging / Gaussian process covariance).

2. **The distal dendrite half of the analogy is absent.** The docstring says "proximal vs distal," but PRUNER only computes a single HZS value. There is no "predicted" signal (distal activation) computed separately.

3. **Lambda = 0.1 has no theoretical justification.** At λ = 0.1, a line 23 positions away from the nearest change just crosses the cold threshold. For a typical git commit touching lines in a cluster of 5 lines, this means ±23 lines of context are "hot." In a 100-line function, that is 46% of the body — far too broad. A biologically motivated rate would require empirical calibration against the actual commit patterns in the target codebase. The value is a magic number.

4. **No temporal memory.** In HTM, distal dendrites encode *sequences* — they fire based on what the system saw in previous time steps. PRUNER is entirely stateless across slices. There is no "last N commits seen" or sequence model.

5. **The fallback is inverted** (inference_bridge.py:199–200):
   ```python
   if not changed_lines:
       return 1.0  # no diff info → treat all lines as Hot
   ```
   When git diff is unavailable, every line is maximally "hot" (HZS = 1.0), which forces **every chunk above the HZS threshold (> 0.10)** to FOREGROUND. In PRUNER, when `hot_lines` is empty (line 80–82), `hzs_score = 0.0` is used (lines 145–148), directly contradicting the InB fallback. This inconsistency means the behaviour of `_classify_chunk` diverges from `hot_zone_score` on the boundary condition for no-diff inputs.

---

## 4. Missing HTM Features — Five Specific Mechanisms

### 4.1 Temporal Memory / Sequence Learning

HTM temporal memory tracks which **cells within a column** fire, enabling the system to disambiguate the same input pattern depending on context (prior sequence). PRUNER has no temporal memory — it classifies each chunk independently regardless of what appeared before it in the file or in prior pipeline runs. 

**Impact:** A helper function `_validate_config()` always gets the same PI score whether it appears in a hot commit path or a cold utility module. A temporal model would recognise that when `_validate_config()` co-occurs with recently-changed callers, it should be elevated to FOREGROUND.

**Implementation sketch:**
```python
# Store per-chunk activation history across pipeline runs in the bus state
self._cell_state: Dict[str, Deque[bool]] = defaultdict(lambda: deque(maxlen=8))

def _temporal_boost(self, chunk_id: str, currently_active: bool) -> float:
    history = self._cell_state[chunk_id]
    if not history:
        return 0.0
    # Fraction of recent activations = predicted probability
    return sum(history) / len(history)
```

### 4.2 Spatial Pooler Normalization / Global Inhibition

HTM's spatial pooler enforces a **global sparsity constraint** via k-WTA (k-Winner-Take-All) inhibition: regardless of how many columns exceed their local overlap threshold, only the top-k% are activated. PRUNER applies two independent per-chunk binary thresholds with no global inhibition.

**Impact:** On a file where all 200 functions are complex (e.g., a game engine physics module), all 200 are FOREGROUND, TER collapses to near 0%. A WTA layer would cap FOREGROUND at the target sparsity.

**Implementation sketch:**
```python
def _apply_wta(self, scored_chunks: List[Dict]) -> List[Dict]:
    k = max(1, int(len(scored_chunks) * self.SDR_SPARSITY_TARGET))
    # Rank by combined score (low PI + high HZS = most foreground)
    ranked = sorted(scored_chunks, key=lambda c: c["pi_score"] - c["hzs_score"])
    foreground_ids = {c["chunk_id"] for c in ranked[:k]}
    for c in scored_chunks:
        c["zone"] = "FOREGROUND" if c["chunk_id"] in foreground_ids else "BACKGROUND"
    return scored_chunks
```

### 4.3 Synaptic Permanence and Online Learning

In HTM, each proximal synapse has a **permanence value** [0, 1] that increases when the synapse is active during a correct prediction and decays otherwise. This allows the spatial pooler to adapt over time. PRUNER is entirely static — it recomputes PI and HZS from scratch on each call with no memory of prior classifications.

**Impact:** PRUNER cannot learn that a particular function is *always* BACKGROUND across hundreds of pipeline runs and short-circuit the expensive `predictability_index` call. Nor can it learn that a class of functions consistently misclassified as BACKGROUND actually causes RQS failures.

**Implementation sketch:** Maintain a `permanence_table: Dict[str, float]` keyed by `chunk_id` (or content hash), updated after downstream RQS feedback:
```python
def update_permanence(self, chunk_id: str, was_correct: bool):
    delta = +0.01 if was_correct else -0.05  # Hebbian: fast forgetting, slow learning
    self.permanence_table[chunk_id] = max(0.0, min(1.0,
        self.permanence_table.get(chunk_id, 0.5) + delta))
```

### 4.4 Anomaly Detection

HTM produces a natural **anomaly score** as the fraction of columns that **burst** (all cells activate, indicating an unpredicted input). PRUNER has no anomaly detection — it cannot flag code chunks that are structurally unlike anything seen before (e.g., a novel design pattern, a first-time use of an external API).

**Impact:** Novel code chunks with moderate PI scores (0.65–0.70) near the threshold boundary may be silently classified as BACKGROUND when they represent genuinely new logic that should be inspected. An anomaly signal would catch these.

**Implementation sketch:**
```python
def _anomaly_score(self, pi_score: float, hzs_score: float, chunk_id: str) -> float:
    expected_pi = self._running_mean_pi.get(chunk_id[:4], 0.75)  # namespace-level prior
    deviation = abs(pi_score - expected_pi)
    return min(1.0, deviation * 2.0)  # normalize to [0,1]
```

### 4.5 Boosting of Under-Represented Columns

HTM boosts columns that rarely win inhibition, preventing the spatial pooler from converging on a small subset of always-active columns ("dead columns"). Without boosting, PRUNER is vulnerable to **systematic classification bias**: if the PI formula always classifies certain chunk types (e.g., short class-level `__init__` functions) as BACKGROUND, those chunk types are never represented in the FOREGROUND output, and downstream agents lose the ability to refine their understanding of those patterns.

**Impact:** On a codebase consisting entirely of small initializer functions, `fg_count` would be 0, confidence would be `1.0 - |0.0 - 0.20| = 0.80`, and the pipeline would silently under-analyse the entire file.

**Implementation sketch:**
```python
def _boost_score(self, chunk_type: str) -> float:
    activations = self._type_activation_counts.get(chunk_type, 0)
    total = self._total_classifications
    if total == 0:
        return 1.0
    actual_rate = activations / total
    # Boost if this type has been under-activated relative to sparsity target
    return 1.0 + max(0.0, (self.SDR_SPARSITY_TARGET - actual_rate) * 5.0)
```

---

## 5. Taxonomy Tag Correctness

### Tags emitted (line 112)
```python
taxonomy_tags = ["code.classification", "sdr.foreground", "sdr.background", "predictability", lang_tag]
```

### Audit against `data/taxonomy.json`

| Tag emitted | In `all_valid_tags`? | Correct usage? |
|---|---|---|
| `code.classification` | Yes | Correct — this is the classification category root |
| `sdr.foreground` | Yes | Incorrect — this should be emitted per-chunk, not on the slice. The slice always contains *both* foreground and background chunks. Emitting both `sdr.foreground` and `sdr.background` on every output slice regardless of actual composition is misleading. |
| `sdr.background` | Yes | Same issue as above |
| `predictability` | Yes | Correct |
| `lang.python` / etc. | Yes | Correct |

### Missing tags that should be emitted

- **`hot_zone`** and **`cold_zone`** are defined in the taxonomy under `code.classification` but are never emitted. They would be the correct per-slice tags when git hot lines are present vs absent.
- If `git_hot_lines` is non-empty and any chunk has HZS > threshold, the slice should include `hot_zone`.
- If all chunks are cold (no git data), the slice should include `cold_zone`.

### Incorrect conditional tag logic (lines 109–110)

```python
lang_tag = f"lang.{language}" if f"lang.{language}" in self._lang_tags() else "lang.unknown"
```

`_lang_tags()` is a static method returning a hardcoded list (line 179–180). This is correct in isolation, but if the orchestrator sends `language = "javascript"` (not in the list), the fallback to `lang.unknown` silently discards the real language. The taxonomy would need extension; currently this is handled gracefully, but there is no warning or log entry emitted when the fallback fires.

---

## 6. FeatureSlice Interface Audit

### Fields read from `slice_in.payload`

| Field | Line | Type assumed | Validated? |
|---|---|---|---|
| `chunks` | 71 | `list` | No — `.get("chunks", [])` silently defaults to empty |
| `git_hot_lines` | 72 | `dict[str, list[int]]` | No |
| `file_path` | 73 | `str` | No |
| `language` | 74 | `str` | No |

Per-chunk fields read inside `_classify_chunk`:

| Field | Line | Default |
|---|---|---|
| `source` | 131 | `""` |
| `start_line` | 132 | `0` |
| `chunk_type` | 133 | `"function"` |
| `chunk_id` | 134 | `""` |
| `end_line` | 166 | `start_line` |
| `original_tokens` | 174 | `0` |

### Fields written to `slice_out.payload`

| Field | Line |
|---|---|
| `classified_chunks` | 114 |
| `foreground_count` | 115 |
| `background_count` | 116 |
| `total_chunks` | 117 |
| `file_path` | 118 |
| `language` | 119 |

### Missing passthrough fields

The output payload does **not** forward `git_hot_lines`. Downstream agents (GRUG, BALANCER) that may want to apply their own hot-line logic receive no git diff data. This forces those agents to re-derive hot lines from scratch or accept that the information is lost after PRUNER.

### `compression_pct` is always `None`

`BaseAgent.emit()` accepts `compression_pct` but PRUNER never computes or passes it. At this stage, PRUNER doesn't compress — it classifies — so this is acceptable, but the field's absence means TER tracking from the bus audit log has no PRUNER-stage contribution.

---

## 7. Error Handling Gaps

### 7.1 Empty `chunks` list

If `payload.get("chunks", [])` returns `[]` (missing key or explicit empty list), the loop at line 84 never executes. The agent emits a valid slice with `total_chunks = 0` and `fg_ratio = 0.0`. The confidence calculation becomes `1.0 - |0.0 - 0.20| = 0.80` — misleadingly high for a degenerate input. No warning is logged. Downstream agents receive an empty `classified_chunks` list and must handle that gracefully themselves.

**Recommended fix:** Add an explicit check and emit a `quality.fail` slice (or at minimum log a WARNING decision) when `total == 0`.

### 7.2 `_INB_AVAILABLE = False` path produces inconsistent results

When `src.inference_bridge` is not importable (line 36–37), the PI fallback (lines 141–142) uses:
```python
n = len([l for l in source.splitlines() if l.strip()])
pi_score = max(0.0, min(1.0, 1.0 - n / 50.0))
```
A 50-line function gives `PI = 0.0`, which is below the 0.70 threshold → FOREGROUND. A 35-line function gives `PI = 0.30` → FOREGROUND. This means in offline mode, all functions longer than ~15 lines (`1.0 - 15/50 = 0.70`, exactly at threshold) are FOREGROUND. With `FOREGROUND_PI_THRESHOLD = 0.70` and `is_complex = pi_score < 0.70`, a 15-line function gives `PI = 0.70`, `is_complex = False` — background. The fallback formula is a different model than the real `predictability_index` and will produce different FOREGROUND/BACKGROUND distributions, potentially breaking the 77.9% TER baseline silently.

### 7.3 `hot_zone_score` inversion on no-diff inputs

As noted in section 3, when `hot_lines` is an empty set, PRUNER sets `hzs_score = 0.0` (line 148). But `hot_zone_score` in InB returns `1.0` when `changed_lines` is empty (inference_bridge.py:199–200). This is a **silent inconsistency**: PRUNER bypasses the InB function entirely when `hot_lines` is empty (the guard at line 145 is `if _INB_AVAILABLE and hot_lines`), so the InB fallback is never invoked. The effect is that with no git data, PRUNER classifies by PI alone and misses the InB intent of treating all lines as hot.

### 7.4 `git_hot` type assumption: `{file_path: set[int]}` vs `{file_path: list[int]}`

Line 72 has the comment `# {file_path: set[int]}` but JSON serialisation (via `to_dict()` / `from_dict()`) will always deserialise the inner value as a `list`, not a `set`. Line 82 calls `set(git_hot.get(file_path, []))`, which correctly converts the list to a set — this is fine. But line 72's comment is misleading and could cause confusion if a future contributor assumes the set is already a set and skips the conversion.

### 7.5 No `InferenceBridge` unavailability warning

Lines 33–37:
```python
try:
    from src.inference_bridge import predictability_index, hot_zone_score
    _INB_AVAILABLE = True
except ImportError:
    _INB_AVAILABLE = False
```

`_INB_AVAILABLE = False` is never logged. An operator running PRUNER in an environment where `src/` is missing will get degraded (and differently calibrated) classifications with no indication in the decision audit log.

---

## 8. Specific Line-Level Code Issues

### 8.1 `pruner.py:63` — Misleading constant comment
```python
SDR_SPARSITY_TARGET = 0.20   # target ~20% of chunks as FOREGROUND
```
The inline comment in the docstring (line 60) reads: `# SDR tuning — these mirror Numenta's ~2% sparsity target`. 20% is not Numenta's 2% target. The comment in the class body is correct ("~20% of chunks"); the module docstring comment is false. The value should either be changed to 0.02 (with a note that this maps code chunks, not neocortical columns, so a looser sparsity is intentional) or the documentation corrected to not claim HTM alignment.

### 8.2 `pruner.py:93` — `fg_ratio` division guard is correct but the confidence formula is not symmetric
```python
fg_ratio = fg_count / total if total > 0 else 0.0
# ...
confidence = 1.0 - abs(fg_ratio - self.SDR_SPARSITY_TARGET)
```
When `fg_ratio = 0.80`, confidence = `1.0 - 0.60 = 0.40`. When `fg_ratio = 0.0`, confidence = `1.0 - 0.20 = 0.80`. A completely empty classification scores higher confidence (0.80) than a heavily loaded one (0.40). This means the agent self-reports higher confidence on degenerate (empty chunk) inputs than on correctly-loaded-but-noisy inputs. Confidence should probably be 0.0 when `total == 0`.

### 8.3 `pruner.py:105` — Confidence can exceed 1.0 conceptually but does not in practice
`abs(fg_ratio - 0.20)` ranges from 0 (perfect) to 0.80 (all foreground), so confidence ranges from `[0.20, 1.0]`. It cannot reach 0.0 with this formula. Minimum confidence is 0.20 (when fg_ratio = 1.0 or 0.0+0.20). This means even catastrophic mis-classification is reported with 0.20 confidence, not 0.0. This is a subtle logical error.

### 8.4 `pruner.py:153` — `sdr_overlap` is computed but used only for return value, not classification
```python
sdr_overlap = pi_score * (1.0 - hzs_score)
```
This value is written into the chunk dict (line 170) and passed downstream, but it is **not used to classify the chunk** (lines 156–160 use `is_hot` and `is_complex` derived from raw threshold comparisons, not from `sdr_overlap`). The value is therefore decorative metadata. Either it should drive classification, or the downstream consumer (GRUG) should be documented as the primary consumer of `sdr_overlap`.

### 8.5 `pruner.py:157` — Strict less-than creates an unintuitive boundary
```python
is_complex = pi_score < self.FOREGROUND_PI_THRESHOLD
```
`FOREGROUND_PI_THRESHOLD = 0.70` and `SKELETAL_PI_THRESHOLD = 0.70` (inference_bridge.py:40). The InB skeletonizer uses `>= pi_threshold` for skeletonizing (i.e., bodies that score 0.70 are skeletonized = treated as predictable = BACKGROUND). PRUNER's `< 0.70` means PI = 0.70 → `is_complex = False` → BACKGROUND. The boundary is consistent, but it relies on exact float equality at 0.70, which is fine for the step-function values returned by `predictability_index` but fragile if the function is ever modified to return continuous values.

### 8.6 `pruner.py:109` — Silent taxonomy fallback without log entry
```python
lang_tag = f"lang.{language}" if f"lang.{language}" in self._lang_tags() else "lang.unknown"
```
No `log_decision` call is made when the fallback fires. An unknown language is silently downgraded to `lang.unknown`. Add a `log_decision` with `decision_type="LANG_FALLBACK"` when this triggers.

### 8.7 `pruner.py:124` — `token_count` sums `original_tokens`, not compressed tokens
```python
token_count = sum(c.get("original_tokens", 0) for c in classified),
```
`FeatureSlice.token_count` is used by the bus for TER accounting. At the PRUNER stage, no compression has occurred yet, so passing `original_tokens` is arguably correct — but it gives downstream agents no signal about how many tokens PRUNER's BACKGROUND classification will save. The token count of BACKGROUND chunks specifically should be computable here and either passed as a separate field or used to populate `compression_pct`.

### 8.8 `inference_bridge.py:189` — `hot_zone_score` signature uses Python 3.9+ type hint syntax
```python
def hot_zone_score(line_idx: int, changed_lines: set[int]) -> float:
```
`set[int]` as a type hint (lowercase built-in generic) requires Python 3.9+. If the project targets Python 3.8, this will raise `TypeError` at import time in 3.8. The rest of the codebase uses `from typing import ...` imports. Should be `Set[int]` from `typing` for compatibility, or explicitly document the Python 3.9+ requirement.

---

## Summary: Priority Improvements

| Priority | Issue | Fix effort |
|---|---|---|
| Critical | SDR_SPARSITY_TARGET = 0.20 is not enforced; on complex files TER collapses | Add WTA k-selection after scoring loop (section 4.2) |
| High | Empty chunks → misleadingly high confidence (0.80) | Add `if total == 0` guard with `quality.fail` emit |
| High | `sdr_overlap` computed but not used for classification | Either wire into classification or document as downstream metadata |
| High | InB unavailability is silent in decision audit log | Add `log_decision("WARN", "INB_UNAVAILABLE", ...)` at import time |
| Medium | `hot_zone_score` fallback inversion: InB returns 1.0, PRUNER uses 0.0 on no-diff | Unify: pass empty set to `hot_zone_score` or replicate its fallback logic |
| Medium | Confidence formula minimum is 0.20, not 0.0 | Fix formula; force `confidence=0.0` when `total==0` |
| Medium | Missing `hot_zone` / `cold_zone` tags — defined in taxonomy, never emitted | Conditionally emit based on `git_hot_lines` presence |
| Low | `git_hot_lines` not forwarded to downstream agents | Include in output payload passthrough |
| Low | `lang_tag` fallback fires silently | Add `log_decision("LANG_FALLBACK", ...)` |
| Low | Python 3.9+ `set[int]` syntax in `hot_zone_score` signature | Replace with `Set[int]` from `typing` |
