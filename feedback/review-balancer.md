# BALANCER Agent — Technical Review
**Date:** 2026-04-06
**Reviewer:** Claude Sonnet 4.6 (automated research audit)
**Files reviewed:**
- `agents/balancer.py`
- `orchestrator/pipeline.py`
- `agents/base_agent.py`
- `.env.example`

---

## Executive Summary

BALANCER is structurally sound but architecturally incomplete relative to its
claimed research basis. The core problem: it claims FrugalGPT-style cascade
routing but implements a one-shot upfront selector. This is not a minor gap — it
is a category error. The agent also carries stale pricing data, a flawed
auto-escalation flag that never triggers action, and a `ACTIVE_TIER_MAX=0`
behavior that silently saturates tier selection rather than short-circuiting
cleanly.

---

## 1. FrugalGPT Cascade Routing Gap

### What FrugalGPT actually does

FrugalGPT (Chen et al. 2024, arXiv:2305.05176) implements a **learned cascade**:

1. Send query to cheapest model (e.g. Haiku).
2. Score the response quality using a learned scorer.
3. If quality >= threshold, return — done.
4. Otherwise escalate to the next model in the chain.
5. Repeat up to model N.

The key property: **the cheap model is actually called first**. Escalation is
driven by *observed output quality*, not pre-flight heuristics.

### What BALANCER does

`_select_tier()` (lines 200-254) picks a tier upfront from static heuristics:
task type floor, context token count, and RQS-L1 from the previous agent. No
model is ever called inside BALANCER. The decision is committed at line 242
(`selected = max(task_floor, context_tier, quality_tier)`).

This is a **pre-flight router**, not a cascade. It is equivalent to Rao (2023)
cost-optimal routing — which is cited — but not FrugalGPT, which is also cited.
The distinction matters: a pre-flight router misses the cases where a cheap
model would have succeeded but static heuristics overestimate complexity. Those
over-escalations are pure waste.

### What's missing

True cascade requires BALANCER to:
1. Emit a routing decision for T1.
2. After T1 responds, receive an RQS-L2 score on the actual model output.
3. If RQS-L2 >= threshold, accept and forward.
4. If RQS-L2 < threshold, re-invoke the pipeline at T2.

This requires a feedback loop that does not exist in the current pipeline
architecture. `orchestrator/pipeline.py` runs BALANCER once (line 131:
`self.balancer.run_once(timeout=2.0)`), collects the routing decision, and
hands it to ZIPPY. There is no post-response quality re-check or re-routing
loop.

**Minimum viable cascade addition** — pseudocode:

```python
# In pipeline.py, replace single BALANCER pass with:
for attempt in range(MAX_CASCADE_DEPTH):
    balancer_out = self.balancer.run_once(timeout=2.0)
    tier = balancer_out.payload["routing_decision"]["tier"]
    if tier == 0:
        break  # local cache, no LLM needed
    llm_response = call_llm(tier, compressed_chunks)
    rqs_l2 = score_response_quality(llm_response, question)
    if rqs_l2 >= self.quality_floor:
        break  # success at this tier
    # inject escalation signal for next BALANCER pass
    grug_out.payload["rqs_l1_score"] = rqs_l2
    grug_out.payload["cascade_attempt"] = attempt + 1
    self.bus.publish("grug.out", grug_out.to_dict())
```

Until this exists, the FrugalGPT citation in the docstring (line 18-19) is
technically misleading. The correct citation is Rao (2023) alone.

---

## 2. Stale Pricing

### Current hardcoded prices (lines 54-76)

| Tier | Model | cost_per_1k_in | cost_per_1k_out |
|------|-------|---------------|----------------|
| T1 | claude-haiku-4-5-20251001 | $0.00025 | $0.00125 |
| T2 | claude-sonnet-4-5 | $0.003 | $0.015 |
| T3 | claude-opus-4-6 | $0.015 | $0.075 |

### Issues

**T1 / Haiku pricing** (line 54-55): The values `$0.00025 / $0.00125` per 1k
tokens appear to be Claude 3 Haiku pricing from early 2024, applied to a model
ID `claude-haiku-4-5-20251001`. These do not match current Anthropic published
rates. The comment on line 54 reads `# Haiku pricing (approximate)` — the word
"approximate" signals the author was uncertain. This approximation propagates
directly into `est_cost` in the routing decision and into downstream cost
tracking.

**T3 / Opus model ID** (line 71): `claude-opus-4-6` is the model being used by
this very session (as confirmed by the system-reminder). T3 being Opus is
correct, but the pricing at `$0.015 in / $0.075 out` per 1k needs verification
against current Anthropic pricing page. As of late 2025, Claude Opus 3 was
$0.015/$0.075; Claude Opus 4 pricing may differ.

**T2 / Sonnet** (line 62): `$0.003 in / $0.015 out` matches claude-sonnet-3.5
pricing from mid-2024. For `claude-sonnet-4-5` this may be stale.

**No last-updated timestamp anywhere in the file.** Add a `# Prices verified:
YYYY-MM-DD` comment to the TIER_CONFIG block so future reviewers know when to
re-check.

**`_estimate_cost()` assumes 200 output tokens** (line 268):

```python
cost_out = (200 / 1000) * cfg["cost_per_1k_out"]
```

200 output tokens is defensible for a brief explanation but wildly low for
`codegen` or `refactor` tasks which may produce 1,000-4,000 output tokens.
The estimate should be task-type-aware. Mispricing output tokens matters most
at T3 where `cost_per_1k_out = $0.075` — a 2,000-token response costs $0.15,
not the $0.015 BALANCER estimates.

**Suggested fix** — parameterize expected output tokens by task type:

```python
TASK_OUTPUT_TOKENS = {
    "search":   50,
    "explain":  300,
    "review":   600,
    "codegen":  1500,
    "refactor": 2000,
    "debug":    800,
    "test":     1200,
    "unknown":  400,
}

def _estimate_cost(self, tier: int, context_tokens: int, task_type: str = "unknown") -> float:
    cfg = TIER_CONFIG[tier]
    out_tokens = TASK_OUTPUT_TOKENS.get(task_type, 400)
    cost_in  = (context_tokens / 1000) * cfg["cost_per_1k_in"]
    cost_out = (out_tokens / 1000)     * cfg["cost_per_1k_out"]
    return round(cost_in + cost_out, 6)
```

---

## 3. Holling Phase Mapping

Holling (1973) defines four phases of the adaptive cycle:

| Phase | Symbol | Character |
|-------|--------|-----------|
| Exploitation | r | Fast growth, opportunistic resource use |
| Conservation | K | Accumulated capital, stable but rigid |
| Release / Crisis | Ω (Omega) | Collapse, rapid release of stored capital |
| Reorganization | α (Alpha) | Chaotic but creative, low connectedness |

### Current tier mapping (lines 210-213 docstring)

The docstring maps BALANCER logic to three phases (Exploit, Disturbance,
Reorganize) but uses non-standard names and collapses K and Ω. The actual tier
behavior maps to Holling as follows:

| Tier | BALANCER behavior | Holling phase | Assessment |
|------|------------------|--------------|------------|
| T0 | Rainbow cache hit, zero LLM calls | r — Exploitation | Correct. Pure opportunistic use of pre-accumulated knowledge. |
| T1 | Haiku, cheap inference | r — late Exploitation / early K | Acceptable. Still harvesting cheaply. |
| T2 | Sonnet, medium inference | K — Conservation | Weak mapping. Conservation in Holling means maintaining a stable, high-capital state — not escalation. The mapping should be Ω (release/crisis): we are releasing token budget reserves under disturbance. |
| T3 | Opus, full inference | Ω + α — Release and Reorganization | The crisis response is correct, but α (reorganization) implies the system should *learn* from this failure and adjust future routing. BALANCER does not feed escalation events back to update its heuristics. |

### The missing α phase

Holling's α phase is the most important and most absent. After an Ω event
(T3 escalation), the system should reorganize: update task-type tier floors,
adjust quality thresholds for specific file patterns, or flag the file for
pre-escalation on future runs. BALANCER treats every run as independent; there
is no memory of past escalations.

Walker et al. (2004) extend this to "transformability" — the ability to
restructure the system when current state is untenable. BALANCER has no
transformability mechanism.

**Minimum α implementation**: log T3 escalation events with file hash and task
type to `StateStore`, and read them back on initialization to pre-warm
`TASK_TIER_FLOOR` for known-hard files.

---

## 4. RQS Gating

### What exists

`rqs_l1_score` is read from payload at line 126 and passed to `_select_tier()`
at line 135. Inside `_select_tier()`, lines 234-239 implement three levels:

```python
if rqs_l1 < 0.70:
    quality_tier = 3   # critically low
elif rqs_l1 < self.quality_floor:   # floor = 0.85 by default
    quality_tier = task_floor + 1
else:
    quality_tier = task_floor
```

This is a **pre-flight quality gate** — it uses RQS from the *compression
pipeline output* (GRUG's estimate of how much quality survived compression),
not from a model response. This is legitimate but semantically different from
the FrugalGPT cascade gate.

### The `auto_escalate` flag is dead code

Lines 146-147:

```python
auto_escalate = rqs_l1 < self.quality_floor and tier < self.tier_max
```

This flag is:
1. Set at line 146.
2. Logged to the decision audit at line 160 (`metadata={"auto_escalate": ...}`).
3. Forwarded in the output payload at line 180.
4. **Never acted upon anywhere in `pipeline.py`.**

`orchestrator/pipeline.py` reads `balancer_out.payload` only to extract
`routing_decision` (line 147 of pipeline.py) and pass chunks to ZIPPY. The
`escalation_policy` dict including `auto_escalate` is never read by any
downstream consumer. The flag documents intent but produces no behavior. This
is the most actionable single bug in the file.

**Fix**: In `pipeline.py`, after BALANCER runs, check `auto_escalate` and
re-invoke the tier selection at `tier + 1` before dispatching to ZIPPY:

```python
balancer_out = self.balancer.run_once(timeout=2.0)
ep = balancer_out.payload.get("escalation_policy", {})
if ep.get("auto_escalate") and ep.get("max_tier", 0) > balancer_out.payload["routing_decision"]["tier"]:
    # Bump tier and re-emit
    balancer_out.payload["routing_decision"]["tier"] += 1
    # re-pick model for new tier
    new_tier = balancer_out.payload["routing_decision"]["tier"]
    balancer_out.payload["routing_decision"]["model_id"] = TIER_CONFIG[new_tier]["model"]
    balancer_out.payload["routing_decision"]["reason"] += " [auto-escalated on RQS]"
```

This is a workaround, not a true cascade, but it at least makes `auto_escalate`
do something.

### Threshold analysis

The 0.70 hard-floor for T3 escalation (line 234) is undocumented. Where does
0.70 come from? The project-wide quality floor is 0.85 (CLAUDE.md). A
critically low RQS of 0.70 means 30% of semantic content is lost in compression
— reasonable for Opus escalation. But the threshold should be named:

```python
CRITICAL_QUALITY_FLOOR = 0.70  # below this, full-source (T3) required
```

---

## 5. `ACTIVE_TIER_MAX=0` Behavior

### Code path

`self.tier_max` is set at line 119:

```python
self.tier_max = int(os.getenv("ACTIVE_TIER_MAX", tier_max))
```

In `_select_tier()`, the cap is applied at line 244:

```python
selected = min(selected, self.tier_max)
```

With `ACTIVE_TIER_MAX=0`, `selected` is always clamped to 0 (T0/LOCAL).

### What happens at T0 with non-rainbow tasks

`_estimate_cost(tier=0, ...)` returns 0.0 (correct). `_pick_model(tier=0)`
returns `None` (line 257-258, correct). The routing decision emits
`{"tier": 0, "model_id": None, ...}`.

**Problem**: `_select_tier()` only returns T0 naturally for rainbow cache hits
on `search` or `explain` tasks (line 217-218). For all other tasks, the initial
`selected` will be >= 1 (e.g., `codegen` has `task_floor = 2`). The `min(...,
0)` clamp then forces T0 regardless. ZIPPY receives `model_id=None` for a
`codegen` task. There is no warning, no error, and no documentation of this
behavior.

**Expected behavior in offline mode**: BALANCER should detect when
`tier_max == 0` and emit a payload flag like `{"offline_mode": True,
"model_id": None, "warning": "ACTIVE_TIER_MAX=0: LLM unavailable"}`. ZIPPY and
the orchestrator can then skip LLM dispatch and return cached/compressed output
only.

**Line-level fix** — add after line 244:

```python
if self.tier_max == 0 and not rainbow_hit:
    return 0, "ACTIVE_TIER_MAX=0 (offline mode) — no LLM available"
```

This makes offline mode explicit rather than a silent clamp artifact.

---

## 6. FeatureSlice Interface Audit

### Conformance

BALANCER uses `self.emit()` (line 165) which calls `self.taxonomy.validate()`
on tags before constructing the FeatureSlice. This is correct per `BaseAgent`
contract (base_agent.py line 200).

`to_dict()` / `from_dict()` round-trip: BALANCER reads `slice_in.payload`
(line 123) and emits via `self.emit(...)`. Payload fields passed through are
all primitive types (strings, floats, lists, dicts). No issues found.

### Issues

**`confidence` is set to `rqs_l1` (lines 192 and 193).** RQS-L1 is a quality
score for the *compression*, not a confidence score for the *routing decision*.
These are different quantities. A high-quality compression (RQS-L1=0.95) of
an ambiguous task does not imply high routing confidence. The routing
confidence should reflect how unambiguous the tier selection was — for example,
high confidence when `task_floor == context_tier == quality_tier`, lower when
they disagree and `max()` was the tiebreaker.

**`decision` string (line 149) is truncated to 100 chars by FeatureSlice
`__init__`** (base_agent.py line 71: `self.decision = decision[:100]`). The
formatted string at line 149-152 is:

```
T2/MID: codegen, 20000ctx, RQS=0.72, TER=81.2%, est=$0.00345
```

This is 62 characters for this example — fits. But for long task type names
and high token counts it could truncate. Low severity.

**`tier_tag = f"routing.tier{tier}"` (line 163)** — these tags
(`routing.tier0` through `routing.tier3`) must be present in `data/taxonomy.json`
or `taxonomy.validate()` will raise. If new tier values are ever added, the
taxonomy file must be updated simultaneously. No validation that `tier` is in
`[0,1,2,3]` before constructing the tag.

**`compression_pct` is never set.** `emit()` is called without `compression_pct`
so FeatureSlice stores `None`. BALANCER does know `overall_ter` from the
incoming payload. This field should be populated:

```python
return self.emit(
    ...
    compression_pct = overall_ter,
)
```

---

## 7. Error Handling Gaps

### Missing input validation

`process()` does no type checking on `payload` keys before using them. For
example:

- **Line 125**: `overall_ter = payload.get("overall_ter", 0.0)` — if GRUG
  emits an error slice (`payload = {"error": "...", "input_slice_id": "..."}`),
  `overall_ter` defaults to 0.0, `rqs_l1` defaults to 1.0, `context_tokens`
  defaults to 0, `rainbow_hit` defaults to `False`. BALANCER will route this
  to T1 with no indication that the upstream agent failed. The error should
  be detected and re-raised or short-circuited.

**Suggested guard** — add at the start of `process()`:

```python
if "error" in payload:
    raise ValueError(f"Upstream error from {slice_in.agent_id}: {payload['error']}")
```

`BaseAgent._error_slice()` will catch this and produce a proper error slice
rather than a spurious routing decision.

- **Line 119**: `self.tier_max = int(os.getenv("ACTIVE_TIER_MAX", tier_max))`
  — if `ACTIVE_TIER_MAX` is set to a non-integer string (e.g. `"disabled"`),
  this raises `ValueError` at construction time with no helpful message.

```python
try:
    self.tier_max = int(os.getenv("ACTIVE_TIER_MAX", tier_max))
except ValueError:
    raise ValueError(
        f"ACTIVE_TIER_MAX must be an integer 0-3, got: {os.getenv('ACTIVE_TIER_MAX')!r}"
    )
```

- **Line 237**: `quality_tier = task_floor + 1` — if `task_floor` is already
  at `self.tier_max`, this produces `task_floor + 1 > tier_max`, which is then
  correctly clamped by line 244. However, `quality_tier` at line 251 is then
  included in the reason string even though it was clamped. The reason message
  could mislead: `"RQS=0.82<floor→T3"` when actual tier is capped at T2.
  Fix: build reason string after applying `min()` clamp.

### No timeout on BALANCER in `run_once`

`pipeline.py` line 131 passes `timeout=2.0`. If BALANCER's `process()` hangs
(unlikely since it's pure Python arithmetic) the pipeline blocks for 2 seconds
before returning `None`. There is no `try/except TimeoutError` around the agent
invocation in `pipeline.py`. The existing `BaseAgent._error_slice()` only
catches exceptions from within `process()`, not bus timeouts. This is a
structural gap in `BaseAgent` — not BALANCER-specific but affects all agents.

---

## 8. Specific Line-Level Issues

| Location | Issue | Severity |
|----------|-------|----------|
| `balancer.py:19` | FrugalGPT citation claims cascade; implementation is pre-flight router | Medium — misleading documentation |
| `balancer.py:54-55` | T1 pricing `$0.00025/$0.00125` is approximate and potentially stale | Medium — cost tracking accuracy |
| `balancer.py:62-64` | T2 Sonnet pricing may not match `claude-sonnet-4-5` current rates | Medium |
| `balancer.py:71-73` | T3 `claude-opus-4-6` — no pricing verification timestamp | Medium |
| `balancer.py:91` | `DEFAULT_TIER_MAX = 2` hardcoded constant; should reference env var default only | Low |
| `balancer.py:119` | `int(os.getenv(...))` can raise `ValueError` on bad env value with no helpful message | Medium |
| `balancer.py:125-130` | No guard for upstream error slice — silently routes errors as T1 tasks | High |
| `balancer.py:146-147` | `auto_escalate` computed but never acted upon anywhere in pipeline | High |
| `balancer.py:163` | `routing.tier{tier}` tags must exist in taxonomy.json — no bounds check on `tier` | Low |
| `balancer.py:192` | `confidence=rqs_l1` conflates compression quality with routing confidence | Low |
| `balancer.py:212-213` | Holling phase comment uses non-standard names; K/Ω phases conflated | Low — documentation |
| `balancer.py:234` | Magic number `0.70` for critical RQS floor is unnamed | Low |
| `balancer.py:244` | Silent clamp to T0 in offline mode — no warning emitted | Medium |
| `balancer.py:251` | Reason string built before `min()` clamp — can cite unclamped tier number | Low |
| `balancer.py:268` | Fixed 200 output tokens in cost estimate regardless of task type | Medium — cost accuracy |
| `pipeline.py:131` | `balancer_out.payload["escalation_policy"]["auto_escalate"]` never read | High — dead signal |

---

## Highest-Value Fixes (Priority Order)

### Fix 1 — Honor `auto_escalate` in the orchestrator (pipeline.py:131-138)

This is the single highest-value fix. The flag is computed, logged, and
forwarded but never read. Until this is wired, RQS-gated escalation is
documentation fiction.

### Fix 2 — Guard upstream error slices (balancer.py:123-130)

Without this, an error from PRUNER or GRUG gets silently routed to a real LLM
tier and billed. Add the `"error" in payload` check as the first line of
`process()`.

### Fix 3 — Task-aware output token estimate in `_estimate_cost()` (balancer.py:268)

The `codegen` and `refactor` tasks have 5-10x higher output token counts than
`search` or `explain`. The current 200-token flat estimate understates T3 cost
by an order of magnitude on generation tasks, which corrupts the cost tracking
signal that downstream monitoring depends on.

---

*End of review.*
