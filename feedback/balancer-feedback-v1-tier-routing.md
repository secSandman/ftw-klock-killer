# BALANCER Agent Feedback — v1 · Tier Routing

**Auditor:** Claude Sonnet 4.6  
**Date:** 2026-04-06  
**File reviewed:** `agents/balancer.py`, `orchestrator/pipeline.py`  
**Review skill:** `.claude/commands/review-balancer.md`

---

## 1. Tier Routing — Not a Cascade

The current `_select_tier()` computes three independent pressure signals and takes `max()` upfront:

```python
selected = max(task_floor, context_tier, quality_tier)
selected = min(selected, self.tier_max)
```

This is a **static routing policy**, not a cascade.

| Pattern | Behavior | Research basis |
|---|---|---|
| Static max-pick (current) | Computes pressures, picks tier once, never retries | Routing heuristic |
| Cascade (FrugalGPT) | Calls T1, evaluates output, escalates to T2 only on failure | Trial-and-escalate |

The docstring labels the three phases as "Exploit → Disturbance check → Reorganize" implying an iterative loop. There is no loop. There is no model call, no output evaluation, no retry.

**Risk:** Over-routing is silent. A `codegen` task with 4,500 tokens and RQS 0.91 routes to T2 because `task_floor=2` dominates, even if T1 would have succeeded.

---

## 2. FrugalGPT Cascade — Non-Compliant

FrugalGPT (Chen et al. 2024, arXiv:2305.05176) achieves 1/50th cost via:
1. Submit to cheapest model
2. Evaluate output quality via a learned estimator
3. If quality passes → return; else re-submit to next tier
4. Repeat until quality passes or final tier exhausted

BALANCER does none of steps 1–4. The `auto_escalate` flag is computed and emitted but **never acted on**:

```python
auto_escalate = rqs_l1 < self.quality_floor and tier < self.tier_max
```

This goes into `escalation_policy` in the payload. Nothing in `pipeline.py` reads `auto_escalate` and re-routes. It is a dead code path.

**Rao (2023) specifically shows** that *post-inference quality gating* delivers 60–85% cost reduction. Pre-inference tier estimation, however accurate, cannot produce those savings because it cannot observe whether the cheap model succeeded.

**BALANCER is a tier predictor, not a tier cascade.** The research citations are aspirationally accurate but mechanistically misleading.

---

## 3. Cost Model — Stale Prices

```python
# T1: claude-haiku-4-5-20251001
"cost_per_1k_in":  0.00025,    # ← likely 3.2× too low
"cost_per_1k_out": 0.00125,    # ← likely 3.2× too low

# T3 alt_model — same as T2
"alt_model": "gpt-4o"          # T2 AND T3 both specify gpt-4o — no effective T3 escalation on OpenAI path
```

As of Q1 2026, Claude 3.5 Haiku is $0.00080/$0.00400 per 1K tokens. The T1 prices in `TIER_CONFIG` are systematically low, biasing routing toward T1.

T2 prices (`$0.003/$0.015`) match Claude 3.5 Sonnet and are plausible. T3 prices (`$0.015/$0.075`) match Claude 3 Opus — if `claude-opus-4-6` is a 4.x model, actual pricing could be 2× higher.

**Output estimate is too coarse:**
```python
cost_out = (200 / 1000) * cfg["cost_per_1k_out"]   # hardcoded 200 tokens
```
`refactor` / `codegen` tasks regularly produce 1,000–2,000 tokens. This underestimates output cost 5–10× for complex tasks.

---

## 4. Holling Adaptive Cycle — Metaphor, Not Implementation

The intended T0–T3 → r/K/Ω/α mapping:

| Tier | Holling Phase |
|---|---|
| T0 | r (exploit free resources) |
| T1 | r→K |
| T2 | K (conservation, accepting cost) |
| T3 | Ω→α (crisis, reorganize) |

**What the code implements:** Three static `if/elif` blocks. No feedback loop. No mechanism to "reorganize" back to T0 after resolving a crisis. Holling's cycle is inherently cyclical; BALANCER's routing is one-directional and stateless across requests.

**The K→Ω gap:** track cumulative cost across a session; when cost exceeds a budget threshold, force a lower tier or pause. That feedback is completely absent.

---

## 5. RQS-L1 Gate — Two Critical Bugs

**Bug 1 — Fail-open default:**

```python
rqs_l1 = payload.get("rqs_l1_score", 1.0)   # default = perfect quality
```

If GRUG fails to compute or emit `rqs_l1_score`, BALANCER treats quality as perfect and never escalates on quality grounds. A chunk with severe semantic degradation silently routes to T1.

**Fix:** Use a conservative default and log a warning:
```python
rqs_l1_raw = payload.get("rqs_l1_score")
if rqs_l1_raw is None:
    rqs_l1 = 0.82   # conservative mid-value
    self.log_decision("WARNING", "rqs_l1_score absent from GRUG payload", ...)
else:
    rqs_l1 = float(rqs_l1_raw)
```

**Bug 2 — Cap silently discards distress signals:**

When `ACTIVE_TIER_MAX` clamps `selected` below the quality-distress tier, no warning is emitted. Operator sees the clamped tier as if it were the correct choice. `auto_escalate` also evaluates to `False` when `tier == tier_max` — the flag inverts its meaning when the ceiling prevents escalation.

**Fix:** Log cap override explicitly:
```python
uncapped = max(task_floor, context_tier, quality_tier)
selected = min(uncapped, self.tier_max)
if uncapped > self.tier_max:
    reason += f" | CAP_OVERRIDE: wanted T{uncapped}, capped at T{self.tier_max}"
# auto_escalate = "should we escalate if we could" (not blocked by cap)
auto_escalate = rqs_l1 < self.quality_floor
escalate_blocked = auto_escalate and (selected == self.tier_max)
```

---

## 6. ACTIVE_TIER_MAX=0 — Dangerous Failure Mode

Scenario: `ACTIVE_TIER_MAX=0` (offline mode). `refactor` task arrives with `rqs_l1=0.65` (critically low). `_select_tier()` computes `selected=3`, clamps to `min(3,0)=0`. The routing decision emits T0/LOCAL with `model_id=None`.

Problems:
1. No warning to operator
2. `auto_escalate = False` (because `0 < 0` is False) — flag inverts meaning
3. ZIPPY receives `model_id=None` for a crisis task with no guard in `pipeline.py`

---

## 7. Five Missing Routing Signals

**a) `overall_ter` ignored.** Low TER (little compression) → dense uncompressed input → needs more capable model. High TER + low RQS → needs better model to handle aggressively compressed context. Neither is captured.

**b) Task-signal interactions not multiplicative.** `debug` + `large file` + `low RQS` is T3 territory but individual signals all max at T2. Signals are each evaluated independently and max'd — interaction effects missed.

**c) `language` ignored.** Languages with high-complexity parsing failure modes (Rust macros, C preprocessor) could usefully bias toward T2/T3.

**d) No session-level cost accumulation.** 20 consecutive T2 calls with no mechanism to notice cumulative spend exceeds a threshold. This is the K→Ω Holling transition that is claimed but unimplemented.

**e) Rainbow hit scope too narrow.** T0 shortcut only fires for `search` and `explain`. A `debug` or `review` task on a file where all functions are stdlib boilerplate should have `rainbow_hit` reduce `context_tier` by one step.

---

## 8. Proposed Improvements (Prioritized)

### Fix 1 — Cap-Override Warning (zero runtime cost, high operator value)
See Bug 2 fix above. ~15 lines. Eliminates silent quality degradation under `ACTIVE_TIER_MAX`.

### Fix 2 — Safe RQS Default + TER Signal
See Bug 1 fix above. Closes fail-open. Add:

```python
# TER pressure in _select_tier():
if overall_ter < 30.0 and context_tokens > 8_000:
    ter_tier = 2   # low compression on large context = raw dense input
elif overall_ter > 75.0 and rqs_l1 < self.quality_floor:
    ter_tier = min(quality_tier + 1, 3)
else:
    ter_tier = task_floor
selected = max(task_floor, context_tier, quality_tier, ter_tier)
```

### Fix 3 — Post-Inference Cascade Loop (delivers actual FrugalGPT savings)

Add retry loop in `pipeline.py` after ZIPPY output:

```python
for attempt in range(MAX_CASCADE_ATTEMPTS):
    # ... run ZIPPY + LLM call ...
    response_rqs = zippy_out.payload.get("response_rqs_l1")
    if response_rqs is None or response_rqs >= quality_floor:
        break
    next_tier = min(current_tier + 1, self.balancer.tier_max)
    if next_tier == current_tier:
        break
    # Re-route at higher tier without re-running GRUG/PRUNER
    balancer_out.payload["routing_decision"]["tier"] = next_tier
```

This is the only change that delivers the claimed 60–85% cost reductions from FrugalGPT. Pre-routing heuristics alone cannot produce those savings.

---

## Summary

| Issue | Severity |
|---|---|
| `auto_escalate` flag computed, logged, never acted on | High |
| Fail-open RQS default (1.0) silently routes degraded output to T1 | High |
| FrugalGPT cascade entirely absent — is a static max-pick | High |
| T1 prices likely 3× too low (biases routing toward T1) | Medium |
| T3 `alt_model` duplicates T2 — no effective T3 on OpenAI path | Medium |
| Cap override silently discards quality-distress signals | Medium |
| Holling cycle metaphor unimplemented (no session-level feedback) | Low |
| TER, language, rainbow scope, session budget all ignored as routing signals | Medium |
