Read the file `agents/balancer.py` in full.
Also read `.env.example` for the full configuration surface.
Also read `orchestrator/pipeline.py` to understand how BALANCER fits in the flow.

You are acting as a research-grounded AI agent auditor. Scrutinize BALANCER — the Resource Balancer / Systems Ecologist — against its claimed research basis.

**Research baseline BALANCER claims to implement:**
- Holling (1973) Adaptive Cycle: exploit cheap resources first, escalate on disturbance (r → K → Ω → α)
- Walker et al. (2004): resilience = capacity to absorb disturbance and reorganize
- FrugalGPT (Chen et al. 2024): cascading small→large models achieves GPT-4 quality at 1/50th cost
- Rao (2023): cost-optimal LLM routing with 60-85% cost reduction

**Your audit must cover:**

1. **Tier Routing Logic** — Read the `_select_tier()` function. Does the routing logic actually implement a cascading strategy (try cheapest first, escalate on failure)? Or does it pick a tier upfront and commit? What does FrugalGPT's cascade look like vs this implementation?

2. **Missing Cascade** — FrugalGPT actually *tries* the cheap model and escalates only if quality is insufficient. BALANCER appears to route upfront without trying cheaper tiers first. Is this correct? What's needed to implement true cascade routing?

3. **Cost Model Accuracy** — Review `_estimate_cost()`. Are the per-token prices accurate for current Anthropic/OpenAI pricing? Update them if stale.

4. **Holling Adaptive Cycle Mapping** — The four phases are: r (growth/exploit), K (conservation), Ω (release/crisis), α (reorganization/recovery). Map each tier (T0-T3) to the appropriate Holling phase. Is the current escalation logic consistent with this mapping?

5. **RQS-L1 Routing Gate** — BALANCER receives `rqs_l1_score` from GRUG. Is it actually used in `_select_tier()`? Should a low RQS-L1 force escalation to T2/T3?

6. **ACTIVE_TIER_MAX** — Is the env cap applied correctly? What happens if a task_type that requires T3 is capped at T0?

7. **Missing features** — What routing signals are currently ignored that could improve tier selection? (e.g. file size, language, rainbow_hit rate, historical quality for this file)

8. **Proposed improvements** — Write the 2-3 highest-value changes as concrete code diffs or pseudocode.

Every unnecessary Opus call costs 100× a Haiku call. This agent is the accountant.
