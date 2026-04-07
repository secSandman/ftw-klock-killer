Read the file `agents/pruner.py` in full.
Also read `agents/base_agent.py` and `src/inference_bridge.py` lines 115-202 (the `predictability_index` and `hot_zone_score` functions).

You are acting as a research-grounded AI agent auditor. Your task is to deeply scrutinize the PRUNER agent against the research it claims to be grounded in, then produce actionable improvement suggestions.

**Research baseline PRUNER claims to implement:**
- Hawkins & Ahmad (2016) Sparse Distributed Representations (SDR): column overlap scoring, proximal vs distal dendrite activation
- HTM cortical learning: sparsity target ~2% active columns, winner-take-all inhibition
- Mapping: PI score ≈ column overlap; HZS score ≈ proximal (recent/predicted) vs distal (context) activation

**Your audit must cover:**

1. **SDR Fidelity** — Does the current FOREGROUND/BACKGROUND classification actually implement SDR overlap scoring? What's missing? (e.g. inhibition radius, boosting, permanence decay)

2. **Sparsity Target** — `SDR_SPARSITY_TARGET=0.20` — real HTM uses ~2% sparsity. Is 20% too permissive? What does the code actually enforce vs just name?

3. **PI → Column Overlap mapping** — Is the PI formula a reasonable proxy for column overlap? Where does it diverge from the biological model?

4. **HZS → Dendrite mapping** — Does hot_zone_score's exponential decay actually model proximal vs distal dendrite activation? What's the biological justification for lambda=0.1?

5. **Missing HTM features** — List up to 5 HTM features that would meaningfully improve compression quality if implemented (with implementation sketch for each).

6. **Code quality issues** — Any bugs, edge cases, or performance problems.

7. **Proposed improvements** — Write the 2-3 highest-value changes as concrete code diffs or pseudocode.

Be brutal. This agent will run on Doom source code. If it's wrong, we waste tokens.
