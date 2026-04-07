The user wants to compare two agent implementations. Their request:
$ARGUMENTS

Read both agent files specified by the user (or infer from context which two agents to compare).

**Your task:**

1. **Interface comparison** — For each agent, extract:
   - IN_CHANNEL / OUT_CHANNEL
   - Taxonomy tags emitted
   - Fields read from input FeatureSlice payload
   - Fields written to output FeatureSlice payload
   - Timeout used in pipeline.run()

2. **Research grounding comparison** — Side-by-side: what research paper drives each agent? Are the implementations faithful to their claimed research basis?

3. **Compression contribution** — For each agent, estimate what percentage of total TER reduction it contributes. Which is doing the most work? Which is potentially redundant?

4. **Overlap analysis** — Do any two agents perform overlapping work? (e.g. if GRUG runs Caveman AND PRUNER pre-classifies, is there double compression on some paths?)

5. **Bus payload diff** — List every key that appears in one agent's output payload but not the other. Are there keys that downstream agents expect but might not always be present?

6. **Error handling comparison** — How does each agent handle: empty input, malformed FeatureSlice, InferenceBridge unavailable, timeout exceeded?

7. **Proposed unification** — If there is meaningful overlap, propose a refactor. If they are cleanly separated, confirm that separation and document the contract between them.
