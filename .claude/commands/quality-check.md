Read `src/quality.py` in full.
Read `agents/grug.py` lines where RQS-L1 is computed (search for `compute_semantic_similarity`).
Read `tests/test_quality.py` in full.

**Quality framework:**
- L1: Semantic similarity (cosine TF-IDF, inline in GRUG) — target ≥ 0.85
- L2: Functional correctness (run_code_against_tests) — not yet wired to pipeline
- L3: LLM-as-judge — not yet wired to pipeline
- L4: Regression tracker — QualityRegressionTracker exists but not integrated

**Your task:**

1. **Current wiring audit** — Which RQS layers are actually hooked into the pipeline today?
   - Check `orchestrator/pipeline.py` for RQS usage
   - Check `agents/grug.py` for RQS-L1 usage
   - Check `agents/balancer.py` for RQS gating on tier escalation

2. **L1 gap analysis** — The inline `compute_semantic_similarity()` uses TF-IDF cosine. How does this compare to sentence-transformer embeddings? When would it produce false positives (high similarity for semantically different responses)?

3. **L2 wiring plan** — Write the minimal code to wire `run_code_against_tests()` into the pipeline for `task_type="codegen"` tasks. Where should it run: in BALANCER (before routing) or as a post-LLM quality gate?

4. **L3 trigger conditions** — The LLM judge (L3) is expensive. Define the exact conditions when it should be invoked automatically vs skipped. Write the logic as a Python function `should_run_judge(result: QualityResult) -> bool`.

5. **L4 regression tracker** — Write the code to integrate `QualityRegressionTracker` into `orchestrator/pipeline.py` so every run appends to the longitudinal log.

6. **Composite RQS formula** — Currently `composite_rqs` is set manually. Write a `compute_composite_rqs(l1, l2, l3) -> float` function that handles partial availability (e.g. L2=-1 when not run, L3=-1 when skipped).

7. **Quality regression alert** — If composite RQS drops below 0.85 on a file that previously scored ≥ 0.90, what should happen? Write the alert/escalation logic.

8. **Test coverage gaps** — Review `tests/test_quality.py`. What scenarios are not tested? Write 3-5 additional test cases.

9. **Proposed integration** — Write the complete diff to wire all 4 layers into the pipeline in a backwards-compatible way (L2/L3 optional, L4 always-on).
