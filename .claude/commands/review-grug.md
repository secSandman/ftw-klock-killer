Read the file `agents/grug.py` in full.
Also read `src/inference_bridge.py` (full file — all four F1-F4 stages).
Also read `src/quality.py` lines 1-170 (the semantic similarity and RQS-L1 functions).

You are acting as a research-grounded AI agent auditor. Scrutinize GRUG — the Semantic Compressor — against the science it claims.

**Research baseline GRUG claims to implement:**
- Zipf's Law (1935): word frequency ∝ 1/rank — shorter words carry more information per character
- MDL principle (Rissanen 1978): best model = shortest description without information loss
- Brown et al. (1992): word clustering — synonyms share distributional context
- LLMLingua (Wu et al. 2024): 20x compression with <5% quality loss via token importance scoring

**Your audit must cover:**

1. **Synonym Table Fidelity** — Read the `_SYNONYM_TABLE` in grug.py. Are these substitutions semantically valid? Compute the average character reduction per substitution. Are there high-frequency Python/C identifiers missing from the table?

2. **Zipf Alignment** — The claim is that shorter synonyms are "mathematically equivalent." Is the current substitution scheme actually Zipf-optimal? (i.e. does it replace the longest words first? Or does it just pick common dev jargon?)

3. **MDL Implementation** — Does GRUG's compression pipeline minimize description length? Specifically: after F4 Caveman runs, is there still redundancy that a true MDL coder would eliminate?

4. **F1-F4 Pipeline Order** — Is running F1 skeleton → F3 mask → F4 caveman the optimal order? What would happen if you ran F4 before F3? Would it affect RQS-L1?

5. **RQS-L1 Inline Check** — GRUG calls `compute_semantic_similarity()` inline. Does it use it to gate decisions, or just log it? Should a low RQS-L1 cause GRUG to skip certain compressions?

6. **BACKGROUND chunk handling** — BACKGROUND chunks only get F4 (Caveman). Is this the right call? What does the research say about skeletonizing vs caveman-only for low-PI code?

7. **Missing LLMLingua features** — LLMLingua uses perplexity-based token importance. What would it take to add a lightweight importance scorer to GRUG?

8. **Proposed improvements** — Write the 2-3 highest-value changes as concrete code diffs or pseudocode.

Me GRUG. Me want better numbers.
