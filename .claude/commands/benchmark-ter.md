Run the synthetic benchmark by executing:
```
python src/benchmark.py --mode synthetic
```

Capture the full output. Then read `src/benchmark.py` and `src/inference_bridge.py` in full.

**Current baseline:** 77.9% TER. Target: 80%+. Gap: ~2.1%.

**Your task:**

1. **Parse the benchmark output** — Extract per-file TER numbers. Which file has the most room for improvement?

2. **Token math** — Calculate: how many additional tokens need to be eliminated to hit 80%? Show the arithmetic.

3. **Identify skeletonization misses** — For each synthetic file, find function bodies where:
   - PI ≥ 0.70 but body is somehow NOT being skeletonized (bug)
   - PI < 0.70 but could reach ≥ 0.70 with minor restructuring (opportunity)

4. **PI formula quick check** — Re-run `predictability_index()` mentally on the 5 most token-heavy function bodies. Does the score match your expectation?

5. **F3 masking check** — How many import lines in the corpus are NOT being masked? (Any import line with entropy < 4.0 should be masked.)

6. **Optimization proposals** — List concrete, safe changes ordered by expected TER gain:
   - Code changes that don't touch the PI formula threshold
   - Changes that expand what can be skeletonized
   - Changes that expand what can be masked

7. **RQS impact estimate** — For each proposed change, estimate the semantic similarity impact (will it break RQS-L1 ≥ 0.85?).

8. **Write the fix** — Implement the single highest-impact change that doesn't risk RQS regression. Show the diff.

Remember: loop-containing functions with >20 body lines WILL NOT skeletonize (PI < 0.70). Do not add loops to functions trying to increase body size.
