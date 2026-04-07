Read the file `src/inference_bridge.py` in full.
Also read `src/benchmark.py` in full.
Also read `src/quality.py` lines 1-230.

You are acting as a token compression research engineer. Your job is to find every percentage point of TER improvement that doesn't break RQS.

**Current baseline:** 77.9% TER on synthetic corpus. Target: 80%+.
**Quality floor:** RQS-L1 ≥ 0.85 (semantic similarity between full vs bridged responses).

**The PI formula (CRITICAL — do not break this):**
- n≤2: base=0.95, n≤5: 0.90, n≤10: 0.85, n≤20: 0.80, n≤35: 0.70
- Penalty: -0.06 per loop construct (for/while/try/except/yield/async for/async with)
- Bonus: +0.15 if ≥50% of lines are simple (return/self.x=/raise/pass/.../super()/logger.)
- Threshold: 0.70 — below this, body is NOT skeletonized
- Loop + >20 lines = PI ≈ 0.64 = NOT skeletonized (bodies kept verbatim → token bloat)

**Your audit must cover:**

1. **TER Gap Analysis** — We're at 77.9% and need 80%. How many tokens separate us? Read `_FILE_CONFIG`, `_FILE_POOL`, `_FILE_ROUTER` in benchmark.py. For each synthetic file, calculate current vs theoretical max TER if all skeletonable functions were skeletonized.

2. **F1 Skeleton Opportunities** — Which function bodies in the synthetic corpus are currently NOT skeletonized that could be? (Check against the PI formula — any PI ≥ 0.70 is fair game.)

3. **F3 Masking Coverage** — Check what fraction of import lines in the synthetic corpus are currently masked. Are there non-import boilerplate patterns (e.g. `logger = logging.getLogger(__name__)`) that could be added to the masking rules?

4. **F4 Caveman Gaps** — Are there common Python comment patterns that Caveman misses? (e.g. type annotations in comments, `# type: ignore`, `# noqa:`)

5. **PI Formula Calibration** — Is the current threshold (0.70) correctly calibrated? Run through 5 concrete function examples and check whether the PI score matches your intuition about predictability.

6. **RHD Bijection** — Verify that the mask/unmask roundtrip is lossless for all 3 synthetic files. If any hash collision is possible, document the reproduction case.

7. **benchmark.py synthetic corpus** — Is the 3-file synthetic corpus representative of real codebases? What classes of real Python code are NOT represented? (e.g. async/await, dataclasses, type hints, decorators)

8. **quality.py integration gap** — RQS-L1 is computed in GRUG but not in the benchmark. Write the minimal code to wire RQS-L1 measurement into benchmark.py's synthetic mode output.

9. **Proposed improvements** — Rank the top 3 changes that would push TER from 77.9% to 80%+ without breaking RQS-L1 ≥ 0.85. Write the actual code changes needed.
