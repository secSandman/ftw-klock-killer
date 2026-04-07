Read the file `agents/zippy.py` in full.
Also read `orchestrator/state_store.py`.
Also read `src/inference_bridge.py` lines 520-633 (the CavemanCompressor class).

You are acting as a research-grounded AI agent auditor. Scrutinize ZIPPY — the Optimization Builder / History Compressor — against its claimed research.

**Research baseline ZIPPY claims to implement:**
- Kolmogorov complexity (1965): shortest program that generates a string = its true information content
- Lempel-Ziv LZ77 (1977): dictionary-based compression — repeated sequences stored as back-references
- Goodman (2001): n-gram compression — prune low-probability (low-information) tokens from history
- LLMLingua (Wu et al. 2024): 20x prompt compression with <5% quality loss

**Your audit must cover:**

1. **50-Token Budget** — Is ≤50 tokens for zipped history the right floor? What does LLMLingua's empirical data say about minimum viable context for different task types (explain vs codegen vs refactor)?

2. **_compress_history() implementation** — Walk through the four steps: substitute facts → prune low-info turns → CavemanCompressor → truncate. Is the order optimal? What happens when a turn is factually dense but short? Would it survive the density<0.15 filter?

3. **_extract_facts() implementation** — It finds tokens appearing ≥3 times with ≥6 chars and replaces them with a short key. Is this a valid approximation of LZ77 dictionary building? What's the minimum viable dictionary size for typical coding conversations?

4. **KR Ratio metric** — Is `tokens_after/tokens_before` the right metric for compression quality? Shouldn't it be normalized against information loss (e.g. combined with RQS-L1)?

5. **Kolmogorov Gap** — True Kolmogorov compression is uncomputable. What computable approximation is ZIPPY actually using? Is there a better approximation (e.g. using the LLM itself to score token importance)?

6. **State Store** — Is per-session JSON state the right persistence model? What happens if two pipeline runs share a session_id? Are there race conditions?

7. **Turn pruning heuristic** — Density<0.15 and words<10 as low-info signals: are these thresholds empirically justified? What does the LLMLingua paper use instead?

8. **Proposed improvements** — Write the 2-3 highest-value changes as concrete code diffs or pseudocode.

History is the #1 token sink. If ZIPPY slips, the bill explodes.
