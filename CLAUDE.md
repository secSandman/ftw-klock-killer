# FTW-KLOC-KILLER — Claude Code Project Instructions

## What This Project Is

A multi-agent token compression system. Four agents (PRUNER, GRUG, BALANCER, ZIPPY)
pipe source code through a compression chain before sending it to an LLM.
The goal: pay less, lose nothing.

Core metric: **TER** (Token Efficiency Ratio) — current baseline **85.4%** (Python synthetic).
Consistency gate: **CCC** (Code Consistency Comparison) — floor **0.85**.
True value = TER × CCC (simulation context). Both must be satisfied simultaneously.

**NOTE:** CCC is a CONSISTENCY metric (token overlap), not a QUALITY metric (answer correctness).
The term "RQS-L1" has been renamed to "CCC" to accurately describe what is measured.
The real-LLM metric `rqs_llm` / `RQS-LLM` from `experiments/local_llm.py` is NOT renamed.

## Project Structure

```
src/                    Core InB pipeline (inference_bridge.py, benchmark.py, quality.py)
agents/                 4-agent system (pruner, grug, balancer, zippy)
bus/                    Local JSONL message bus + decision audit log
languages/              Multi-language parsers (Python, C, Go, Rust)
rag/                    Chunker, hasher, embedder, Qdrant ETL
rainbow/                Pre-computed stdlib hash → compressed form tables
orchestrator/           Pipeline wiring + session state
tests/                  pytest suite covering all layers
.claude/commands/       AI-powered review and scaffolding slash commands
```

## Critical Constraints

### The PI Formula — DO NOT BREAK
`predictability_index()` in `src/inference_bridge.py`:
- Loop-containing functions (for/while/try/except/yield) with **>20 body lines**
  get PI = 0.70 - 0.06 = **0.64 < 0.70 threshold** → NOT skeletonized → skeleton output bloats
- This is the root cause of failed benchmarks. Always check this before adding
  functions to synthetic test data.
- Safe: non-loop functions up to 35 lines (PI = 0.70, exactly at threshold)
- List comprehensions `[x for x in ...]` on a single line do NOT count as loops

### Benchmark Baseline
- **79.3% TER** is the accepted baseline (default config, `PI_THRESHOLD=0.70`, synthetic corpus)
- Do not regress below this without explicit user approval
- Override via `KLOC_BASELINE_TER` env var — this is the single source of truth (defined in `src/benchmark.py`)
- 85.4% was an experimental result at `PI_THRESHOLD=0.85`, not the default baseline

### Message Bus Channels (fixed order)
```
pruner.in → pruner.out → grug.out → balancer.out → zippy.out
```

### Tier Routing
- T0: local/rainbow cache (0 cost)
- T1: Haiku (cheap)
- T2: Sonnet (mid)
- T3: Opus (full — crisis only)
- `ACTIVE_TIER_MAX=0` = fully offline (test/CI mode)

## Coding Standards

- All agents extend `BaseAgent` in `agents/base_agent.py`
- All inter-agent messages use `FeatureSlice` with `to_dict()` / `from_dict()`
- All decisions logged via `DecisionTracker.log(agent_id, decision_type, decision_value, rationale, confidence)`
- All taxonomy tags validated against `data/taxonomy.json`
- Windows-compatible: no bare `import fcntl` (already guarded in message_bus.py)
- `ACTIVE_TIER_MAX` read from env; never hardcode tier ceiling in logic

## Running Things

```bash
# Master CLI — one entry point for everything
python kloc.py setup                              # build tables + dep check
python kloc.py benchmark --mode synthetic         # TER + RQS combined
python kloc.py benchmark --file path/to/game.c   # benchmark any file
python kloc.py ter  --file game.c                 # token count only
python kloc.py rqs  --file game.c                 # quality only
python kloc.py pipeline --file game.c --question "explain render loop"
python kloc.py build-metadata --repo doom/        # build function metadata index
python kloc.py build-metadata --repo doom/ --summaries  # + T1 Haiku summaries
python kloc.py test                               # full pytest suite
python kloc.py test --filter "skeleton or caveman"

# Direct (legacy)
python src/benchmark.py --mode synthetic
pytest tests/ -v
ACTIVE_TIER_MAX=0 pytest tests/test_pipeline.py -v
```

## Experiments

### Overview

All experiment infrastructure lives in `experiments/`. Results are persisted to
`experiments/results/YYYY-MM-DD/results.jsonl`. Every algorithm constant that matters
is injectable via environment variable — subprocess isolation guarantees clean reads
between runs. 81 experiments run to date.

```bash
# Run a sweep
python experiments/sweep.py --param KLOC_PI_THRESHOLD 0.65 0.70 0.75 0.80
python experiments/sweep.py --grid experiments/grids/pi_quick.json
python kloc.py experiment sweep --grid pi_quick.json --workers 4

# Coordinate-descent optimizer (TV reward signal)
python experiments/optimize.py --param pi --rounds 3
python kloc.py experiment optimize --param pi

# Bayesian TPE optimizer (Optuna — joint param search)
python experiments/bayesian.py --param pi+tfidf --trials 50
python kloc.py experiment bayesian --param pi --trials 30

# Leaderboard
python experiments/leaderboard.py --top 20 --diff
python kloc.py experiment leaderboard --diff
```

---

### Current Theories

#### Theory 1 — PI Threshold Is the Primary TER Lever
The Predictability Index threshold (`KLOC_PI_THRESHOLD`, default 0.70) is a
binary gate: above it a function body is replaced with `...`, below it the full
body is kept. Raising the threshold → fewer functions skeletonized → lower TER,
higher RQS. Lowering it → more skeletonized → higher TER, lower RQS.
The TV optimum sits where marginal TER gain equals marginal RQS loss — empirically
at threshold ≈ **0.75** for the Python synthetic corpus.

#### Theory 2 — TER × RQS Anti-Correlation Is Structural, Not Tunable
True Value = TER × RQS. In the simulation benchmark, RQS is measured as TF cosine
similarity between original source and compressed source. Compression removes tokens
→ TF overlap falls. No parameter tuning can escape this: any config that raises TER
will lower the TF-based RQS, and vice versa. TV converges to a plateau near **0.528**
regardless of which PI knob is adjusted. Breaking this ceiling requires:
- A real LLM in the loop (T1/T2 call), measuring response quality directly
- A corpus where rare domain terms survive compression (e.g., signature-only code)
- A different quality metric that measures semantic preservation, not token overlap

#### Theory 3 — TF-IDF Gives a Stricter, More Accurate RQS
Raw TF cosine is inflated by high-frequency boilerplate (`self`, `return`, `logger`)
that appears identically in both original and compressed. TF-IDF down-weights these
common terms and up-weights rare domain identifiers (`pw_hash`, `A_Chase`, `record_id`).
The effect: TF-IDF RQS = **0.579** vs raw TF RQS = **0.650** for the same compression.
TV drops from 0.516 to 0.459 — not a regression, but a correction. The old metric was
overestimating quality by ~12%. Enable: `KLOC_USE_TFIDF_RQS=1`.

#### Theory 4 — Skeleton Signature Retention Composes With TF-IDF
If TF-IDF penalizes the absence of rare terms, and those terms live in function bodies
that get skeletonized, then appending the rarest body identifiers as a comment on the
`...` line partially restores TF-IDF similarity at a token cost.
```python
# Before:  ...
# After:   ...  # pw_hash token_expiry record_id last_login_at
```
In the simulation this is net-negative (token cost > RQS gain) because the comment
adds each term only once while the original body had it multiple times. With a real
LLM, the model sees the identifiers and produces responses that reference them, which
is the intended benefit. Enable: `KLOC_SKELETON_SIG_RETAIN=1`.

#### Theory 5 — C Corpus Has a Different TV Optimum
The C PI threshold (`KLOC_CPI_THRESHOLD`, default 0.70) is too conservative for
Doom/Quake-style game code. Game functions tend to be long, complex, loop-heavy —
the default threshold skeletonizes very few of them. Sweeping CPI_THRESHOLD lower
reveals the TV optimum is near **0.35–0.40**: TER jumps from 36% to 54%, RQS falls
from 0.870 to 0.731, and TV rises from 0.313 to 0.398 (+27%).

---

### Testing Framework

#### Tools

| Tool | File | Purpose |
|------|------|---------|
| `sweep.py` | `experiments/sweep.py` | Full grid — all permutations of a param list, parallel `ThreadPoolExecutor` |
| `optimize.py` | `experiments/optimize.py` | Coordinate descent — sweep one param at a time, fix best, repeat |
| `bayesian.py` | `experiments/bayesian.py` | Optuna TPE — joint search, escapes coordinate-descent local optima |
| `runner.py` | `experiments/runner.py` | Single-experiment executor — subprocess isolation, JSONL persistence |
| `leaderboard.py` | `experiments/leaderboard.py` | Ranked table from all JSONL, `--diff` vs baseline |

#### Param Spaces (env-var injectable)

| Group | Key params | File |
|-------|-----------|------|
| `pi` | `KLOC_PI_THRESHOLD`, `KLOC_PI_BOILERPLATE_BONUS`, `KLOC_PI_LOOP_FREE_BONUS`, `KLOC_PI_LOOP_PENALTY`, `KLOC_PI_SIMPLE_RATIO_THRESHOLD` | `experiments/params/pi_formula.json` |
| `c_pi` | `KLOC_CPI_THRESHOLD`, `KLOC_CPI_BOILERPLATE_BONUS`, `KLOC_CPI_LOOP_FREE_BONUS`, `KLOC_CPM_KQ_THRESHOLD` | `experiments/params/c_pi.json` |
| `retrieval` | `KLOC_RAG_NAME_WEIGHT`, `KLOC_RAG_SYMBOL_BOOST`, `KLOC_RAG_LARGE_FN_LINES`, `KLOC_RAG_CHAIN_EXPAND_K` | `experiments/params/retrieval.json` |
| `f4` | `KLOC_F4_VOWEL_PRUNE_MIN_LEN` | `experiments/params/caveman_f4.json` |
| `tfidf` | `KLOC_USE_TFIDF_RQS`, `KLOC_TFIDF_IDF_FLOOR` | — |
| `sig` | `KLOC_SKELETON_SIG_RETAIN`, `KLOC_SKELETON_SIG_MAX_IDS` | — |
| `rqs_version` | `KLOC_RQS_VERSION=v1\|v2` | `v1` = CSO (legacy, circular); `v2` = ROUGE-L/Haiku (non-circular) |

#### Grid files in `experiments/grids/`

| File | Combos | Purpose |
|------|--------|---------|
| `pi_quick.json` | 10 | Fast threshold × bonus sweep |
| `pi_full.json` | 60 | Full PI parameter grid |
| `c_ter_boost.json` | 48 | C PI + pattern miner combos |
| `retrieval_weights.json` | 48 | RAG field weight combos |

#### Subprocess Isolation Design
Each experiment runs as a child process:
```python
subprocess.run([sys.executable, "kloc.py", "benchmark", "--mode", "synthetic"],
               env={**os.environ, **env_overrides}, ...)
```
Module-level constants (e.g. `SKELETAL_PI_THRESHOLD`) are re-read from env at import
time in each subprocess. This guarantees no state bleed between parallel workers and
makes results exactly reproducible from the saved `env` dict in JSONL.

---

### Current Results

#### Confirmed Leaderboard (2026-04-07, 248 experiments)

**Python synthetic corpus** — baseline TER=79.3%, CCC=0.650, TV=0.516

| Rank | Config | TER | CCC | TV | TV×T | ΔTV |
|------|--------|-----|-----|----|------|-----|
| 1 (max TV) | `PI=0.75, F4=8` | 60.5% | ~0.875 | **~0.529** | — | +2.5% |
| 1 (max TV×T) | `PI=0.70, F4=4` | 79.3% | 0.639 | 0.507 | **0.577** | −1.7% |
| — | Baseline (default PI=0.70) | 79.3% | 0.650 | 0.516 | — | — |

**C corpus (doom/src/strife/p_enemy.c)** — baseline TER=36.0%, CCC=0.870, TV=0.313

| Rank | Config | TER | CCC | TV | ΔTV |
|------|--------|-----|-----|----|-----|
| **1 (Bayesian 2026-04-07)** | `CPI=0.35, BPLATE=0.15, LF=0.10, KQ=3, F4=8` | 54.4% | 0.753 | **0.410** | +31.0% |
| 2 | `CPI=0.35, F4=6` (grid) | 54.4% | 0.736 | 0.400 | +27.8% |
| — | Baseline (CPI=0.70) | 36.0% | 0.870 | 0.313 | — |

**Recommended configs:**
```
# Python — max TV
KLOC_PI_THRESHOLD=0.75
KLOC_F4_VOWEL_PRUNE_MIN_LEN=8

# C — Bayesian-confirmed best (2026-04-07)
KLOC_CPI_THRESHOLD=0.35
KLOC_CPI_BOILERPLATE_BONUS=0.15
KLOC_CPI_LOOP_FREE_BONUS=0.10
KLOC_CPM_KQ_THRESHOLD=3
KLOC_F4_VOWEL_PRUNE_MIN_LEN=8
```

#### Key Lessons from 248 Experiments

- `KLOC_F4_VOWEL_PRUNE_MIN_LEN=8` was found by Bayesian search for both Python and C independently. It is universally better. The default of 5 is suboptimal.
- `KLOC_PI_LOOP_FREE_BONUS` has zero effect on the Python synthetic corpus (dead param — corpus has no loop-free long functions). Do not tune it for Python.
- TV×TIME (`--sort tv_time` in leaderboard) favours PI=0.70 over PI=0.75 because it is 10–15% faster at near-identical quality.
- The TV ceiling of ~0.529 is structural (TF cosine anti-correlation with TER). 248 experiments confirmed it. Breaking it requires real LLM calls.

#### The TV Plateau
248 total experiments (grid sweep, coordinate descent, Bayesian TPE) all converge to
TV≈0.529. This is a structural ceiling for the simulation benchmark — confirmed, not
a local optimum. The next meaningful improvement requires real LLM calls or a semantic
quality metric that isn't anti-correlated with token removal.

**Next step:** `python kloc.py experiment bayesian --param all --trials 100
--storage sqlite:///experiments/optuna.db` with `ACTIVE_TIER_MAX=1` (real Haiku).

---

## Slash Commands Available

| Command | Purpose |
|---------|---------|
| `/review-pruner` | AI critique of PRUNER agent vs HTM/SDR research |
| `/review-grug` | AI critique of GRUG agent vs Zipf/MDL research |
| `/review-balancer` | AI critique of BALANCER agent vs FrugalGPT/Holling research |
| `/review-zippy` | AI critique of ZIPPY agent vs Kolmogorov/LZW research |
| `/review-pipeline` | End-to-end pipeline architectural review |
| `/review-inb` | Core InferenceBridge F1-F4 critique |
| `/add-agent` | Scaffold a new agent from the BaseAgent template |
| `/benchmark-ter` | Run TER benchmark and get optimization suggestions |
| `/optimize-retrieval` | Compare BM25 vs metadata retrieval for a query, diagnose failures |
| `/analyze-chunks` | Inspect chunk size distribution, flag giant/tiny/unnamed chunks |
| `/build-metadata` | Build function metadata index for a repo (field-weighted RAG) |
| `/quality-check` | Run RQS quality check and identify regression sources |
| `/diff-agents` | Compare two agent implementations side-by-side |
