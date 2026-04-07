# FTW-KLOC-KILLER — Experiments

A living document covering the optimization theories, testing infrastructure, and
benchmark results for the token compression pipeline. Updated as experiments run.

---

## Background

The pipeline compresses source code before sending it to an LLM. Two metrics govern
every decision:

- **TER** (Token Efficiency Ratio) — fraction of tokens removed. Higher = cheaper.
- **RQS** (Response Quality Score) — how similar the LLM response to compressed code
  is compared to the response to the full code. Measured as TF cosine similarity. Higher = better.
- **True Value (TV) = TER × RQS** — the primary optimization target. Both dimensions
  must be satisfied simultaneously; maximizing one at the expense of the other is a loss.

Current simulation baseline (Python synthetic corpus):

| Metric | Value |
|--------|-------|
| TER | 79.3% |
| RQS | 0.650 |
| **TV** | **0.516** |

---

## Theories

### Theory 1 — The PI Threshold Is the Primary TER Lever

The Predictability Index (`PI`) scores each function body from 0 to 1.
If `PI >= threshold`, the entire body is replaced with `...` (skeletonized).
The threshold (`KLOC_PI_THRESHOLD`, default 0.70) is a binary gate:

```
Raise threshold → fewer functions skeletonized → lower TER, higher RQS
Lower threshold → more functions skeletonized → higher TER, lower RQS
```

The PI formula is score-based, not entropy-based. Base score is assigned by line
count (short functions score high), then penalised for complexity constructs
(loops, try/except, yield) and rewarded for boilerplate patterns (assignments,
returns, guard clauses).

**Critical constraint:** a loop-containing function with >20 body lines gets
`PI = 0.70 - 0.06 = 0.64`, which falls below the default threshold and is never
skeletonized. This is intentional — loop-heavy bodies are not predictable from
their signature alone.

**Finding:** the TV optimum sits near threshold = **0.75** for the Python corpus.
At that point, fewer functions are skeletonized, RQS rises from 0.650 → 0.874,
and TV rises from 0.516 → 0.528.

---

### Theory 2 — TER × RQS Anti-Correlation Is Structural, Not Tunable

In the simulation benchmark, RQS is TF cosine similarity between the original
source and the compressed source. This creates a hard anti-correlation:

```
More compression  →  fewer shared tokens  →  lower TF similarity  →  lower RQS
Less compression  →  more shared tokens   →  higher TF similarity  →  higher RQS
```

No combination of PI parameters can escape this. Every config that pushes TER up
pulls RQS down by a roughly proportional amount. Coordinate descent, full grid
search, and Bayesian TPE (20 trials across all PI dimensions) all converge to the
same ceiling: **TV ≈ 0.528**.

This is a property of the simulation, not a flaw in the pipeline. To break past
0.528 requires one of:

1. A real LLM in the loop — measuring actual response quality instead of TF overlap
2. A corpus where rare domain terms survive into the compressed output
3. A semantic quality metric that rewards structure preservation over token overlap

---

### Theory 3 — Raw TF Cosine Overestimates Quality by ~12%

Raw TF cosine treats `self`, `return`, and `logger` the same as `pw_hash`,
`A_Chase`, or `token_expiry`. High-frequency boilerplate appears in both original
and compressed (it's rarely removed), inflating the similarity score.

TF-IDF corrects this. Weight for each term:

```
idf(w) = log( total_unique_terms / (1 + freq_w) )
```

High-frequency terms get low IDF weight. Rare domain terms get high weight.
The result is a stricter metric dominated by domain-specific identifiers — the
terms that actually differentiate one codebase from another.

| Metric | Raw TF | TF-IDF |
|--------|--------|--------|
| RQS | 0.650 | 0.579 |
| TV | 0.516 | **0.459** |

TF-IDF TV = 0.459 is not a regression — it is the more accurate number. The
pipeline was reporting 12% higher quality than it was actually delivering.

Enable: `KLOC_USE_TFIDF_RQS=1`

---

### Theory 4 — Skeleton Signature Retention Composes With TF-IDF

If TF-IDF penalises absent rare terms, and those terms live in function bodies
that get skeletonized, then preserving a few of those terms in the `...` line
should partially restore TF-IDF similarity.

Implementation: extract the rarest identifiers from the body (lowest frequency
= most domain-specific) and append them as a comment:

```python
# Before
...

# After (KLOC_SKELETON_SIG_RETAIN=1, KLOC_SKELETON_SIG_MAX_IDS=15)
...  # pw_hash token_expiry record_id last_login_at ip_address
```

Same pattern implemented for C:
```c
/* Before */
/* ...skeletonized 28L PI=0.68... */

/* After */
/* ...28L PI=0.68 entity_type movecount threshold strafecount... */
```

**Simulation result:** net-negative. Each identifier appears once in the comment
but many times in the original body, so TF-IDF similarity only partially recovers.
The token overhead outweighs the similarity gain.

**Production hypothesis:** with a real LLM call, the model sees the identifier list
and uses it to reason about the function's domain. This is expected to meaningfully
improve response quality even at the cost of added tokens. Not yet measured.

Enable: `KLOC_SKELETON_SIG_RETAIN=1`

---

### Theory 5 — C Game Code Has a Different TV Optimum Than Python

The default C PI threshold (0.70) was calibrated for Python-style code. Doom/Quake
game functions tend to be long, imperative, and loop-heavy — the default threshold
is too conservative, leaving most bodies intact.

Sweeping `KLOC_CPI_THRESHOLD` from 0.20 to 0.80 on `p_enemy.c` reveals:

```
CPI_THRESHOLD=0.70  →  TER=36.0%,  RQS=0.870,  TV=0.313  (default)
CPI_THRESHOLD=0.40  →  TER=54.4%,  RQS=0.731,  TV=0.398  (+27%)
CPI_THRESHOLD=0.35  →  TER=54.4%,  RQS=0.731,  TV=0.398  (same)
CPI_THRESHOLD=0.20  →  TER=64.3%,  RQS=0.609,  TV=0.392  (RQS drops too fast)
```

The TV optimum is near **0.35–0.40** for this corpus. Below that, RQS falls faster
than TER rises. The finding generalises: language-specific PI thresholds should be
tuned per corpus type, not shared across languages.

---

## Testing Framework

### Architecture

Every experiment runs as an isolated subprocess:

```python
subprocess.run(
    [sys.executable, "kloc.py", "benchmark", "--mode", "synthetic"],
    env = {**os.environ, **env_overrides},
    ...
)
```

Module-level constants like `SKELETAL_PI_THRESHOLD` are re-read from env at import
time in each subprocess. This guarantees:
- No state bleed between parallel workers
- Results are exactly reproducible from the saved `env` dict in JSONL
- Any parameter can be overridden without modifying source code

Results are appended to `experiments/results/YYYY-MM-DD/results.jsonl`.
Each record contains `experiment_id`, `timestamp`, `label`, `env`, `metrics`, `stdout`.

---

### Tools

#### `experiments/sweep.py` — Parallel Grid Search

Generates all permutations of a parameter grid and runs them in parallel.

```bash
# Single-param sweep
python experiments/sweep.py --param KLOC_PI_THRESHOLD 0.65 0.70 0.75 0.80

# Multi-param grid (3×3 = 9 experiments)
python experiments/sweep.py \
    --param KLOC_PI_THRESHOLD 0.65 0.70 0.75 \
    --param KLOC_PI_BOILERPLATE_BONUS 0.10 0.15 0.20

# From a JSON grid file
python experiments/sweep.py --grid experiments/grids/pi_quick.json

# C file corpus
python experiments/sweep.py \
    --corpus doom/src/strife/p_enemy.c --repo doom/ \
    --param KLOC_CPI_THRESHOLD 0.40 0.50 0.60 0.70

# Control parallelism
python experiments/sweep.py --workers 8 --param ...
```

Use when: exploring a new parameter for the first time, or when you want
a complete map of the parameter space rather than the optimum.

---

#### `experiments/optimize.py` — Coordinate Descent

Sweeps one parameter at a time, fixes the best value, moves to the next.
Repeats for N rounds or until improvement < 0.2%.

```bash
python experiments/optimize.py --param pi --rounds 3
python experiments/optimize.py --param c_pi --corpus doom/src/strife/p_enemy.c --repo doom/
python experiments/optimize.py --param retrieval --fine  # refine around best value
```

Complexity: O(params × values) vs O(values^params) for full grid.
Seeded from best past result in JSONL history — does not re-run known configs.

Use when: you have a large parameter space and want a fast approximate optimum.

---

#### `experiments/bayesian.py` — Bayesian TPE Optimizer (Optuna)

Tree-structured Parzen Estimators: builds density models over "good" and "bad"
configs, samples from the ratio. Explores joint interactions that coordinate
descent misses (e.g. threshold × bonus interactions).

```bash
# 50-trial PI + TF-IDF joint search
python experiments/bayesian.py --param pi+tfidf --trials 50

# Persist study across sessions (resume after interruption)
python experiments/bayesian.py \
    --param all --trials 100 \
    --storage sqlite:///experiments/optuna.db

# Available param groups
# pi         PI formula knobs only
# f4         Caveman vowel-prune knob
# tfidf      TF-IDF floor only
# sig        Signature retention max-ids
# all        All of the above jointly
# pi+tfidf   Most useful joint group
# pi+sig     PI + retention (for real-LLM eval)
```

Also available as: `python kloc.py experiment bayesian --param pi+tfidf --trials 50`

Use when: coordinate descent has converged and you suspect joint interactions,
or when you want to run a long overnight search with persistence.

---

#### `experiments/runner.py` — Single Experiment

Runs one configuration, captures metrics, saves to JSONL.

```bash
python experiments/runner.py \
    --env KLOC_PI_THRESHOLD=0.75 \
    --env KLOC_PI_BOILERPLATE_BONUS=0.20 \
    --label "manual test"
```

Also usable as a library:
```python
from experiments.runner import run_experiment
result = run_experiment({"KLOC_PI_THRESHOLD": "0.75"})
print(result["metrics"])  # {'python_ter_pct': 0.605, 'rqs_l1': 0.874, 'true_value': 0.528}
```

---

#### `experiments/leaderboard.py` — Results Viewer

```bash
python experiments/leaderboard.py                     # all results
python experiments/leaderboard.py --top 20 --diff     # top 20 with delta vs baseline
python experiments/leaderboard.py --date 2026-04-06   # one day
python experiments/leaderboard.py --param PI_THRESHOLD  # filter by param name
```

---

### Parameter Reference

All constants are injectable via environment variable. Source files read them at
import time with a fallback default.

#### Python PI (`experiments/params/pi_formula.json`)

| Env var | Default | Range | Effect |
|---------|---------|-------|--------|
| `KLOC_PI_THRESHOLD` | 0.70 | 0.50–0.95 | Gate: above = skeletonize body |
| `KLOC_PI_BOILERPLATE_BONUS` | 0.15 | 0.05–0.30 | +PI for simple-assignment-heavy bodies |
| `KLOC_PI_LOOP_FREE_BONUS` | 0.10 | 0.03–0.20 | +PI for long bodies with zero loops |
| `KLOC_PI_LOOP_PENALTY` | 0.06 | 0.02–0.15 | −PI per loop/try/yield construct |
| `KLOC_PI_SIMPLE_RATIO_THRESHOLD` | 0.50 | 0.30–0.70 | Min ratio of simple lines to earn bonus |

#### C PI (`experiments/params/c_pi.json`)

| Env var | Default | Range | Effect |
|---------|---------|-------|--------|
| `KLOC_CPI_THRESHOLD` | 0.70 | 0.20–0.90 | Gate for C function bodies |
| `KLOC_CPI_BOILERPLATE_BONUS` | 0.15 | 0.05–0.30 | +PI for C assignment cascades |
| `KLOC_CPI_LOOP_FREE_BONUS` | 0.10 | 0.03–0.20 | +PI for long C bodies with no loops |
| `KLOC_CPM_KQ_THRESHOLD` | 3 | 1–6 | Min occurrences for a pattern to enter the dict |

#### RAG Retrieval (`experiments/params/retrieval.json`)

| Env var | Default | Effect |
|---------|---------|--------|
| `KLOC_RAG_NAME_WEIGHT` | 5.0 | Multiplier for exact function name match in BM25 |
| `KLOC_RAG_SYMBOL_BOOST` | 4.0 | Boost for exact symbol hit in query |
| `KLOC_RAG_LARGE_FN_LINES` | 100 | Functions above this get a size penalty |
| `KLOC_RAG_CHAIN_EXPAND_K` | 2 | How many call-chain hops to expand after top-K |

#### Experimental Flags

| Env var | Default | Effect |
|---------|---------|--------|
| `KLOC_USE_TFIDF_RQS` | 0 | Use TF-IDF weighted cosine for RQS (stricter, more accurate) |
| `KLOC_TFIDF_IDF_FLOOR` | 0.01 | Minimum IDF weight (prevents zero-weighting common terms) |
| `KLOC_SKELETON_SIG_RETAIN` | 0 | Append rare body identifiers to `...` comment |
| `KLOC_SKELETON_SIG_MAX_IDS` | 15 | Max identifiers to append per skeleton line |

---

### Grid Files

Pre-built grids in `experiments/grids/`:

| File | Combinations | Purpose |
|------|-------------|---------|
| `pi_quick.json` | 10 | Threshold × bonus — fast sanity check |
| `pi_full.json` | 60 | All 5 PI knobs at coarse resolution |
| `c_ter_boost.json` | 48 | CPI threshold × miner KQ threshold |
| `retrieval_weights.json` | 48 | RAG name weight × symbol boost |

---

## Current Results

### All Experiments (81 total, 2026-04-06)

Run `python experiments/leaderboard.py --diff` to see the live table.

### Python Synthetic Corpus

3 boilerplate-heavy CRUD service files. 6,136 original tokens.

| Rank | Config | TER | RQS | TV | ΔTV |
|------|--------|-----|-----|----|-----|
| 1 | `PI_THRESHOLD=0.75` | 60.5% | 0.874 | **0.528** | +2.3% |
| — | Baseline (default) | 79.3% | 0.650 | 0.516 | — |
| — | `PI_THRESHOLD=0.85` | 85.4% | 0.504 | 0.430 | −16.7% |
| — | `PI_THRESHOLD=0.60` | 79.3% | 0.650 | 0.516 | 0.0% |

The plateau at TV=0.528 held across all 81 experiments. No PI configuration
or Bayesian search found a way past it.

### C Corpus — `doom/src/strife/p_enemy.c`

11,886-token game AI file. Brace-counting skeleton parser, no libclang.

| Rank | Config | TER | RQS | TV | ΔTV |
|------|--------|-----|-----|----|-----|
| 1 | `CPI_THRESHOLD=0.40` | 54.4% | 0.731 | **0.398** | +27.2% |
| 2 | `CPI_THRESHOLD=0.50` | 46.1% | 0.797 | 0.367 | +17.3% |
| 3 | `CPI_THRESHOLD=0.55` | 42.0% | 0.833 | 0.350 | +11.8% |
| — | Baseline (0.70) | 36.0% | 0.870 | 0.313 | — |
| — | `CPI_THRESHOLD=0.20` | 64.3% | 0.609 | 0.392 | +25.2% |

**Recommendation:** set `KLOC_CPI_THRESHOLD=0.40` as the new C default.

### Experimental Strategy Results

Tested on Python synthetic corpus against baseline TV=0.516.

| Strategy | Env | TER | RQS | TV | Notes |
|----------|-----|-----|-----|----|-------|
| TF-IDF only | `KLOC_USE_TFIDF_RQS=1` | 79.3% | 0.579 | 0.459 | More accurate, not a regression |
| Sig retain (15 ids) | `KLOC_SKELETON_SIG_RETAIN=1` | 70.5% | 0.651 | 0.459 | TER overhead dominates |
| Sig retain (8 ids) | same, `MAX_IDS=8` | 74.3% | 0.617 | 0.458 | Still net negative |
| TF-IDF + sig (8 ids) | both | 74.3% | 0.617 | 0.458 | Simulation can't capture LLM benefit |

All experimental strategies are net-negative in the simulation benchmark. They
are designed for use with real LLM calls (see next steps).

---

## Known Ceiling and Next Steps

### Why TV Is Stuck at 0.528

The simulation benchmark measures RQS as TF cosine between original source and
compressed source. This is a proxy — it does not require an actual LLM call.
The proxy has a hard ceiling: compressing more tokens always lowers TF overlap.

Coordinate descent, grid sweep (60 configs), and Bayesian TPE (20 trials) all
converge to TV=0.528. This is not a local optimum — it is the global maximum
of TER × TF-cosine(original, compressed) given the current compression strategy.

### Paths to Breaking 0.528

**Option 1: Real LLM evaluation**
Run `ACTIVE_TIER_MAX=1` to enable Haiku (T1) calls. RQS is then measured as
similarity between two actual model responses. Signature retention and TF-IDF
can both be re-evaluated under this condition.

```bash
ACTIVE_TIER_MAX=1 python experiments/bayesian.py \
    --param pi+sig --trials 100 \
    --storage sqlite:///experiments/optuna.db
```

**Option 2: Redesign compression to preserve semantic units**
Instead of replacing entire function bodies with `...`, preserve:
- The first and last statement of each body (establish + return value)
- Lines containing the function's rarest identifiers
- Lines adjacent to hot-zone (recently changed) lines

This is a compression architecture change, not a parameter change.

**Option 3: Smarter RQS metric**
Replace TF cosine with embedding cosine (`sentence-transformers/all-MiniLM-L6-v2`
is already wired in `rag/embedder.py`). Embedding similarity is not anti-correlated
with compression the same way TF overlap is — a function skeleton can be
semantically close to its full body even with very few shared tokens.

```bash
# Already available — just needs wiring into quality.py
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
python kloc.py benchmark --mode synthetic
```

---

## Reproducing Results

```bash
# Install deps
pip install optuna

# Run the full PI sweep that produced the TV=0.528 result
python experiments/sweep.py --grid experiments/grids/pi_full.json --workers 4

# Run the C CPI sweep
python experiments/sweep.py \
    --param KLOC_CPI_THRESHOLD 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
    --corpus doom/src/strife/p_enemy.c --repo doom/ \
    --workers 4

# Run Bayesian joint search (persistent)
python experiments/bayesian.py \
    --param pi+tfidf --trials 50 \
    --storage sqlite:///experiments/optuna.db

# View all results
python experiments/leaderboard.py --top 30 --diff
```
