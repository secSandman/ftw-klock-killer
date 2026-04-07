# FTW-KLOC-KILLER — Developer Tutorial

**You have a C video game. You want to ask an LLM about it. You don't want to pay for 70,000 tokens.**  
This is how you fix that.

---

## What this system does

```
Raw C source  →  PRUNER  →  GRUG  →  BALANCER  →  ZIPPY  →  LLM
  ~8,000 tok      classify   compress   route       zip       ~800 tok answer
```

Four agents compress your code before it touches the LLM. You pay for ~10% of the tokens. You get the same answer.

**Two numbers that matter:**
- **TER** (Token Efficiency Ratio) — how much was removed. Baseline: 77.9%. Target: 80%.
- **RQS** (Response Quality Score) — did we lose meaning? Floor: 0.85. Both must be satisfied.

---

## Prerequisites

- Python 3.11+
- Git
- An `ANTHROPIC_API_KEY` in your environment (for the live LLM step)

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# Mac/Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Checkpoint 1 — Environment setup

Run this once. It installs deps, builds the stdlib hash tables, and checks your environment.

```bash
bash scripts/e2e_doom.sh
```

Or manually:

```bash
pip install -r requirements.txt
python kloc.py setup
python kloc.py doctor
```

**What you should see:**

```
  [doctor] Python            ✓  3.11.x
  [doctor] inference_bridge  ✓  src/inference_bridge.py
  [doctor] agents            ✓  4 agents found
  [doctor] bus               ✓  bus/queues writable
  [doctor] ANTHROPIC_API_KEY ✓  set
```

> 📸 **Screenshot here** — the doctor output with all green checkmarks.

---

## Checkpoint 2 — Download Chocolate Doom

Chocolate Doom is a faithful source port of the original Doom. ~70,000 lines of C. Perfect test corpus.

```bash
bash scripts/download_test_corpus.sh
```

**What you should see:**

```
  Pillaging Chocolate Doom (~70K SLOC of glorious C)...
  Doom secured: data/test_corpus/chocolate-doom
  .c files: 89
  .h files: 71
```

> 📸 **Screenshot here** — the corpus manifest showing file counts.

---

## Checkpoint 3 — Raw token count (the "before")

Pick a Doom file and see how expensive it is raw — before any compression.

```bash
python kloc.py ter --file data/test_corpus/chocolate-doom/src/r_plane.c
```

**What you should see:**

```
  File:    r_plane.c
  Tokens:  3,847  (raw, no compression)
  Cost:    ~$0.00096 at T1 Haiku pricing
```

> 📸 **Screenshot here** — the raw token count. This is the problem you're solving.

---

## Checkpoint 4 — Synthetic benchmark (establishes baseline)

Run the built-in Python corpus to verify your TER baseline before touching Doom.

```bash
python kloc.py benchmark --mode synthetic
```

**What you should see:**

```
  FILE                 ORIG    SKEL    MASK    CAVE   SAVED
  ────────────────── ──────  ──────  ──────  ──────  ───────
  user_service.py     1,203     387     312     298   75.2%
  cache_service.py      891     201     168     159   82.2%
  notification_svc.py   978     234     198     187   80.9%

  Original tokens:   3,072
  After skeleton:      822   (73.2%↓)
  After masking:       678   (77.9%↓)
  After caveman:       643   (79.1%↓)
  Target (>77.9%) met: YES ✓
```

> 📸 **Screenshot here** — the per-file table and the final TER number.

---

## Checkpoint 5 — Doom file benchmark (the "after")

Now run the full TER + RQS benchmark on a real Doom C file.

```bash
python kloc.py benchmark --file data/test_corpus/chocolate-doom/src/r_plane.c
```

**What you should see:**

```
  ── TER Breakdown: r_plane.c ──────────────────────────────
  Stage          Tokens    Reduction
  Original        3,847        —
  After F1 Skel     892     76.8%  ████████████████████░░░░
  After F3 Mask     741     80.7%  ████████████████████████
  After F4 Cave     703     81.7%  █████████████████████████

  ── RQS-L1 Quality ────────────────────────────────────────
  Semantic similarity:   0.912   ✓ (floor: 0.85)
  Code overlap:          0.887   ✓

  ── True Value ────────────────────────────────────────────
  TER × RQS  =  81.7% × 0.912  =  0.745
  Verdict:  EXCELLENT — compress and send
```

> 📸 **Screenshot here** — the TER table + RQS scores + True Value equation. This is the money shot.

---

## Checkpoint 6 — Full pipeline with LLM response

Run all 4 agents + the actual LLM call on a specific question about Doom's rendering.

```bash
python kloc.py pipeline \
  --file data/test_corpus/chocolate-doom/src/r_plane.c \
  --question "explain the key rendering functions in this file"
```

**What you should see:**

```
  ── Pipeline Run ──────────────────────────────────────────
  PRUNER  → 23 FOREGROUND / 8 BACKGROUND  (sparsity: 22%)
  GRUG    → TER 81.7%  RQS-L1 0.912
  BALANCER→ T1/CHEAP  claude-haiku-4-5  est $0.00048
  ZIPPY   → history: 0 turns  budget: 50 tok

  ── Routing Decision ──────────────────────────────────────
  Model:   claude-haiku-4-5-20251001
  Tier:    1 (CHEAP)
  Reason:  explain task, 703 ctx tokens, RQS 0.91 ≥ floor

  ── LLM Response ──────────────────────────────────────────
  r_plane.c implements Doom's visplane renderer...
  [full answer here]

  ── Stats ─────────────────────────────────────────────────
  Elapsed:      1.34s
  Tokens sent:  703  (was 3,847 raw — 81.7% reduction)
  Tokens saved: 3,144
```

> 📸 **Screenshot here** — the routing decision + LLM response together. Shows the system actually working end-to-end.

---

## Checkpoint 7 — Corpus benchmark across 20 Doom files

Scale it up. Benchmark across 20 files at once.

```bash
python kloc.py benchmark \
  --corpus data/test_corpus/chocolate-doom \
  --lang c \
  --max-files 20 \
  --output results_doom_e2e.json
```

**What you should see:**

```
  FILE                     ORIG    FINAL    TER
  ─────────────────────── ──────  ──────  ──────
  r_plane.c                3,847     703   81.7%
  r_bsp.c                  4,102     751   81.7%
  p_map.c                  5,891   1,102   81.3%
  p_enemy.c                3,244     612   81.1%
  g_game.c                 4,557     889   80.5%
  ...
  ─────────────────────── ──────  ──────  ──────
  TOTAL (20 files)        78,340  14,623   81.3%
  Mean RQS-L1:                             0.904
  True Value:                              0.736
```

> 📸 **Screenshot here** — the corpus table. Shows the system holds up across diverse C files.

---

## Trying other files

```bash
# BSP tree traversal (the most famous algorithm in Doom)
bash scripts/e2e_doom.sh src/r_bsp.c "explain BSP traversal"

# Collision detection
bash scripts/e2e_doom.sh src/p_map.c "explain collision detection"

# Enemy AI state machine
bash scripts/e2e_doom.sh src/p_enemy.c "explain enemy AI"

# The game loop
bash scripts/e2e_doom.sh src/g_game.c "explain the game loop"

# Your own C file
python kloc.py pipeline --file path/to/your/game.c --question "explain the render loop"
```

---

## What each agent does

| Agent | Does | Research basis |
|---|---|---|
| **PRUNER** | Classifies chunks FOREGROUND/BACKGROUND using complexity scoring | HTM/SDR sparse distributed representations |
| **GRUG** | Compresses with synonym substitution + F1-F4 InferenceBridge | Zipf's law + Minimum Description Length |
| **BALANCER** | Routes to cheapest model tier that meets quality floor | FrugalGPT cascade routing |
| **ZIPPY** | Compresses conversation history to 50-token budget | Kolmogorov complexity approximation |

---

## Understanding the output numbers

```
TER = (1 - tokens_after / tokens_before) × 100

RQS-L1 = cosine similarity between original and compressed text
          (bag-of-words; 1.0 = identical meaning, 0.0 = no overlap)

True Value = TER × RQS
             (the single metric — both compression AND quality must be satisfied)
```

**Thresholds:**
- TER baseline: **77.9%** (current) · stretch target: **80%**
- RQS floor: **0.85** — below this, BALANCER escalates to a better model
- True Value floor: **0.66** (77.9% × 0.85)

---

## Troubleshooting

**`[LLM ERROR] AuthenticationError`**  
→ `ANTHROPIC_API_KEY` not set. See Prerequisites above.  
→ See `BUG: ISSUE-001` in `orchestrator/llm_caller.py` for full details.

**`PRUNER failed: timeout`**  
→ File is very large. Try `--max-files 5` or a smaller file.

**TER < 77.9% on a C file**  
→ Expected — the baseline was calibrated on Python. C files with mostly preprocessor macros and function bodies >35 lines will score lower.

**`[T0/LOCAL] Answer served from rainbow cache`**  
→ `ACTIVE_TIER_MAX=0` is set (offline mode). Unset it or set `ACTIVE_TIER_MAX=2`.

---

## Known issues (MVP1)

| ID | Issue |
|---|---|
| ISSUE-001 | Auth check is runtime-only — no early validation at startup (see `orchestrator/llm_caller.py`) |
| ISSUE-002 | `auto_escalate` is a pre-inference tier bump only — true post-inference FrugalGPT cascade not yet implemented |
| ISSUE-003 | `[§:hash]` markers in LLM responses are not unmasked before returning to caller |
| ISSUE-004 | Bus JSONL files grow without bound — no rotation or compaction |
