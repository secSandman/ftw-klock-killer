# FTW-KLOC-KILLER — Feature Roadmap

> Last updated: 2026-04-06
> Owner: txsan
> Core metric: **TER × RQS ≥ 0.85** (True Value) at every layer

---

## Current State (Baseline)

| Language | File | TER | RQS-L1 | True Value | Status |
|----------|------|-----|--------|------------|--------|
| Python   | synthetic corpus | 77.9% | — | — | ✅ passing, below 80% target |
| C/C++    | doom/p_enemy.c | 35.0% | 0.862 | 0.3017 | ✅ algorithms verified |
| Go       | — | — | — | — | ⬜ parser built, no benchmark |
| Rust     | — | — | — | — | ⬜ parser built, no benchmark |

**Known regressions / open bugs:**
- ISSUE-002: Post-inference RQS cascade not implemented (quality checked pre-send only)
- ISSUE-003: `[§:hash]` placeholders not unmasked in LLM responses
- ISSUE-004: Bus JSONL rotation/compaction not implemented (files grow unbounded)

---

## Savings Staircase — Theoretical Ceiling

Each layer compounds multiplicatively on what remains:

```
Layer                   Technique                        Status        Expected cost reduction (cumulative)
─────────────────────────────────────────────────────────────────────────────────────────────────────────
Original                —                                —             0%
F1 Skeleton             Predictable fn bodies removed    ✅ Live        49% (C), 79% (Python) token reduction
F3 Include Masking      KQ #include/#define → [§:hash]   ✅ Live        negligible additional on C
F4 Caveman              // and /* */ comment compression  ✅ Live        54% cumulative (C)
T0.5 Local LLM          Ollama answers query at $0.00    ✅ Live*       cost = $0 when accepted (~X% queries)
RAG Retrieval           Send only relevant chunks        ⚠ Planned     ~80% cumulative token reduction
Context Caching         Provider prefix cache            ⚠ Planned     ~92% cumulative
Semantic Compression    LLMLingua-style rewrite          ⚠ Planned     ~95% cumulative
─────────────────────────────────────────────────────────────────────────────────────────────────────────
Theoretical floor on p_enemy.c: ~836 tokens from 17,886 original

* T0.5 is wired and runs. The COMBINED metric (acceptance rate × $0  +  rejection rate × compressed
  cost) is NOT yet measured. See PROTO-001 below.
```

---

## Feature Backlog — Prioritised

### PROTOTYPE STATUS — What is wired vs what is not yet measured

> **This is a prototype.** The individual layers (compression, local LLM quality) are implemented
> and measured separately. The combined end-to-end production metric does not exist yet.

| Layer | Implemented | Measured | Notes |
|-------|-------------|----------|-------|
| F1+F3+F4 compression | ✅ | ✅ TER=54.4% on p_enemy.c | Works in production |
| T0.5 local LLM routing | ✅ | ⚠ partially | Quality measured; acceptance rate not tracked over batch |
| **Blended cost metric** | ❌ | ❌ | **THIS IS THE GOAL — see PROTO-001** |
| Compression × T0.5 combined | ❌ | ❌ | We measure each in isolation, not together |

**The gap (PROTO-001):** We currently measure:
- Compression quality in isolation (TER=54.4%, RQS-LLM=0.796)
- Local LLM answer quality in isolation

We do NOT yet measure the **cumulative production pipeline**:
```
blended_cost    = acceptance_rate × $0.00
                + (1 - acceptance_rate) × compressed_token_cost
blended_quality = acceptance_rate × local_llm_quality
                + (1 - acceptance_rate) × anthropic_quality
```
This requires running a batch of real queries end-to-end and tracking which tier answered each one.

---

### P0 — Fix existing regressions (must do before new features)

| ID | Feature | Success Criteria | Expected Result |
|----|---------|-----------------|-----------------|
| BUG-001 | Python benchmark regression check | `python kloc.py benchmark --mode synthetic` returns TER ≥ 77.9% | Confirm no regressions from C additions |
| BUG-002 | Unmask `[§:hash]` in LLM responses (ISSUE-003) | LLM answer contains readable imports, not `[§:INC_a1b2c3d4]` | Zero masked tokens visible to end user |
| BUG-003 | Bus JSONL rotation (ISSUE-004) | Queue files capped at 10k lines, old entries rotated to `.archive` | No unbounded disk growth in production |

---

### P1 — Close the 80% Python TER gap

| ID | Feature | Success Criteria | Expected Result |
|----|---------|-----------------|-----------------|
| TER-001 | Expand `user_service.py` non-loop functions by ~7 lines each | `benchmark --mode synthetic` returns TER ≥ 80.0% | 80%+ TER, True Value > 0.68 |
| TER-002 | C TER target: reach 40% on real Doom code | `report-c --file doom/src/strife/p_enemy.c` TER ≥ 40% | Requires better skeleton detection on K&R-style functions |
| TER-003 | F3 masking: token-aware pattern expansion | Masking never increases token count (no negative savings) | Already fixed; add regression test |

---

### P2 — RAG Pipeline wired end-to-end

**What it is:** Instead of sending a whole file, embed all chunks into Qdrant, retrieve only the top-K chunks relevant to the user's question, send those.

**Why it matters:** RAG alone can cut 70-90% of tokens because most of a file is irrelevant to any given question.

| ID | Feature | Success Criteria | Expected Result |
|----|---------|-----------------|-----------------|
| RAG-001 | Wire RAG retrieval into `kloc.py pipeline` command | Pipeline sends ≤5 chunks instead of whole file | ~70% additional reduction post-F1/F3/F4 |
| RAG-002 | C ETL verified end-to-end (mine → chunk → compress → embed → upsert) | `kloc.py etl-c --repo doom` completes with no errors, chunks queryable | Doom codebase searchable by semantic query |
| RAG-003 | Top-K retrieval quality gate | Retrieved chunks answer question at RQS-L1 ≥ 0.85 vs whole-file baseline | Confirms retrieval doesn't lose critical context |
| RAG-004 | Report: show "with RAG" column in toggle comparison | Report shows tokens with and without RAG layer | Visual proof of compound savings |

---

### P3 — Provider Context Caching

**What it is:** Anthropic and OpenAI both support prompt caching — if the same prefix (system prompt + file header) is sent repeatedly, subsequent calls cost 10% of normal input price.

**Why it matters:** In a dev workflow, the same file header is sent dozens of times per session. Structuring prompts to maximise cache hits = ~90% reduction on cached portions.

| ID | Feature | Success Criteria | Expected Result |
|----|---------|-----------------|-----------------|
| CACHE-001 | Detect repeated prompt prefixes across session | `SessionState` tracks hash of last N system prompts | Identify cache hit opportunities |
| CACHE-002 | Structure prompts so static prefix comes first | Prompt builder puts system prompt + file header before dynamic question | Anthropic cache hit rate ≥ 80% in simulated session |
| CACHE-003 | Cache hit savings shown in report | Report shows "estimated cache savings" based on session replay rate | ~60% additional savings on cached prefix at T2 Sonnet |

---

### P4 — Semantic Compression (LLMLingua-style)

**What it is:** Use a small cheap model (T1 Haiku) to intelligently rewrite/compress the prompt before sending to the expensive model (T2 Sonnet / T3 Opus). The cheap model understands meaning; our rule-based compression doesn't.

**Why it matters:** Rule-based compression hits a ceiling (we can't remove semantically important code). A learned compressor can go further.

**Risk:** Quality degradation. The small model may misunderstand and drop important context. Must be gated strictly by RQS.

| ID | Feature | Success Criteria | Expected Result |
|----|---------|-----------------|-----------------|
| SEM-001 | Haiku-based prompt rewriter agent | New agent rewrites chunk using structured prompt: "Compress this code for an LLM. Keep all logic, remove verbosity." | 20-40% additional TER on top of F1+F3+F4 |
| SEM-002 | Quality gate on semantic compression output | RQS-L1(original, rewritten) ≥ 0.85 before allowing | Rejects bad compressions automatically |
| SEM-003 | Cost model: verify Haiku compression cost < Sonnet savings | cost(Haiku rewrite) < (orig_tokens - compressed_tokens) × Sonnet_price | Net positive ROI at ≥500 req/day |

---

### P5 — Multi-Language Parity

| ID | Feature | Success Criteria | Expected Result |
|----|---------|-----------------|-----------------|
| LANG-001 | Go benchmark: TER on real Go repo | `report-c`-equivalent for Go on `golang/go` stdlib | Baseline TER established |
| LANG-002 | Rust benchmark: TER on real Rust repo | Rust report on `rust-lang/rust` or `tokio-rs/tokio` | Baseline TER established |
| LANG-003 | TypeScript/JS support | Parser + compressor for `.ts`/`.js` | Opens web frontend codebases |

---

### P5.5 — Blended Pipeline Metric (PROTO-001 — the actual goal)

**What it is:** Run a batch of real queries through the full pipeline (compress → T0.5 → fallback to Anthropic) and measure the *combined* cost and quality, not each layer in isolation.

**Why it matters:** Today we know compression saves 54.4% tokens and the local LLM answers at ~80% quality — but these are measured separately. The real question is: across 100 production queries, what is the average cost per query and average quality? The cumulative effect of T0.5 acceptance + compressed fallback is what makes this system worth deploying.

**What needs to be built:**

| ID | Feature | Success Criteria | Expected Result |
|----|---------|-----------------|-----------------|
| PROTO-001 | `pipeline_batch` sweep — run N queries through full pipeline | Tracks per-query: tier used, tokens sent to Anthropic (0 if T0.5 accepted), quality, latency | First true blended cost measurement |
| PROTO-002 | T0.5 acceptance rate metric | % of queries answered by local LLM without escalation, tracked per query batch | Expect 40–70% acceptance on C code questions |
| PROTO-003 | Blended cost report column | Leaderboard shows: sim_cost / t05_cost / blended_cost | Single number: "this config costs $X per 100 queries" |
| PROTO-004 | Quality comparison: local LLM answer vs Anthropic answer | For escalated queries: how much quality gain did Anthropic provide over T0.5? | Quantify when escalation is worth it |

**How to run once built:**
```bash
python kloc.py experiment pipeline-batch \
    --file doom/src/strife/p_enemy.c \
    --questions experiments/questions/c_game_questions.txt \
    --n 20
# → acceptance_rate, blended_cost/query, blended_quality
```

---

### P6 — Production Hardening

| ID | Feature | Success Criteria | Expected Result |
|----|---------|-----------------|-----------------|
| PROD-001 | Post-inference RQS cascade (ISSUE-002) | After LLM response received, compare with un-compressed baseline RQS | Detect quality regressions in production, not just pre-send |
| PROD-002 | Global pattern dictionary across repos | Patterns seen in 10+ repos promoted to global dict | Better F3 masking on first-seen repos |
| PROD-003 | Continuous benchmark CI | GitHub Actions runs `benchmark --mode synthetic` on every PR | Never regress below 77.9% TER |
| PROD-004 | Dashboard: live TER + cost savings over session | Web UI showing running totals per session | Product-quality demo capability |

---

## Success Criteria Summary

| Milestone | Definition of Done |
|-----------|-------------------|
| **M1 — Python 80%** | `benchmark --mode synthetic` TER ≥ 80%, RQS ≥ 0.85 |
| **M2 — C 40%** | `report-c` on Doom p_enemy.c TER ≥ 40%, RQS ≥ 0.85 |
| **M3 — RAG wired** | `pipeline` command uses RAG retrieval, cumulative TER ≥ 75% on C |
| **M4 — True Value 0.70** | Full pipeline TER × RQS ≥ 0.70 on any real-world file |
| **M5 — 95% ceiling** | All staircase layers live, theoretical ceiling demonstrated in report |
| **M6 — Blended metric** | `pipeline_batch` runs 20+ queries; blended_cost and acceptance_rate reported (PROTO-001) |

---

## What We Are NOT Building (Scope Exclusions)

- We do not build a paid SaaS product (no auth, billing, multi-tenancy)
- We do not retrain or fine-tune any LLM
- We do not implement our own vector DB (Qdrant is the dependency)
- We do not compress output tokens (only input/context window is in scope)
- We do not guarantee lossless reconstruction of skeletonized code (F1 is lossy by design)
