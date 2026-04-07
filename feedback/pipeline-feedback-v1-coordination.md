# Pipeline Feedback — v1 · Coordination Architecture

**Auditor:** Claude Sonnet 4.6  
**Date:** 2026-04-06  
**Files reviewed:** `orchestrator/pipeline.py`, `bus/message_bus.py`, `agents/base_agent.py`  
**Review skill:** `.claude/commands/review-pipeline.md`

---

## CRITICAL: Error Slices Propagate Silently Through the Full Pipeline

When any agent raises an exception, `_error_slice` returns a valid `FeatureSlice` with `confidence=0.0` and `{"error": ..., "input_slice_id": ...}` payload. `run_once` publishes this error slice to `OUT_CHANNEL`. The orchestrator checks only for `None`:

```python
pruner_out = self.pruner.run_once(timeout=2.0)
if pruner_out is None:                          # only None check
    return self._error("PRUNER produced no output")
# pruner_out could be an error slice — falls through
pruner_out.payload["task_type"] = task_type    # KeyError risk: payload only has "error" key
```

If PRUNER raises, `pruner_out.payload["task_type"]` is injected into `{"error": ..., "input_slice_id": ...}`. GRUG receives garbage and produces its own error slice. The final result dict contains `routing_decision: {}` and `compressed_chunks: []` with **no `error` key** — the caller sees an empty success response.

**Fix:**
```python
pruner_out = self.pruner.run_once(timeout=2.0)
if pruner_out is None or "error" in pruner_out.payload:
    err = pruner_out.payload.get("error", "no output") if pruner_out else "timeout"
    return self._error(f"PRUNER failed: {err}")
```

---

## 1. Message Bus Failure Modes

### Partial-Write / Malformed Line (no recovery)
`publish()` appends `line + "\n"` in one write. If the process is killed mid-write, the JSONL file contains a truncated line. `_read_next` calls `json.loads(lines[cursor])` with no `try/except` — a truncated line crashes the consuming agent.

**Fix:** Wrap `json.loads` in `_read_next` with `try/except json.JSONDecodeError: advance cursor + log warning`.

### Non-Atomic Cursor Update
`_read_next` reads `lines[cursor]`, then calls `_set_cursor(cursor + 1)` as a separate write. A crash between these two operations re-delivers the message on next run (silent duplicate delivery).

**Fix:** Use `os.replace()` (write to `.tmp`, then rename) for the cursor file — atomic on POSIX and near-atomic on Windows NTFS.

### Windows Has No Locking
`fcntl.flock` is conditionally imported only on non-Windows. The publish path on Windows has no locking. Fine for the current single-process model, but any future threading or subprocess use will silently race.

### Cursor Is a Line Index, Not a Byte Offset
The `_cursors` type comment says "byte offset" (line 55). It is actually a **line index**. `_read_next` uses `lines[cursor]`, not `file.seek(cursor)`. This mismatch will mislead any developer who tries to optimize with `file.seek()`.

### Full File Read Per `consume()` Call
`_read_next` calls `path.read_text()` on every message poll — a full file read. After 1,000 pipeline runs, `pruner_in.jsonl` has 1,000 lines. Every `consume()` reads all 1,000 to retrieve 1. This is O(total_messages) per message consumed — will degrade significantly at scale.

---

## 2. Double-Publish Bug

Every agent's output is published **twice**:
1. Inside `run_once()` → `self.bus.publish(self.OUT_CHANNEL, ...)` (base_agent.py:166)
2. In `pipeline.py` after payload mutation → `self.bus.publish("pruner.out", pruner_out.to_dict())`

The second publish is the one the next agent reads (cursor at correct position). The first publish writes a message that is never consumed — it sits in the queue file and doubles dead message accumulation. The cursor skips message #1 and reads message #2.

---

## 3. Orchestrator Payload Mutations — Schema Debt

`pipeline.py` directly mutates slice payloads before republishing:

```python
pruner_out.payload["task_type"] = task_type   # line 117
pruner_out.payload["question"]  = question    # line 118
```

This happens 3× (after PRUNER, GRUG, BALANCER). Agents write **incomplete** slices; the orchestrator backfills missing fields. There is no schema document capturing what each channel's payload must contain — easy for a new agent to miss a required field.

---

## 4. `auto_escalate` Is Dead Code

```python
# balancer.py line 146
auto_escalate = rqs_l1 < self.quality_floor and tier < self.tier_max
```

This is computed, logged, and included in the output payload. Nothing in `pipeline.py` reads `escalation_policy.auto_escalate` and re-routes. ZIPPY receives it but does not act on it. It is a flag that does nothing — and when `tier == tier_max`, it evaluates to `False` even when quality is critically low, inverting its intended meaning.

---

## 5. Missing LLM Call

The pipeline terminates at `zippy_out` and returns `routing_decision` (with `model_id`, `tier`, `context_token_budget`) and `compressed_chunks` — but **never makes an API call**. The README Token Journey diagram shows `ZIP → LLM → OUT`. That `LLM` node has no code.

**Minimal wiring** (new file `orchestrator/llm_caller.py`):

```python
import anthropic

def call_llm(pipeline_result: dict) -> str:
    routing  = pipeline_result.get("routing_decision", {})
    model_id = routing.get("model_id")
    tier     = routing.get("tier", 1)

    if tier == 0 or model_id is None:
        return "[T0/LOCAL] Answer served from rainbow cache."

    chunks = pipeline_result.get("compressed_chunks", [])
    context = "\n---\n".join(
        f"[{c.get('zone','?')}]\n{c.get('compressed_src','')}"
        for c in chunks if c.get("compressed_src","").strip()
    )
    history = pipeline_result.get("zipped_history", "")
    question = pipeline_result.get("question", "")

    messages = []
    if history and history != "no history":
        messages += [
            {"role": "user",      "content": f"[prior context, compressed]\n{history}"},
            {"role": "assistant", "content": "Understood."},
        ]
    messages.append({"role": "user", "content": f"{context}\n\n---\nQuestion: {question}"})

    resp = anthropic.Anthropic().messages.create(
        model=model_id, max_tokens=1024,
        system=f"Precise code assistant. Task: {pipeline_result.get('task_type','explain')}.",
        messages=messages,
    )
    return resp.content[0].text
```

Integration in `pipeline.py` after `zippy_out`:

```python
from orchestrator.llm_caller import call_llm
result["llm_response"] = call_llm(result)
```

Also: the RHD `§hash → original` **unmasking** referenced in the README Token Journey diagram (`OUT` node) is not implemented anywhere. `rhd_registry_delta` from GRUG is passed through but never used to unmask the LLM response before returning it to the caller.

---

## 6. Timeout Is Not a Processing Deadline

```python
# base_agent.py:158
raw = self.bus.consume(self.IN_CHANNEL, timeout=timeout)
```

Once `consume` returns (immediately after orchestrator publishes), the timeout is no longer enforced. `process()` can run for arbitrarily long. The timeout is effectively only a queue-empty wait timeout.

**Consequence:** PRUNER's 2s timeout will not contain large C files. A 50 KLOC C file chunked into ~500 functions at 20ms/chunk = 10 seconds processing time, no abort. The timeout parameter is misleading — it implies a processing deadline that doesn't exist.

---

## 7. No Taxonomy Validation on Initial Slice

`Taxonomy.validate()` is called inside `BaseAgent.emit()`. The orchestrator constructs the initial slice directly:

```python
self.bus.publish("pruner.in", init_slice.to_dict())   # no emit(), no validation
```

`init_slice` uses tags `["code.classification", f"lang.{language}"]`. If `language` is anything other than `python/c/go/rust/unknown`, the tag `lang.{language}` silently passes through unvalidated. Taxonomy validation is also bypassed on all three orchestrator republishes.

If `data/taxonomy.json` is missing, all validation silently no-ops (no warning log):

```python
if not self._valid_tags:
    return  # silent skip
```

---

## 8. Bus Is Not Session-Scoped

All sessions share the same JSONL files and cursor files. Two concurrent `Pipeline` instances with different `session_id` values race on the same `pruner.in` queue. A new `Pipeline()` instance starts with `_cursors = {}` — if cursor files from a prior run exist, it starts at the right position; if cursor files are missing (tmpdir, docker volume), it replays all messages from all previous runs.

**Minimal fix for concurrent safety:** Add a `job_id` prefix to queue directory per pipeline run:

```python
# In Pipeline.__init__:
queue_dir = queue_dir / f"job_{session_id}"
self.bus = MessageBus(queue_dir=queue_dir)
```

The `MessageBus` constructor already accepts `queue_dir` — this is a two-line change.

---

## 9. Audit Log Silent Decisions

| Location | What happens | Logged? |
|---|---|---|
| `pipeline.py:95` | `rainbow_hit` computed and injected into payload | No |
| `pipeline.py:83-87` | Fallback from `chunk_file` to `chunk_source` on empty chunks | No |
| `balancer.py:146` | `auto_escalate=True` set but never acted on | Logged but dead |
| `zippy.py:244` | `_extract_facts` hardcodes up to 10 tokens per run | Count only, not keys |
| `pruner.py:109` | `_lang_tags()` fallback to `lang.unknown` | No |
| `grug.py:259-263` | InB exception swallowed silently; TER silently 0% | **No — critical gap** |

The GRUG silent exception is the most dangerous: `except Exception: pass` around `InferenceBridge.process_file()`. TER can be 0% with no operator visibility.

---

## 10. README Accuracy Issues

- **Sequence diagram omits orchestrator payload mutations.** The clean `PRUNER → B2 → GRUG` arrow hides the 3× payload mutation step — architecturally significant.
- **`LLM` and `Qdrant` nodes exist in diagrams but have no code in `pipeline.py`.** The system is presented as end-to-end functional; it is not.
- **Module docstring F1→F2→F3→F4 ordering** conflicts with actual F2→F1→F3→F4 execution.

---

## Summary Table

| Issue | Severity |
|---|---|
| Error slices propagate silently — caller sees empty success | Critical |
| No LLM call wired; `auto_escalate` and RHD unmasking are dead code | Critical |
| Bus not session-scoped; concurrent sessions race on same queues | Critical |
| Cursor write non-atomic (duplicate delivery on crash) | High |
| Full file read per `consume()` — O(total_messages) | High |
| Double-publish per agent — dead message accumulates | Medium |
| Orchestrator bypasses `emit()` — no taxonomy validation on initial slice | High |
| GRUG InB exception silently swallowed — TER silently 0% | High |
| `auto_escalate` computes False under cap, inverts intended meaning | High |
| Timeout is queue-wait only, not processing deadline | Medium |
| `rainbow_hit` routing decision not logged | Medium |
