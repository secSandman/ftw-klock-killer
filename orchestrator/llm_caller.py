"""
llm_caller.py — Minimal LLM wiring for the FTW-KLOC-KILLER pipeline.

MVP1 assumption: anthropic SDK is installed and ANTHROPIC_API_KEY is set
in the environment (same session that runs Claude Code).

# BUG: ISSUE-001 — No explicit auth validation before API call
# Authentication relies entirely on the Anthropic SDK auto-reading
# ANTHROPIC_API_KEY from the environment at call time. If the variable
# is not set, the error surfaces at runtime (AuthenticationError), not
# at startup or import time. A user running this pipeline outside of a
# Claude Code session will get a confusing runtime crash instead of a
# clear "please set ANTHROPIC_API_KEY" message.
# Fix: add an explicit os.getenv("ANTHROPIC_API_KEY") check at Pipeline
# init time and raise a descriptive ConfigurationError with setup instructions.
# Also: endpoint is hardcoded to Anthropic default (api.anthropic.com/v1/messages).
# No support for custom base_url, proxy, or OpenAI alt_model fallback yet.

TODO: add provider abstraction (OpenAI alt_model fallback), streaming,
      response RQS-L1 scoring for post-inference cascade escalation.

## Prompt caching (Anthropic)
When the model is a claude-* model, ACTIVE_TIER_MAX > 0, and the system
prompt or context block is >= 1024 tokens, we attach cache_control to those
blocks so Anthropic can reuse the KV cache across calls.  Cache hits are
logged via DecisionTracker and a `cache_savings` field is appended to the
returned token report.

Caching is a no-op for non-Anthropic models (never breaks OpenAI calls).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

# ── T0.5 Local LLM settings (mirror of balancer.py constants) ───────────────
# Read directly from env — no import from balancer to avoid circular deps.
_LOCAL_LLM_ENABLED           = os.getenv("KLOC_LOCAL_LLM_ENABLED", "1") == "1"
_LOCAL_LLM_MAX_CTX           = int(os.getenv("KLOC_LOCAL_LLM_MAX_CTX", "8000"))
_LOCAL_LLM_MODEL             = os.getenv("KLOC_LLM_MODEL", "qwen2.5-coder:14b")
_LOCAL_LLM_PORT              = int(os.getenv("KLOC_LLM_PORT", "11434"))
_LOCAL_LLM_RAG_TOP_K         = int(os.getenv("KLOC_LOCAL_LLM_RAG_TOP_K", "0"))
_LOCAL_LLM_QUALITY_THRESHOLD = int(os.getenv("KLOC_LOCAL_LLM_QUALITY_THRESHOLD", "20"))
_LOCAL_LLM_TEMPERATURE       = float(os.getenv("KLOC_LOCAL_LLM_TEMPERATURE", "0.1"))
_LOCAL_LLM_TWO_STAGE         = os.getenv("KLOC_LOCAL_LLM_TWO_STAGE", "0") == "1"

_local_available_cache: Optional[bool] = None  # cached per process


def _local_llm_available() -> bool:
    """Probe Ollama once per process; cache result."""
    global _local_available_cache
    if _local_available_cache is not None:
        return _local_available_cache
    if not _LOCAL_LLM_ENABLED:
        _local_available_cache = False
        return False
    try:
        import httpx
        r = httpx.get(f"http://localhost:{_LOCAL_LLM_PORT}/api/tags", timeout=2.0)
        _local_available_cache = r.status_code == 200
    except Exception:
        _local_available_cache = False
    return _local_available_cache


def _local_response_quality_ok(response: str) -> bool:
    """
    Lightweight post-inference quality gate.
    Rejects empty, very short, or refusal responses.
    Threshold is configurable via KLOC_LOCAL_LLM_QUALITY_THRESHOLD (default 20).
    """
    if not response or len(response.split()) < _LOCAL_LLM_QUALITY_THRESHOLD:
        return False
    refusals = [
        "i don't have", "i cannot", "i'm unable",
        "insufficient information", "i do not have access",
        "no context provided",
    ]
    low = response.lower()
    return not any(p in low for p in refusals)


def _build_rag_context(question: str, context_block: str, file_path: str = "") -> str:
    """
    P1: Prepend top-k BM25 chunks from metadata index to the local LLM context.
    Falls back to unmodified context_block if metadata index unavailable or RAG disabled.
    """
    if _LOCAL_LLM_RAG_TOP_K == 0 or not question.strip():
        return context_block
    try:
        from rag.retriever import RAGRetriever
        from rag.chunker import Chunker
        # Locate metadata index relative to file_path
        repo_hint = None
        if file_path:
            p = Path(file_path).resolve()
            for parent in p.parents:
                if (parent / ".kloc" / "metadata_index.json").exists():
                    repo_hint = parent
                    break
        retriever = RAGRetriever(force_bm25=True, repo_path=repo_hint)
        # Build minimal chunks from context_block for BM25 ranking
        from rag.chunker import Chunk
        import hashlib
        lines = context_block.split("\n---\n")
        chunks = []
        for i, src in enumerate(lines):
            cid = hashlib.md5(src.encode()).hexdigest()[:16]
            c = Chunk.__new__(Chunk)
            c.chunk_id = cid
            c.source = src
            c.compressed_src = src
            c.name = f"chunk_{i}"
            c.zone = "FOREGROUND"
            c.language = "unknown"
            c.token_count = len(src.split())
            chunks.append(c)
        if not chunks:
            return context_block
        selected = retriever.retrieve(question=question, chunks=chunks, top_k=_LOCAL_LLM_RAG_TOP_K)
        if not selected:
            return context_block
        rag_parts = [f"[RAG/{c.name}]\n{c.source}" for c in selected]
        return "\n---\n".join(rag_parts) + "\n---\n" + context_block
    except Exception:
        return context_block


def _score_response_confidence(question: str, response: str) -> float:
    """
    P2: TF cosine between question key terms and response.
    Returns 0.0–1.0. Used as a soft quality signal above the hard word-count gate.
    """
    import math
    _TOK = re.compile(r"[a-zA-Z_]\w*|[0-9]+")

    def tf(text):
        freq = {}
        for w in _TOK.findall(text.lower()):
            freq[w] = freq.get(w, 0) + 1
        return freq

    q_tf = tf(question)
    r_tf = tf(response)
    if not q_tf or not r_tf:
        return 0.0
    keys = set(q_tf) | set(r_tf)
    dot  = sum(q_tf.get(k, 0) * r_tf.get(k, 0) for k in keys)
    mq   = math.sqrt(sum(v**2 for v in q_tf.values()))
    mr   = math.sqrt(sum(v**2 for v in r_tf.values()))
    return round(dot / (mq * mr), 4) if mq and mr else 0.0


def _try_local_llm(
    context_block: str,
    question: str,
    task_type: str,
    pipeline_result: Dict[str, Any],
) -> Optional[str]:
    """
    T0.5 shim: attempt local Ollama inference before calling Anthropic.

    Returns the response string if accepted, None otherwise.
    Writes local_llm_attempted / local_llm_accepted metadata into pipeline_result.
    """
    pipeline_result["local_llm_attempted"] = True
    pipeline_result["local_llm_accepted"]  = False

    if not _local_llm_available():
        pipeline_result["local_llm_skip_reason"] = "unavailable"
        return None

    # Skip if context exceeds local model's comfort zone
    if _wc(context_block) > _LOCAL_LLM_MAX_CTX:
        pipeline_result["local_llm_skip_reason"] = (
            f"context too large ({_wc(context_block)} words > {_LOCAL_LLM_MAX_CTX})"
        )
        return None

    # P1: inject RAG context from metadata index
    file_path = pipeline_result.get("file_path", "")
    context_block = _build_rag_context(question, context_block, file_path)

    try:
        import time as _time
        from experiments.local_llm import LocalLLMHarness
        harness = LocalLLMHarness(model=_LOCAL_LLM_MODEL, verbose=False)
        _t = _time.time()
        response = harness.ask(question, context=context_block, temperature=_LOCAL_LLM_TEMPERATURE)
        pipeline_result["local_llm_latency_ms"] = round((_time.time() - _t) * 1000)
    except Exception as exc:
        pipeline_result["local_llm_skip_reason"] = f"harness error: {exc}"
        return None

    if not _local_response_quality_ok(response):
        pipeline_result["local_llm_skip_reason"] = "quality gate failed"
        return None

    # P2: record confidence score
    conf = _score_response_confidence(question, response)
    pipeline_result["local_llm_confidence"] = conf

    # Accepted — annotate result metadata
    pipeline_result["local_llm_accepted"] = True
    pipeline_result["local_llm_model"]    = _LOCAL_LLM_MODEL
    pipeline_result["local_llm_tier"]     = "T0.5/LOCAL_LLM"

    # Log to audit trail
    try:
        from bus.decision_tracker import DecisionTracker
        DecisionTracker().log(
            agent_id       = "LLM_CALLER",
            decision_type  = "LOCAL_LLM_HIT",
            decision_value = {"model": _LOCAL_LLM_MODEL, "task_type": task_type},
            rationale      = f"T0.5 local LLM accepted response ({len(response.split())} words)",
            confidence     = 0.80,
        )
    except Exception:
        pass

    return response

# Minimum block size (tokens) Anthropic requires for cache eligibility
_CACHE_MIN_TOKENS = 1024

# Fast word-count approximation for cache eligibility check.
# Real token count is higher, but this avoids a tiktoken import.
# Over-counting is safe: we only skip caching if clearly below threshold.
_WC_RE = re.compile(r"\S+")


def _wc(text: str) -> int:
    """Rough word count as a proxy for token count."""
    return len(_WC_RE.findall(text))


def _is_anthropic_model(model_id: str) -> bool:
    return bool(model_id) and model_id.startswith("claude-")


def _caching_enabled(model_id: str) -> bool:
    """True when all three conditions for cache_control are met."""
    tier_max = int(os.environ.get("ACTIVE_TIER_MAX", "2"))
    return tier_max > 0 and _is_anthropic_model(model_id)


def _maybe_cache(block: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """
    Attach cache_control to a content block if it is eligible.
    Mutates and returns the block dict.
    """
    if not _caching_enabled(model_id):
        return block
    text = block.get("text", "")
    if _wc(text) >= _CACHE_MIN_TOKENS:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def _try_two_stage(
    context_block: str,
    question: str,
    task_type: str,
    model_id: str,
    pipeline_result: Dict[str, Any],
) -> Optional[str]:
    """
    P3: Two-stage Anthropic→Local reconstruction.

    Only activates when:
      - KLOC_LOCAL_LLM_TWO_STAGE=1
      - task_type in (refactor, codegen) — high output token tasks
      - local Ollama is available

    Stage 1: Anthropic (model_id) produces a structured plan (max_tokens=512).
    Stage 2: Local 14B executes the plan (generates code/diff from the plan).

    Reduces Anthropic output tokens from ~800-1000 to ~512 while keeping
    plan quality. Local model handles the mechanical generation.
    """
    if not _LOCAL_LLM_TWO_STAGE:
        return None
    if task_type not in ("refactor", "codegen"):
        return None
    if not _local_llm_available():
        return None

    try:
        # Stage 1: Anthropic produces compact plan
        plan_prompt = (
            f"{context_block}\n\n---\n"
            f"Task: {question}\n\n"
            "Respond with a numbered action plan only (no code). "
            "Each step should be one line. Max 8 steps."
        )
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model_id,
            max_tokens=512,
            messages=[{"role": "user", "content": plan_prompt}],
        )
        plan = resp.content[0].text.strip()
        pipeline_result["two_stage_plan"] = plan

        # Stage 2: Local model executes the plan
        from experiments.local_llm import LocalLLMHarness
        harness = LocalLLMHarness(model=_LOCAL_LLM_MODEL, verbose=False)
        execution_prompt = (
            f"Given this code context:\n{context_block[:3000]}\n\n"
            f"Execute this plan step by step and produce the result:\n{plan}"
        )
        result = harness.ask(execution_prompt, temperature=_LOCAL_LLM_TEMPERATURE)
        if _local_response_quality_ok(result):
            pipeline_result["two_stage_accepted"] = True
            pipeline_result["two_stage_model"]    = _LOCAL_LLM_MODEL
            try:
                from bus.decision_tracker import DecisionTracker
                DecisionTracker().log(
                    agent_id       = "LLM_CALLER",
                    decision_type  = "TWO_STAGE_HIT",
                    decision_value = {"plan_tokens": len(plan.split()), "task_type": task_type},
                    rationale      = "P3 two-stage: Anthropic plan + local execution",
                    confidence     = 0.75,
                )
            except Exception:
                pass
            return result
    except Exception as exc:
        pipeline_result["two_stage_error"] = str(exc)

    return None


def call_llm(pipeline_result: Dict[str, Any]) -> str:
    """
    Execute the LLM call described by pipeline_result["routing_decision"].
    Returns the model's text response.

    Side-effects:
    - Populates pipeline_result["cache_savings"] with estimated cost savings
      when a cache hit is detected in the response usage object.
    """
    routing   = pipeline_result.get("routing_decision", {})
    tier      = routing.get("tier", 1)
    model_id  = routing.get("model_id")
    question  = pipeline_result.get("question", "")
    task_type = pipeline_result.get("task_type", "explain")

    # T0 = rainbow cache / local — no LLM needed
    if tier == 0 or model_id is None:
        return "[T0/LOCAL] Answer served from rainbow cache."

    # Build context block from compressed chunks
    chunks = pipeline_result.get("compressed_chunks", [])
    context_parts = []
    for c in chunks:
        zone = c.get("zone", "?")
        src  = c.get("compressed_src", "") or c.get("source", "")
        if src.strip():
            context_parts.append(f"[{zone}]\n{src}")
    context_block = "\n---\n".join(context_parts)

    # T0.5 — try local Ollama before spending Anthropic tokens
    local_resp = _try_local_llm(context_block, question, task_type, pipeline_result)
    if local_resp is not None:
        return _unmask_response(local_resp, pipeline_result)

    # P3 — two-stage: Anthropic plan + local execution (for refactor/codegen)
    if tier >= 2:
        two_stage_resp = _try_two_stage(context_block, question, task_type, model_id, pipeline_result)
        if two_stage_resp is not None:
            return _unmask_response(two_stage_resp, pipeline_result)

    # Build messages: zipped history context + current question
    zipped_history = pipeline_result.get("zipped_history", "")
    hardcoded      = pipeline_result.get("hardcoded_state", {})

    messages: List[Dict[str, Any]] = []
    if zipped_history and zipped_history not in ("no history", ""):
        messages += [
            {"role": "user",      "content": f"[prior context, compressed]\n{zipped_history}"},
            {"role": "assistant", "content": "Understood. Continuing with compressed context."},
        ]

    # Context block as a structured content block so we can attach cache_control
    user_msg_content: Any
    if context_block:
        context_content_block = _maybe_cache(
            {"type": "text", "text": f"{context_block}\n\n---\nQuestion: {question}"},
            model_id,
        )
        user_msg_content = [context_content_block]
    else:
        user_msg_content = question

    messages.append({"role": "user", "content": user_msg_content})

    # System prompt — hint about compressed format and hardcoded expansions
    hardcoded_hint = ""
    if hardcoded:
        pairs = ", ".join(f"{v}={k}" for k, v in list(hardcoded.items())[:8])
        hardcoded_hint = f"\nKey expansions (short→full): {pairs}"

    system_prompt_text = (
        f"You are a precise, concise code assistant. Task: {task_type}.\n"
        f"Source context is token-compressed. [ZONE] = FOREGROUND/BACKGROUND. "
        f"[§:hash] = masked boilerplate."
        f"{hardcoded_hint}"
    )

    # Build system parameter: plain string for short prompts, structured list
    # with cache_control for long ones (Anthropic requirement: list of blocks).
    system_param: Any
    if _caching_enabled(model_id) and _wc(system_prompt_text) >= _CACHE_MIN_TOKENS:
        system_param = [
            _maybe_cache({"type": "text", "text": system_prompt_text}, model_id)
        ]
    else:
        system_param = system_prompt_text

    client   = anthropic.Anthropic()
    response = client.messages.create(
        model      = model_id,
        max_tokens = 1024,
        system     = system_param,
        messages   = messages,
    )

    # ── Cache hit detection ──────────────────────────────────────────────────
    _handle_cache_usage(response, pipeline_result, model_id)

    raw_text = response.content[0].text

    # ── ISSUE-003: unmask [§:hash] tokens in the LLM response ──────────────
    # The LLM sometimes quotes back the compressed [§:INC_xxxxxxxx] tokens
    # verbatim.  Restore them to their original text before returning so the
    # caller (and end user) never sees bare hash tokens.
    unmasked_text = _unmask_response(raw_text, pipeline_result)
    return unmasked_text


# ---------------------------------------------------------------------------
# Cache usage reporting
# ---------------------------------------------------------------------------

# Approximate cost per million tokens for Anthropic models (USD).
# Cache reads cost ~10% of normal input price.
# These are rough estimates — actual pricing may differ by model.
_INPUT_COST_PER_MTK: Dict[str, float] = {
    "claude-haiku":  0.25,
    "claude-sonnet": 3.00,
    "claude-opus":  15.00,
}


def _model_input_cost(model_id: str) -> float:
    """Return input cost per million tokens for the given model ID."""
    model_id_lower = model_id.lower()
    for key, cost in _INPUT_COST_PER_MTK.items():
        if key in model_id_lower:
            return cost
    return 3.00  # default: Sonnet pricing


def _handle_cache_usage(
    response: Any,
    pipeline_result: Dict[str, Any],
    model_id: str,
) -> None:
    """
    Inspect the response usage object for cache hits.
    Logs via DecisionTracker and writes cache_savings into pipeline_result.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    cache_read_tokens  = getattr(usage, "cache_read_input_tokens",  0) or 0
    cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

    # Compute estimated savings: cache reads cost 10% of normal input price
    input_cost_per_m   = _model_input_cost(model_id)
    normal_cost        = (cache_read_tokens / 1_000_000) * input_cost_per_m
    cached_cost        = (cache_read_tokens / 1_000_000) * input_cost_per_m * 0.10
    savings_usd        = round(normal_cost - cached_cost, 8)

    pipeline_result["cache_savings"] = {
        "cache_read_tokens":  cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "savings_usd":        savings_usd,
        "model_id":           model_id,
    }

    if cache_read_tokens > 0:
        # Log cache hit to the audit trail
        try:
            from bus.decision_tracker import DecisionTracker
            tracker = DecisionTracker()
            tracker.log(
                agent_id       = "LLM_CALLER",
                decision_type  = "CACHE_HIT",
                decision_value = cache_read_tokens,
                rationale      = (
                    f"Anthropic prompt cache hit: {cache_read_tokens} tokens read "
                    f"from cache, saved ~${savings_usd:.6f}"
                ),
                confidence     = 1.0,
                metadata       = {
                    "cache_read_tokens":  cache_read_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "savings_usd":        savings_usd,
                    "model_id":           model_id,
                },
            )
        except Exception:
            pass  # never crash on audit logging


# ---------------------------------------------------------------------------
# Response unmasking
# ---------------------------------------------------------------------------

def _unmask_response(text: str, pipeline_result: Dict[str, Any]) -> str:
    """
    Restore [§:hash] markers in *text* back to their original strings.

    Two unmask passes are applied in order:
      1. Python RHD registry  — ChromatophoricMasker (inference_bridge.py F3)
         Uses the persisted inb_registry.json plus any deltas carried in
         pipeline_result["rhd_registry_delta"].
      2. C include/define registry — C_IncludeMasker._savings reverse map,
         carried in pipeline_result["_c_masker"] when available.

    Unknown tokens are left in place (safe — never corrupts output).
    """
    import re as _re

    # Fast path: no mask tokens present at all
    _MASK_RE = _re.compile(r"\[§:[a-f0-9]{8}\]")
    if not _MASK_RE.search(text):
        return text

    # ── Pass 1: Python RHD (ChromatophoricMasker) ───────────────────────────
    try:
        from src.inference_bridge import ChromatophoricMasker
        masker = ChromatophoricMasker()  # loads inb_registry.json
        # Merge in any session-local delta (new masks registered this run)
        rhd_delta = pipeline_result.get("rhd_registry_delta", {})
        if rhd_delta:
            masker._registry.update(rhd_delta)
        text = masker.unmask(text)
    except Exception:
        pass  # never crash on unmask — degraded output is better than no output

    # ── Pass 2: C include/define masker ─────────────────────────────────────
    try:
        c_masker = pipeline_result.get("_c_masker")
        if c_masker is not None:
            text = c_masker.unmask(text)
    except Exception:
        pass

    return text
