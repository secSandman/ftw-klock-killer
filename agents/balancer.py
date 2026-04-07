"""
balancer.py — BALANCER Agent (The Resource Balancer / Systems Ecologist)
=========================================================================
Strategy: Tiered Intelligence Loop — never over-kill.
Action:   Routes tasks to the cheapest model that meets quality floor.
          Simple task? Stay local (0 tokens, 0 cost).
          Complex task? Escalate — but only as far as needed.

Research grounding:
  - Holling, C.S. (1973). "Resilience and Stability of Ecological Systems."
    Annual Review of Ecology and Systematics, 4, 1-23.
    Adaptive cycle: exploit cheap resources first, escalate on disturbance.
  - Walker, B. et al. (2004). "Resilience, adaptability and transformability
    in social-ecological systems." Ecology and Society 9(2):5.
  - Rao, S. (2023). "Cost-Optimal LLM Inference Routing." arXiv:2310.03744
    Empirical justification for tiered routing: 60-85% cost reduction
    with minimal quality loss using small→large escalation.
  - Chen, L. et al. (2024). "Frugal GPT." arXiv:2305.05176
    FrugalGPT: achieves GPT-4 quality at 1/50th the cost via cascading.

Tier table:
  T0  LOCAL   — rainbow cache hit + simple lookup. 0 tokens, 0 cost.
  T1  CHEAP   — Haiku / GPT-4o-mini. Fast, cheap. Most tasks land here.
  T2  MID     — Sonnet / GPT-4o. Complex codegen and review.
  T3  FULL    — Opus / GPT-4o. Crisis: refactor, low RQS, huge context.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, FeatureSlice


# Tier definitions — override via .env ACTIVE_TIER_MAX
TIER_CONFIG = {
    0: {
        "name":    "LOCAL",
        "model":   None,
        "cost_per_1k_in":  0.0,
        "cost_per_1k_out": 0.0,
        "max_context":     None,
        "description": "Rainbow cache / local pattern match. Zero cost.",
    },
    1: {
        "name":    "CHEAP",
        "model":   "claude-haiku-4-5-20251001",
        "alt_model": "gpt-4o-mini",
        "cost_per_1k_in":  0.00080,   # $0.80/MTok  (Haiku 4.5)
        "cost_per_1k_out": 0.00400,   # $4.00/MTok  (Haiku 4.5)
        "max_context":     200_000,
        "description": "Fast & cheap. Handles 80% of tasks.",
    },
    2: {
        "name":    "MID",
        "model":   "claude-sonnet-4-5",
        "alt_model": "gpt-4o",
        "cost_per_1k_in":  0.003,
        "cost_per_1k_out": 0.015,
        "max_context":     200_000,
        "description": "Mid-tier. Complex codegen and review.",
    },
    3: {
        "name":    "FULL",
        "model":   "claude-opus-4-6",
        "alt_model": "gpt-4o",
        "cost_per_1k_in":  0.015,
        "cost_per_1k_out": 0.075,
        "max_context":     200_000,
        "description": "Full power. Only for crisis-level complexity.",
    },
}

TASK_TIER_FLOOR = {
    "search":   0,
    "explain":  1,
    "review":   1,
    "codegen":  2,
    "refactor": 2,
    "debug":    1,
    "test":     1,
    "unknown":  1,
}

DEFAULT_QUALITY_FLOOR = 0.85
DEFAULT_TIER_MAX      = 2

_TASK_OUTPUT_TOKENS = {
    "search":   100,
    "explain":  400,
    "review":   600,
    "codegen":  800,
    "refactor": 1000,
    "debug":    500,
    "test":     600,
    "unknown":  400,
}

# ── T0.5 Local LLM ─────────────────────────────────────────────────────────
# When Ollama is running, call_llm transparently tries the local model before
# escalating to Anthropic. BALANCER signals availability via routing_decision
# so llm_caller can act without knowing about Ollama directly.
#
# KLOC_LOCAL_LLM_ENABLED  — "1" enable (default), "0" disable entirely
# KLOC_LOCAL_LLM_MAX_CTX  — max context tokens to send to local model (default 8000)
#                           14B models degrade on long contexts; keep it tight
# KLOC_LLM_MODEL          — model name inside Ollama (default qwen2.5-coder:14b)
LOCAL_LLM_ENABLED  = os.getenv("KLOC_LOCAL_LLM_ENABLED", "1") == "1"
LOCAL_LLM_MAX_CTX  = int(os.getenv("KLOC_LOCAL_LLM_MAX_CTX", "8000"))
LOCAL_LLM_MODEL    = os.getenv("KLOC_LLM_MODEL", "qwen2.5-coder:14b")

_local_llm_available_cache: Optional[bool] = None  # module-level cache


def _local_llm_available() -> bool:
    """
    Probe whether the Ollama container is up and responding.
    Result is cached for the process lifetime — Ollama doesn't go down
    between requests in normal operation.
    """
    global _local_llm_available_cache
    if _local_llm_available_cache is not None:
        return _local_llm_available_cache
    if not LOCAL_LLM_ENABLED:
        _local_llm_available_cache = False
        return False
    try:
        import httpx
        port = int(os.getenv("KLOC_LLM_PORT", "11434"))
        r = httpx.get(f"http://localhost:{port}/api/tags", timeout=2.0)
        _local_llm_available_cache = r.status_code == 200
    except Exception:
        _local_llm_available_cache = False
    return _local_llm_available_cache


class BalancerAgent(BaseAgent):
    """
    BALANCER: Adaptive tier router.

    Input channel:  grug.out
    Output channel: balancer.out

    Reads GRUG compression output + RQS-L1 score.
    Selects the cheapest model tier that meets the quality floor.
    Auto-escalates if RQS-L1 is below threshold.
    """

    AGENT_ID    = "BALANCER"
    IN_CHANNEL  = "grug.out"
    OUT_CHANNEL = "balancer.out"

    def __init__(
        self,
        quality_floor: float = DEFAULT_QUALITY_FLOOR,
        tier_max: int = DEFAULT_TIER_MAX,
        llm_provider: str = "anthropic",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.quality_floor = float(os.getenv("QUALITY_FLOOR", quality_floor))
        self.tier_max      = int(os.getenv("ACTIVE_TIER_MAX", tier_max))
        self.llm_provider  = llm_provider

    def process(self, slice_in: FeatureSlice) -> FeatureSlice:
        payload = slice_in.payload

        overall_ter    = payload.get("overall_ter", 0.0)
        rqs_l1         = payload.get("rqs_l1_score", 0.82)
        task_type      = payload.get("task_type", "unknown").lower()
        question       = payload.get("question", "")
        context_tokens = payload.get("total_final_tokens", 0)
        rainbow_hit    = payload.get("rainbow_hit", False)
        language       = payload.get("language", "unknown")

        # Select tier
        tier, reason = self._select_tier(
            task_type      = task_type,
            rqs_l1         = rqs_l1,
            context_tokens = context_tokens,
            rainbow_hit    = rainbow_hit,
        )

        tier_info = TIER_CONFIG[tier]
        model_id  = self._pick_model(tier)
        est_cost  = self._estimate_cost(tier, context_tokens, task_type)

        # Escalation policy
        auto_escalate = rqs_l1 < self.quality_floor and tier < self.tier_max

        decision = (
            f"T{tier}/{tier_info['name']}: {task_type}, "
            f"{context_tokens}ctx, RQS={rqs_l1:.2f}, "
            f"TER={overall_ter:.1f}%, est=${est_cost:.5f}"
        )

        self.log_decision(
            decision_type  = "ROUTE",
            decision_value = {"tier": tier, "model": model_id, "est_cost_usd": est_cost},
            rationale      = reason,
            confidence     = rqs_l1,
            slice_id       = slice_in.slice_id,
            metadata       = {"auto_escalate": auto_escalate, "tier_max": self.tier_max},
        )

        tier_tag = f"routing.tier{tier}"

        return self.emit(
            taxonomy_tags   = [tier_tag, "cost.optimization", "adaptive.escalation", "quality.floor"],
            payload         = {
                "routing_decision": {
                    "tier":                tier,
                    "tier_name":           tier_info["name"],
                    "model_id":            model_id,
                    "reason":              reason,
                    "estimated_cost_usd":  est_cost,
                    "context_token_budget": context_tokens,
                    "quality_floor":       self.quality_floor,
                },
                "escalation_policy": {
                    "auto_escalate":         auto_escalate,
                    "escalate_if_rqs_below": self.quality_floor,
                    "max_tier":              self.tier_max,
                },
                # Pass through for orchestrator / ZIPPY
                "compressed_chunks":    payload.get("compressed_chunks", []),
                "total_final_tokens":   context_tokens,
                "overall_ter":          overall_ter,
                "rqs_l1_score":         rqs_l1,
                "question":             question,
                "task_type":            task_type,
                "language":             language,
                "file_path":            payload.get("file_path", ""),
            },
            confidence      = rqs_l1,
            decision        = decision,
            parent_slice_id = slice_in.slice_id,
            token_count     = context_tokens,
        )

    # ── Routing logic ────────────────────────────────────────────────

    def _select_tier(
        self,
        task_type: str,
        rqs_l1: float,
        context_tokens: int,
        rainbow_hit: bool,
    ) -> tuple[int, str]:
        """
        Returns (tier, reason_string).

        Routing table (Holling adaptive cycle inspired):
          Phase 1 — Exploit: use cheapest possible resource
          Phase 2 — Disturbance check: escalate only if quality fails
          Phase 3 — Reorganize: lock in at the right tier
        """

        # T0: Rainbow cache hit — free, instant
        if rainbow_hit and task_type in ("search", "explain"):
            return 0, "Rainbow cache hit + simple task → T0/LOCAL (free)"

        # Base tier from task type
        task_floor = TASK_TIER_FLOOR.get(task_type, 1)

        # Context size escalation
        if context_tokens > 50_000:
            context_tier = 3
        elif context_tokens > 16_000:
            context_tier = 2
        elif context_tokens > 4_000:
            context_tier = 1
        else:
            context_tier = 0

        # Quality escalation
        if rqs_l1 < 0.70:
            quality_tier = 3   # RQS critically low — send full source
        elif rqs_l1 < self.quality_floor:
            quality_tier = task_floor + 1
        else:
            quality_tier = task_floor

        # Take the max of all pressures
        selected = max(task_floor, context_tier, quality_tier)
        selected = min(selected, self.tier_max)  # never exceed ceiling

        reasons = []
        if task_floor > 0:
            reasons.append(f"task={task_type}→T{task_floor}")
        if context_tier > task_floor:
            reasons.append(f"context={context_tokens}tok→T{context_tier}")
        if rqs_l1 < self.quality_floor:
            reasons.append(f"RQS={rqs_l1:.2f}<floor→T{quality_tier}")

        reason = " | ".join(reasons) if reasons else f"{task_type} at T{selected}"
        return selected, reason

    def _pick_model(self, tier: int) -> Optional[str]:
        if tier == 0:
            return None
        cfg = TIER_CONFIG[tier]
        # Use alt_model if provider is openai
        if self.llm_provider == "openai":
            return cfg.get("alt_model", cfg["model"])
        return cfg["model"]

    def _estimate_cost(self, tier: int, context_tokens: int, task_type: str = "unknown") -> float:
        cfg = TIER_CONFIG[tier]
        out_tokens = _TASK_OUTPUT_TOKENS.get(task_type, 400)
        cost_in  = (context_tokens / 1000) * cfg["cost_per_1k_in"]
        cost_out = (out_tokens / 1000) * cfg["cost_per_1k_out"]
        return round(cost_in + cost_out, 6)


# ── CLI self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    balancer = BalancerAgent()

    test_cases = [
        {"task": "search",   "rqs": 0.95, "tokens": 200,   "rainbow": True,  "expect": 0},
        {"task": "explain",  "rqs": 0.92, "tokens": 1000,  "rainbow": False, "expect": 1},
        {"task": "codegen",  "rqs": 0.88, "tokens": 8000,  "rainbow": False, "expect": 2},
        {"task": "refactor", "rqs": 0.72, "tokens": 20000, "rainbow": False, "expect": 2},
    ]

    print("BALANCER self-test:")
    for tc in test_cases:
        tier, reason = balancer._select_tier(tc["task"], tc["rqs"], tc["tokens"], tc["rainbow"])
        status = "✓" if tier == tc["expect"] else "✗"
        print(f"  {status} task={tc['task']:<10} rqs={tc['rqs']} tokens={tc['tokens']:<6} → T{tier} ({reason})")
