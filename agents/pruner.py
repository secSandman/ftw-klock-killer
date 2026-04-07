"""
pruner.py — PRUNER Agent (The Neural Pruner / Biomimicry Architect)
====================================================================
Strategy: Sparse Distributed Representations (SDR)
Action:   Classifies code chunks as FOREGROUND (moving parts, needs attention)
          or BACKGROUND (static/predictable, skip expensive analysis).

Research grounding:
  - Hawkins, J. & Ahmad, S. (2016). "Why Neurons Have Thousands of Synapses,
    a Theory of Sequence Memory in Neocortex." Frontiers in Neural Circuits.
    https://doi.org/10.3389/fncir.2016.00023
  - Ahmad, S. & Hawkins, J. (2016). "How Do Neurons Operate on Sparse
    Distributed Representations?" arXiv:1601.00720
  - Numenta HTM School: https://numenta.org/resources/htm-school/
  - Mapping: PI score ≈ column overlap score in HTM.
             HZS score ≈ proximal vs distal dendrite activation (recent vs predicted).

Like peripheral vision: the brain doesn't focus on static background.
PRUNER tells the other agents where to spend the expensive tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, FeatureSlice

# Import from existing InB engine
try:
    from src.inference_bridge import predictability_index, hot_zone_score
    _INB_AVAILABLE = True
except ImportError:
    _INB_AVAILABLE = False


class PrunerAgent(BaseAgent):
    """
    PRUNER: Sparse Distributed Representation classifier.

    Input channel:  pruner.in
    Output channel: pruner.out

    Classifies each code chunk as FOREGROUND or BACKGROUND using:
      - PI (Predictability Index) from InferenceBridge
      - HZS (Hot Zone Score) from InferenceBridge RetinalDelta
      - SDR overlap: simulated column activation density

    FOREGROUND = complex + recently changed = needs expensive processing
    BACKGROUND = predictable + cold = safe to skip or lighten
    """

    AGENT_ID   = "PRUNER"
    IN_CHANNEL  = "pruner.in"
    OUT_CHANNEL = "pruner.out"

    # SDR tuning — these mirror Numenta's ~2% sparsity target
    FOREGROUND_PI_THRESHOLD   = 0.70   # below this PI → function is complex → FOREGROUND
    FOREGROUND_HZS_THRESHOLD  = 0.10   # above this HZS → recently changed → FOREGROUND
    SDR_SPARSITY_TARGET       = 0.20   # target ~20% of chunks as FOREGROUND

    def process(self, slice_in: FeatureSlice) -> FeatureSlice:
        """
        Classify all chunks from the orchestrator input.
        Returns a slice with classified_chunks list.
        """
        payload     = slice_in.payload
        chunks      = payload.get("chunks", [])
        git_hot     = payload.get("git_hot_lines", {})  # {file_path: set[int]}
        file_path   = payload.get("file_path", "")
        language    = payload.get("language", "unknown")
        task_type   = payload.get("task_type", "unknown")
        question    = payload.get("question", "")
        rainbow_hit = payload.get("rainbow_hit", False)

        classified = []
        fg_count   = 0
        bg_count   = 0

        hot_lines = set()
        if git_hot and file_path:
            hot_lines = set(git_hot.get(file_path, []))

        for chunk in chunks:
            result = self._classify_chunk(chunk, hot_lines)
            classified.append(result)
            if result["zone"] == "FOREGROUND":
                fg_count += 1
            else:
                bg_count += 1

        total = len(chunks)
        fg_ratio = fg_count / total if total > 0 else 0.0

        decision = (
            f"{fg_count}/{total} FOREGROUND "
            f"({fg_ratio:.0%}), {bg_count} BACKGROUND. "
            f"lang={language}"
        )

        self.log_decision(
            decision_type  = "CLASSIFY",
            decision_value = {"foreground": fg_count, "background": bg_count, "total": total},
            rationale      = decision,
            confidence     = 1.0 - abs(fg_ratio - self.SDR_SPARSITY_TARGET),
            slice_id       = slice_in.slice_id,
        )

        lang_tag = f"lang.{language}" if f"lang.{language}" in self._lang_tags() else "lang.unknown"

        return self.emit(
            taxonomy_tags   = ["code.classification", "sdr.foreground", "sdr.background", "predictability", lang_tag],
            payload         = {
                "classified_chunks": classified,
                "foreground_count":  fg_count,
                "background_count":  bg_count,
                "total_chunks":      total,
                "file_path":         file_path,
                "language":          language,
                "task_type":         task_type,
                "question":          question,
                "rainbow_hit":       rainbow_hit,
            },
            confidence      = 0.90,
            decision        = decision,
            parent_slice_id = slice_in.slice_id,
            token_count     = sum(c.get("original_tokens", 0) for c in classified),
        )

    # ── Classification logic ───────────────────────────────────────────

    def _classify_chunk(self, chunk: Dict[str, Any], hot_lines: set) -> Dict[str, Any]:
        """Classify a single code chunk using PI + HZS + SDR overlap."""
        source      = chunk.get("source", "")
        start_line  = chunk.get("start_line", 0)
        chunk_type  = chunk.get("chunk_type", "function")
        chunk_id    = chunk.get("chunk_id", "")

        # Compute PI score via existing InB function
        if _INB_AVAILABLE and source:
            pi_score = predictability_index(source)
        else:
            # Fallback: line-count heuristic
            n = len([l for l in source.splitlines() if l.strip()])
            pi_score = max(0.0, min(1.0, 1.0 - n / 50.0))

        # Compute HZS score via existing InB function
        if _INB_AVAILABLE and hot_lines:
            hzs_score = hot_zone_score(start_line, hot_lines)
        else:
            hzs_score = 0.0

        # SDR overlap: simulate % active columns
        # High PI + cold = well-predicted background (like HTM predicted state)
        # Low PI + hot  = anomalous foreground (like HTM burst state)
        sdr_overlap = pi_score * (1.0 - hzs_score)

        # Classify: FOREGROUND if complex OR recently changed
        is_hot        = hzs_score > self.FOREGROUND_HZS_THRESHOLD
        is_complex    = pi_score  < self.FOREGROUND_PI_THRESHOLD
        is_foreground = is_hot or is_complex

        zone = "FOREGROUND" if is_foreground else "BACKGROUND"

        return {
            "chunk_id":               chunk_id,
            "chunk_type":             chunk_type,
            "start_line":             start_line,
            "end_line":               chunk.get("end_line", start_line),
            "zone":                   zone,
            "pi_score":               round(pi_score, 4),
            "hzs_score":              round(hzs_score, 4),
            "sdr_overlap":            round(sdr_overlap, 4),
            "is_hot":                 is_hot,
            "is_complex":             is_complex,
            "skip_expensive_analysis": zone == "BACKGROUND",
            "original_tokens":        chunk.get("original_tokens", 0),
            "source":                 source,
        }

    @staticmethod
    def _lang_tags() -> List[str]:
        return ["lang.python", "lang.c", "lang.go", "lang.rust", "lang.unknown"]


# ── CLI for manual inspection ──────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("PRUNER self-test — pushing a synthetic chunk through the bus...")

    from bus.message_bus import MessageBus
    bus = MessageBus()
    bus.clear("pruner.in")
    bus.clear("pruner.out")

    test_slice = FeatureSlice(
        agent_id      = "ORCHESTRATOR",
        taxonomy_tags = ["code.classification", "lang.python"],
        payload       = {
            "file_path": "test_module.py",
            "language":  "python",
            "chunks": [
                {
                    "chunk_id": "fn_init",
                    "chunk_type": "function",
                    "start_line": 10,
                    "end_line":   14,
                    "source": "self._db = db\nself._cache = {}\nself._ops = []",
                    "original_tokens": 15,
                },
                {
                    "chunk_id": "fn_complex",
                    "chunk_type": "function",
                    "start_line": 50,
                    "end_line":   90,
                    "source": "\n".join([f"line_{i} = process(data[{i}])" for i in range(40)]),
                    "original_tokens": 280,
                },
            ],
            "git_hot_lines": {"test_module.py": [51, 52, 53]},
        },
        confidence = 1.0,
        decision   = "orchestrator dispatch",
    )

    bus.publish("pruner.in", test_slice.to_dict())

    agent = PrunerAgent(bus=bus)
    result = agent.run_once(timeout=1.0)

    if result:
        print(f"\n  Decision: {result.decision}")
        for c in result.payload["classified_chunks"]:
            print(f"  {c['chunk_id']:<20} zone={c['zone']:<12} PI={c['pi_score']:.3f}  HZS={c['hzs_score']:.3f}")
    else:
        print("  No output (bus empty)")
