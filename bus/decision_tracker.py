"""
decision_tracker.py — Append-only decision audit log
=====================================================
Every agent decision is written here. Never truncated. Read with grep.
This is your black box recorder. When things go sideways, read this first.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

_DEFAULT_LOG = Path(__file__).parent / "audit" / "decisions.jsonl"


class DecisionTracker:
    """Append-only audit log for agent routing and compression decisions."""

    def __init__(self, log_path: str | Path = _DEFAULT_LOG):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        agent_id: str,
        decision_type: str,
        decision_value: Any,
        rationale: str,
        confidence: float = 1.0,
        slice_id: Optional[str] = None,
        parent_slice_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Append a decision record.

        Parameters
        ----------
        agent_id        : "PRUNER" | "GRUG" | "BALANCER" | "ZIPPY" | "ORCHESTRATOR"
        decision_type   : "CLASSIFY" | "COMPRESS" | "ROUTE" | "ZIP" | "ESCALATE" | "SKIP"
        decision_value  : the actual decision (tier number, "FOREGROUND", compression%, etc.)
        rationale       : short human-readable explanation (max 100 chars enforced)
        confidence      : 0.0–1.0
        slice_id        : the FeatureSlice this decision is attached to
        parent_slice_id : the upstream slice that triggered this
        metadata        : any extra agent-specific fields

        Returns the decision_id (uuid4).
        """
        decision_id = str(uuid.uuid4())[:8]
        record = {
            "decision_id":     decision_id,
            "agent_id":        agent_id,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "decision_type":   decision_type,
            "decision_value":  decision_value,
            "rationale":       rationale[:100],
            "confidence":      round(float(confidence), 4),
            "slice_id":        slice_id,
            "parent_slice_id": parent_slice_id,
            "metadata":        metadata or {},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return decision_id

    def tail(self, n: int = 20) -> list[Dict[str, Any]]:
        """Return the last N decisions."""
        if not self.log_path.exists():
            return []
        lines = [l for l in self.log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [json.loads(l) for l in lines[-n:]]

    def by_agent(self, agent_id: str) -> list[Dict[str, Any]]:
        """Return all decisions from a specific agent."""
        if not self.log_path.exists():
            return []
        results = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("agent_id") == agent_id:
                results.append(rec)
        return results

    def summary(self) -> Dict[str, Any]:
        """Return aggregate stats across all decisions."""
        if not self.log_path.exists():
            return {"total": 0, "by_agent": {}, "by_type": {}}
        records = [json.loads(l) for l in self.log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_agent: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        for r in records:
            by_agent[r["agent_id"]] = by_agent.get(r["agent_id"], 0) + 1
            by_type[r["decision_type"]] = by_type.get(r["decision_type"], 0) + 1
        return {"total": len(records), "by_agent": by_agent, "by_type": by_type}
