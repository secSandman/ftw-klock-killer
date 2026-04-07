"""
base_agent.py — Abstract BaseAgent + FeatureSlice schema
=========================================================
Every agent in the crew inherits from BaseAgent.
Every message on the bus is a FeatureSlice.

Research grounding:
  - Sparse Distributed Representations (Hawkins, Numenta)
  - Information Theory (Shannon 1948)
  - Multi-Agent Systems (Wooldridge & Jennings 1995)
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, field_validator
    PYDANTIC = True
except ImportError:
    PYDANTIC = False
    BaseModel = object  # type: ignore

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from bus.message_bus import MessageBus
from bus.decision_tracker import DecisionTracker
from bus.taxonomy import Taxonomy


# ─────────────────────────────────────────────────────────────────────────────
# FeatureSlice — the universal message format
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSlice:
    """
    The atomic unit of communication between agents.

    Every message on the bus is a FeatureSlice.
    Inspired by HTM Sparse Distributed Representations — a slice carries
    only the "active columns" (high-information features), not the full source.
    """
    __slots__ = [
        "agent_id", "slice_id", "parent_slice_id", "timestamp",
        "taxonomy_tags", "payload", "confidence", "decision",
        "token_count", "compression_pct",
    ]

    def __init__(
        self,
        agent_id: str,
        taxonomy_tags: List[str],
        payload: Dict[str, Any],
        confidence: float = 1.0,
        decision: str = "",
        parent_slice_id: Optional[str] = None,
        token_count: int = 0,
        compression_pct: Optional[float] = None,
    ):
        self.agent_id        = agent_id
        self.slice_id        = str(uuid.uuid4())[:8]
        self.parent_slice_id = parent_slice_id
        self.timestamp       = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.taxonomy_tags   = taxonomy_tags
        self.payload         = payload
        self.confidence      = round(float(confidence), 4)
        self.decision        = decision[:100]  # enforce 100-char max
        self.token_count     = token_count
        self.compression_pct = compression_pct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":        self.agent_id,
            "slice_id":        self.slice_id,
            "parent_slice_id": self.parent_slice_id,
            "timestamp":       self.timestamp,
            "taxonomy_tags":   self.taxonomy_tags,
            "payload":         self.payload,
            "confidence":      self.confidence,
            "decision":        self.decision,
            "token_count":     self.token_count,
            "compression_pct": self.compression_pct,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSlice":
        s = cls(
            agent_id        = d["agent_id"],
            taxonomy_tags   = d.get("taxonomy_tags", []),
            payload         = d.get("payload", {}),
            confidence      = d.get("confidence", 1.0),
            decision        = d.get("decision", ""),
            parent_slice_id = d.get("parent_slice_id"),
            token_count     = d.get("token_count", 0),
            compression_pct = d.get("compression_pct"),
        )
        s.slice_id  = d.get("slice_id", s.slice_id)
        s.timestamp = d.get("timestamp", s.timestamp)
        return s

    def __repr__(self) -> str:
        return (
            f"FeatureSlice(agent={self.agent_id!r}, id={self.slice_id!r}, "
            f"tags={self.taxonomy_tags}, confidence={self.confidence}, "
            f"decision={self.decision!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BaseAgent
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all KLOC-KILLER agents.

    Subclasses implement `process(slice_in) -> FeatureSlice`.
    BaseAgent handles: bus subscribe/publish, taxonomy validation,
    decision logging, and error isolation.

    Usage:
        class PrunerAgent(BaseAgent):
            AGENT_ID = "PRUNER"
            IN_CHANNEL = "pruner.in"
            OUT_CHANNEL = "pruner.out"

            def process(self, slice_in):
                ...
                return self.emit(tags, payload, confidence, decision, slice_in.slice_id)
    """

    AGENT_ID: str = "UNKNOWN"
    IN_CHANNEL: str = ""
    OUT_CHANNEL: str = ""

    def __init__(
        self,
        bus: Optional[MessageBus] = None,
        tracker: Optional[DecisionTracker] = None,
        taxonomy: Optional[Taxonomy] = None,
    ):
        self.bus      = bus or MessageBus()
        self.tracker  = tracker or DecisionTracker()
        self.taxonomy = taxonomy or Taxonomy()
        self._run_count = 0

    # ── Core loop ──────────────────────────────────────────────────────

    def run_once(self, timeout: float = 0.0) -> Optional[FeatureSlice]:
        """
        Consume one slice from IN_CHANNEL, process it, publish to OUT_CHANNEL.
        Returns the output FeatureSlice, or None if no input was available.
        """
        raw = self.bus.consume(self.IN_CHANNEL, timeout=timeout)
        if raw is None:
            return None
        slice_in = FeatureSlice.from_dict(raw)
        try:
            slice_out = self.process(slice_in)
        except Exception as exc:
            slice_out = self._error_slice(slice_in, exc)
        self.bus.publish(self.OUT_CHANNEL, slice_out.to_dict())
        self._run_count += 1
        return slice_out

    def run_loop(self, max_iterations: Optional[int] = None, poll_interval: float = 0.1) -> None:
        """Run continuously until stopped or max_iterations reached."""
        i = 0
        while True:
            self.run_once(timeout=poll_interval)
            i += 1
            if max_iterations and i >= max_iterations:
                break

    @abstractmethod
    def process(self, slice_in: FeatureSlice) -> FeatureSlice:
        """
        Core agent logic. Receives a FeatureSlice, returns a FeatureSlice.
        Must call self.emit() to construct the output slice.
        """
        ...

    # ── Helpers ────────────────────────────────────────────────────────

    def emit(
        self,
        taxonomy_tags: List[str],
        payload: Dict[str, Any],
        confidence: float = 1.0,
        decision: str = "",
        parent_slice_id: Optional[str] = None,
        token_count: int = 0,
        compression_pct: Optional[float] = None,
    ) -> FeatureSlice:
        """Validate taxonomy tags and construct an output FeatureSlice."""
        self.taxonomy.validate(taxonomy_tags)
        return FeatureSlice(
            agent_id        = self.AGENT_ID,
            taxonomy_tags   = taxonomy_tags,
            payload         = payload,
            confidence      = confidence,
            decision        = decision,
            parent_slice_id = parent_slice_id,
            token_count     = token_count,
            compression_pct = compression_pct,
        )

    def log_decision(
        self,
        decision_type: str,
        decision_value: Any,
        rationale: str,
        confidence: float = 1.0,
        slice_id: Optional[str] = None,
        parent_slice_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Log a decision to the audit trail."""
        return self.tracker.log(
            agent_id        = self.AGENT_ID,
            decision_type   = decision_type,
            decision_value  = decision_value,
            rationale       = rationale,
            confidence      = confidence,
            slice_id        = slice_id,
            parent_slice_id = parent_slice_id,
            metadata        = metadata,
        )

    def _error_slice(self, slice_in: FeatureSlice, exc: Exception) -> FeatureSlice:
        """Produce an error slice so the pipeline doesn't deadlock on exceptions."""
        self.log_decision(
            decision_type  = "ERROR",
            decision_value = str(exc)[:100],
            rationale      = f"{self.AGENT_ID} raised {type(exc).__name__}",
            confidence     = 0.0,
            slice_id       = slice_in.slice_id,
        )
        return self.emit(
            taxonomy_tags   = ["quality.fail"],
            payload         = {"error": str(exc), "input_slice_id": slice_in.slice_id},
            confidence      = 0.0,
            decision        = f"ERROR: {str(exc)[:80]}",
            parent_slice_id = slice_in.slice_id,
        )

    def status(self) -> Dict[str, Any]:
        return {
            "agent_id":    self.AGENT_ID,
            "in_channel":  self.IN_CHANNEL,
            "out_channel": self.OUT_CHANNEL,
            "run_count":   self._run_count,
        }
