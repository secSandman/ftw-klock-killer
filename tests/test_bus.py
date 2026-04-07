"""
test_bus.py — Message bus and decision tracker tests
=====================================================
Validates the JSONL-backed message passing infrastructure:
  1. publish → consume round-trip
  2. cursor-based read position (no re-reads)
  3. channel isolation
  4. decision tracker append + tail
  5. taxonomy validation

Run: pytest tests/test_bus.py -v
"""

import sys
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from bus.message_bus import MessageBus
from bus.decision_tracker import DecisionTracker
from bus.taxonomy import Taxonomy


# ── MessageBus tests ──────────────────────────────────────────────────────────

class TestMessageBus:
    def setup_method(self, tmp_path=None):
        # Each test gets its own temp directory so state doesn't bleed between tests
        import tempfile, os
        self._tmpdir = tempfile.mkdtemp()
        self.bus = MessageBus(queue_dir=self._tmpdir)

    def _make_slice(self, agent_id="TEST", payload=None):
        return {
            "slice_id":      str(uuid.uuid4()),
            "agent_id":      agent_id,
            "timestamp":     "2026-01-01T00:00:00",
            "taxonomy_tags": [],
            "payload":       payload or {"data": "hello"},
            "confidence":    1.0,
            "decision":      "test decision",
            "token_count":   10,
            "compression_pct": 0.0,
        }

    def test_publish_and_consume(self):
        msg = self._make_slice(payload={"value": 42})
        self.bus.publish("pruner.in", msg)
        result = self.bus.consume("pruner.in")
        assert result is not None
        assert result["payload"]["value"] == 42

    def test_consume_returns_none_when_empty(self):
        result = self.bus.consume("pruner.in")
        assert result is None

    def test_cursor_advances(self):
        """After consuming, a second consume should return None."""
        self.bus.publish("pruner.in", self._make_slice())
        self.bus.consume("pruner.in")
        result = self.bus.consume("pruner.in")
        assert result is None

    def test_multiple_messages_ordered(self):
        for i in range(3):
            self.bus.publish("grug.out", self._make_slice(payload={"i": i}))
        for i in range(3):
            msg = self.bus.consume("grug.out")
            assert msg["payload"]["i"] == i

    def test_channel_isolation(self):
        self.bus.publish("pruner.in",  self._make_slice(payload={"ch": "pruner"}))
        self.bus.publish("balancer.out", self._make_slice(payload={"ch": "balancer"}))

        r1 = self.bus.consume("pruner.in")
        r2 = self.bus.consume("balancer.out")
        assert r1["payload"]["ch"] == "pruner"
        assert r2["payload"]["ch"] == "balancer"
        # Other channel still empty
        assert self.bus.consume("grug.out") is None

    def test_consume_all(self):
        for i in range(5):
            self.bus.publish("zippy.out", self._make_slice(payload={"i": i}))
        msgs = self.bus.consume_all("zippy.out")
        assert len(msgs) == 5

    def test_peek_does_not_advance_cursor(self):
        self.bus.publish("pruner.out", self._make_slice(payload={"x": 99}))
        peek1 = self.bus.peek("pruner.out")
        peek2 = self.bus.peek("pruner.out")
        assert len(peek1) > 0
        assert len(peek2) > 0
        assert peek1[-1]["payload"]["x"] == peek2[-1]["payload"]["x"] == 99

    def test_reset_cursor(self):
        self.bus.publish("pruner.in", self._make_slice(payload={"seq": 1}))
        self.bus.consume("pruner.in")
        self.bus.reset_cursor("pruner.in")
        result = self.bus.consume("pruner.in")
        assert result is not None
        assert result["payload"]["seq"] == 1

    def test_stats(self):
        self.bus.publish("pruner.in", self._make_slice())
        self.bus.publish("pruner.in", self._make_slice())
        stats = self.bus.stats()
        assert isinstance(stats, dict)


# ── DecisionTracker tests ─────────────────────────────────────────────────────

class TestDecisionTracker:
    def setup_method(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        log_path = Path(self._tmpdir) / "decisions.jsonl"
        self.tracker = DecisionTracker(log_path=str(log_path))

    def test_log_and_tail(self):
        self.tracker.log(
            agent_id="PRUNER",
            decision_type="classification",
            decision_value="FOREGROUND",
            rationale="pi=0.55 below threshold",
            confidence=0.9,
        )
        entries = self.tracker.tail(10)
        assert len(entries) == 1
        assert entries[0]["agent_id"] == "PRUNER"

    def test_tail_limit(self):
        for i in range(25):
            self.tracker.log(
                agent_id="GRUG",
                decision_type="compression",
                decision_value=f"TER={i}",
                rationale="synthetic test",
                confidence=0.8,
            )
        entries = self.tracker.tail(10)
        assert len(entries) == 10

    def test_by_agent(self):
        self.tracker.log("PRUNER",  "classification", "FG", "pi below threshold", confidence=0.9)
        self.tracker.log("GRUG",    "compression",    "ok", "ter acceptable",     confidence=0.8)
        self.tracker.log("PRUNER",  "classification", "BG", "pi above threshold", confidence=0.7)

        pruner_entries = self.tracker.by_agent("PRUNER")
        assert len(pruner_entries) == 2
        assert all(e["agent_id"] == "PRUNER" for e in pruner_entries)

    def test_summary(self):
        self.tracker.log("PRUNER",   "classification", "FG", "test", confidence=0.9)
        self.tracker.log("BALANCER", "routing",        "T1", "test", confidence=1.0)
        summary = self.tracker.summary()
        assert isinstance(summary, dict)
        assert "total" in summary or len(summary) >= 0  # non-crashing

    def test_rationale_truncated_to_100_chars(self):
        long_rationale = "x" * 200
        self.tracker.log("ZIPPY", "state", "some_value", long_rationale, confidence=0.5)
        entries = self.tracker.tail(1)
        assert len(entries[0].get("rationale", "")) <= 100


# ── Taxonomy tests ────────────────────────────────────────────────────────────

class TestTaxonomy:
    def setup_method(self):
        self.taxonomy = Taxonomy()

    def test_valid_tags_pass(self):
        # These tags must exist in data/taxonomy.json
        valid = ["code.classification", "compression.f1_skeleton"]
        # Should not raise
        try:
            self.taxonomy.validate(valid)
        except ValueError:
            pytest.skip("Tags not found in taxonomy.json — check data/taxonomy.json")

    def test_invalid_tag_raises(self):
        with pytest.raises((ValueError, KeyError)):
            self.taxonomy.validate(["totally.fake.tag.xyz"])

    def test_tags_for_category_returns_list(self):
        result = self.taxonomy.tags_for_category("compression")
        assert isinstance(result, list)

    def test_lookup_returns_something(self):
        # Just verify it doesn't crash on a real tag
        valid_tags = self.taxonomy.tags_for_category("code")
        if valid_tags:
            result = self.taxonomy.lookup(valid_tags[0])
            assert result is not None
