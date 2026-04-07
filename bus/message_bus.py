"""
message_bus.py — FTW-KLOC-KILLER Local Feature Bus
====================================================
JSONL file-based message queue. No broker. No network. No mercy.
Each channel is one append-only .jsonl file.

Channels:
  pruner.in   → orchestrator → PRUNER
  pruner.out  → PRUNER       → GRUG
  grug.out    → GRUG         → BALANCER
  balancer.out→ BALANCER     → ZIPPY
  zippy.out   → ZIPPY        → orchestrator
"""

from __future__ import annotations

import os
import sys
import time

if sys.platform != "win32":
    import fcntl  # type: ignore
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:
    import ujson as json
except ImportError:
    import json  # type: ignore

CHANNELS = [
    "pruner.in",
    "pruner.out",
    "grug.out",
    "balancer.out",
    "zippy.out",
]

_DEFAULT_QUEUE_DIR = Path(__file__).parent / "queues"


class MessageBus:
    """
    Local JSONL message bus.

    Thread/process safety: uses a simple cursor file per channel to track
    the read position. Works fine for single-process multi-agent pipelines.
    For concurrent agents, each agent opens its own MessageBus instance
    pointed at the same queue_dir.
    """

    def __init__(self, queue_dir: str | Path = _DEFAULT_QUEUE_DIR):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._cursors: Dict[str, int] = {}  # channel → byte offset

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    # Maximum number of lines a queue file may hold before it is rotated.
    # When the live file exceeds this threshold on write, it is renamed to
    # *.jsonl.1 (overwriting any prior .1) and a fresh *.jsonl is started.
    ROTATION_LINE_LIMIT = 10_000

    def publish(self, channel: str, slice_dict: Dict[str, Any]) -> None:
        """Append a FeatureSlice dict to the channel queue.

        Rotation (ISSUE-004): if the queue file already has >= ROTATION_LINE_LIMIT
        lines before this write, the current file is renamed to <name>.jsonl.1
        (overwriting any prior .1 archive) and a new empty <name>.jsonl is
        started.  The cursor is NOT reset — the in-memory / on-disk cursor
        continues to track the new file from line 0 after rotation.
        """
        self._assert_channel(channel)
        path = self._path(channel)

        # ── rotation check (ISSUE-004) ────────────────────────────────────
        if path.exists():
            try:
                line_count = sum(
                    1 for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                )
                if line_count >= self.ROTATION_LINE_LIMIT:
                    archive = path.with_suffix(".jsonl.1")
                    path.replace(archive)   # rename (overwrites any prior .1)
                    self._set_cursor(channel, 0)   # new file starts at line 0
            except OSError:
                pass  # rotation failure is non-fatal; continue writing

        line = json.dumps(slice_dict, ensure_ascii=False) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            if sys.platform != "win32":
                fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)

    def consume(self, channel: str, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Read the next unread message from channel.
        Returns None if queue is empty.
        Advances internal cursor so next call returns the following message.
        timeout: if > 0, poll until a message appears or timeout expires.
        """
        self._assert_channel(channel)
        deadline = time.time() + timeout
        while True:
            msg = self._read_next(channel)
            if msg is not None:
                return msg
            if time.time() >= deadline:
                return None
            time.sleep(0.05)

    def consume_all(self, channel: str) -> list[Dict[str, Any]]:
        """Drain all unread messages from channel."""
        msgs = []
        while True:
            m = self._read_next(channel)
            if m is None:
                break
            msgs.append(m)
        return msgs

    def peek(self, channel: str, n: int = 5) -> list[Dict[str, Any]]:
        """Return the last N messages from channel WITHOUT advancing the cursor."""
        path = self._path(channel)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines[-n:] if l.strip()]

    def reset_cursor(self, channel: str) -> None:
        """Rewind read cursor to beginning (useful for replay / debug)."""
        self._cursors[channel] = 0
        cur_path = self._cursor_path(channel)
        cur_path.write_text("0")

    def clear(self, channel: str) -> None:
        """Delete all messages in channel and reset cursor. Use with care."""
        self._path(channel).unlink(missing_ok=True)
        self.reset_cursor(channel)

    def stats(self) -> Dict[str, Dict]:
        """Return message counts for all channels."""
        result = {}
        for ch in CHANNELS:
            path = self._path(ch)
            if path.exists():
                lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
                cursor = self._get_cursor(ch)
                result[ch] = {"total": len(lines), "unread": max(0, len(lines) - cursor)}
            else:
                result[ch] = {"total": 0, "unread": 0}
        return result

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _path(self, channel: str) -> Path:
        return self.queue_dir / f"{channel.replace('.', '_')}.jsonl"

    def _cursor_path(self, channel: str) -> Path:
        return self.queue_dir / f"{channel.replace('.', '_')}.cursor"

    def _get_cursor(self, channel: str) -> int:
        if channel in self._cursors:
            return self._cursors[channel]
        cur_path = self._cursor_path(channel)
        if cur_path.exists():
            try:
                val = int(cur_path.read_text().strip())
            except (ValueError, OSError):
                val = 0
        else:
            val = 0
        self._cursors[channel] = val
        return val

    def _set_cursor(self, channel: str, value: int) -> None:
        self._cursors[channel] = value
        self._cursor_path(channel).write_text(str(value))

    def _read_next(self, channel: str) -> Optional[Dict[str, Any]]:
        path = self._path(channel)
        if not path.exists():
            return None
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        cursor = self._get_cursor(channel)
        if cursor >= len(lines):
            return None
        msg = json.loads(lines[cursor])
        self._set_cursor(channel, cursor + 1)
        return msg

    @staticmethod
    def _assert_channel(channel: str) -> None:
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel '{channel}'. Valid: {CHANNELS}")
