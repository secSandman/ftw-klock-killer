"""
state_store.py — Persistent session state for ZIPPY.
JSON-backed key/value store per session_id.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict

_DEFAULT_PATH = Path(os.getenv("STATE_STORE_PATH", "bus/audit/state.json"))


class StateStore:
    def __init__(self, path: str | Path = _DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str = "default") -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8")).get(session_id, {})
        except Exception:
            return {}

    def save(self, session_id: str, state: Dict[str, Any]) -> None:
        all_state: Dict[str, Any] = {}
        if self.path.exists():
            try:
                all_state = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        all_state[session_id] = state
        # Atomic write: write to .tmp then rename — prevents partial-write corruption
        # and is safe for single-machine concurrent access (os.replace is atomic on
        # POSIX and near-atomic on Windows NTFS).
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(all_state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, session_id: str, key: str, default=None):
        return self.load(session_id).get(key, default)

    def set(self, session_id: str, key: str, value: Any) -> None:
        state = self.load(session_id)
        state[key] = value
        self.save(session_id, state)

    def clear(self, session_id: str) -> None:
        self.save(session_id, {})
