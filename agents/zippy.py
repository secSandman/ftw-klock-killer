"""
zippy.py — ZIPPY Agent (The Optimization Builder / AI Software Engineer)
=========================================================================
Strategy: Hard-coded state management + conversation compression.
Action:   "Zips" the full conversation history into a ≤50-token summary
          before each new turn. Hardcodes stable facts so they're never
          resent in full — they're referenced by a 3-token key instead.

The #1 cause of high LLM bills is resending the whole chat history on
every turn. ZIPPY kills this. Dead. Buried. Gone.

Research grounding:
  - Kolmogorov, A.N. (1965). "Three approaches to the quantitative definition
    of information." Problems of Information Transmission, 1(1), 1-7.
    Kolmogorov complexity: the shortest program that generates a string IS
    its information content. ZIPPY hunts for that shortest program.
  - Lempel, A. & Ziv, J. (1977). "A Universal Algorithm for Sequential Data
    Compression." IEEE Transactions on Information Theory 23(3), 337-343.
    LZ77/LZ78: the dictionary-based compression model ZIPPY's state store
    is literally implementing for conversation history.
  - Goodman, J. (2001). "A Bit of Progress in Language Modeling."
    MSR Technical Report. n-gram compression of sequences: prune low-
    probability (low-information) tokens from history.
  - Wu, Y. et al. (2024). "LLMLingua: Compressing Prompts for Accelerated
    Inference of Large Language Models." arXiv:2310.05736.
    Empirical: 20x compression with <5% quality loss on most tasks.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, FeatureSlice

try:
    from src.inference_bridge import CavemanCompressor, count_tokens
    _INB_AVAILABLE = True
except ImportError:
    _INB_AVAILABLE = False

_DEFAULT_STATE_PATH = Path(__file__).parent.parent / "bus" / "audit" / "state.json"
_MAX_ZIP_TOKENS     = 50    # the hard ceiling on history summary
_PRUNE_THRESHOLD    = 0.15  # turns with information density below this are dropped


class ZippyAgent(BaseAgent):
    """
    ZIPPY: Conversation state manager + history compressor.

    Input channel:  balancer.out
    Output channel: zippy.out

    Loads full conversation history from state_store.
    Compresses to ≤50 token "ZIP" using CavemanCompressor.
    Hardcodes stable facts into state store as key→short_value pairs.
    Prunes low-information turns (information density < threshold).
    Commits new state. Returns the ZIP + state delta.
    """

    AGENT_ID    = "ZIPPY"
    IN_CHANNEL  = "balancer.out"
    OUT_CHANNEL = "zippy.out"

    def __init__(self, state_path: str | Path = _DEFAULT_STATE_PATH, **kwargs):
        super().__init__(**kwargs)
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._caveman = CavemanCompressor() if _INB_AVAILABLE else None

    def process(self, slice_in: FeatureSlice) -> FeatureSlice:
        payload = slice_in.payload
        history = payload.get("conversation_history", [])
        session_id = payload.get("session_id", "default")

        # Load existing hardcoded state
        state = self._load_state(session_id)
        hardcoded = state.get("hardcoded", {})

        # Compress history → ZIP
        zip_text, pruned_turns, tokens_before, tokens_after, kr_ratio = (
            self._compress_history(history, hardcoded)
        )

        # Extract new facts to hardcode
        delta_state, patches = self._extract_facts(history, hardcoded)

        # Commit new state
        state["hardcoded"] = {**hardcoded, **delta_state}
        state["last_zip"]  = zip_text
        state["session_id"] = session_id
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._save_state(session_id, state)

        decision = (
            f"{len(history)} turns → {tokens_after} tok ZIP. "
            f"{len(delta_state)} facts hardcoded. "
            f"KR={kr_ratio:.3f}. "
            f"{len(pruned_turns)} turns pruned."
        )

        self.log_decision(
            decision_type  = "ZIP",
            decision_value = {
                "tokens_before": tokens_before,
                "tokens_after":  tokens_after,
                "kr_ratio":      kr_ratio,
                "pruned":        len(pruned_turns),
                "hardcoded":     len(delta_state),
            },
            rationale  = decision,
            confidence = 1.0 - kr_ratio,  # lower KR = better compression = higher confidence
            slice_id   = slice_in.slice_id,
        )

        return self.emit(
            taxonomy_tags   = ["state.hardcoded", "history.compressed", "kolmogorov", "session.zip", "turn.pruned"],
            payload         = {
                "zipped_history":    zip_text,
                "hardcoded_state":   state["hardcoded"],
                "delta_state":       delta_state,
                "state_patches":     patches,
                "pruned_turns":      pruned_turns,
                "tokens_before":     tokens_before,
                "tokens_after":      tokens_after,
                "kolmogorov_ratio":  kr_ratio,
                "session_id":        session_id,
                # Pass-through for orchestrator
                "routing_decision":  payload.get("routing_decision", {}),
                "compressed_chunks": payload.get("compressed_chunks", []),
                "question":          payload.get("question", ""),
                "task_type":         payload.get("task_type", "unknown"),
                "file_path":         payload.get("file_path", ""),
            },
            confidence      = 1.0 - kr_ratio,
            decision        = decision,
            parent_slice_id = slice_in.slice_id,
            token_count     = tokens_after,
        )

    # ── Compression internals ─────────────────────────────────────────

    def _compress_history(
        self,
        history: List[Dict],
        hardcoded: Dict[str, str],
    ) -> Tuple[str, List[int], int, int, float]:
        """
        Returns: (zip_text, pruned_turn_ids, tokens_before, tokens_after, kolmogorov_ratio)
        """
        if not history:
            return "no history", [], 0, 0, 0.0

        # Tokens before
        all_text = " ".join(
            f"{turn.get('role','?')}: {turn.get('content','')}"
            for turn in history
        )
        if _INB_AVAILABLE:
            tokens_before = count_tokens(all_text)
        else:
            tokens_before = len(all_text.split())

        # Step 1: Replace hardcoded facts with short keys
        substituted = self._substitute_hardcoded(all_text, hardcoded)

        # Step 2: Prune low-information turns
        pruned_turns = self._prune_turns(history)
        kept_history = [t for t in history if t.get("turn_id", -1) not in pruned_turns]

        # Step 3: Caveman compression on the kept text
        kept_text = " ".join(
            f"{t.get('role','?')}: {t.get('content','')}"
            for t in kept_history
        )
        kept_text = self._substitute_hardcoded(kept_text, hardcoded)

        if self._caveman:
            try:
                zipped = self._caveman.compress(kept_text)
            except Exception:
                zipped = kept_text
        else:
            zipped = self._naive_compress(kept_text)

        # Step 4: Hard-truncate to token budget
        zipped = self._truncate_to_budget(zipped, _MAX_ZIP_TOKENS)

        if _INB_AVAILABLE:
            tokens_after = count_tokens(zipped)
        else:
            tokens_after = len(zipped.split())

        kr_ratio = round(tokens_after / max(tokens_before, 1), 4)
        return zipped, pruned_turns, tokens_before, tokens_after, kr_ratio

    def _prune_turns(self, history: List[Dict]) -> List[int]:
        """
        Identify turns with low information density.
        Low density = short, generic, no code/identifiers.
        These are safe to drop from the ZIP.
        """
        pruned = []
        for turn in history:
            content = turn.get("content", "")
            turn_id = turn.get("turn_id", -1)
            if turn_id < 0:
                continue
            # Information density heuristic: ratio of identifiers to total words
            words = content.split()
            if len(words) < 3:
                pruned.append(turn_id)  # too short to matter
                continue
            # Count identifier-like tokens (camelCase, snake_case, code patterns)
            import re
            identifiers = re.findall(r'[a-zA-Z_]\w{3,}', content)
            density = len(identifiers) / len(words) if words else 0
            if density < _PRUNE_THRESHOLD and len(words) < 10:
                pruned.append(turn_id)
        return pruned

    def _extract_facts(
        self, history: List[Dict], existing_hardcoded: Dict[str, str]
    ) -> Tuple[Dict[str, str], List[Dict]]:
        """
        Extract stable facts from history for hardcoding.
        A "fact" is a key=value pattern that recurs across turns.
        """
        import re
        from collections import Counter

        # Count repeated tokens/phrases across turns
        all_text = " ".join(t.get("content", "") for t in history)
        tokens   = re.findall(r'[a-zA-Z_]\w{4,}', all_text)
        counts   = Counter(tokens)

        delta: Dict[str, str] = {}
        patches: List[Dict]   = []

        for token, count in counts.most_common(10):
            if count >= 3 and token not in existing_hardcoded and len(token) > 6:
                # Long repeated token → compress to short key
                short_key = token[:3] + str(len(token))
                delta[token] = short_key
                patches.append({
                    "key":              token,
                    "value":            short_key,
                    "source_turn":      -1,
                    "compression_method": "caveman",
                    "occurrences":      count,
                })

        return delta, patches

    def _substitute_hardcoded(self, text: str, hardcoded: Dict[str, str]) -> str:
        """Replace full forms with their hardcoded short keys."""
        import re
        for long_form, short_key in sorted(hardcoded.items(), key=lambda x: -len(x[0])):
            text = re.sub(r'\b' + re.escape(long_form) + r'\b', short_key, text)
        return text

    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Hard truncate text to token budget."""
        words = text.split()
        # Rough: 1 word ≈ 1.3 tokens for compressed caveman-speak
        max_words = int(max_tokens / 1.3)
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + " [...]"

    def _naive_compress(self, text: str) -> str:
        """Fallback compressor when InB not available."""
        # Remove stop words + shorten
        stop = {"the","a","an","is","are","was","were","be","been","have","has","had",
                "do","does","did","will","would","could","should","may","might","shall",
                "that","this","these","those","then","than","so","up","out","no","not"}
        words = [w for w in text.split() if w.lower() not in stop]
        return " ".join(words)

    # ── State store ───────────────────────────────────────────────────

    def _load_state(self, session_id: str) -> Dict[str, Any]:
        import json
        if not self.state_path.exists():
            return {}
        try:
            all_state = json.loads(self.state_path.read_text(encoding="utf-8"))
            return all_state.get(session_id, {})
        except Exception:
            return {}

    def _save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        import json
        all_state: Dict[str, Any] = {}
        if self.state_path.exists():
            try:
                all_state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        all_state[session_id] = state
        self.state_path.write_text(json.dumps(all_state, indent=2, ensure_ascii=False), encoding="utf-8")


# ── CLI self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    zippy = ZippyAgent()

    history = [
        {"role": "user",      "content": "What does the UserService.authenticate method do?",    "turn_id": 1},
        {"role": "assistant", "content": "The UserService.authenticate method validates credentials.", "turn_id": 2},
        {"role": "user",      "content": "Can you refactor UserService.authenticate?",            "turn_id": 3},
        {"role": "assistant", "content": "Sure, here is the refactored UserService implementation.", "turn_id": 4},
        {"role": "user",      "content": "ok",                                                    "turn_id": 5},
    ]

    zip_text, pruned, t_before, t_after, kr = zippy._compress_history(history, {})
    print(f"Tokens before: {t_before}")
    print(f"Tokens after:  {t_after}")
    print(f"KR ratio:      {kr:.3f}")
    print(f"Pruned turns:  {pruned}")
    print(f"ZIP:           {zip_text!r}")
