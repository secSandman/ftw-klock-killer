"""
pipeline.py — Main orchestrator: wires all 4 agents in sequence.

Usage:
    from orchestrator.pipeline import Pipeline
    pipe = Pipeline()
    result = pipe.run(file_path="doom/src/r_plane.c", question="explain BSP traversal")
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from bus.message_bus import MessageBus
from bus.decision_tracker import DecisionTracker
from bus.taxonomy import Taxonomy
from agents.base_agent import FeatureSlice
from agents.pruner import PrunerAgent
from agents.grug import GrugAgent
from agents.balancer import BalancerAgent
from agents.zippy import ZippyAgent
from rag.chunker import Chunker
from rag.hasher import Hasher
from languages.language_registry import LanguageRegistry
from orchestrator.state_store import StateStore
from orchestrator.llm_caller import call_llm


class Pipeline:
    """
    Orchestrates the 4-agent pipeline:
      chunker → PRUNER → GRUG → BALANCER → ZIPPY → LLM

    Each agent reads from its IN_CHANNEL and writes to its OUT_CHANNEL.
    The orchestrator publishes the initial slice and reads the final output.
    """

    def __init__(self, session_id: str = "default", tier_max: int = 2):
        self.session_id = session_id
        self.bus        = MessageBus()
        self.tracker    = DecisionTracker()
        self.taxonomy   = Taxonomy()
        self.state      = StateStore()
        self.chunker    = Chunker()
        self.hasher     = Hasher()
        self.registry   = LanguageRegistry()

        # Instantiate agents with shared infrastructure
        kwargs = dict(bus=self.bus, tracker=self.tracker, taxonomy=self.taxonomy)
        self.pruner   = PrunerAgent(**kwargs)
        self.grug     = GrugAgent(**kwargs)
        self.balancer = BalancerAgent(tier_max=tier_max, **kwargs)
        self.zippy    = ZippyAgent(**kwargs)

    def run(
        self,
        file_path: str,
        question: str = "explain this code",
        task_type: str = "explain",
        conversation_history: list = None,
        git_hot_lines: dict = None,
        rag_chunk_ids: set = None,
        top_k: int = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline run. Returns:
          {
            routing_decision, compressed_chunks, zipped_history,
            hardcoded_state, question, task_type,
            decision_audit: [last 20 decisions],
            elapsed_s: float,
            rag_chunks_used: int   (only present when top_k filtering occurred),
            rag_total_chunks: int  (only present when top_k filtering occurred),
          }

        rag_chunk_ids : optional set of chunk_id strings pre-selected by the
                        RAG retriever.  When provided, only those chunks are
                        forwarded into the agent chain (the rest are dropped
                        before PRUNER sees them).

        top_k         : when set, RAG-filter chunks to the top-k most relevant
                        to *question* using BM25 (force_bm25=True — no
                        sentence-transformers dependency at runtime).
                        Ignored when top_k is None (pipeline runs unchanged).
                        # DECISION: top_k=None keeps the no-question path
                        # identical so existing callers are not affected.
        """
        t0 = time.time()
        p  = Path(file_path)
        language = self.registry.language_for(p) or "python"

        # 1. Chunk the file
        chunks = self.chunker.chunk_file(p)
        if not chunks:
            # Try treating file_path as source snippet
            chunks = self.chunker.chunk_source(
                p.read_text(encoding="utf-8", errors="replace") if p.exists() else file_path,
                file_path, language
            )

        # Track RAG metadata for result dict (populated below if filtering occurs)
        _rag_total   = len(chunks)
        _rag_used    = None   # None means no filtering happened

        # 1b. Inline RAG filter — when top_k is provided, rank chunks against
        #     the question and keep only the top-k most relevant ones.
        #     Prefers metadata-augmented field-weighted BM25 when a metadata
        #     index exists for the repo (.kloc/metadata_index.json).
        #     Falls back to plain BM25 (force_bm25=True) — zero runtime
        #     dependency on sentence-transformers; works fully offline.
        if top_k is not None and question and question.strip():
            from rag.retriever import RAGRetriever
            # Try to locate metadata index relative to the file being queried
            _repo_hint = Path(file_path).parent
            while _repo_hint != _repo_hint.parent:
                if (_repo_hint / ".kloc" / "metadata_index.json").exists():
                    break
                _repo_hint = _repo_hint.parent
            else:
                _repo_hint = None

            retriever = RAGRetriever(
                force_bm25  = True,
                repo_path   = _repo_hint,
            )
            selected  = retriever.retrieve(
                question=question, chunks=chunks, top_k=top_k
            )
            # Safety: never produce an empty payload
            if selected:
                chunks   = selected
                _rag_used = len(chunks)
            self.tracker.log(
                agent_id       = "ORCHESTRATOR",
                decision_type  = "RAG_FILTER",
                decision_value = f"{len(chunks)}/{_rag_total} chunks kept",
                rationale      = (
                    f"Inline RAG top-{top_k} BM25 filter: "
                    f"kept {len(chunks)} of {_rag_total} chunks"
                ),
                confidence     = 1.0,
            )

        # 1c. Legacy RAG pre-filter — keep only the retrieved chunks when
        #     a pre-computed set of chunk_ids is passed by the caller.
        elif rag_chunk_ids:
            total_before = len(chunks)
            filtered = [c for c in chunks if c.chunk_id in rag_chunk_ids]
            # Safety: never produce an empty payload (fall back to all chunks)
            chunks = filtered if filtered else chunks
            _rag_used = len(chunks)
            self.tracker.log(
                agent_id       = "ORCHESTRATOR",
                decision_type  = "RAG_FILTER",
                decision_value = f"{len(chunks)}/{total_before} chunks kept",
                rationale      = f"RAG pre-filter: kept {len(chunks)} of {total_before} chunks",
                confidence     = 1.0,
            )

        # 2. Hash + rainbow check
        for chunk in chunks:
            h = self.hasher.hash_chunk(chunk)
            chunk.__dict__["sha256"]      = h["sha256"]
            chunk.__dict__["rainbow_hit"] = h["rainbow_hit"]

        rainbow_hit = any(c.__dict__.get("rainbow_hit") for c in chunks)

        # 3. Build orchestrator payload and publish to pruner.in
        # task_type, question, and rainbow_hit are injected here so they
        # flow through the entire agent chain without needing re-publish mutations.
        pruner_payload = self.chunker.to_pruner_payload(chunks, file_path, language)
        pruner_payload["git_hot_lines"] = git_hot_lines or {}
        pruner_payload["task_type"]     = task_type
        pruner_payload["question"]      = question
        pruner_payload["rainbow_hit"]   = rainbow_hit

        init_slice = FeatureSlice(
            agent_id      = "ORCHESTRATOR",
            taxonomy_tags = ["code.classification", f"lang.{language}"],
            payload       = pruner_payload,
            confidence    = 1.0,
            decision      = f"dispatch {p.name} {len(chunks)} chunks",
        )

        self.bus.publish("pruner.in", init_slice.to_dict())

        # 4. Run agents in sequence — each agent reads and forwards context fields
        pruner_out = self.pruner.run_once(timeout=2.0)
        if pruner_out is None or "error" in pruner_out.payload:
            err = pruner_out.payload.get("error", "timeout") if pruner_out else "timeout"
            return self._error(f"PRUNER failed: {err}")

        grug_out = self.grug.run_once(timeout=10.0)
        if grug_out is None or "error" in grug_out.payload:
            err = grug_out.payload.get("error", "timeout") if grug_out else "timeout"
            return self._error(f"GRUG failed: {err}")

        balancer_out = self.balancer.run_once(timeout=2.0)
        if balancer_out is None or "error" in balancer_out.payload:
            err = balancer_out.payload.get("error", "timeout") if balancer_out else "timeout"
            return self._error(f"BALANCER failed: {err}")

        # Act on auto_escalate: if RQS is below floor and headroom exists,
        # bump tier in the routing decision before ZIPPY sees it.
        # MVP1: pre-inference tier bump only (post-inference cascade is TODO).
        escalation = balancer_out.payload.get("escalation_policy", {})
        if escalation.get("auto_escalate"):
            rd   = balancer_out.payload.get("routing_decision", {})
            cur  = rd.get("tier", 1)
            next_tier = min(cur + 1, self.balancer.tier_max)
            if next_tier > cur:
                from agents.balancer import TIER_CONFIG
                rd["tier"]      = next_tier
                rd["tier_name"] = TIER_CONFIG[next_tier]["name"]
                rd["model_id"]  = self.balancer._pick_model(next_tier)
                rd["reason"]    = rd.get("reason", "") + f" | AUTO_ESCALATED T{cur}→T{next_tier}"

        # Inject conversation history for ZIPPY
        balancer_out.payload["conversation_history"] = conversation_history or []
        balancer_out.payload["session_id"] = self.session_id
        self.bus.publish("balancer.out", balancer_out.to_dict())

        zippy_out = self.zippy.run_once(timeout=5.0)
        if zippy_out is None or "error" in zippy_out.payload:
            err = zippy_out.payload.get("error", "timeout") if zippy_out else "timeout"
            return self._error(f"ZIPPY failed: {err}")

        elapsed = round(time.time() - t0, 3)

        result = {
            "routing_decision":  zippy_out.payload.get("routing_decision", {}),
            "compressed_chunks": zippy_out.payload.get("compressed_chunks", []),
            "zipped_history":    zippy_out.payload.get("zipped_history", ""),
            "hardcoded_state":   zippy_out.payload.get("hardcoded_state", {}),
            "question":          question,
            "task_type":         task_type,
            "overall_ter":       grug_out.payload.get("overall_ter", 0),
            "rqs_l1_score":      grug_out.payload.get("rqs_l1_score", 1.0),
            # ISSUE-003: carry RHD delta so llm_caller can unmask [§:hash] tokens
            "rhd_registry_delta": grug_out.payload.get("rhd_registry_delta", {}),
            "decision_audit":    self.tracker.tail(20),
            "elapsed_s":         elapsed,
        }

        # Attach RAG filtering metadata when filtering was applied
        if _rag_used is not None:
            result["rag_chunks_used"]  = _rag_used
            result["rag_total_chunks"] = _rag_total

        # 5. Make the actual LLM call with the compressed context.
        # T0.5: llm_caller tries local Ollama first (free), falls through to
        # Anthropic only if unavailable or quality gate fails.
        # TODO: add post-inference RQS-L1 scoring + cascade escalation loop.
        # ISSUE-003: unmask [§:hash] markers in the LLM response — handled in
        # llm_caller._unmask_response() using ChromatophoricMasker + C_IncludeMasker.
        try:
            # Inject C masker instance (if present) so llm_caller can unmask
            # [§:hash] tokens produced by C_IncludeMasker (ISSUE-003).
            from agents.grug import _c_compressor_cache
            if _c_compressor_cache:
                # Use the first cached compressor — one repo per pipeline run.
                result["_c_masker"] = next(iter(_c_compressor_cache.values())).masker
            result["llm_response"] = call_llm(result)
        except Exception as exc:
            result["llm_response"] = f"[LLM ERROR] {type(exc).__name__}: {exc}"

        # Expose local LLM routing metadata (populated by llm_caller._try_local_llm)
        result["local_llm_accepted"]  = result.pop("local_llm_accepted",  False)
        result["local_llm_attempted"] = result.pop("local_llm_attempted", False)
        if result.get("local_llm_accepted"):
            result["local_llm_model"] = result.pop("local_llm_model", None)
            result["local_llm_tier"]  = result.pop("local_llm_tier",  "T0.5/LOCAL_LLM")

        return result

    @staticmethod
    def _error(msg: str) -> Dict[str, Any]:
        return {"error": msg, "routing_decision": {}, "compressed_chunks": [], "elapsed_s": 0}
