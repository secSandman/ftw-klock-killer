"""
local_llm.py — Ollama container harness for real-LLM RQS evaluation
=====================================================================
Spins up an Ollama Docker container, pulls a code-focused model, and
provides a simple ask() interface. Used by the benchmark to replace
TF cosine similarity with actual LLM response comparison.

Why this matters:
  The simulation benchmark measures RQS as TF cosine(original, compressed).
  This is anti-correlated with TER by construction — removing tokens always
  lowers TF overlap. Real-LLM RQS breaks this ceiling: if the compressed
  context contains enough signal for the model to produce an equivalent answer,
  RQS stays high regardless of how many tokens were removed.

  Expected TV with real LLM: 0.65–0.80 vs simulation ceiling of 0.528.

Prerequisites:
  - Docker Desktop running (Windows/Mac) or Docker daemon (Linux)
  - pip install ollama httpx

Usage (library):
  from experiments.local_llm import LocalLLMHarness
  h = LocalLLMHarness()
  h.ensure_running()          # start container if needed
  h.ensure_model()            # pull model if not cached
  response = h.ask("What does this function return?", context="def foo(): return 42")
  h.teardown()                # optional — container persists by default

Usage (CLI):
  python experiments/local_llm.py --setup               # pull + verify
  python experiments/local_llm.py --ask "what is this?" --file src/inference_bridge.py
  python experiments/local_llm.py --benchmark           # run real-LLM RQS benchmark
  python experiments/local_llm.py --teardown            # stop container
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Defaults ──────────────────────────────────────────────────────────────────

import os as _os

OLLAMA_IMAGE      = "ollama/ollama"
CONTAINER_NAME    = _os.environ.get("KLOC_LLM_CONTAINER",  "kloc-ollama")
OLLAMA_PORT       = int(_os.environ.get("KLOC_LLM_PORT",   "11434"))

# Default model tuned for RTX 5080 (16GB GDDR7) — 14B fits in VRAM with room to spare.
# Override via env: KLOC_LLM_MODEL=qwen2.5-coder:32b for spill-to-RAM run,
#                   KLOC_LLM_MODEL=qwen2.5-coder:7b for speed-over-quality.
DEFAULT_MODEL     = _os.environ.get("KLOC_LLM_MODEL",      "qwen2.5-coder:14b")

# Timeout is generous — 14B on RTX 5080 GDDR7 should respond in <5s per call.
# Set higher only for 32B spill-to-RAM runs.
REQUEST_TIMEOUT   = int(_os.environ.get("KLOC_LLM_TIMEOUT", "60"))
MAX_TOKENS        = int(_os.environ.get("KLOC_LLM_MAX_TOKENS", "400"))

# Smaller fallback for CPU-only machines where 14B is too slow
FAST_MODEL        = _os.environ.get("KLOC_LLM_FAST_MODEL", "qwen2.5-coder:1.5b")

# OLLAMA env vars forwarded into the container for performance tuning.
# OLLAMA_NUM_PARALLEL: concurrent request slots (safe to set to 2+ on 16GB VRAM)
# OLLAMA_MAX_LOADED_MODELS: keep model hot between calls (default 1)
# OLLAMA_FLASH_ATTENTION: enable FlashAttention2 (RTX 5080 supports it, ~30% speedup)
OLLAMA_CONTAINER_ENV = {
    "OLLAMA_NUM_PARALLEL":      _os.environ.get("KLOC_LLM_NUM_PARALLEL",    "2"),
    "OLLAMA_MAX_LOADED_MODELS": _os.environ.get("KLOC_LLM_MAX_MODELS",      "1"),
    "OLLAMA_FLASH_ATTENTION":   _os.environ.get("KLOC_LLM_FLASH_ATTENTION", "1"),
    "OLLAMA_KV_CACHE_TYPE":     _os.environ.get("KLOC_LLM_KV_CACHE_TYPE",   "q8_0"),
}

# System prompt used for all code Q&A calls
CODE_SYSTEM_PROMPT = (
    "You are a concise code analysis assistant. "
    "When given source code and a question, answer in 2-4 sentences. "
    "Focus only on what the code actually does — do not invent details. "
    "If you cannot determine the answer from the provided code, say so briefly."
)

# Standard questions for benchmark evaluation
BENCHMARK_QUESTIONS = [
    "What is the primary purpose of this code? Describe in one sentence.",
    "What are the main functions or methods, and what does each return?",
    "What data structures or types are passed in and returned?",
    "What conditions or inputs would cause this code to raise an error or fail?",
]


class LocalLLMHarness:
    """
    Manages the lifecycle of an Ollama container for real-LLM RQS evaluation.

    Designed for batch use: start once, ask many, optionally teardown.
    Container is named and persists across sessions by default so the
    model stays warm (avoids re-loading ~4GB weights each run).
    """

    def __init__(
        self,
        model:          str = DEFAULT_MODEL,
        container_name: str = CONTAINER_NAME,
        port:           int = OLLAMA_PORT,
        verbose:        bool = True,
    ):
        self.model              = model
        self.container_name     = container_name
        self.port               = port
        self.base_url           = f"http://localhost:{port}"
        self.verbose            = verbose
        self._client            = None   # lazy-init httpx client
        self._last_latency_ms   = None   # set after each ask() call

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def ensure_running(self, pull_image: bool = True) -> bool:
        """
        Start the Ollama container if it isn't running.
        Returns True if container is healthy after the call.
        """
        if self._container_running():
            if self.verbose:
                print(f"  [llm] Container '{self.container_name}' already running.")
            return self._wait_healthy()

        # Container exists but is stopped — restart it
        if self._container_exists():
            if self.verbose:
                print(f"  [llm] Restarting stopped container '{self.container_name}'...")
            subprocess.run(["docker", "start", self.container_name],
                           capture_output=True)
            return self._wait_healthy()

        # Container doesn't exist — create it
        if self.verbose:
            print(f"  [llm] Starting new Ollama container on port {self.port}...")
            if pull_image:
                print(f"  [llm] Pulling {OLLAMA_IMAGE} (first time only, ~1GB)...")

        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-v", "kloc-ollama-models:/root/.ollama",
            "-p", f"{self.port}:11434",
        ]
        # GPU passthrough — always attempt; fall back gracefully if unavailable
        cmd += ["--gpus", "all"] if self._docker_has_gpu() else []
        # Forward Ollama performance tuning env vars into the container
        for k, v in OLLAMA_CONTAINER_ENV.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [OLLAMA_IMAGE]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [llm] ERROR starting container: {result.stderr.strip()}")
            return False

        return self._wait_healthy(retries=15, interval=3)

    def ensure_model(self, model: Optional[str] = None) -> bool:
        """
        Pull the model if it isn't already cached in the container.
        Returns True on success.
        """
        model = model or self.model
        if self._model_cached(model):
            if self.verbose:
                print(f"  [llm] Model '{model}' already cached.")
            return True

        if self.verbose:
            m = model.lower()
            if "32b" in m or "34b" in m:   size_hint = "~19GB"
            elif "14b" in m or "13b" in m: size_hint = "~8GB"
            elif "7b" in m or "8b" in m:   size_hint = "~4GB"
            elif "3b" in m:                size_hint = "~2GB"
            else:                          size_hint = "~1GB"
            print(f"  [llm] Pulling '{model}' ({size_hint}, please wait)...")

        result = subprocess.run(
            ["docker", "exec", self.container_name, "ollama", "pull", model],
            capture_output=False,  # let pull progress stream to terminal
            text=True,
        )
        if result.returncode != 0:
            print(f"  [llm] ERROR pulling model '{model}'")
            return False

        if self.verbose:
            print(f"  [llm] Model '{model}' ready.")
        return True

    def setup(self, fast: bool = False) -> bool:
        """
        One-call setup: start container + pull model.
        `fast=True` pulls the smaller 1.5B model instead of 7B.
        """
        model = FAST_MODEL if fast else self.model
        self.model = model
        ok = self.ensure_running()
        if ok:
            ok = self.ensure_model(model)
        return ok

    def teardown(self, remove: bool = False) -> None:
        """Stop (and optionally remove) the container."""
        if self._container_running():
            subprocess.run(["docker", "stop", self.container_name],
                           capture_output=True)
            if self.verbose:
                print(f"  [llm] Container '{self.container_name}' stopped.")
        if remove and self._container_exists():
            subprocess.run(["docker", "rm", self.container_name],
                           capture_output=True)

    # ── Inference ─────────────────────────────────────────────────────────────

    def ask(
        self,
        question:    str,
        context:     str = "",
        system:      str = CODE_SYSTEM_PROMPT,
        max_tokens:  int = MAX_TOKENS,
        temperature: float = 0.1,
    ) -> str:
        """
        Ask the model a question, optionally with source code as context.
        Returns the model's response string, or "" on failure.

        temperature=0.1 for near-deterministic responses — reproducibility matters
        for RQS comparison (full vs bridge must be evaluated consistently).
        """
        user_msg = f"{context}\n\n{question}" if context else question
        payload  = {
            "model": self.model,
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user_msg},
            ],
            "stream": False,
            "options": {
                "temperature":  temperature,
                "num_predict":  max_tokens,
            },
        }

        try:
            import httpx
            t0 = time.time()
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                resp = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer ollama"},
                )
                resp.raise_for_status()
                data = resp.json()
            self._last_latency_ms = round((time.time() - t0) * 1000)
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            if self.verbose:
                print(f"  [llm] ask() failed: {exc}")
            self._last_latency_ms = None
            return ""

    def ask_batch(
        self,
        questions: list[str],
        context:   str = "",
        **kwargs,
    ) -> list[str]:
        """Ask multiple questions against the same context."""
        return [self.ask(q, context=context, **kwargs) for q in questions]

    # ── Health checks ─────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """True if the container is up and the API is responding."""
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _wait_healthy(self, retries: int = 20, interval: float = 2.0) -> bool:
        for i in range(retries):
            if self.health_check():
                if self.verbose:
                    print(f"  [llm] API healthy.")
                return True
            if self.verbose and i == 0:
                print(f"  [llm] Waiting for API to come up", end="", flush=True)
            if self.verbose:
                print(".", end="", flush=True)
            time.sleep(interval)
        if self.verbose:
            print(f"\n  [llm] ERROR: API did not become healthy after {retries} retries.")
        return False

    # ── Docker helpers ─────────────────────────────────────────────────────────

    def _container_running(self) -> bool:
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", self.container_name],
            capture_output=True, text=True,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"

    def _container_exists(self) -> bool:
        r = subprocess.run(
            ["docker", "inspect", self.container_name],
            capture_output=True, text=True,
        )
        return r.returncode == 0

    def _model_cached(self, model: str) -> bool:
        """Check if model is already pulled inside the container."""
        r = subprocess.run(
            ["docker", "exec", self.container_name, "ollama", "list"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return False
        # Match full model name including tag (e.g. "qwen2.5-coder:14b")
        # Fall back to checking base name only if no tag specified
        model_lower = model.lower()
        for line in r.stdout.splitlines():
            line_lower = line.lower()
            if model_lower in line_lower:
                return True
        return False

    @staticmethod
    def _docker_has_gpu() -> bool:
        """
        Probe whether Docker can actually pass through the NVIDIA GPU.
        Uses a real test container rather than checking daemon config — the
        config check is unreliable on Windows + Docker Desktop + WSL2.
        Result is cached for the process lifetime.
        """
        if LocalLLMHarness._gpu_available_cache is not None:
            return LocalLLMHarness._gpu_available_cache
        r = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.3.1-base-ubuntu22.04", "nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=30,
        )
        result = r.returncode == 0 and "GPU" in r.stdout
        LocalLLMHarness._gpu_available_cache = result
        return result

    _gpu_available_cache: Optional[bool] = None  # class-level cache

    @staticmethod
    def docker_available() -> bool:
        r = subprocess.run(["docker", "info"], capture_output=True)
        return r.returncode == 0

    def gpu_status(self) -> dict:
        """
        Return a dict describing GPU visibility inside Docker.
        Runs nvidia-smi inside a throwaway container.
        """
        r = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.3.1-base-ubuntu22.04", "nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return {"available": False, "error": r.stderr.strip() or "nvidia-smi failed"}
        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "vram": parts[1], "driver": parts[2]})
        return {"available": True, "gpus": gpus}

    def ollama_processor(self) -> str:
        """
        Ask a running Ollama container which processor it's using for the loaded model.
        Returns 'GPU', 'CPU', or 'unknown'.
        """
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/api/ps", timeout=5)
            data = r.json()
            models = data.get("models", [])
            if not models:
                return "no model loaded"
            processor = models[0].get("details", {}).get("processor", "")
            if not processor:
                # Older Ollama versions — infer from size_vram
                size_vram = models[0].get("size_vram", 0)
                return "GPU" if size_vram > 0 else "CPU"
            return processor
        except Exception:
            return "unknown"


# ── Real-LLM RQS computation ─────────────────────────────────────────────────

def compute_rqs_llm(
    full_src:   str,
    bridge_src: str,
    questions:  list[str] = BENCHMARK_QUESTIONS,
    harness:    Optional[LocalLLMHarness] = None,
    model:      str = DEFAULT_MODEL,
) -> dict:
    """
    Compute real-LLM RQS by comparing model responses to full vs compressed source.

    For each question:
      1. Ask model: question + full_src   → response_full
      2. Ask model: question + bridge_src → response_bridge
      3. Compute TF-IDF cosine(response_full, response_bridge)

    Returns a dict with:
      rqs_llm        — mean similarity across all questions
      per_question   — list of {question, sim, resp_full_len, resp_bridge_len}
      model          — model name used
    """
    import math, re

    if harness is None:
        harness = LocalLLMHarness(model=model)
        if not harness.health_check():
            return {"rqs_llm": None, "error": "Ollama not running — call harness.ensure_running() first"}

    def _tf(text: str) -> dict:
        words = re.findall(r"[a-zA-Z_]\w*|[0-9]+", text.lower())
        freq: dict = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return freq

    def _cosine(a: dict, b: dict) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        dot  = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        ma   = math.sqrt(sum(v**2 for v in a.values()))
        mb   = math.sqrt(sum(v**2 for v in b.values()))
        return dot / (ma * mb) if ma and mb else 0.0

    per_q     = []
    latencies = []
    for q in questions:
        resp_full   = harness.ask(q, context=full_src[:3000])
        lat_full    = harness._last_latency_ms
        resp_bridge = harness.ask(q, context=bridge_src[:3000])
        lat_bridge  = harness._last_latency_ms
        if not resp_full or not resp_bridge:
            continue
        # Accumulate combined latency for both asks per question
        pair_lat = (lat_full or 0) + (lat_bridge or 0)
        latencies.append(pair_lat)
        sim = round(_cosine(_tf(resp_full), _tf(resp_bridge)), 4)
        per_q.append({
            "question":         q,
            "sim":              sim,
            "resp_full_len":    len(resp_full.split()),
            "resp_bridge_len":  len(resp_bridge.split()),
        })

    rqs = round(sum(r["sim"] for r in per_q) / len(per_q), 4) if per_q else None
    mean_lat = round(sum(latencies) / len(latencies)) if latencies else None
    return {
        "rqs_llm":          rqs,
        "per_question":     per_q,
        "model":            harness.model,
        "mean_latency_ms":  mean_lat,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Local LLM harness for real-RQS evaluation")
    parser.add_argument("--setup",     action="store_true", help="Start container + pull model")
    parser.add_argument("--fast",      action="store_true", help="Use smaller 1.5B model")
    parser.add_argument("--teardown",  action="store_true", help="Stop the container")
    parser.add_argument("--status",    action="store_true", help="Show container + model status")
    parser.add_argument("--gpu-check", action="store_true", help="Verify GPU is visible inside Docker")
    parser.add_argument("--ask",       default=None,        help="Ask a question")
    parser.add_argument("--file",      default=None,        help="Source file to use as context")
    parser.add_argument("--benchmark", action="store_true", help="Run real-LLM RQS benchmark")
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not LocalLLMHarness.docker_available():
        print("  ERROR: Docker is not running or not installed.")
        print("  Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
        sys.exit(1)

    h = LocalLLMHarness(model=args.model)

    if args.teardown:
        h.teardown()
        sys.exit(0)

    if getattr(args, "gpu_check", False):
        print("\n  Checking NVIDIA GPU visibility inside Docker...")
        print("  (pulls a small CUDA base image on first run — ~200MB)\n")
        status = h.gpu_status()
        if not status["available"]:
            print(f"  GPU NOT visible inside Docker.")
            print(f"  Error: {status.get('error', 'unknown')}")
            print()
            print("  Fix checklist:")
            print("  1. Open WSL (wsl) and run:")
            print("       distribution=$(. /etc/os-release;echo $ID$VERSION_ID)")
            print("       curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg")
            print("       curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
            print("       sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit")
            print("       sudo nvidia-ctk runtime configure --runtime=docker")
            print("  2. Restart Docker Desktop")
            print("  3. Re-run this check")
            sys.exit(1)
        else:
            print("  GPU IS visible inside Docker:\n")
            for g in status.get("gpus", []):
                print(f"    {g['name']}  |  VRAM: {g['vram']}  |  Driver: {g['driver']}")
            print()
            print("  Model size guide for your VRAM:")
            for g in status.get("gpus", []):
                vram_str = g.get("vram", "0 MiB")
                try:
                    vram_mb = int(vram_str.split()[0].replace(",", ""))
                    if vram_mb >= 20000:
                        print(f"    {vram_mb//1024}GB → can run: 7B, 13B, 30B (4-bit) — try codellama:34b or llama3.1:70b-q4")
                    elif vram_mb >= 10000:
                        print(f"    {vram_mb//1024}GB → can run: 7B, 13B (4-bit) — try qwen2.5-coder:14b")
                    elif vram_mb >= 6000:
                        print(f"    {vram_mb//1024}GB → can run: 7B (4-bit) comfortably — qwen2.5-coder:7b recommended")
                    else:
                        print(f"    {vram_mb//1024}GB → use 1.5B–3B model — try: qwen2.5-coder:1.5b or phi3:mini")
                except (ValueError, IndexError):
                    pass
            sys.exit(0)

    if args.status:
        running = h._container_running()
        healthy = h.health_check() if running else False
        cached  = h._model_cached(args.model) if running else False
        print(f"  Container '{h.container_name}': {'running' if running else 'stopped'}")
        print(f"  API health:  {'ok' if healthy else 'not responding'}")
        print(f"  Model cache: {'yes' if cached else 'no'} ({args.model})")
        sys.exit(0)

    if args.setup:
        ok = h.setup(fast=args.fast)
        sys.exit(0 if ok else 1)

    if args.ask:
        if not h.health_check():
            print("  Container not running. Run: python experiments/local_llm.py --setup")
            sys.exit(1)
        context = ""
        if args.file:
            p = Path(args.file)
            if p.exists():
                context = p.read_text(encoding="utf-8", errors="replace")[:4000]
        resp = h.ask(args.ask, context=context)
        print(f"\n  Model: {h.model}")
        print(f"  Q: {args.ask}")
        print(f"  A: {resp}\n")
        sys.exit(0)

    if args.benchmark:
        if not h.health_check():
            print("  Container not running. Run: python experiments/local_llm.py --setup")
            sys.exit(1)
        print(f"\n  Real-LLM RQS benchmark using {h.model}")
        print(f"  Running full synthetic compression pipeline...\n")

        # Import and run the InB pipeline on synthetic files
        from src.benchmark import SYNTHETIC_FILES
        from src.inference_bridge import InferenceBridge

        bridge = InferenceBridge()
        total_sim = []

        for fname, src in SYNTHETIC_FILES:
            compressed = bridge.compress(src)
            result = compute_rqs_llm(src, compressed, harness=h)
            rqs = result.get("rqs_llm")
            if rqs is not None:
                total_sim.append(rqs)
                print(f"  {fname:<35} RQS-LLM = {rqs:.4f}")
                for pq in result["per_question"]:
                    print(f"    [{pq['sim']:.3f}] {pq['question'][:60]}")

        if total_sim:
            mean = sum(total_sim) / len(total_sim)
            print(f"\n  Mean RQS-LLM: {mean:.4f}  (simulation RQS-L1 baseline: 0.650)")
            print(f"  Model: {h.model}")

    parser.print_help()
