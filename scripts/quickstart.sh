#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  quickstart.sh — Zero to benchmark in 5 commands                        ║
# ║  For the poor soul who just sat down with a C video game and wants       ║
# ║  to know how expensive their LLM context is about to be.                 ║
# ║                                                                          ║
# ║  Usage:                                                                  ║
# ║    bash scripts/quickstart.sh                     # Python baseline only ║
# ║    bash scripts/quickstart.sh path/to/game.c      # + your C file        ║
# ║    bash scripts/quickstart.sh --full              # + Doom corpus         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
set -e

GAME_FILE="${1:-}"
FULL_MODE="${1:-}"

echo ""
echo "  ☠️  FTW-KLOC-KILLER  Quickstart"
echo "  ════════════════════════════════════════════════════"
echo ""

# ── Step 1: Dependencies ──────────────────────────────────────────────────────
echo "  [1/5] Installing Python dependencies..."
pip install -q -r requirements.txt
echo "        ✅ done"
echo ""

# ── Step 2: Rainbow tables ───────────────────────────────────────────────────
echo "  [2/5] Building rainbow tables (stdlib hash cache)..."
python rainbow/build_rainbow.py
echo ""

# ── Step 3: Doctor check ─────────────────────────────────────────────────────
echo "  [3/5] Environment health check..."
python kloc.py doctor
echo ""

# ── Step 4: Synthetic Python baseline ────────────────────────────────────────
echo "  [4/5] Running synthetic Python benchmark (TER + RQS baseline)..."
python kloc.py benchmark --mode synthetic
echo ""

# ── Step 5: Your file (if provided) ─────────────────────────────────────────
if [ -f "$GAME_FILE" ]; then
    echo "  [5/5] Benchmarking your file: $GAME_FILE"
    echo ""
    python kloc.py benchmark --file "$GAME_FILE"
    echo ""
    echo "  Running full 4-agent pipeline on: $GAME_FILE"
    python kloc.py pipeline --file "$GAME_FILE" --question "explain the key functions in this file"
elif [ "$FULL_MODE" = "--full" ]; then
    echo "  [5/5] Downloading Doom + Quake corpus and running C benchmark..."
    bash scripts/download_test_corpus.sh
    echo ""
    python kloc.py benchmark \
        --corpus data/test_corpus/chocolate-doom \
        --lang c \
        --max-files 20 \
        --output results_doom_benchmark.json
    echo ""
    echo "  Results saved → results_doom_benchmark.json"
else
    echo "  [5/5] Skipping custom file benchmark."
    echo "        Pass your file to benchmark it:"
    echo "          bash scripts/quickstart.sh path/to/your/game.c"
    echo "        Or download the Doom/Quake test corpus:"
    echo "          bash scripts/quickstart.sh --full"
fi

echo ""
echo "  ════════════════════════════════════════════════════"
echo "  Quickstart complete. Commands to explore further:"
echo ""
echo "    python kloc.py ter      --file game.c               # token count only"
echo "    python kloc.py rqs      --file game.c               # quality check only"
echo "    python kloc.py benchmark --file game.c              # combined TER + RQS"
echo "    python kloc.py pipeline  --file game.c \\            # full agent pipeline"
echo "                   --question 'explain BSP traversal'"
echo "    python kloc.py test                                  # full test suite"
echo "    python kloc.py rag      --action build-rainbow       # rebuild hash cache"
echo ""
echo "  AI review skills (in Claude Code):"
echo "    /review-pruner    /review-grug    /review-balancer"
echo "    /review-zippy     /review-pipeline  /benchmark-ter"
echo "  ════════════════════════════════════════════════════"
echo ""
