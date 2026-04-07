#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  e2e_doom.sh — End-to-end demo on Chocolate Doom source                     ║
# ║                                                                              ║
# ║  Runs the full FTW-KLOC-KILLER pipeline on real C game code.                ║
# ║  Designed for new-developer onboarding and tutorial screenshots.             ║
# ║                                                                              ║
# ║  Usage:                                                                      ║
# ║    bash scripts/e2e_doom.sh                    # default: r_plane.c          ║
# ║    bash scripts/e2e_doom.sh src/p_map.c        # any Doom .c file            ║
# ║    bash scripts/e2e_doom.sh src/r_bsp.c "explain the BSP traversal"          ║
# ║                                                                              ║
# ║  Screenshot checkpoints are marked  ► CHECKPOINT N  in the output.          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
set -e

DOOM_DIR="data/test_corpus/chocolate-doom"
TARGET_FILE="${1:-src/r_plane.c}"          # relative to DOOM_DIR
QUESTION="${2:-explain the key rendering functions in this file}"
FULL_PATH="$DOOM_DIR/$TARGET_FILE"

# ── colours ──────────────────────────────────────────────────────────────────
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RESET='\033[0m'

checkpoint() {
    echo ""
    echo -e "${YELLOW}${BOLD}┌──────────────────────────────────────────────────────────────┐${RESET}"
    echo -e "${YELLOW}${BOLD}│  ► CHECKPOINT $1 — $2 ${RESET}"
    echo -e "${YELLOW}${BOLD}└──────────────────────────────────────────────────────────────┘${RESET}"
    echo ""
}

banner() {
    echo -e "${CYAN}${BOLD}"
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║  FTW-KLOC-KILLER  ·  End-to-End Doom Demo                   ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
}

banner

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 1 — Environment setup
# ─────────────────────────────────────────────────────────────────────────────
checkpoint 1 "Environment setup"

echo "  Installing dependencies..."
pip install -q -r requirements.txt
echo "  ${GREEN}✓ Dependencies installed${RESET}"
echo ""

echo "  Building rainbow tables (stdlib hash cache)..."
python rainbow/build_rainbow.py
echo ""

echo "  Running doctor check..."
python kloc.py doctor
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 2 — Download Doom corpus
# ─────────────────────────────────────────────────────────────────────────────
checkpoint 2 "Download Chocolate Doom source (~70K SLOC of C)"

bash scripts/download_test_corpus.sh

echo ""
C_COUNT=$(find "$DOOM_DIR/src" -name "*.c" 2>/dev/null | wc -l)
H_COUNT=$(find "$DOOM_DIR/src" -name "*.h" 2>/dev/null | wc -l)
echo -e "  ${GREEN}✓ Doom corpus ready${RESET}"
echo "    .c files: $C_COUNT"
echo "    .h files: $H_COUNT"
echo "    Target:   $FULL_PATH"
echo ""

if [ ! -f "$FULL_PATH" ]; then
    # fallback to first .c file we can find
    FULL_PATH=$(find "$DOOM_DIR/src" -name "*.c" | head -1)
    echo "  Note: $TARGET_FILE not found — using $FULL_PATH instead"
fi

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 3 — Token count baseline (TER only, no LLM)
# ─────────────────────────────────────────────────────────────────────────────
checkpoint 3 "Token count baseline — how expensive is this file raw?"

python kloc.py ter --file "$FULL_PATH"

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 4 — Synthetic Python benchmark (establishes TER baseline)
# ─────────────────────────────────────────────────────────────────────────────
checkpoint 4 "Synthetic benchmark — TER baseline on Python corpus"

python kloc.py benchmark --mode synthetic

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 5 — Full benchmark on target Doom file
# ─────────────────────────────────────────────────────────────────────────────
checkpoint 5 "Doom file benchmark — TER + RQS on real C code"

python kloc.py benchmark --file "$FULL_PATH"

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 6 — Full 4-agent pipeline + LLM response
# ─────────────────────────────────────────────────────────────────────────────
checkpoint 6 "Full pipeline — PRUNER→GRUG→BALANCER→ZIPPY→LLM"

echo "  File:     $FULL_PATH"
echo "  Question: $QUESTION"
echo ""

python kloc.py pipeline \
    --file "$FULL_PATH" \
    --question "$QUESTION"

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT 7 — Corpus benchmark (top 20 Doom files)
# ─────────────────────────────────────────────────────────────────────────────
checkpoint 7 "Corpus benchmark — top 20 Doom .c files"

python kloc.py benchmark \
    --corpus "$DOOM_DIR" \
    --lang c \
    --max-files 20 \
    --output results_doom_e2e.json

echo ""
echo -e "  ${GREEN}${BOLD}Results saved → results_doom_e2e.json${RESET}"

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║  End-to-end demo complete.                                   ║"
echo "  ║                                                              ║"
echo "  ║  7 checkpoints hit. Good screenshots to take:               ║"
echo "  ║    CP3 — raw token count (the 'before')                     ║"
echo "  ║    CP5 — TER + RQS table on Doom file (the 'after')         ║"
echo "  ║    CP6 — LLM response on compressed context                 ║"
echo "  ║    CP7 — corpus table across 20 files                       ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo ""
echo "  Try other Doom files:"
echo "    bash scripts/e2e_doom.sh src/r_bsp.c    \"explain BSP traversal\""
echo "    bash scripts/e2e_doom.sh src/p_map.c    \"explain collision detection\""
echo "    bash scripts/e2e_doom.sh src/p_enemy.c  \"explain enemy AI state machine\""
echo "    bash scripts/e2e_doom.sh src/g_game.c   \"explain the game loop\""
echo ""
