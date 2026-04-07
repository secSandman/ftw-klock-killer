#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  download_test_corpus.sh — Steal the best C code the seas have seen ║
# ║  Clones Chocolate Doom + QuakeSpasm (both GPLv2, no lawyers needed) ║
# ║  Usage: bash scripts/download_test_corpus.sh                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -e

CORPUS_DIR="data/test_corpus"
mkdir -p "$CORPUS_DIR"

# ── Chocolate Doom ────────────────────────────────────────────────────
DOOM_URL="https://github.com/chocolate-doom/chocolate-doom.git"
DOOM_DIR="$CORPUS_DIR/chocolate-doom"
if [ -d "$DOOM_DIR" ]; then
    echo "  Chocolate Doom already plundered at $DOOM_DIR"
else
    echo "  Pillaging Chocolate Doom (~70K SLOC of glorious C)..."
    git clone --depth=5 "$DOOM_URL" "$DOOM_DIR"
    echo "  Doom secured: $DOOM_DIR"
fi

# ── QuakeSpasm ────────────────────────────────────────────────────────
QUAKE_URL="https://github.com/sezero/quakespasm.git"
QUAKE_DIR="$CORPUS_DIR/quakespasm"
if [ -d "$QUAKE_DIR" ]; then
    echo "  QuakeSpasm already plundered at $QUAKE_DIR"
else
    echo "  Boarding QuakeSpasm (~80K SLOC, BSP trees and dark magic)..."
    git clone --depth=5 "$QUAKE_URL" "$QUAKE_DIR"
    echo "  Quake secured: $QUAKE_DIR"
fi

echo ""
echo "  Corpus manifest:"
echo "    Chocolate Doom: $(find $DOOM_DIR -name '*.c' 2>/dev/null | wc -l) .c files"
echo "    QuakeSpasm:     $(find $QUAKE_DIR -name '*.c' 2>/dev/null | wc -l) .c files"
echo ""
echo "  Run the ETL:"
echo "    python rag/etl_pipeline.py --repo $DOOM_DIR --language c"
echo "    python rag/etl_pipeline.py --repo $QUAKE_DIR --language c"
