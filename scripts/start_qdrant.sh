#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════╗
# ║  start_qdrant.sh — Spin up the vector DB treasure chest  ║
# ║  Usage: bash scripts/start_qdrant.sh                     ║
# ╚══════════════════════════════════════════════════════════╝
set -e

echo "  Hoisting the Qdrant flag..."
docker-compose up -d qdrant

echo "  Waiting for the ship to set sail (health check)..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
        echo "  Vector DB is live at http://localhost:6333"
        echo "  Collections: $(curl -s http://localhost:6333/collections | python3 -m json.tool 2>/dev/null | head -5)"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "  Qdrant didn't respond after 30s. Check: docker logs kloc_qdrant"
exit 1
