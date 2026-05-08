#!/usr/bin/env bash
set -e
PORT="${PORT:-8080}"
FOLDER="${1:-.}"
cd "$(dirname "$0")"

OLD_PID=$(lsof -ti :"$PORT" 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
    echo "Killing old process on port $PORT (PID $OLD_PID)"
    kill -9 $OLD_PID 2>/dev/null || true
    sleep 1
fi

echo "Starting markdown-search — folder: $FOLDER"
echo ""
uv run python serve.py "$FOLDER" --port "$PORT"
