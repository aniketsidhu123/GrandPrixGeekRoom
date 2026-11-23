#!/usr/bin/env bash
# macOS / Linux equivalent of start.bat
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

echo "Starting Crowd Flow Optimiser on http://localhost:8001 ..."
exec ./.venv/bin/python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
