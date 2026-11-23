#!/usr/bin/env bash
# macOS / Linux equivalent of setup.bat
#
# Differences from the Windows setup.bat:
#   - No CUDA. PyTorch installs from standard PyPI (Apple Silicon gets MPS support).
#   - llama-cpp-python is built with Metal instead of CUDA, and is OPTIONAL.
#   - The simulation runs without any AI models; vision deps are a separate step.
set -e
cd "$(dirname "$0")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Crowd Flow Optimiser - macOS Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -x ".venv/bin/python" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv .venv
else
    echo "[1/3] Virtual environment already exists."
fi

./.venv/bin/python -m pip install --upgrade pip -q

echo "[2/3] Installing core dependencies (simulation + web server)..."
./.venv/bin/pip install -q \
    fastapi uvicorn pydantic numpy websockets \
    torch torchvision opencv-python

echo "[3/3] Core setup complete."
echo
echo "Run ./start.sh to launch the dashboard at http://localhost:8001"
echo
echo "OPTIONAL — AI vision pipeline (adds ~2GB of model weights):"
echo "  ./.venv/bin/pip install ultralytics filterpy huggingface-hub"
echo "  ./.venv/bin/python download_models.py"
echo
echo "OPTIONAL — VLM scene descriptions (Metal-accelerated build):"
echo "  CMAKE_ARGS=\"-DGGML_METAL=on\" ./.venv/bin/pip install llama-cpp-python --no-cache-dir"
