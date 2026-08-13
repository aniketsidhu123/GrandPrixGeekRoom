@echo off
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Crowd Flow Optimiser - AI Pipeline Setup
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
) else (
    echo [1/4] Virtual environment already exists.
)

echo [2/4] Activating virtual environment...
call .\.venv\Scripts\activate.bat

echo [3/4] Installing dependencies...

REM Install PyTorch with CUDA support (RTX 4060 needs CUDA 12.x)
echo.
echo Installing PyTorch with CUDA 12.8 (GPU acceleration)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

REM Install remaining dependencies (excluding torch/torchvision since we just installed them)
pip install -r requirements.txt

REM Install llama-cpp-python with CUDA support for GPU-accelerated VLM
echo.
echo Installing llama-cpp-python with CUDA support...
set CMAKE_ARGS=-DGGML_CUDA=on
set FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --no-cache-dir

echo.
echo [4/4] Downloading all YOLO, Depth, and Vision Language Models (SmolVLM2)...
python download_models.py

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Setup complete! Run start.bat to launch.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
