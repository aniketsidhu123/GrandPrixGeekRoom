@echo off
setlocal
if not exist "C:\temp" mkdir "C:\temp"
set TMP=C:\temp
set TEMP=C:\temp

echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Crowd Flow Optimiser - AI Pipeline Setup
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    py -3.12 -m venv .venv
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

REM Install llama-cpp-python FIRST from prebuilt CUDA wheel (before requirements.txt)
echo.
echo Installing llama-cpp-python with CUDA support...
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

REM Install remaining dependencies (llama-cpp-python already installed above, so exclude it)
pip install -r requirements.txt --no-deps llama-cpp-python 2>nul
pip install fastapi uvicorn pydantic numpy websockets ultralytics opencv-python filterpy huggingface-hub

echo.
echo [4/4] Downloading all YOLO, Depth, and Vision Language Models (SmolVLM2)...
python download_models.py

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Setup complete! Run start.bat to launch.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
