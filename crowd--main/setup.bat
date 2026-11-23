@echo off
setlocal
if not exist "C:\temp" mkdir "C:\temp"
set TMP=C:\temp
set TEMP=C:\temp

echo ==============================================
echo   Crowd Guard - AI Pipeline Setup
echo ==============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not create the virtual environment with Python 3.11.
        echo         Prebuilt llama-cpp-python CUDA wheels only exist for
        echo         Python 3.10, 3.11 and 3.12. Install one of those and retry.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Virtual environment already exists.
)

echo [2/5] Activating virtual environment...
call .\.venv\Scripts\activate.bat

echo [3/5] Installing dependencies...

REM ---- PyTorch with CUDA ----
echo.
echo Installing PyTorch with CUDA 12.8 (GPU acceleration)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

REM ---- llama-cpp-python from the prebuilt CUDA wheel index ----
REM NOTE: llama-cpp-python publishes wheels for cu118, cu121-cu125, cu130, cu132.
REM There is NO cu128 index, so we use cu130 even though torch uses cu128.
REM --no-cache-dir matters: a cached CPU-only wheel is the most common reason
REM this silently installs without GPU support.
echo.
echo Installing llama-cpp-python with CUDA support...
pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130

REM ---- Remaining dependencies ----
pip install -r requirements.txt --no-deps llama-cpp-python 2>nul
pip install fastapi uvicorn pydantic numpy websockets ultralytics opencv-python huggingface-hub

echo.
echo [4/5] Verifying GPU support...
echo.

python -c "import torch; ok = torch.cuda.is_available(); print('  PyTorch  :', torch.__version__, '| CUDA available:', ok); exit(0 if ok else 1)"
if errorlevel 1 (
    echo.
    echo   [ERROR] PyTorch cannot see a CUDA device.
    echo           If the version above has no '+cuXXX' suffix, the CPU-only
    echo           wheel was installed. Fix with:
    echo             pip uninstall torch torchvision -y
    echo             pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    echo.
    pause
)

python -c "from llama_cpp import llama_supports_gpu_offload as g; ok = g(); print('  llama-cpp: CUDA offload supported:', ok); exit(0 if ok else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo   [WARNING] llama-cpp-python was installed WITHOUT CUDA support.
    echo             The vision-language model will run on CPU and will be very slow.
    echo             Try the cu132 wheel index instead:
    echo               pip install llama-cpp-python --force-reinstall --no-cache-dir ^
    echo                 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu132
    echo             See README for the source-build fallback.
    echo.
    pause
)

python -c "import torch; d=torch.cuda.get_device_properties(0); print('  GPU      :', d.name, '|', round(d.total_memory/1048576), 'MB VRAM')" 2>nul

echo.
echo [5/5] Downloading YOLO, Depth and Vision Language models...
python download_models.py

echo.
echo ==============================================
echo   Setup complete! Run start.bat to launch.
echo   No NVIDIA GPU? Use start-cpu.bat instead.
echo ==============================================
pause