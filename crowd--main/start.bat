@echo off
echo Starting Crowd Guard (GPU mode)...

if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first!
    exit /b 1
)

REM ---- Device placement -------------------------------------------------
REM Fail loudly if no CUDA device is present, instead of silently running
REM everything on CPU. Use start-cpu.bat on machines without an NVIDIA GPU.
set CFO_REQUIRE_GPU=1

REM Which device each model loads onto.
REM Sizing on an 8 GB card: everything below fits at once (~3.5 GB total).
REM   vlm     ~2000 MB  - biggest win from GPU, runs every 30th frame
REM   crowd    ~200 MB  - runs on EVERY frame, 5 fps CPU vs 80 fps GPU
REM   depth    ~300 MB
REM   traffic  ~350 MB  - drop these three to cpu first if VRAM is tight
REM   anomaly  ~700 MB
REM   flow     ~250 MB
set CFO_DEVICE_VLM=cuda
set CFO_DEVICE_CROWD=cuda
set CFO_DEVICE_DEPTH=cuda
set CFO_DEVICE_TRAFFIC=cpu
set CFO_DEVICE_ANOMALY=cpu
set CFO_DEVICE_FLOW=cpu

echo   GPU mode: VLM + crowd + depth on CUDA, remaining YOLO variants on CPU.
echo   Starting server on http://localhost:8001/static/index.html
echo.

.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001