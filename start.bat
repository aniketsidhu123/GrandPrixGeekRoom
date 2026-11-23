@echo off
echo Starting Crowd Flow Optimiser...
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first!
    exit /b 1
)
echo Starting server...
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
