@echo off
echo Starting Crowd Guard (CPU mode)...

if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first!
    exit /b 1
)

echo   No NVIDIA GPU: all models run on CPU (slower, but works).
echo   Dashboard: http://localhost:8001/static/index.html
echo.

start "" "http://localhost:8001/static/index.html"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001