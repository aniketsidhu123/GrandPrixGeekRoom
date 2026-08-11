@echo off
echo Starting Crowd Flow Optimiser...
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)
echo Starting server...
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
pause
