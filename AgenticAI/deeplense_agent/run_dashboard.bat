@echo off
echo =========================================
echo Starting DeepLense Agent Dashboard...
echo =========================================

cd /d "%~dp0"

IF NOT EXIST .venv (
    echo [1/3] Creating Python virtual environment...
    python -m venv .venv
    
    echo [2/3] Installing DeepLense Agent and dependencies...
    call .venv\Scripts\activate.bat
    pip install -e .
    pip install -r frontend\backend\requirements.txt
) ELSE (
    echo [1/3] Virtual environment found. Activating...
    call .venv\Scripts\activate.bat
)

echo.
echo [3/3] Starting Servers...
echo.
echo Starting FastAPI Backend on port 8000...
start cmd /k "call .venv\Scripts\activate.bat && set PYTHONPATH=%~dp0 && cd frontend\backend && uvicorn main:app --host 0.0.0.0 --port 8000"

echo Serving Frontend on port 8080...
start cmd /k "python -m http.server 8080 -d frontend\web"

echo =========================================
echo DeepLense Dashboard running!
echo.
echo Backend API: http://localhost:8000
echo Web Interface: http://localhost:8080
echo.
echo IMPORTANT: Ensure you have your GROQ_API_KEY (or OpenAI/Anthropic key) set!
echo Example: set GROQ_API_KEY=your_key_here before running the backend.
echo =========================================
