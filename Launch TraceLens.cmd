@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating a local Python environment...
    python -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import tracelens, torch, fastapi, uvicorn" >nul 2>nul
if errorlevel 1 (
    echo Installing TraceLens and CPU dependencies. First launch may take a few minutes...
    ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto failed
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 goto failed
)
echo Open http://127.0.0.1:8765 in your browser after the server starts.
echo Keep this window open while using TraceLens. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m tracelens.cli serve
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo TraceLens could not start. Check that Python 3.11+ is installed and read README.md.
pause
exit /b 1
