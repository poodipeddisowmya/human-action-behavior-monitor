@echo off
REM Start the monitor and open the UI. Run setup.bat once before this.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: no virtual environment found.
  echo Run setup.bat first.
  pause
  exit /b 1
)

echo Starting... the browser will open once the server is up.
REM Give uvicorn a moment to bind the port before the browser opens, so the
REM client does not land on a connection-refused page.
start "" /b cmd /c "timeout /t 12 /nobreak >nul & start http://localhost:8000"

.venv\Scripts\python.exe scripts\run_pipeline.py %*

pause
