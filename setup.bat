@echo off
REM One-time setup on Windows. Creates a virtual environment, installs
REM dependencies, and runs the install check.
setlocal
cd /d "%~dp0"

echo.
echo === Human Action ^& Behavior Monitor : setup ===
echo.

where py >nul 2>&1
if %ERRORLEVEL%==0 ( set "PY=py -3" ) else ( set "PY=python" )

%PY% --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python not found on PATH.
  echo Install Python 3.11 from https://www.python.org/downloads/
  echo and tick "Add Python to PATH" during installation.
  pause
  exit /b 1
)

REM Check the version BEFORE building a venv and downloading ~2 GB of wheels.
REM On 3.13+ numpy has no wheel and the install dies in a compiler error after
REM several minutes, which is a miserable way to find this out.
%PY% -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
  echo.
  %PY% --version
  echo ERROR: unsupported Python version. This needs Python 3.10, 3.11 or 3.12.
  echo.
  echo Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
  echo tick "Add Python to PATH", then delete the .venv folder and run this again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 ( echo ERROR: could not create .venv & pause & exit /b 1 )
) else (
  echo Virtual environment already exists, reusing it.
)

echo.
echo Installing dependencies ^(this takes a few minutes the first time^)...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 ( echo. & echo ERROR: dependency install failed. & pause & exit /b 1 )

echo.
echo Compiling the models for this computer's CPU (a few minutes; optional)...
.venv\Scripts\python.exe -m pip install -r requirements-edge.txt --quiet
if not errorlevel 1 .venv\Scripts\python.exe scripts\optimise_for_device.py
if errorlevel 1 echo   (skipped: the monitor uses the standard models)

echo.
echo Running install check...
echo.
.venv\Scripts\python.exe scripts\verify_install.py
if errorlevel 1 ( pause & exit /b 1 )

echo.
echo Setup complete. Double-click run.bat to start.
echo.
pause
