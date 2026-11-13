@echo off
REM One-time setup for a Windows machine with an NVIDIA GPU (e.g. RTX 3050).
REM Fully automatic where Windows allows it:
REM   - installs Python 3.11 itself if it is missing (via winget)
REM   - installs the GPU build of PyTorch and every library (requirements-gpu.txt)
REM   - downloads the CLIP model once
REM   - checks the NVIDIA driver and finishes with a GPU self-check
REM The only thing it cannot do is install the NVIDIA driver: that needs the
REM NVIDIA app / Windows Update and a reboot. Without it, the app still runs,
REM on the CPU.
setlocal
cd /d "%~dp0"

echo.
echo === Human Action ^& Behavior Monitor : GPU setup ===
echo.

REM ---------------------------------------------------------------- driver
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo NOTE: no NVIDIA driver found. The app will run on the CPU until one is
  echo installed: NVIDIA app or https://www.nvidia.com/Download/index.aspx
  echo.
) else (
  echo NVIDIA GPU and driver:
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo.
)

REM ---------------------------------------------------------------- python
set "PY="
where py >nul 2>&1 && py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11"
if not defined PY (
  where python >nul 2>&1 && python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3.11 not found - installing it now with winget...
  winget install -e --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo ERROR: automatic Python install failed. Install Python 3.11 from
    echo https://www.python.org/downloads/release/python-3119/ and run this again.
    pause
    exit /b 1
  )
  REM A fresh install is not on this window's PATH yet; use its known location.
  if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  ) else (
    set "PY=py -3.11"
  )
)
echo Using Python: %PY%

REM ---------------------------------------------------------------- venv + libraries
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv || ( echo ERROR: could not create .venv & pause & exit /b 1 )
)
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo.
echo Installing PyTorch with GPU support and all libraries ^(about 3 GB, one time^)...
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
if errorlevel 1 (
  echo Pinned GPU build not available, installing the latest CUDA 12.6 build instead...
  .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || ( echo ERROR: PyTorch install failed. & pause & exit /b 1 )
  .venv\Scripts\python.exe -m pip install -r requirements.txt || ( echo ERROR: dependency install failed. & pause & exit /b 1 )
)

echo.
echo Downloading the CLIP model used for place names and sitting/standing ^(one time^)...
.venv\Scripts\python.exe -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); print('CLIP ready')"
echo Downloading VideoMAE, the GPU activity model ^(one time, about 350 MB^)...
.venv\Scripts\python.exe -c "from transformers import VideoMAEForVideoClassification as M, VideoMAEImageProcessor as P; c='MCG-NJU/videomae-base-finetuned-kinetics'; M.from_pretrained(c); P.from_pretrained(c); print('VideoMAE ready')"
echo Downloading SmolVLM2, the GPU detailed-subtitle model ^(one time, about 1 GB^)...
.venv\Scripts\python.exe -c "from transformers import AutoModelForImageTextToText as M, AutoProcessor as P; c='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'; M.from_pretrained(c); P.from_pretrained(c); print('SmolVLM2 ready')"

echo.
echo === GPU self-check ===
.venv\Scripts\python.exe scripts\gpu_check.py

echo Setup complete. Double-click run.bat to start, then open http://localhost:8000
pause
