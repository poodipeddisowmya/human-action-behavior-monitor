@echo off
REM One click: upload the latest project to Kaggle, test it end-to-end on a
REM free T4 GPU, and put the live-demo notebook in your Kaggle account.
REM Report: dist\kaggle-output\gpu_report.md
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe scripts\kaggle_auto.py all %*
echo.
echo Live demo: open https://www.kaggle.com/code/ and the "HAM GPU Demo" notebook, then Run All.
pause
