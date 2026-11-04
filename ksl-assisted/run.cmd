@echo off
setlocal
set "PYTHONUTF8=1"
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.cmd first, or use your existing Python environment with main.py.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
