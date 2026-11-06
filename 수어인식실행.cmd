@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
set "SIGN_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0..\experiments\ksl-assisted_253d2e3\.venv\Scripts\python.exe" set "SIGN_PYTHON=%~dp0..\experiments\ksl-assisted_253d2e3\.venv\Scripts\python.exe"
"%SIGN_PYTHON%" main.py %*
if errorlevel 1 pause
