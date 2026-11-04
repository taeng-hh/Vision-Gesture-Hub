@echo off
setlocal
set "PYTHONUTF8=1"
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-fusion.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\setup_assets.py
if errorlevel 1 goto failed
echo Ready. Run run.cmd
pause
exit /b 0
:failed
echo Setup failed. See the error above.
pause
exit /b 1
