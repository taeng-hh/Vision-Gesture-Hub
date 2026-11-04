@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m unittest tests.test_fusion tests.test_fusion_preview tests.test_fusion_speed tests.test_live_sign tests.test_assisted_policy -v
if errorlevel 1 pause
