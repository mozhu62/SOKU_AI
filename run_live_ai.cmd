@echo off
setlocal
cd /d "%~dp0"
python scripts\run_live_ai.py
if errorlevel 1 pause
