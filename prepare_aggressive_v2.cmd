@echo off
setlocal
cd /d "%~dp0"
python scripts\preprocess_replays.py --config configs\dqfd_suika_aggressive_v2.yaml
if errorlevel 1 pause
