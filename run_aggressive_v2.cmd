@echo off
setlocal
cd /d "%~dp0"
if not exist "data\processed_aggressive_v2\normalization.json" (
  echo Run prepare_aggressive_v2.cmd first.
  pause
  exit /b 1
)
python scripts\train_dqfd_demo.py --config configs\dqfd_suika_aggressive_v2.yaml --init-from checkpoints\noop_finetune\snapshots\step_000021000.pt
if errorlevel 1 pause
