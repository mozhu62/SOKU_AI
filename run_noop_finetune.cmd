@echo off
setlocal
cd /d "%~dp0"
python scripts\train_dqfd_demo.py --config configs\dqfd_suika_noop_finetune.yaml --init-from checkpoints\snapshots\step_000051512.pt
if errorlevel 1 pause
