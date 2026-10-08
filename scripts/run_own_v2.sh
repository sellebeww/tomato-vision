#!/bin/bash
# Resumable driver for the own_v2 study and the demo release (detached; survives closing the terminal).
#   nohup caffeinate -i bash scripts/run_own_v2.sh > outputs/experiments/own_v2/driver.log 2>&1 &
# Step 1 (long, resumable): the stage-1 sweep, 25 runs per config; finished runs are skipped.
# Step 2: python -m scripts.finish_own_v2 (summarize, select, final refit, locked test once, export, parity, docs).
# Every step refuses to overwrite earlier artifacts. Stop with: pkill -f run_own_v2.sh; pkill -f src.cv_study
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
echo "[$(date '+%F %T')] sweep start"
$PY -u -m src.cv_study sweep --configs ref lr0003 bn09 aug regularized --workers 2 >> outputs/experiments/own_v2/sweep_stage1.log 2>&1
echo "[$(date '+%F %T')] sweep complete"
$PY -u -m scripts.finish_own_v2
echo "[$(date '+%F %T')] driver done"
