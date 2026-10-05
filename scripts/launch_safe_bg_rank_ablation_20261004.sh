#!/usr/bin/env bash
set -euo pipefail
ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
SESSION=safe_bg_rank_ablation_20261004
BOOTLOG="$ROOT/experiments/safe_bg_ablation_20261004/formal_startup.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    printf 'Session already exists: %s\n' "$SESSION" >&2
    exit 2
fi
if [ -e "$ROOT/experiments/safe_bg_ablation_20261004/formal" ] || [ -e "$BOOTLOG" ]; then
    printf 'Formal output collision; refusing to restart\n' >&2
    exit 3
fi
COMMAND="cd '$ROOT' && env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='$ROOT' YOLO_CONFIG_DIR='$ROOT/.ultralytics' MPLCONFIGDIR='$ROOT/.mpl' XDG_CACHE_HOME='$ROOT/.cache' TORCH_HOME='$ROOT/.torch' WANDB_MODE=disabled '$PYTHON' scripts/run_safe_bg_rank_ablation_pipeline_20261004.py > '$BOOTLOG' 2>&1"
tmux new-session -d -s "$SESSION" "$COMMAND"
printf 'Started %s\n' "$SESSION"
