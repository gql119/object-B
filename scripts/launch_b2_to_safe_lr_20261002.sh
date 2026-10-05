#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="b2_to_safe_lr_pipeline_20261002"
NOISE="$ROOT/noise_full_b2_to_safe_lr_eps8_person_seed0_20261002"
VICTIM="$ROOT/victim_b2_to_safe_lr_eps8_person_seed0_20261002"
REPORT="$ROOT/experiments/b2_to_safebg_lr_20261002/formal_retry3"
BOOTLOG="$ROOT/experiments/b2_to_safebg_lr_20261002/pipeline_startup_retry3.log"
CONFIG="$ROOT/ue_framework/configs/exp_voc_person_b2_to_safe_lr_p5_rank8_eps8_seed0_20261002.yaml"
SANITY="$ROOT/experiments/b2_to_safebg_lr_20261002/lowrank_sanity_gate.json"
ADMISSION="$ROOT/experiments/b2_to_safebg_lr_20261002/prerun_review.json"
CLEAN="$ROOT/../../baseline_dev_20260928/clean_baseline_seed0_20260928"
B2TO="$ROOT/../../victim_b2_to_lr001_orth0005_seed0_20260927"
B2TO_CSV="$ROOT/../../noise_full_b2_to_lr001_orth0005_seed0_20260927/diagnostics.csv"
PYTHON="/project/guoqinglong/miniconda3/envs/cfbdm/bin/python"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Refusing duplicate pipeline session: $SESSION" >&2
    exit 2
fi
for path in "$NOISE" "$VICTIM" "$REPORT" "$BOOTLOG"; do
    if [[ -e "$path" ]]; then
        echo "Refusing to overwrite existing output: $path" >&2
        exit 2
    fi
done
for path in "$CONFIG" "$SANITY" "$ADMISSION" "$CLEAN/run_spec.json" \
            "$CLEAN/victim/weights/last.pt" "$B2TO/run_spec.json" \
            "$B2TO/victim/weights/last.pt" "$B2TO_CSV"; do
    if [[ ! -f "$path" ]]; then
        echo "Required read-only input is missing: $path" >&2
        exit 3
    fi
done

COMMAND="cd '$ROOT' && env PYTHONPATH='$ROOT' PYTHONDONTWRITEBYTECODE=1 YOLO_CONFIG_DIR='$ROOT/.ultralytics' MPLCONFIGDIR='$ROOT/.mpl' XDG_CACHE_HOME='$ROOT/.cache' TORCH_HOME='$ROOT/.torch' WANDB_MODE=disabled '$PYTHON' scripts/run_b2_to_safe_lr_pipeline_20261002.py --config '$CONFIG' --sanity-gate '$SANITY' --admission '$ADMISSION' --noise '$NOISE' --victim '$VICTIM' --report-dir '$REPORT' --clean-victim '$CLEAN' --b2to-victim '$B2TO' --formal-b2to-noise-csv '$B2TO_CSV' > '$BOOTLOG' 2>&1"
tmux new-session -d -s "$SESSION" "$COMMAND"
echo "Started $SESSION"
echo "RunID: b2_to_safe_lr_eps8_person_seed0_20261002"
echo "Primary checkpoint: last.pt"
