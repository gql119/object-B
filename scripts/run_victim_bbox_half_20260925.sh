#!/usr/bin/env bash
set -u
root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
output="$root/victim_bbox_half_seed0_20260925"
cd "$root" || exit 2
if [ -e "$output" ]; then
  echo "Victim output already exists: $output" >&2
  exit 2
fi
env YOLO_CONFIG_DIR="$root/.ultralytics" WANDB_MODE=disabled PYTHONPATH=. \
  "${PYTHON:-python}" scripts/run_isolated_victim.py \
  --config ue_framework/configs/exp_voc_person_legacy_kproto_ret_bbox_half.yaml \
  --output "$output" --noise "$root/noise_full_bbox_half_seed0_20260925" --seed 0 \
  > "$root/victim_bbox_half_seed0_20260925.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$root/victim_bbox_half_seed0_20260925.exitcode"
exit "$code"
