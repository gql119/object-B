#!/usr/bin/env bash
set -u
root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$root" || exit 2
if [ -e "$root/runs_bbox_half/poisoned_datasets/legacy_kproto_ret/steps40/seed0/status.json" ]; then
  echo "Materialization status already exists; refusing duplicate launch" >&2
  exit 2
fi
env YOLO_CONFIG_DIR="$root/.ultralytics" PYTHONPATH=. \
  "${PYTHON:-python}" scripts/materialize_noise_once.py \
  --config ue_framework/configs/exp_voc_person_legacy_kproto_ret_bbox_half.yaml \
  --noise "$root/noise_full_bbox_half_seed0_20260925" --seed 0 \
  > "$root/materialization_bbox_half_20260925.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$root/materialization_bbox_half_20260925.exitcode"
exit "$code"
