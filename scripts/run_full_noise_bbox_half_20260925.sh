#!/usr/bin/env bash
set -u
root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
output="$root/noise_full_bbox_half_seed0_20260925"
cd "$root" || exit 2
if [ -e "$output" ]; then
  echo "Output already exists: $output" >&2
  exit 2
fi
/usr/bin/timeout -s INT -k 30s 5h /usr/bin/env \
  YOLO_CONFIG_DIR="$root/.ultralytics" PYTHONPATH=. \
  "${PYTHON:-python}" scripts/train_noise_once.py \
  --config ue_framework/configs/exp_voc_person_legacy_kproto_ret_bbox_half.yaml \
  --output "$output" --seed 0 \
  > "$root/noise_full_bbox_half_seed0_20260925.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$root/noise_full_bbox_half_seed0_20260925.exitcode"
exit "$code"
