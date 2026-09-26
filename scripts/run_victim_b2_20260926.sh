#!/usr/bin/env bash
set -u
root=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926
output="$root/victim_b2_seed0_20260926"
cd "$root" || exit 2
if [ -e "$output" ]; then
  echo "Victim output exists: $output" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '^[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 2
fi
YOLO_CONFIG_DIR="$root/.ultralytics" WANDB_MODE=disabled PYTHONPATH=. \
  /project/guoqinglong/miniconda3/envs/cfbdm/bin/python scripts/run_isolated_victim.py \
  --config ue_framework/configs/exp_voc_person_legacy_kproto_ret_b2.yaml \
  --output "$output" --noise "$root/noise_full_b2_seed0_20260926" --seed 0 \
  > "$root/victim_b2_seed0_20260926.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$root/victim_b2_seed0_20260926.exitcode"
exit "$code"
