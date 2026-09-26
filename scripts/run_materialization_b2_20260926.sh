#!/usr/bin/env bash
set -u
root=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926
cd "$root" || exit 2
YOLO_CONFIG_DIR="$root/.ultralytics" PYTHONPATH=. \
  /project/guoqinglong/miniconda3/envs/cfbdm/bin/python scripts/materialize_noise_once.py \
  --config ue_framework/configs/exp_voc_person_legacy_kproto_ret_b2.yaml \
  --noise "$root/noise_full_b2_seed0_20260926" --seed 0 \
  > "$root/materialization_b2_20260926.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$root/materialization_b2_20260926.exitcode"
exit "$code"
