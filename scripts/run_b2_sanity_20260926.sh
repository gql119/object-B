#!/usr/bin/env bash
set -u
root=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926
output="$root/noise_sanity_b2_24epoch_seed0_5steps_20260926"
cd "$root" || exit 2
if [ -e "$output" ]; then
  echo "Sanity output exists: $output" >&2
  exit 2
fi
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '^[0-9]'; do
  sleep 60
done
if [ -e "$output" ]; then
  echo "Sanity output appeared while waiting: $output" >&2
  exit 2
fi
/usr/bin/timeout -s INT -k 30s 30m /usr/bin/env \
  YOLO_CONFIG_DIR="$root/.ultralytics" PYTHONPATH=. \
  /project/guoqinglong/miniconda3/envs/cfbdm/bin/python scripts/train_noise_once.py \
  --config ue_framework/configs/exp_voc_person_legacy_kproto_ret_b2.yaml \
  --output "$output" --seed 0 --smoke-max-images 16 --smoke-max-steps 5 \
  > "$root/noise_sanity_b2_24epoch_seed0_5steps_20260926.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$root/noise_sanity_b2_24epoch_seed0_5steps_20260926.exitcode"
exit "$code"
