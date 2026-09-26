#!/usr/bin/env bash
set -u
root=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926
cd "$root" || exit 2
python=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
export YOLO_CONFIG_DIR="$root/.ultralytics" PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
reference=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926
output="$root/noise_sanity_b2_original_5steps_final_20260926"
[ ! -e "$output" ] || exit 2
(cd "$reference" && "$python" scripts/train_noise_once.py --config ue_framework/configs/exp_voc_person_legacy_kproto_ret_b2.yaml --output "$output" --seed 0 --smoke-max-images 16 --smoke-max-steps 5) > "${output}.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "${output}.exitcode"
[ "$code" = 0 ] || exit "$code"
for method in b2 b2_t b2_to; do
  if [ "$method" = b2 ]; then config=ue_framework/configs/exp_voc_person_legacy_kproto_ret_b2.yaml; else config="ue_framework/configs/exp_voc_person_${method}.yaml"; fi
  output="$root/noise_sanity_${method}_5steps_final_20260926"
  [ ! -e "$output" ] || exit 2
  /usr/bin/timeout -s INT -k 30s 30m "$python" scripts/train_noise_once.py --config "$config" --output "$output" --seed 0 --smoke-max-images 16 --smoke-max-steps 5 > "${output}.console.log" 2>&1
  code=$?
  printf '%s\n' "$code" > "${output}.exitcode"
  [ "$code" = 0 ] || exit "$code"
done
