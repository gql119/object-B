#!/usr/bin/env bash
set -u
root=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926
cd "$root" || exit 2
export YOLO_CONFIG_DIR="$root/.ultralytics" MPLCONFIGDIR="$root/.mpl" XDG_CACHE_HOME="$root/.cache" TORCH_HOME="$root/.torch" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
bash scripts/run_full_noise_b2_t_20260926.sh
code=$?
[ "$code" = 0 ] || exit "$code"
bash scripts/run_materialization_b2_t_20260926.sh
code=$?
[ "$code" = 0 ] || exit "$code"
/project/guoqinglong/miniconda3/envs/cfbdm/bin/python scripts/preflight_b2_t.py > preflight_b2_t_20260926.console.log 2>&1
code=$?
printf '%s\n' "$code" > preflight_b2_t_20260926.exitcode
[ "$code" = 0 ] || exit "$code"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '^[0-9]'; do
  sleep 30
done
bash scripts/run_victim_b2_t_20260926.sh
code=$?
printf '%s\n' "$code" > pipeline_b2_t_20260926.exitcode
exit "$code"
