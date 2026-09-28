#!/usr/bin/env bash
set -u
ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926
cd "$ROOT" || exit 2
CONFIG=ue_framework/configs/exp_voc_person_b2_to_lr001_orth0005_eps8_overlap4_20260927.yaml
RUN=noise_sanity_b2_to_eps8_overlap4_5steps_20260927
for path in "$ROOT/$RUN" "$ROOT/$RUN.console.log" "$ROOT/$RUN.exitcode"; do
  [ ! -e "$path" ] || { echo "Refusing to overwrite $path" >&2; exit 2; }
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '^[0-9]'; then
  echo 'GPU compute process is active' >&2
  exit 2
fi
/usr/bin/timeout -s INT -k 30s 30m /usr/bin/env \
  YOLO_CONFIG_DIR="$ROOT/.ultralytics" MPLCONFIGDIR="$ROOT/.mpl" \
  XDG_CACHE_HOME="$ROOT/.cache" TORCH_HOME="$ROOT/.torch" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. WANDB_MODE=disabled \
  /project/guoqinglong/miniconda3/envs/cfbdm/bin/python scripts/train_noise_once.py \
  --config "$CONFIG" --output "$ROOT/$RUN" --seed 0 \
  --smoke-max-images 16 --smoke-max-steps 5 > "$ROOT/$RUN.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$ROOT/$RUN.exitcode"
exit "$code"
