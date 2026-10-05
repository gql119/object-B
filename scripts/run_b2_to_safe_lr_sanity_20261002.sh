#!/usr/bin/env bash
set -uo pipefail
ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
CONFIG="$ROOT/ue_framework/configs/exp_voc_person_b2_to_safe_lr_p5_rank8_eps8_seed0_20261002.yaml"
ARTIFACT="$ROOT/artifacts_b2_to_safe_ls_p5_eps8_seed0_20261002"
NOISE="$ROOT/noise_sanity_b2_to_safe_lr_p5_rank8_5steps_20261002"
AUDIT="$ROOT/sanity_b2_to_safe_lr_p5_rank8_20261002"
cd "$ROOT" || exit 2
if test -e "$NOISE" || test -e "$AUDIT" || test -e "$ROOT/safe_lr_sanity.console.log" || test -e "$ROOT/safe_lr_sanity.exitcode"; then
  printf 'Safe-LR sanity output exists; refusing to restart.\n' >&2
  exit 3
fi
if ! test -f "$ARTIFACT/status.json" || ! grep -q '"state": "complete"' "$ARTIFACT/status.json"; then
  printf 'Frozen P5 coordinates are incomplete.\n' >&2
  exit 4
fi
mkdir "$AUDIT" || exit 5
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT"
export YOLO_CONFIG_DIR="$ROOT/.ultralytics" MPLCONFIGDIR="$ROOT/.mpl"
export XDG_CACHE_HOME="$ROOT/.cache" TORCH_HOME="$ROOT/.torch"
"$PYTHON" scripts/train_noise_once.py --config "$CONFIG" --seed 0 \
  --smoke-max-images 5 --smoke-max-steps 5 --output "$NOISE" \
  > "$ROOT/safe_lr_sanity.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$ROOT/safe_lr_sanity.train.exitcode"
if test "$code" -ne 0; then
  printf '%s\n' "$code" > "$ROOT/safe_lr_sanity.exitcode"
  exit "$code"
fi
"$PYTHON" scripts/check_b2_to_safe_lr_sanity_20261002.py --config "$CONFIG" \
  --noise "$NOISE" --output "$AUDIT" \
  > "$ROOT/safe_lr_sanity_gate.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$ROOT/safe_lr_sanity.exitcode"
exit "$code"
