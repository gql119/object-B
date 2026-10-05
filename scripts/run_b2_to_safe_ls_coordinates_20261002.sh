#!/usr/bin/env bash
set -uo pipefail
ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
CONFIG="$ROOT/ue_framework/configs/exp_voc_person_b2_to_safe_ls_p5_eps8_seed0_20261002.yaml"
REFERENCE=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/artifacts_specificity1_fraction75
OUTPUT="$ROOT/artifacts_b2_to_safe_ls_p5_eps8_seed0_20261002"
cd "$ROOT" || exit 2
if test -e "$OUTPUT" || test -e "$ROOT/safe_ls_coordinates.exitcode" || test -e "$ROOT/safe_ls_coordinates.console.log"; then
  printf 'Safe-LS coordinate output exists; refusing to restart.\n' >&2
  exit 3
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT"
export YOLO_CONFIG_DIR="$ROOT/.ultralytics" MPLCONFIGDIR="$ROOT/.mpl"
export XDG_CACHE_HOME="$ROOT/.cache" TORCH_HOME="$ROOT/.torch"
"$PYTHON" scripts/build_b2_to_safe_coordinates.py \
  --config "$CONFIG" --reference-artifact-dir "$REFERENCE" \
  --output "$OUTPUT" --seed 0 > "$ROOT/safe_ls_coordinates.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$ROOT/safe_ls_coordinates.exitcode"
exit "$code"
