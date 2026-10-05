#!/usr/bin/env bash
set -uo pipefail
ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
CONFIG="$ROOT/ue_framework/configs/exp_voc_person_b2_to_strict_tsfl_safebg_eps8_person_seed0_20261002.yaml"
REFERENCE=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/artifacts_specificity1_fraction75
OUTPUT="$ROOT/artifacts_feasibility_scan_20261002"
cd "$ROOT" || exit 2
if test -e "$OUTPUT" || test -e "$ROOT/feasibility_scan.exitcode" || test -e "$ROOT/feasibility_scan.console.log"; then
  printf 'Feasibility scan output already exists; refusing to restart.\n' >&2
  exit 3
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT"
export YOLO_CONFIG_DIR="$ROOT/.ultralytics" MPLCONFIGDIR="$ROOT/.mpl"
export XDG_CACHE_HOME="$ROOT/.cache" TORCH_HOME="$ROOT/.torch"
"$PYTHON" scripts/scan_b2_to_safe_feasibility_20261002.py \
  --config "$CONFIG" --reference-artifact-dir "$REFERENCE" \
  --output "$OUTPUT" --seed 0 --feasibility-only \
  > "$ROOT/feasibility_scan.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$ROOT/feasibility_scan.exitcode"
exit "$code"
