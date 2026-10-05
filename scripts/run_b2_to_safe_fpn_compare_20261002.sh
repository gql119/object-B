#!/usr/bin/env bash
set -uo pipefail
ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
CONFIG="$ROOT/ue_framework/configs/exp_voc_person_b2_to_strict_tsfl_safebg_eps8_person_seed0_20261002.yaml"
REFERENCE=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/artifacts_specificity1_fraction75
PREVIOUS="$ROOT/artifacts_feasibility_scan_20261002/feasibility_scan.json"
OUTPUT="$ROOT/artifacts_fpn_separability_20261002"
cd "$ROOT" || exit 2
if test -e "$OUTPUT" || test -e "$ROOT/fpn_separability.exitcode" || test -e "$ROOT/fpn_separability.console.log"; then
  printf 'P3/P4/P5 comparison output already exists; refusing to restart.\n' >&2
  exit 3
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT"
export YOLO_CONFIG_DIR="$ROOT/.ultralytics" MPLCONFIGDIR="$ROOT/.mpl"
export XDG_CACHE_HOME="$ROOT/.cache" TORCH_HOME="$ROOT/.torch"
"$PYTHON" scripts/compare_b2_to_safe_fpn_feasibility_20261002.py \
  --config "$CONFIG" --reference-artifact-dir "$REFERENCE" \
  --previous-p3-scan "$PREVIOUS" --output "$OUTPUT" --seed 0 \
  > "$ROOT/fpn_separability.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$ROOT/fpn_separability.exitcode"
exit "$code"
