#!/usr/bin/env bash
set -u
ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926
cd "$ROOT" || exit 2
CONFIG=ue_framework/configs/exp_voc_person_b2_to_lr001_orth0005_eps8_overlap4_20260927.yaml
ADMISSION=experiments/b2_to_eps8_overlap4_20260927/admission.json
NOISE="$ROOT/noise_full_b2_to_eps8_overlap4_seed0_20260927"
VICTIM="$ROOT/victim_b2_to_eps8_overlap4_seed0_20260927"
STATUS="$ROOT/pipeline_b2_to_eps8_overlap4_seed0_20260927.json"
REFERENCE="$ROOT/runs_b2_to_lr001_orth0005/poisoned_datasets/legacy_kproto_ret/steps24/seed0"
if [ ! -f "$ADMISSION" ]; then
  echo "Missing reviewed admission receipt: $ADMISSION" >&2
  exit 2
fi
for path in "$NOISE" "$VICTIM" "$STATUS" "$ROOT/pipeline_b2_to_eps8_overlap4_seed0_20260927.console.log" "$ROOT/pipeline_b2_to_eps8_overlap4_seed0_20260927.exitcode"; do
  [ ! -e "$path" ] || { echo "Refusing to overwrite $path" >&2; exit 2; }
done
/usr/bin/env YOLO_CONFIG_DIR="$ROOT/.ultralytics" MPLCONFIGDIR="$ROOT/.mpl" \
  XDG_CACHE_HOME="$ROOT/.cache" TORCH_HOME="$ROOT/.torch" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. WANDB_MODE=disabled \
  /project/guoqinglong/miniconda3/envs/cfbdm/bin/python scripts/run_b2_to_auto_pipeline.py \
  --config "$CONFIG" --admission "$ADMISSION" --noise "$NOISE" \
  --victim "$VICTIM" --status "$STATUS" --seed 0 \
  --budget-reference-poison-root "$REFERENCE" \
  > "$ROOT/pipeline_b2_to_eps8_overlap4_seed0_20260927.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$ROOT/pipeline_b2_to_eps8_overlap4_seed0_20260927.exitcode"
exit "$code"
