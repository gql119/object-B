#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/kdcls_20261007/ue_project
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
RUNROOT="$ROOT/runs_safe_soft05_kdcls_objbal_eps8_person_seed0_20261008"
CONFIG="$ROOT/ue_framework/configs/exp_voc_person_b2_to_safe_soft05_kdcls_objbal_eps8_seed0_20261008_locked.yaml"
CAL="$ROOT/calibration_objbal_seed0_20261008/kd_preregistered_lambda.json"
GATE="$ROOT/objbal_sanity_gate_det1_20261008.json"
REVIEW="$ROOT/prerun_review_objbal_20261008.json"
NOISE="$RUNROOT/noise_full_safe_soft05_kdcls_objbal_seed0_20261008"
VICTIM="$RUNROOT/victim_safe_soft05_kdcls_objbal_seed0_20261008"
REPORT="$RUNROOT/formal_safe_soft05_kdcls_objbal_seed0_20261008"
POISON="$RUNROOT/poisoned_datasets/legacy_kproto_ret/steps24/seed0"
EXITCODE="$ROOT/kd_objbal_formal_20261008.exitcode"

cd "$ROOT"
for output in "$NOISE" "$VICTIM" "$REPORT" "$POISON" "$EXITCODE"; do
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite $output" >&2
        exit 3
    fi
done
set +e
env PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR="$ROOT/.ultralytics" MPLCONFIGDIR="$ROOT/.mpl" \
    XDG_CACHE_HOME="$ROOT/.cache" TORCH_HOME="$ROOT/.torch" WANDB_MODE=disabled \
    "$PYTHON" scripts/run_kd_objbal_formal_pipeline_20261008.py \
    --config "$CONFIG" --calibration "$CAL" --sanity-gate "$GATE" \
    --review "$REVIEW" --noise "$NOISE" --victim "$VICTIM" --report "$REPORT"
result=$?
printf '%s\n' "$result" > "$EXITCODE"
exit "$result"
