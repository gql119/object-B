#!/usr/bin/env bash
set -u

ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/kdcls_20261007/ue_project
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
RUN=calibration_objbal_seed0_20261008
CONFIG=ue_framework/configs/exp_voc_person_b2_to_safe_soft05_kdcls_objbal_eps8_seed0_20261008.yaml
FIXED=calibration_multibatch_6ae9258_seed0_20261007/calibration.json
OLD=gradient_audit_mb16_a270038_seed0_20261007/kd_gradient_calibration_summary.json

cd "$ROOT" || exit 2
if [[ -e "$RUN" || -e "$RUN.console.log" || -e "$RUN.exitcode" ]]; then
    echo "Refusing to reuse object-balanced calibration RunID" >&2
    exit 3
fi
PYTHONPATH=.:scripts "$PYTHON" scripts/calibrate_safe_soft05_kdcls_objbal_gradients.py \
    --config "$CONFIG" --fixed-set "$FIXED" --old-gradient-summary "$OLD" \
    --output "$RUN" > "$RUN.console.log" 2>&1
result=$?
printf '%s\n' "$result" > "$RUN.exitcode"
exit "$result"
