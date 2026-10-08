#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/kdcls_20261007/ue_project
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
TEMPLATE=ue_framework/configs/exp_voc_person_b2_to_safe_soft05_kdcls_objbal_eps8_seed0_20261008.yaml
CONFIG=ue_framework/configs/exp_voc_person_b2_to_safe_soft05_kdcls_objbal_eps8_seed0_20261008_locked.yaml
PARENT=ue_framework/configs/exp_voc_person_b2_to_safe_soft05_eps8_seed0_20261004.yaml
CAL=calibration_objbal_seed0_20261008/kd_preregistered_lambda.json
FIXED=calibration_multibatch_6ae9258_seed0_20261007/calibration.json
SMOKE=runs_safe_soft05_kdcls_objbal_eps8_person_seed0_20261008/smoke5_objbal_seed0_20261008
PROBE=probe_parent_ret_objbal_seed0_20261008.json
PIXEL=smoke5_objbal_preview_audit_20261008.json
GATE=objbal_sanity_gate_20261008.json

cd "$ROOT"
if [[ -e "$CONFIG" || -e "$SMOKE" || -e "$PROBE" || -e "$PIXEL" || -e "$GATE" ]]; then
    echo "Refusing to overwrite any post-calibration output" >&2
    exit 3
fi
PYTHONPATH=.:scripts "$PYTHON" -m unittest discover -s tests -p test_kd_object_balanced.py
PYTHONPATH=.:scripts "$PYTHON" scripts/bind_kd_objbal_lambda_20261008.py \
    --template "$TEMPLATE" --calibration "$CAL" --output "$CONFIG"
PYTHONPATH=.:scripts "$PYTHON" scripts/probe_kd_objbal_parent_equivalence.py \
    --parent-config "$PARENT" --candidate-config "$CONFIG" \
    --fixed-set "$FIXED" --output "$PROBE"
mkdir -p "$(dirname "$SMOKE")"
PYTHONPATH=.:scripts "$PYTHON" scripts/train_noise_once.py \
    --config "$CONFIG" --seed 0 --output "$SMOKE" \
    --smoke-max-images 16 --smoke-max-steps 5
PYTHONPATH=.:scripts "$PYTHON" scripts/audit_safe_soft05_kdcls_preview.py \
    --config "$CONFIG" --noise "$SMOKE" --output "$PIXEL"
PYTHONPATH=.:scripts "$PYTHON" scripts/check_kd_objbal_sanity_20261008.py \
    --config "$CONFIG" --parent-config "$PARENT" --calibration "$CAL" \
    --smoke "$SMOKE" --preview-audit "$PIXEL" --output "$GATE"
