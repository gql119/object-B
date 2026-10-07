#!/usr/bin/env bash
set -u

ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/kdcls_20261007/ue_project
RUN=gradient_audit_mb16_a270038_seed0_20261007
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
CONFIG=ue_framework/configs/exp_voc_person_b2_to_safe_soft05_kdcls_mb16_eps8_seed0_20261007.yaml

cd "$ROOT" || exit 2
if [[ -e "$RUN" || -e "$RUN.console.log" || -e "$RUN.exitcode" ]]; then
    echo "Refusing to reuse KD gradient audit RunID" >&2
    exit 3
fi
PYTHONPATH=.:scripts "$PYTHON" scripts/audit_safe_soft05_kdcls_gradients.py \
    --config "$CONFIG" --output "$RUN" > "$RUN.console.log" 2>&1
result=$?
printf '%s\n' "$result" > "$RUN.exitcode"
exit "$result"
