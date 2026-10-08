#!/usr/bin/env bash
set -u

ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/kdcls_20261007/ue_project
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
RUN=kd_mechanism_diagnostic_20261009
CONFIG=$ROOT/runs_safe_soft05_kdcls_objbal_eps8_person_seed0_20261008/noise_full_safe_soft05_kdcls_objbal_seed0_20261008/source_config.yaml
FIXED=$ROOT/calibration_multibatch_6ae9258_seed0_20261007/calibration.json
CACHE=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt
GLOBAL=$ROOT/runs_safe_soft05_kdcls_objbal_eps8_person_seed0_20261008/noise_full_safe_soft05_kdcls_objbal_seed0_20261008/global_params.pt
COMPARISON=$ROOT/runs_safe_soft05_kdcls_objbal_eps8_person_seed0_20261008/formal_safe_soft05_kdcls_objbal_seed0_20261008/comparison/comparison.json

cd "$ROOT" || exit 2
if [[ -e "$RUN" || -e "$RUN.console.log" || -e "$RUN.exitcode" ]]; then
    echo "Refusing to reuse KD diagnostic RunID" >&2
    exit 3
fi
env PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR="$ROOT/.ultralytics" MPLCONFIGDIR="$ROOT/.mpl" \
    XDG_CACHE_HOME="$ROOT/.cache" TORCH_HOME="$ROOT/.torch" WANDB_MODE=disabled \
    "$PYTHON" scripts/kd_mechanism_diagnostic_20261009.py \
    --config "$CONFIG" --fixed-set "$FIXED" --parent-cache "$CACHE" \
    --global-params "$GLOBAL" --comparison "$COMPARISON" --output "$RUN" \
    > "$RUN.console.log" 2>&1
result=$?
printf '%s\n' "$result" > "$RUN.exitcode"
exit "$result"
