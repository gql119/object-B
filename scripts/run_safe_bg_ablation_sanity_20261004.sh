#!/usr/bin/env bash
set -u
ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate
PYTHON=/project/guoqinglong/miniconda3/envs/cfbdm/bin/python
cd "$ROOT" || exit 2
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT" WANDB_MODE=disabled
export YOLO_CONFIG_DIR="$ROOT/.ultralytics" MPLCONFIGDIR="$ROOT/.mpl"
export XDG_CACHE_HOME="$ROOT/.cache" TORCH_HOME="$ROOT/.torch"
for variant in r4 soft05; do
    if [ "$variant" = r4 ]; then
        run_id=b2_to_safe_r4_eps8_person_seed0_20261004
        config="$ROOT/ue_framework/configs/exp_voc_person_b2_to_safe_r4_eps8_seed0_20261004.yaml"
    else
        run_id=b2_to_safe_soft05_eps8_person_seed0_20261004
        config="$ROOT/ue_framework/configs/exp_voc_person_b2_to_safe_soft05_eps8_seed0_20261004.yaml"
    fi
    noise="$ROOT/noise_sanity_${run_id}_5steps"
    report="$ROOT/experiments/safe_bg_ablation_20261004/sanity_${variant}"
    if [ -e "$noise" ] || [ -e "$report" ]; then
        printf 'Sanity path collision for %s\n' "$variant" >&2
        exit 3
    fi
    mkdir -p "$ROOT/experiments/safe_bg_ablation_20261004"
    "$PYTHON" scripts/train_noise_once.py --config "$config" --seed 0 \
        --smoke-max-images 5 --smoke-max-steps 5 --output "$noise" \
        > "$ROOT/experiments/safe_bg_ablation_20261004/sanity_${variant}.console.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$ROOT/experiments/safe_bg_ablation_20261004/sanity_${variant}.train.exitcode"
    if [ "$code" -ne 0 ]; then exit "$code"; fi
    "$PYTHON" scripts/check_safe_bg_ablation_sanity_20261004.py --variant "$variant" \
        --config "$config" --noise "$noise" --output "$report" \
        > "$ROOT/experiments/safe_bg_ablation_20261004/sanity_${variant}.gate.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$ROOT/experiments/safe_bg_ablation_20261004/sanity_${variant}.gate.exitcode"
    if [ "$code" -ne 0 ]; then exit "$code"; fi
done
