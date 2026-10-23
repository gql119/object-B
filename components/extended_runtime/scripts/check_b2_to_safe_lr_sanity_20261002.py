"""Audit the locked P5/rank-8 five-step experiment; never launch downstream work."""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from ue_framework.config import load_config


ROOT = Path(__file__).resolve().parents[1]
LS_CONFIG = ROOT / "ue_framework/configs/exp_voc_person_b2_to_safe_ls_p5_eps8_seed0_20261002.yaml"
LS_INDICES = [0, 6, 10, 29, 41, 68, 77, 86, 89, 112, 116, 125, 128, 133, 140, 167, 173, 184, 220, 229, 237, 238, 239, 240]


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def number(record, key):
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}")
    return value


def check(args):
    cfg = load_config(args.config)
    ls = load_config(str(LS_CONFIG))
    candidate = flatten(cfg)
    reference = flatten(ls)
    changed = {key for key in candidate.keys() | reference.keys()
               if candidate.get(key) != reference.get(key)}
    expected = {"platform.run_root", "methods.legacy_kproto_ret.semantic.safe_bg.mode",
                "methods.legacy_kproto_ret.semantic.safe_bg.rank"}
    if changed != expected:
        raise ValueError(f"Unexpected Safe-LR config changes: {sorted(changed ^ expected)}")
    sem = cfg["methods"]["legacy_kproto_ret"]["semantic"]
    if (sem["safe_bg"]["mode"] != "low_rank" or sem["safe_bg"]["rank"] != 8
            or sem["safe_bg"]["active_k"] != 3):
        raise ValueError("Prerecorded rank/mode/active classes changed")
    artifact = Path(sem["artifact_dir"])
    selected = torch.load(artifact / "target_specific_channels.pt", map_location="cpu", weights_only=True)["indices"].tolist()
    if selected != LS_INDICES:
        raise ValueError("P5 selected channels differ from the passed LS preflight")
    fixed = json.loads((Path(args.noise) / "fixed_pair.json").read_text(encoding="utf-8"))
    initial, final = fixed["initial"], fixed["after_5"]
    with (Path(args.noise) / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise ValueError("Five outer diagnostic steps required")
    for row in rows:
        for key, value in row.items():
            if key in ("selected_proto_ids",):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(parsed):
                raise ValueError(f"Nonfinite diagnostic {key}")
    # The original LS gate performs decoded-pixel, source, frozen-hash, and P3
    # formal-orth-reference checks. Keep that gate intact and add rank checks.
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_b2_to_safebg_sanity.py"),
                             "--config", args.config, "--noise", args.noise,
                             "--output", args.output], cwd=ROOT, text=True,
                            capture_output=True)
    if result.returncode:
        raise ValueError("Inherited integrity gate failed: " + (result.stderr or result.stdout)[-3000:])
    inherited = json.loads((Path(args.output) / "sanity_gate.json").read_text(encoding="utf-8"))
    rank = number(final, "safe_projector_rank")
    columns = number(final, "safe_projector_basis_columns")
    if abs(rank - 8) > 1e-6 or abs(columns - 8) > 1e-6:
        raise ValueError(f"Rank-8 projector failed: rank={rank}, columns={columns}")
    if number(final, "safe_projector_orthonormal_error") > 1e-5:
        raise ValueError("Q_r is not orthonormal")
    if number(final, "decomposition_error") > 1e-5:
        raise ValueError("Projection decomposition failed")
    if number(final, "safe_direction_retention_raw") < .1:
        raise ValueError("Median raw retention below 0.10")
    if number(final, "removed_component_ratio") <= 0:
        raise ValueError("No non-target component removed")
    if number(final, "safe_direction_norm") <= 0:
        raise ValueError("Safe direction collapsed")
    if abs(number(final, "cos_d_safe_projected_nt")) > 1e-3:
        raise ValueError("Removed and retained directions are not orthogonal")
    for key in ("L_BG_safe_lr", "L_rel_safe_lr", "grad_BG_safe_norm", "grad_rel_safe_norm",
                "grad_orth_norm", "cos_grad_attack_orth", "L_ret", "L_orth_objective"):
        number(final, key)
    if number(final, "grad_BG_safe_norm") <= 0 or number(final, "grad_rel_safe_norm") <= 0:
        raise ValueError("Safe loss has zero delta gradient")
    if (number(final, "safe_bg_feature_layer_id") != 21
            or number(final, "orth_feature_layer_id") != 15
            or number(final, "adaptive_feature_layer_id") != 15):
        raise ValueError("BG/orth/adaptive feature layers changed")
    if number(final, "safe_bg_rank_configured") != 8:
        raise ValueError("Configured Safe-BG rank changed")
    if abs(float(sem["lambda_orth"]) - .0005) > 1e-12:
        raise ValueError("Formal B2-TO lambda_orth changed")
    singular = [number(final, f"safe_projector_singular_{i}") for i in range(1, 17)]
    receipt = dict(inherited)
    receipt.update({
        "gate": "B2-TO-Safe-LR locked rank-8 five-step correctness",
        "selected_channel_indices_match_safe_ls": True,
        "safe_bg_rank": 8, "Q_r_shape": [24, 8], "projector_rank": rank,
        "top_singular_values": singular,
        "top8_energy_ratio": number(final, "safe_projector_top8_energy_ratio"),
        "median_safe_direction_retention_raw": number(final, "safe_direction_retention_raw"),
        "retention_quantiles": {q: number(final, f"safe_retention_{q}")
                                 for q in ("q10", "q25", "q75", "q90")},
        "mean_removed_component_ratio": number(final, "removed_component_ratio"),
        "cos_removed_safe": number(final, "cos_d_safe_projected_nt"),
        "decomposition_error": number(final, "decomposition_error"),
        "Q_r_orthonormal_error": number(final, "safe_projector_orthonormal_error"),
        "L_BG_safe_lr_initial_last": [number(initial, "L_BG_safe_lr"), number(final, "L_BG_safe_lr")],
        "L_rel_safe_lr_initial_last": [number(initial, "L_rel_safe_lr"), number(final, "L_rel_safe_lr")],
        "grad_BG_safe_lr_norm": number(final, "grad_BG_safe_norm"),
        "grad_rel_safe_lr_norm": number(final, "grad_rel_safe_norm"),
        "grad_orth_norm": number(final, "grad_orth_norm"),
        "grad_attack_vs_orth_cosine": number(final, "cos_grad_attack_orth"),
        "lambda_orth_formal_B2_TO_unchanged": True,
        "adaptive_unchanged": True, "L_ret_unchanged": True,
        "configured_eps": float(cfg["experiment"]["eps"]),
        "effective_target_eps": float(cfg["experiment"]["eps"]),
        "effective_overlap_eps": float(cfg["experiment"]["eps"]) * .5,
        "outside_target_max": inherited["decoded_preview_pixel_maxima"]["outside"],
        "BG_rel_layer": "P5/model.21", "orth_layer": "P3/model.15",
        "adaptive_layer": "P3/model.15",
        "raw_diagnostics": {key: number(final, key) for key in
                            ("L_BG_raw", "L_rel_raw", "cos_poison_raw_bg",
                             "cos_poison_safe_bg", "cos_clean_raw_bg", "cos_clean_safe_bg")},
    })
    (Path(args.output) / "lowrank_sanity_gate.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    try:
        report = check(args)
    except Exception as exc:
        report = {"state": "failed", "gate": "B2-TO-Safe-LR five-step correctness",
                  "reason": str(exc), "noise": args.noise}
        (out / "lowrank_safebg_failure_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit(1)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
