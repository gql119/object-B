"""Gate the separate R4 and Soft05 five-step production smokes."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from ue_framework.config import load_config


ROOT = Path(__file__).resolve().parents[1]
PARENT_CONFIG = ROOT / "ue_framework/configs/exp_voc_person_b2_to_safe_lr_p5_rank8_eps8_seed0_20261002.yaml"
SELECTED = [0, 6, 10, 29, 41, 68, 77, 86, 89, 112, 116, 125, 128, 133,
            140, 167, 173, 184, 220, 229, 237, 238, 239, 240]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result.update(flatten(value, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: obj}


def finite(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}")
    return value


def check(args):
    config, noise, output = (Path(args.config).resolve(), Path(args.noise).resolve(),
                             Path(args.output).resolve())
    if not all(path.is_relative_to(ROOT.resolve()) for path in (config, noise, output)):
        raise ValueError("A sanity path escapes the isolated candidate")
    if output.exists():
        raise FileExistsError(output)
    cfg, parent = load_config(str(config)), load_config(str(PARENT_CONFIG))
    diff = {key for key in flatten(cfg).keys() | flatten(parent).keys()
            if flatten(cfg).get(key) != flatten(parent).get(key)}
    allowed = {"platform.run_root", "methods.legacy_kproto_ret.semantic.artifact_dir",
               "methods.legacy_kproto_ret.semantic.safe_bg.rank",
               "methods.legacy_kproto_ret.semantic.safe_bg.removal_alpha"}
    expected = allowed - {"methods.legacy_kproto_ret.semantic.safe_bg.removal_alpha"} if args.variant == "r4" else allowed - {"methods.legacy_kproto_ret.semantic.safe_bg.rank"}
    if diff != expected:
        raise ValueError(f"Unexpected scientific config diff: {sorted(diff ^ expected)}")
    sem = cfg["methods"]["legacy_kproto_ret"]["semantic"]
    safe = sem["safe_bg"]
    rank, alpha = (4, 1.0) if args.variant == "r4" else (8, 0.5)
    if (safe["rank"] != rank or float(safe.get("removal_alpha", 1.0)) != alpha
            or safe["active_k"] != 3 or sem["lambda_orth"] != 0.0005
            or sem["orth_worst_k"] != 3 or sem["strict_tsfl"]["strict_tsfl_feature_level"] != "P5"
            or abs(float(cfg["experiment"]["eps"]) - 8/255) > 1e-9):
        raise ValueError("Frozen method parameters changed")
    selected = torch.load(Path(sem["artifact_dir"]) / "target_specific_channels.pt",
                          map_location="cpu", weights_only=True)["indices"].tolist()
    if selected != SELECTED:
        raise ValueError("P5 selected channel indices differ from parent")
    status = json.loads((noise / "status.json").read_text())
    if status.get("state") != "complete" or status.get("steps") != 5 or not status.get("smoke"):
        raise ValueError("Five complete production outer steps are required")
    output.mkdir(parents=True)
    inherited = subprocess.run([sys.executable, str(ROOT / "scripts/check_b2_to_safebg_sanity.py"),
        "--config", str(config), "--noise", str(noise), "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True)
    if inherited.returncode:
        raise ValueError((inherited.stderr or inherited.stdout)[-3000:])
    gate = json.loads((output / "sanity_gate.json").read_text())
    pair = json.loads((noise / "fixed_pair.json").read_text())
    before, after = pair["initial"], pair["after_5"]
    rows = list(csv.DictReader((noise / "diagnostics.csv").open(newline="", encoding="utf-8")))
    if len(rows) != 5:
        raise ValueError("Diagnostics do not contain five outer steps")
    for row in rows:
        if int(round(finite(row, "safe_projector_rank"))) != rank:
            raise ValueError("Projector rank drifted")
        if finite(row, "decomposition_error") > 1e-5 or finite(row, "safe_projector_orthonormal_error") > 1e-5:
            raise ValueError("Projector numerical check failed")
    if (finite(after, "safe_projector_rank") != rank or
            finite(after, "safe_projector_basis_columns") != rank or
            finite(after, "safe_direction_retention_raw") < 0.10 or
            finite(after, "removed_component_ratio") <= 0 or
            finite(after, "safe_direction_norm") <= 0 or
            finite(after, "grad_BG_safe_norm") <= 0 or
            finite(after, "grad_rel_safe_norm") <= 0 or
            finite(after, "safe_projector_orthonormal_error") > 1e-5 or
            finite(after, "decomposition_error") > 1e-5):
        raise ValueError("Correctness or gradient gate failed")
    projected = finite(after, "projected_component_norm")
    removed = finite(after, "removed_component_norm")
    retained = finite(after, "retained_projected_component_norm")
    if projected <= 0 or abs(removed / projected - alpha) > 1e-5 or abs(retained / projected - (1-alpha)) > 1e-5:
        raise ValueError("Attenuation diagnostics disagree with the preregistered alpha")
    if abs(finite(after, "safe_bg_removal_alpha") - alpha) > 1e-6:
        raise ValueError("The alpha configuration did not reach the computation sink")
    for key in ("L_BG_safe_lr", "L_rel_safe_lr", "L_ret", "L_orth_objective",
                "L_adaptive", "target_displacement_norm", "mean_non_target_feature_drift",
                "grad_orth_norm", "grad_adapt_norm", "cos_grad_attack_orth"):
        finite(after, key)
    receipt = {"state": "passed", "variant": args.variant, "config_sha256": sha256(config),
               "noise_status_sha256": sha256(noise / "status.json"), "steps": 5, "seed": 0,
               "selected_layer": "P5", "selected_channel_indices": selected, "rank": rank,
               "Q_shape": [24, rank], "alpha": alpha,
               "median_retention": finite(after, "safe_direction_retention_raw"),
               "removed_ratio": finite(after, "removed_component_ratio"),
               "effective_removal_ratio": finite(after, "effective_removal_ratio"),
               "retained_projected_ratio": retained / projected,
               "L_BG_first_last": [finite(before, "L_BG_safe_lr"), finite(after, "L_BG_safe_lr")],
               "L_rel_first_last": [finite(before, "L_rel_safe_lr"), finite(after, "L_rel_safe_lr")],
               "attack_orth_cosine": finite(after, "cos_grad_attack_orth"),
               "grad_BG": finite(after, "grad_BG_safe_norm"),
               "grad_rel": finite(after, "grad_rel_safe_norm"),
               "projector_orthonormal_error": finite(after, "safe_projector_orthonormal_error"),
               "decomposition_error": finite(after, "decomposition_error"),
               "pixel_maxima": gate["decoded_preview_pixel_maxima"],
               "lambda_orth_unchanged": True, "adaptive_unchanged": True,
               "epsilon_unchanged": True, "final_validation_used_for_selection": False}
    (output / "ablation_sanity_gate.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("r4", "soft05"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
