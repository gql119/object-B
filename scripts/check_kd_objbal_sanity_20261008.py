"""Fail-closed five-step gate for the calibrated object-balanced KD candidate."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from ue_framework.config import load_config
from ue_framework.methods.b2_safe_kd import SAFE_SOFT05_CHANNELS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "parent-config", "calibration", "smoke", "preview-audit", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    parent = load_config(args.parent_config)
    kd_cfg = cfg["methods"]["legacy_kproto_ret"]["semantic"]["kd_cls"]
    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    smoke = Path(args.smoke)
    status = json.loads((smoke / "status.json").read_text(encoding="utf-8"))
    preview = json.loads(Path(args.preview_audit).read_text(encoding="utf-8"))
    with (smoke / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parent["platform"]["run_root"] = cfg["platform"]["run_root"]
    parent["methods"]["legacy_kproto_ret"]["semantic"]["artifact_dir"] = (
        cfg["methods"]["legacy_kproto_ret"]["semantic"]["artifact_dir"])
    parent["methods"]["legacy_kproto_ret"]["semantic"]["kd_cls"] = kd_cfg
    scientific_config_equal = parent == cfg
    sem = cfg["methods"]["legacy_kproto_ret"]["semantic"]
    safe = sem["safe_bg"]
    checks = {
        "five_outer_steps": status.get("state") == "complete" and status.get("smoke")
                            and status.get("steps") == 5 and len(rows) == 5,
        "config_sha256": status.get("config_sha256") == sha256(args.config),
        "calibration_sha256": kd_cfg["calibration_sha256"] == sha256(args.calibration)
                               and status.get("kd_calibration_sha256") == sha256(args.calibration),
        "calibration_pass": cal.get("gate") == "KD_OBJECT_BALANCED_CALIBRATION_PASS",
        "parent_scientific_config_unchanged": scientific_config_equal,
        "parent_L_ret_weight": cfg["methods"]["legacy_kproto_ret"]["loss"]["lambda_ret"] == .25,
        "parent_L_ret_source": status.get("code_sha256") ==
                               "6bfd9155ea4ed4aa90dba655d835407c0bfad92737b153adc0060215c0020ccb",
        "lambda_orth": sem.get("lambda_orth") == .0005,
        "safe_bg": safe.get("rank") == 8 and safe.get("active_k") == 3
                   and safe.get("removal_alpha") == .5,
        "kd_settings": kd_cfg.get("reduction") == "object_balanced"
                       and kd_cfg.get("tau") == 2.0
                       and kd_cfg.get("scale_mode") == "fixed_gradient_lambda",
        "source_hashes_frozen": bool(status.get("frozen_state_sha256_before"))
                                and bool(status.get("adaptive_state_sha256_before"))
                                and status.get("frozen_state_sha256_before")
                                == status.get("frozen_state_sha256_after")
                                and status.get("adaptive_state_sha256_before")
                                == status.get("adaptive_state_sha256_after"),
        "object_accounting": all(float(r["non_target_gt_count"]) + 1e-6 >=
                                 float(r["valid_kd_object_count"]) for r in rows),
        "KD_finite": all(math.isfinite(float(r["L_KD_raw"])) for r in rows),
        "person_GT_excluded": all(float(r["kd_target_gt_count"]) == 0 for r in rows),
        "person_logit_excluded": all(float(r["kd_person_logit_in_distribution"]) == 0 for r in rows),
        "teacher_assignment_shared": all(0 <= float(r["kd_teacher_student_index_identical"]) <= 1
                                         and (float(r["kd_positive_count"]) == 0 or
                                              float(r["kd_teacher_student_index_identical"]) > 0)
                                         for r in rows),
        "selected_24_channels": all(round(float(r["num_selected_channels"])) ==
                                    len(SAFE_SOFT05_CHANNELS) for r in rows),
        "budget_preview": preview.get("all_pass") is True,
        "all_finite": all(math.isfinite(float(value)) for r in rows for key, value in r.items()
                          if key != "selected_proto_ids"),
        "calibrated_lambda_consumed": all(abs(float(r["lambda_KD"])
                                           - cal["final_lambda_KD"]) < 1e-6 for r in rows),
    }
    receipt = {"state": "passed" if all(checks.values()) else "failed",
               "checks": checks, "config_sha256": sha256(args.config),
               "calibration_sha256": sha256(args.calibration),
               "smoke_status_sha256": sha256(smoke / "status.json"),
               "lambda_KD": cal["final_lambda_KD"],
               "selected_channel_indices": SAFE_SOFT05_CHANNELS,
               "smoke_steps": len(rows),
               "loss_series": [{key: float(r[key]) for key in
                                ("L_BG", "L_rel", "L_ret", "L_KD_raw",
                                 "L_orth_objective", "L_adaptive", "L_total",
                                 "grad_ret_norm", "grad_KD_norm", "grad_util_norm",
                                 "grad_attack_norm")} for r in rows]}
    output = Path(args.output)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps({"state": receipt["state"], "checks": checks}))
    if receipt["state"] != "passed":
        raise RuntimeError("Five-step object-balanced KD sanity gate failed")


if __name__ == "__main__":
    main()
