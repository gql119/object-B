"""Fail-closed Safe-Soft05-KD-ObjBal noise, audit, victim and evaluation pipeline."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from scripts.run_b2_to_safe_lr_pipeline_20261002 import run_stage, wait_gpu
from ue_framework.config import load_config


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent.parent
EXPECTED_STEPS = 9144
REVIEWED_FILES = {
    "ue_framework/methods/kd_utility.py",
    "ue_framework/methods/b2_safe_kd.py",
    "ue_framework/methods/b2_semantic.py",
    "ue_framework/methods/adaptive_learner.py",
    "ue_framework/methods/legacy_kproto_ret.py",
    "scripts/train_noise_once.py",
    "ue_framework/configs/exp_voc_person_b2_to_safe_soft05_kdcls_objbal_eps8_seed0_20261008_locked.yaml",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within_root(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Formal output escapes the isolated KD candidate: {resolved}")
    return resolved


def verify_noise(noise, config, calibration):
    status = json.loads((noise / "status.json").read_text(encoding="utf-8"))
    if (status.get("state") != "complete" or status.get("smoke")
            or status.get("steps") != EXPECTED_STEPS
            or status.get("epochs_completed") != 24
            or status.get("config_sha256") != sha256(config)
            or status.get("kd_calibration_sha256") != sha256(calibration)
            or status.get("kd_scale_mode") != "fixed_gradient_lambda"
            or status.get("kd_reduction") != "object_balanced"):
        raise RuntimeError("Formal noise identity, complete step count or KD binding failed")
    for prefix in ("frozen_state", "adaptive_state"):
        if status.get(prefix + "_sha256_before") != status.get(prefix + "_sha256_after"):
            raise RuntimeError(f"{prefix} changed during poison optimization")
    for name, key in (("kd_utility.py", "kd_utility_sha256"),
                      ("b2_safe_kd.py", "kd_trainer_sha256")):
        if status.get(key) != sha256(ROOT / "ue_framework/methods" / name):
            raise RuntimeError(f"KD source changed after noise generation: {name}")
    with (noise / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_STEPS:
        raise RuntimeError("Incomplete poison step diagnostics")
    for row in rows:
        for key, value in row.items():
            if key != "selected_proto_ids" and not math.isfinite(float(value)):
                raise RuntimeError(f"Nonfinite formal noise diagnostic: {key}")
        if (round(float(row["safe_projector_rank"])) != 8
                or round(float(row["num_selected_channels"])) != 24
                or float(row["kd_target_gt_count"]) != 0
                or float(row["kd_person_logit_in_distribution"]) != 0):
            raise RuntimeError("Safe-BG/TSFL/KD scientific invariant failed in noise")
    return {"steps": len(rows), "epochs": 24,
            "duration_seconds": status.get("duration_seconds"),
            "noise_status_sha256": sha256(noise / "status.json"),
            "diagnostics_csv_sha256": sha256(noise / "diagnostics.csv")}


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "calibration", "sanity-gate", "review", "noise", "victim", "report"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config, calibration, gate, review, noise, victim, report = [within_root(x) for x in
        (args.config, args.calibration, args.sanity_gate, args.review,
         args.noise, args.victim, args.report)]
    for path in (noise, victim, report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite formal output: {path}")
    cfg = load_config(str(config))
    method = cfg["methods"]["legacy_kproto_ret"]
    admission = json.loads(review.read_text(encoding="utf-8"))
    sanity = json.loads(gate.read_text(encoding="utf-8"))
    cal = json.loads(calibration.read_text(encoding="utf-8"))
    if (cal["gate"] != "KD_OBJECT_BALANCED_CALIBRATION_PASS"
            or sanity.get("state") != "passed"
            or sanity.get("config_sha256") != sha256(config)
            or admission.get("verdict") != "scientifically_correct"
            or admission.get("decision") != "allow_run"
            or admission.get("config_sha256") != sha256(config)
            or admission.get("calibration_sha256") != sha256(calibration)
            or method["universal_epochs"] != 24
            or method["universal_batch_size"] != 16
            or cfg["experiment"]["steps"] != [24]
            or cfg["victim"]["epochs"] != 200):
        raise RuntimeError("Locked calibration, sanity, review or protocol is not admitted")
    if set(admission.get("file_sha256", {})) != REVIEWED_FILES:
        raise RuntimeError("Scientific review receipt does not bind the complete file set")
    for name, digest in admission.get("file_sha256", {}).items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Reviewed scientific source changed: {name}")
    report.mkdir(parents=True, exist_ok=False)
    status_path = report / "pipeline_status.json"
    status = {"state": "running", "stage": "preflight", "started_at_unix": time.time(),
              "config_sha256": sha256(config), "calibration_sha256": sha256(calibration),
              "noise": str(noise), "victim": str(victim),
              "cleanup_policy": "retain_all"}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT),
               YOLO_CONFIG_DIR=str(ROOT / ".ultralytics"),
               MPLCONFIGDIR=str(ROOT / ".mpl"),
               XDG_CACHE_HOME=str(ROOT / ".cache"),
               TORCH_HOME=str(ROOT / ".torch"), WANDB_MODE="disabled")
    try:
        wait_gpu()
        run_stage("noise", [sys.executable, "scripts/train_noise_once.py", "--config",
                  str(config), "--seed", "0", "--output", str(noise)],
                  report, status_path, status, env, timeout=8 * 3600)
        status["poison_generation"] = verify_noise(noise, config, calibration)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        run_stage("materialization", [sys.executable, "scripts/materialize_noise_once.py",
                  "--config", str(config), "--seed", "0", "--noise", str(noise)],
                  report, status_path, status, env)
        audit = report / "dataset_audit"
        audit.mkdir()
        run_stage("dataset_audit", [sys.executable,
                  "scripts/audit_b2_to_safe_lr_materialization_20261002.py",
                  "--config", str(config), "--noise", str(noise),
                  "--output", str(audit), "--sample-count", "256"],
                  report, status_path, status, env)
        integrity = json.loads((audit / "dataset_integrity.json").read_text(encoding="utf-8"))
        if (integrity.get("state") != "passed" or integrity.get("source_image_count") != 16551
                or integrity.get("poisoned_image_count") != 16551
                or not integrity.get("labels_byte_identical")):
            raise RuntimeError("Full materialized data integrity gate did not pass")
        status["dataset_audit_sha256"] = sha256(audit / "dataset_integrity.json")
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        wait_gpu()
        run_stage("victim", [sys.executable, "scripts/run_isolated_victim.py",
                  "--config", str(config), "--seed", "0", "--noise", str(noise),
                  "--output", str(victim)], report, status_path, status, env)
        victim_status = json.loads((victim / "status.json").read_text(encoding="utf-8"))
        spec = json.loads((victim / "run_spec.json").read_text(encoding="utf-8"))
        with (victim / "victim/results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        weights = victim / "victim/weights"
        if (victim_status.get("state") != "complete" or len(rows) != 200
                or spec["train_args"]["epochs"] != 200
                or not (weights / "last.pt").is_file()
                or not (weights / "best.pt").is_file()):
            raise RuntimeError("Random-init victim did not complete 200 epochs")
        run_stage("checkpoint_evaluation", [sys.executable,
                  "scripts/evaluate_b2_to_safebg_checkpoints.py", "--config", str(config),
                  "--victim", str(victim), "--output", str(report / "checkpoint_evaluation")],
                  report, status_path, status, env)
        reference = PARENT / "experiments/safe_bg_ablation_20261004/formal"
        run_stage("comparison", [sys.executable, "-m", "scripts.compare_safe_soft05_kd_objbal",
                  "--clean-last", str(reference / "reference_last_eval/clean_last_metrics.json"),
                  "--safe-last", str(reference / "soft05_checkpoint_evaluation/primary_last_metrics.json"),
                  "--kd-last", str(report / "checkpoint_evaluation/primary_last_metrics.json"),
                  "--kd-best", str(report / "checkpoint_evaluation/secondary_best_metrics.json"),
                  "--output", str(report / "comparison")],
                  report, status_path, status, env)
        status.update(state="complete", stage="finished", completed_at_unix=time.time(),
                      primary_checkpoint="last.pt", secondary_checkpoint="best.pt")
    except BaseException as error:
        status.update(state="failed", error=repr(error), completed_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
