"""Run two independent Safe-BG ablations after separate five-step admission."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import torch

from scripts.run_b2_to_safe_lr_pipeline_20261002 import run_stage, wait_gpu
from ue_framework.config import load_config
from ue_framework.methods.semantic_coordinates import semantic_identity


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent.parent
SEMANTIC_ROOT = PARENT.parent.parent
REPORT = ROOT / "experiments/safe_bg_ablation_20261004/formal"
ADMISSION = ROOT / "experiments/safe_bg_ablation_20261004/prerun_review.json"
CLEAN = SEMANTIC_ROOT / "baseline_dev_20260928/clean_baseline_seed0_20260928"
B2TO = SEMANTIC_ROOT / "victim_b2_to_lr001_orth0005_seed0_20260927"
SAFE_LR = PARENT / "victim_b2_to_safe_lr_p5_rank8_eps8_resume2_seed0_20261003"
LOCK = SEMANTIC_ROOT / "baseline_dev_20260928/protocol_lock.json"
SELECTED = [0, 6, 10, 29, 41, 68, 77, 86, 89, 112, 116, 125, 128, 133,
            140, 167, 173, 184, 220, 229, 237, 238, 239, 240]
VARIANTS = {
    "r4": ("b2_to_safe_r4_eps8_person_seed0_20261004",
           "exp_voc_person_b2_to_safe_r4_eps8_seed0_20261004.yaml", 4, 1.0),
    "soft05": ("b2_to_safe_soft05_eps8_person_seed0_20261004",
               "exp_voc_person_b2_to_safe_soft05_eps8_seed0_20261004.yaml", 8, 0.5),
}
PINNED_SOURCE = (
    "ue_framework/methods/safebg.py", "ue_framework/methods/b2_semantic.py",
    "scripts/train_noise_once.py", "scripts/materialize_noise_once.py",
    "scripts/run_isolated_victim.py", "scripts/audit_b2_to_safe_lr_materialization_20261002.py",
    "scripts/evaluate_b2_to_safebg_checkpoints.py",
    "scripts/evaluate_safe_lr_reference_last_20261002.py",
    "scripts/analyze_safe_bg_rank_ablation_20261004.py",
    "scripts/audit_safe_bg_ablation_fingerprint_20261004.py",
    "scripts/run_safe_bg_rank_ablation_pipeline_20261004.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locations(variant):
    run_id, config_name, rank, alpha = VARIANTS[variant]
    return {"run_id": run_id, "rank": rank, "alpha": alpha,
            "config": ROOT / "ue_framework/configs" / config_name,
            "gate": ROOT / "experiments/safe_bg_ablation_20261004" / f"sanity_{variant}/ablation_sanity_gate.json",
            "noise": ROOT / f"noise_full_{run_id}",
            "victim": ROOT / f"victim_{run_id}",
            "report": REPORT / variant}


def validate():
    if REPORT.exists():
        raise FileExistsError(f"Formal report already exists: {REPORT}")
    proof = json.loads(ADMISSION.read_text(encoding="utf-8"))
    if (proof.get("independent_review_verdict") != "scientifically_incorrect"
            or proof.get("blockers_closed") is not True
            or proof.get("decision") != "allow_run_after_blocker_closure"):
        raise RuntimeError("Independent review blockers have not been closed")
    expected = {name: sha256(ROOT / name) for name in PINNED_SOURCE}
    if proof.get("candidate_file_sha256") != expected:
        raise RuntimeError("Reviewed scientific candidate changed")
    for variant in VARIANTS:
        item = locations(variant)
        if item["noise"].exists() or item["victim"].exists():
            raise FileExistsError(f"Output path collision for {variant}")
        gate = json.loads(item["gate"].read_text(encoding="utf-8"))
        if (gate.get("state") != "passed" or gate.get("variant") != variant
                or gate.get("config_sha256") != sha256(item["config"])
                or gate.get("rank") != item["rank"] or gate.get("alpha") != item["alpha"]
                or gate.get("selected_layer") != "P5"
                or gate.get("selected_channel_indices") != SELECTED):
            raise RuntimeError(f"Unbound five-step correctness receipt: {variant}")
        if proof.get("config_sha256", {}).get(variant) != sha256(item["config"]):
            raise RuntimeError(f"Reviewed config changed: {variant}")
        cfg = load_config(str(item["config"]))
        method = cfg["methods"]["legacy_kproto_ret"]
        safe = method["semantic"]["safe_bg"]
        artifact = Path(method["semantic"]["artifact_dir"])
        artifact_indices = torch.load(artifact / "target_specific_channels.pt",
                                      map_location="cpu", weights_only=True)["indices"].tolist()
        smoke_status = ROOT / f"noise_sanity_{item['run_id']}_5steps/status.json"
        smoke = json.loads(smoke_status.read_text(encoding="utf-8"))
        if (artifact_indices != SELECTED or sha256(smoke_status) != gate.get("noise_status_sha256")
                or smoke.get("state") != "complete" or smoke.get("steps") != 5
                or smoke.get("semantic_identity") != semantic_identity(cfg, ROOT)):
            raise RuntimeError(f"Sanity/semantic artifact binding failed: {variant}")
        if (method["universal_epochs"] != 24 or method["universal_batch_size"] != 16
                or cfg["experiment"]["steps"] != [24] or abs(cfg["experiment"]["eps"] - 8/255) > 1e-9
                or safe["rank"] != item["rank"] or float(safe.get("removal_alpha", 1.0)) != item["alpha"]):
            raise RuntimeError(f"Formal noise settings drifted: {variant}")
        item["cfg"] = cfg
    for name, root in (("clean", CLEAN), ("B2-TO", B2TO), ("Safe-LR", SAFE_LR)):
        if not (root / "victim/weights/last.pt").is_file():
            raise FileNotFoundError(f"Missing {name} last.pt reference")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    clean_spec = json.loads((CLEAN / "run_spec.json").read_text(encoding="utf-8"))
    if (lock["clean_baseline"]["protocol_lock_sha256_at_training"] != clean_spec["protocol_lock_sha256"]
            or lock["dataset"]["fingerprint"] != clean_spec["dataset_fingerprint"]):
        raise RuntimeError("Clean baseline protocol lock lineage or VOC fingerprint differs")
    return {variant: locations(variant) for variant in VARIANTS}


def verify_noise(item):
    noise = item["noise"]
    status = json.loads((noise / "status.json").read_text(encoding="utf-8"))
    expected_steps = 24 * math.ceil(6095 / 16)
    if (status.get("state") != "complete" or status.get("smoke")
            or status.get("steps") != expected_steps or status.get("epochs_completed") != 24
            or status.get("config_sha256") != sha256(item["config"])):
        raise RuntimeError(f"Incomplete or mismatched noise: {item['run_id']}")
    for name in ("frozen_state_sha256", "adaptive_state_sha256"):
        if status.get(name + "_before") != status.get(name + "_after"):
            raise RuntimeError(f"Frozen {name} drifted")
    count, min_retention = 0, float("inf")
    with (noise / "diagnostics.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            count += 1
            for key, raw in row.items():
                if key == "selected_proto_ids":
                    continue
                if not math.isfinite(float(raw)):
                    raise RuntimeError(f"Nonfinite {key} in {item['run_id']}")
            if (int(round(float(row["safe_projector_rank"]))) != item["rank"]
                    or abs(float(row["safe_bg_removal_alpha"]) - item["alpha"]) > 1e-6
                    or float(row["decomposition_error"]) > 1e-5
                    or float(row["safe_projector_orthonormal_error"]) > 1e-5):
                raise RuntimeError(f"Projection drift in {item['run_id']}")
            min_retention = min(min_retention, float(row["safe_direction_retention_raw"]))
    if count != expected_steps:
        raise RuntimeError(f"Expected {expected_steps} noise rows, found {count}")
    return {"run_id": item["run_id"], "state": "complete", "epochs": 24,
            "outer_steps": count, "rank": item["rank"], "alpha": item["alpha"],
            "wall_clock_seconds": status.get("duration_seconds"),
            "min_step_median_retention": min_retention,
            "global_params_sha256": sha256(noise / "global_params.pt"),
            "diagnostics_sha256": sha256(noise / "diagnostics.csv")}


def verify_victim(item):
    victim = item["victim"]
    status = json.loads((victim / "status.json").read_text(encoding="utf-8"))
    spec = json.loads((victim / "run_spec.json").read_text(encoding="utf-8"))
    with (victim / "victim/results.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if (status.get("state") != "complete" or len(rows) != 200
            or spec["train_args"]["epochs"] != 200
            or spec["train_args"]["seed"] != 0
            or spec["train_args"]["pretrained"] is not False
            or not (victim / "victim/weights/last.pt").is_file()
            or not (victim / "victim/weights/best.pt").is_file()):
        raise RuntimeError(f"Victim incomplete: {item['run_id']}")


def main():
    items = validate()
    REPORT.mkdir(parents=True, exist_ok=False)
    status_path = REPORT / "pipeline_status.json"
    status = {"state": "running", "stage": "preflight", "started_at_unix": time.time(),
              "order": ["r4_noise", "soft05_noise", "r4_materialization", "soft05_materialization",
                        "r4_integrity", "soft05_integrity", "r4_victim", "soft05_victim", "evaluation"],
              "primary_checkpoint": "last.pt", "secondary_checkpoint": "best.pt",
              "cleanup_policy": "retain_all"}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT),
               YOLO_CONFIG_DIR=str(ROOT / ".ultralytics"), MPLCONFIGDIR=str(ROOT / ".mpl"),
               XDG_CACHE_HOME=str(ROOT / ".cache"), TORCH_HOME=str(ROOT / ".torch"),
               WANDB_MODE="disabled")
    try:
        run_stage("locked_source_fingerprint", [sys.executable,
            "scripts/audit_safe_bg_ablation_fingerprint_20261004.py", "--config", str(items["r4"]["config"]),
            "--lock", str(LOCK), "--output", str(REPORT / "source_fingerprint.json")],
            REPORT, status_path, status, env, timeout=2*3600)
        for variant in VARIANTS:
            item = items[variant]
            wait_gpu()
            run_stage(f"{variant}_noise", [sys.executable, "scripts/train_noise_once.py",
                "--config", str(item["config"]), "--seed", "0", "--output", str(item["noise"])],
                REPORT, status_path, status, env, timeout=8*3600)
            summary = verify_noise(item)
            (REPORT / f"{variant}_noise_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for variant in VARIANTS:
            item = items[variant]
            wait_gpu()
            run_stage(f"{variant}_materialization", [sys.executable,
                "scripts/materialize_noise_once.py", "--config", str(item["config"]),
                "--seed", "0", "--noise", str(item["noise"])], REPORT, status_path, status, env)
        for variant in VARIANTS:
            item = items[variant]
            audit_dir = REPORT / f"{variant}_integrity"
            audit_dir.mkdir(exist_ok=False)
            run_stage(f"{variant}_integrity", [sys.executable,
                "scripts/audit_b2_to_safe_lr_materialization_20261002.py", "--config", str(item["config"]),
                "--noise", str(item["noise"]), "--output", str(audit_dir), "--sample-count", "256"],
                REPORT, status_path, status, env, timeout=4*3600)
            audit = json.loads((audit_dir / "dataset_integrity.json").read_text(encoding="utf-8"))
            if audit.get("state") != "passed" or audit.get("run_id") != item["run_id"]:
                raise RuntimeError(f"Dataset audit or result attribution failed: {variant}")
        run_stage("locked_source_fingerprint_before_victims", [sys.executable,
            "scripts/audit_safe_bg_ablation_fingerprint_20261004.py", "--config", str(items["r4"]["config"]),
            "--lock", str(LOCK), "--output", str(REPORT / "source_fingerprint_before_victims.json")],
            REPORT, status_path, status, env, timeout=2*3600)
        first_fingerprint = json.loads((REPORT / "source_fingerprint.json").read_text())
        last_fingerprint = json.loads((REPORT / "source_fingerprint_before_victims.json").read_text())
        if first_fingerprint != last_fingerprint:
            raise RuntimeError("Locked VOC source changed during poison generation")
        for variant in VARIANTS:
            item = items[variant]
            wait_gpu()
            run_stage(f"{variant}_victim", [sys.executable, "scripts/run_isolated_victim.py",
                "--config", str(item["config"]), "--seed", "0", "--noise", str(item["noise"]),
                "--output", str(item["victim"])], REPORT, status_path, status, env)
            verify_victim(item)
        for variant in VARIANTS:
            item = items[variant]
            wait_gpu()
            run_stage(f"{variant}_checkpoint_evaluation", [sys.executable,
                "scripts/evaluate_b2_to_safebg_checkpoints.py", "--config", str(item["config"]),
                "--victim", str(item["victim"]), "--output", str(REPORT / f"{variant}_checkpoint_evaluation")],
                REPORT, status_path, status, env)
        r4, soft = items["r4"], items["soft05"]
        wait_gpu()
        run_stage("reference_last_evaluation", [sys.executable,
            "scripts/evaluate_safe_lr_reference_last_20261002.py", "--config", str(r4["config"]),
            "--clean-victim", str(CLEAN), "--b2to-victim", str(B2TO),
            "--safe-lr-victim", str(SAFE_LR), "--data-yaml", str(r4["victim"] / "train_data.yaml"),
            "--output", str(REPORT / "reference_last_eval")], REPORT, status_path, status, env)
        run_stage("four_method_analysis", [sys.executable,
            "scripts/analyze_safe_bg_rank_ablation_20261004.py",
            "--clean", str(REPORT / "reference_last_eval/clean_last_metrics.json"),
            "--b2to", str(REPORT / "reference_last_eval/b2to_last_metrics.json"),
            "--safe-lr", str(REPORT / "reference_last_eval/safe_lr_last_metrics.json"),
            "--r4", str(REPORT / "r4_checkpoint_evaluation/primary_last_metrics.json"),
            "--soft05", str(REPORT / "soft05_checkpoint_evaluation/primary_last_metrics.json"),
            "--r4-best", str(REPORT / "r4_checkpoint_evaluation/secondary_best_metrics.json"),
            "--soft05-best", str(REPORT / "soft05_checkpoint_evaluation/secondary_best_metrics.json"),
            "--output", str(REPORT / "comparison")], REPORT, status_path, status, env)
        status.update(state="complete", stage="complete", completed_at_unix=time.time(),
                      final_report=str(REPORT / "comparison/ablation_metrics.json"))
    except BaseException as error:
        status.update(state="failed", error=repr(error), completed_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
