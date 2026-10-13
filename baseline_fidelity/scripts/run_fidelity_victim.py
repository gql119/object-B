"""Train and evaluate an isolated Full-BBox-8 fidelity baseline victim."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_locked_clean_baseline import (
    check_installed_protocol,
    locked_train_args,
)
from ue_framework.baselines_det.fidelity import validate_spatial_protocol_lock
from ue_framework.baselines_det.locked_data import locked_source_inventory, sha256_file


ROOT = Path(__file__).resolve().parent.parent
ALLOWED_ROOT = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/"
    "ue_project/ablation_safe_bg_20261004/candidate/utility_feasibility_20261009/"
    "baseline_fidelity_20261011"
)
NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
TARGET_ID = 14


def validate_materialization(lock: dict, run_root: Path, method: str, run_id: str,
                             protocol_lock_path: Path) -> tuple[Path, dict]:
    materialization = run_root / "materialization"
    preflight_path = materialization / "full_preflight.json"
    manifest_path = materialization / "materialization_manifest.jsonl"
    if not preflight_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Full materialization audit evidence is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("state") != "passed":
        raise RuntimeError("Full materialization audit did not pass")
    if (preflight.get("method") != method or preflight.get("run_id") != run_id):
        raise RuntimeError("Materialization evidence belongs to another method or RunID")
    if preflight.get("protocol_lock_sha256") != sha256_file(protocol_lock_path):
        raise RuntimeError("Materialization used a different protocol lock")
    if preflight.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Materialization manifest changed after its audit")
    validate_spatial_protocol_lock(lock, "new_full_bbox8")
    counts = lock["dataset"]
    expected = {
        "train_images": int(counts["train_images"]),
        "val_images": int(counts["val_images"]),
        "target_images": int(counts["target_images"]),
        "target_objects": int(counts["target_objects"]),
    }
    for key, value in expected.items():
        if preflight.get(key) != value:
            raise RuntimeError(f"Materialization count mismatch for {key}")
    if preflight.get("dataset_fingerprint") != counts["fingerprint"]:
        raise RuntimeError("Materialized data fingerprint differs from lock")
    for flag in ("label_bytes_identical", "non_target_train_bytes_identical",
                 "clean_val_bytes_identical"):
        if preflight.get(flag) is not True:
            raise RuntimeError(f"Materialization integrity check failed: {flag}")
    if preflight.get("changed_person_gt_boxes") != expected["target_objects"]:
        raise RuntimeError("Not every person GT object had a decoded pixel change")
    if preflight.get("changed_outside_pixels") != 0:
        raise RuntimeError("Decoded changes extend outside person support")
    maxima = preflight.get("max_abs_delta_byte", {})
    if maxima.get("person_only", 256) > 8 or maxima.get("person_overlap", 256) > 8:
        raise RuntimeError("Full-BBox-8 decoded-pixel budget was exceeded")

    data = Path(preflight["materialized_root"]).resolve()
    expected_data = (materialization / "poisoned_data").resolve()
    if data != expected_data:
        raise RuntimeError("Materialized dataset escaped its isolated RunID")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != expected["train_images"] + expected["val_images"]:
        raise RuntimeError("Materialization manifest is incomplete")
    seen = set()
    for row in rows:
        identity = (row.get("split"), row.get("stem"))
        if identity in seen or identity[0] not in ("train", "val"):
            raise RuntimeError("Duplicate or invalid manifest identity")
        seen.add(identity)
        image = Path(row["output_image"]).resolve()
        label = Path(row["output_label"]).resolve()
        if image.parent != (data / "images" / row["split"]).resolve():
            raise RuntimeError("Manifest image points outside its split")
        if label.parent != (data / "labels" / row["split"]).resolve():
            raise RuntimeError("Manifest label points outside its split")
        if not image.is_file() or not label.is_file():
            raise RuntimeError(f"Materialized file is missing: {row['stem']}")
        if sha256_file(image) != row["output_image_sha256"]:
            raise RuntimeError(f"Materialized image changed after audit: {row['stem']}")
        if sha256_file(label) != row["label_sha256"]:
            raise RuntimeError(f"Materialized label changed after audit: {row['stem']}")
        if row["split"] == "val" and row["source_image_sha256"] != row["output_image_sha256"]:
            raise RuntimeError("Validation image is not byte-identical to clean source")
        if row["split"] == "train" and not row.get("target", False):
            if row["source_image_sha256"] != row["output_image_sha256"]:
                raise RuntimeError("Non-target-only train image is not byte-identical")
    for split, count in (("train", expected["train_images"]), ("val", expected["val_images"])):
        images = list((data / "images" / split).iterdir())
        labels = list((data / "labels" / split).glob("*.txt"))
        if len(images) != count or len(labels) != count:
            raise RuntimeError(f"Materialized {split} file count changed")
        if {p.stem for p in images} != {p.stem for p in labels}:
            raise RuntimeError(f"Materialized {split} image/label stems differ")
    yaml_path = Path(preflight["data_yaml"]).resolve()
    if not yaml_path.is_file():
        raise RuntimeError("Materialized YOLO data YAML is missing")
    source, source_summary = locked_source_inventory(lock, Path(counts["root"]))
    if source_summary["dataset_fingerprint"] != counts["fingerprint"]:
        raise RuntimeError("Current read-only source data no longer matches the lock")
    source_maps = {
        split: {path.stem: path for path in source[split]["images"]}
        for split in ("train", "val")
    }
    label_maps = {
        split: {path.stem: path for path in source[split]["labels"]}
        for split in ("train", "val")
    }
    for row in rows:
        if row.get("stem") not in source_maps[row["split"]]:
            raise RuntimeError(f"Manifest stem is absent from locked source: {row.get('stem')}")
        source_image = source_maps[row["split"]][row["stem"]]
        source_label = label_maps[row["split"]][row["stem"]]
        if (str(source_image.resolve()) != str(Path(row["source_image"]).resolve()) or
                sha256_file(source_image) != row["source_image_sha256"]):
            raise RuntimeError(f"Manifest source image does not match locked inventory: {row['stem']}")
        if sha256_file(source_label) != row["label_sha256"]:
            raise RuntimeError(f"Materialized label differs from locked source: {row['stem']}")
    return yaml_path, {"preflight": preflight, "source_summary": source_summary}


def summarize_metrics(box) -> dict:
    classes = np.asarray(box.ap_class_index, dtype=int).reshape(-1)
    # Ultralytics 8.4 exposes ``box.ap`` as per-class AP averaged over IoU
    # thresholds (shape [nc,]); ``all_ap`` preserves the [nc, 10] threshold
    # matrix needed for AP50 and AP75 extraction.
    ap = np.asarray(box.all_ap, dtype=float)
    precision = np.asarray(box.p, dtype=float).reshape(-1)
    recall = np.asarray(box.r, dtype=float).reshape(-1)
    if (classes.tolist() != list(range(20)) or ap.ndim != 2 or ap.shape[0] != 20 or
            ap.shape[1] < 6 or precision.shape != (20,) or recall.shape != (20,)):
        raise RuntimeError("Clean validation did not return VOC20 AP50/AP75/P/R arrays")
    if not (np.isfinite(ap).all() and np.isfinite(precision).all() and np.isfinite(recall).all()):
        raise RuntimeError("Validation metrics contain NaN or Inf")
    if np.any((ap < 0) | (ap > 1)) or np.any((precision < 0) | (precision > 1)) or np.any((recall < 0) | (recall > 1)):
        raise RuntimeError("Validation metrics fall outside [0, 1]")
    ap50, ap75 = ap[:, 0], ap[:, 5]
    return {
        "mAP50_all": float(np.mean(ap50)),
        "mAP50_target": float(ap50[TARGET_ID]),
        "mAP50_non_target": float(np.mean(np.delete(ap50, TARGET_ID))),
        "mAP75_all": float(np.mean(ap75)),
        "mAP75_target": float(ap75[TARGET_ID]),
        "mAP75_non_target": float(np.mean(np.delete(ap75, TARGET_ID))),
        "AP50_per_class": ap50.tolist(),
        "AP75_per_class": ap75.tolist(),
        "Precision_per_class_best_f1": precision.tolist(),
        "precision_definition": "Ultralytics per-class precision at the validator's best-F1 operating point",
        "Recall_per_class": recall.tolist(),
    }


def selective_metrics(clean: dict, poison: dict) -> dict:
    clean_ap = np.asarray(clean["AP50_per_class"], dtype=float)
    poison_ap = np.asarray(poison["AP50_per_class"], dtype=float)
    if clean_ap.shape != (20,) or poison_ap.shape != (20,) or np.any(clean_ap <= 0):
        raise RuntimeError("Cannot calculate selective metrics from incomplete clean/poison AP50")
    damage = np.maximum(0.0, (clean_ap - poison_ap) / clean_ap)
    nt_damage = np.delete(damage, TARGET_ID)
    target_damage = float(damage[TARGET_ID])
    nt_retention = float(poison["mAP50_non_target"] / clean["mAP50_non_target"])
    score = (2.0 * target_damage * nt_retention / (target_damage + nt_retention)
             if target_damage + nt_retention else 0.0)
    nt_ids = [idx for idx in range(20) if idx != TARGET_ID]
    worst_ids = sorted(nt_ids, key=lambda idx: damage[idx], reverse=True)
    return {"P_target_AP50_damage": target_damage, "U_non_target_AP50_retention": nt_retention,
            "S_harmonic_mean": score,
            "WCD": float(max(nt_damage)), "WCD_class": NAMES[worst_ids[0]],
            "TCD3": float(np.mean([damage[idx] for idx in worst_ids[:3]])),
            "TCD3_classes": [NAMES[idx] for idx in worst_ids[:3]]}


def validate_clean_reference(lock: dict) -> dict:
    clean = lock["clean_baseline"]
    if clean.get("status") != "matched_complete" or clean.get("evaluation_checkpoint_role") != "last.pt primary":
        raise RuntimeError("Matched clean-last baseline reference is missing")
    checkpoint = Path(clean["checkpoint"])
    metrics_path = Path(clean["metrics_json"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != clean.get("checkpoint_sha256"):
        raise RuntimeError("Matched clean last.pt is missing or has a different SHA-256")
    if not metrics_path.is_file() or sha256_file(metrics_path) != clean.get("metrics_sha256"):
        raise RuntimeError("Matched clean metrics JSON is missing or has a different SHA-256")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    key_pairs = {"mAP50_all": "AP50_all", "mAP50_target": "AP50_target",
                 "mAP50_non_target": "AP50_non_target", "AP50_per_class": "AP50_per_class"}
    for metric_key, lock_key in key_pairs.items():
        if metric_key not in metrics or lock_key not in clean:
            raise RuntimeError(f"Matched clean reference is missing {metric_key}/{lock_key}")
        if isinstance(clean[lock_key], list):
            if not np.allclose(np.asarray(metrics[metric_key]), np.asarray(clean[lock_key]), rtol=0, atol=1e-12):
                raise RuntimeError(f"Matched clean reference metric differs from protocol lock: {metric_key}")
        elif not np.isclose(float(metrics[metric_key]), float(clean[lock_key]), rtol=0, atol=1e-12):
            raise RuntimeError(f"Matched clean reference metric differs from protocol lock: {metric_key}")
    return metrics


def write_class_csv(path: Path, role: str, values: dict, clean: dict) -> None:
    clean_ap = np.asarray(clean["AP50_per_class"], dtype=float)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "class_name", "checkpoint_role", "AP50", "AP75",
                         "Precision_best_F1", "Recall", "relative_AP50_damage_vs_clean_last"])
        for idx, name in enumerate(NAMES):
            writer.writerow([idx, name, role, values["AP50_per_class"][idx],
                             values["AP75_per_class"][idx],
                             values["Precision_per_class_best_f1"][idx],
                             values["Recall_per_class"][idx],
                             (clean_ap[idx] - values["AP50_per_class"][idx]) / clean_ap[idx]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--protocol-lock", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("em_det", "sem_det", "entf_push_det"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    run_root.relative_to(ALLOWED_ROOT)
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    _, source_summary = locked_source_inventory(lock, args.dataset_root)
    if (run_root / "victim_status.json").exists() or (run_root / "victim").exists() or (run_root / "evaluation").exists():
        raise FileExistsError("Victim/evaluation stage already exists; refusing duplicate execution")
    yaml_path, materialized = validate_materialization(
        lock, run_root, args.method, args.run_id, args.protocol_lock.resolve())
    if materialized["source_summary"]["dataset_fingerprint"] != source_summary["dataset_fingerprint"]:
        raise RuntimeError("Materialized and locked source inventories differ")
    ultralytics_version = check_installed_protocol(lock)
    victim = lock["victim"]
    init_yaml = Path(victim["init_yaml"])
    if sha256_file(init_yaml) != victim["init_yaml_sha256"]:
        raise RuntimeError("Victim YAML hash differs from lock")
    train_args = locked_train_args(lock, run_root, yaml_path, smoke=False)
    spec = {
        "run_id": args.run_id, "method": args.method, "official_reproduction": False,
        "detection_adaptation": True, "candidate_code_commit": args.code_commit,
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "dataset_fingerprint": source_summary["dataset_fingerprint"],
        "materialization_preflight_sha256": sha256_file(run_root / "materialization" / "full_preflight.json"),
        "materialization_manifest_sha256": materialized["preflight"]["manifest_sha256"],
        "initialization": "random-init from YAML; pretrained=false",
        "init_yaml": str(init_yaml), "init_yaml_sha256": sha256_file(init_yaml),
        "ultralytics_version": ultralytics_version, "train_args": train_args,
        "evaluation_checkpoint_order": ["last.pt primary", "best.pt secondary"],
        "started_at_unix": time.time(),
    }
    status_path = run_root / "victim_status.json"
    status = {"state": "running", "stage": "victim_training", "run_id": args.run_id,
              "method": args.method, "started_at_unix": time.time()}
    (run_root / "victim_run_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    os.environ["YOLO_CONFIG_DIR"] = str(run_root / ".ultralytics")
    os.environ["WANDB_MODE"] = "disabled"
    try:
        seed = int(victim["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        from ultralytics import YOLO
        model = YOLO(str(init_yaml))
        model.train(**train_args)
        results_path = run_root / "victim" / "results.csv"
        with results_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        epochs = int(victim["epochs"])
        if len(rows) != epochs or int(float(rows[-1]["epoch"])) != epochs:
            raise RuntimeError(f"Victim training incomplete: {len(rows)}/{epochs} epochs")
        checkpoints = {
            "last_primary": run_root / "victim" / "weights" / "last.pt",
            "best_secondary": run_root / "victim" / "weights" / "best.pt",
        }
        if not all(path.is_file() for path in checkpoints.values()):
            raise RuntimeError("One or both required evaluation checkpoints are missing")
        clean = lock["clean_baseline"]
        clean_metrics = validate_clean_reference(lock)
        clean_reference = {"AP50_per_class": clean_metrics["AP50_per_class"],
                           "mAP50_non_target": clean_metrics["mAP50_non_target"]}
        eval_root = run_root / "evaluation"
        eval_root.mkdir(exist_ok=False)
        evaluations = {}
        status.update(stage="evaluating_last_primary")
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        for role, checkpoint in checkpoints.items():
            name = role
            metrics = YOLO(str(checkpoint)).val(
                data=str(yaml_path), imgsz=int(victim["imgsz"]), batch=int(victim["batch"]),
                workers=int(victim["workers"]), device=victim["device"], verbose=False,
                project=str(eval_root), name=name, exist_ok=False,
            )
            values = summarize_metrics(metrics.box)
            values.update({"run_id": args.run_id, "method": args.method, "checkpoint_role": role,
                           "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                           "clean_validation_images": int(lock["dataset"]["val_images"]),
                           "protocol_lock_sha256": sha256_file(args.protocol_lock),
                           "dataset_fingerprint": source_summary["dataset_fingerprint"],
                           "materialization_preflight_sha256": spec["materialization_preflight_sha256"]})
            values["selective_metrics_vs_clean_last"] = selective_metrics(clean_reference, values)
            metrics_path = eval_root / f"{role}_metrics.json"
            metrics_path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
            write_class_csv(eval_root / f"{role}_voc20_metrics.csv", role, values, clean_reference)
            evaluations[role] = str(metrics_path)
            status.update(stage="evaluating_best_secondary" if role == "last_primary" else "evaluation_complete",
                          checkpoints_evaluated=len(evaluations), evaluation_paths=evaluations)
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        status.update(state="complete", stage="complete", epochs_completed=len(rows),
                      evaluation_paths=evaluations, completed_at_unix=time.time())
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), completed_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
