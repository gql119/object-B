"""Re-evaluate saved fidelity-victim checkpoints after a metrics-parser fix.

This recovery path never trains a victim or changes a checkpoint. It writes
only to a fresh evaluation-recovery directory under the source RunID.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from scripts.run_fidelity_victim import (
    ALLOWED_ROOT,
    sha256_file,
    selective_metrics,
    summarize_metrics,
    validate_clean_reference,
    write_class_csv,
)


def write_json_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--protocol-lock", required=True, type=Path)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--recovery-id", required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    lock_path = args.protocol_lock.resolve()
    if ALLOWED_ROOT.resolve() not in run_root.parents:
        raise RuntimeError("Source RunID is outside the isolated project root")
    if args.source_run_id == args.recovery_id:
        raise RuntimeError("Evaluation recovery must use a fresh RunID")

    pipeline = json.loads((run_root / "pipeline_status.json").read_text(encoding="utf-8"))
    victim_status = json.loads((run_root / "victim_status.json").read_text(encoding="utf-8"))
    run_spec = json.loads((run_root / "victim_run_spec.json").read_text(encoding="utf-8"))
    materialization = run_root / "materialization"
    preflight_path = materialization / "full_preflight.json"
    manifest_path = materialization / "materialization_manifest.jsonl"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))

    if pipeline.get("run_id") != args.source_run_id or pipeline.get("state") != "failed":
        raise RuntimeError("Source pipeline is not the expected failed RunID")
    if victim_status.get("run_id") != args.source_run_id or victim_status.get("state") != "failed":
        raise RuntimeError("Source victim status is not the expected failed RunID")
    if "VOC20 AP50/AP75/P/R arrays" not in victim_status.get("error", ""):
        raise RuntimeError("Failure was not the known VOC20 metrics-shape parser error")
    if run_spec.get("run_id") != args.source_run_id or run_spec.get("method") != "em_det":
        raise RuntimeError("Saved victim run specification does not match source RunID/method")
    if sha256_file(lock_path) != pipeline["registered_identity"]["protocol_lock_sha256"]:
        raise RuntimeError("Protocol lock hash differs from the source pipeline")
    if (sha256_file(preflight_path) != pipeline.get("materialization_audit_sha256") or
            sha256_file(preflight_path) != run_spec.get("materialization_preflight_sha256")):
        raise RuntimeError("Materialization audit hash differs from source run provenance")
    if sha256_file(manifest_path) != run_spec.get("materialization_manifest_sha256"):
        raise RuntimeError("Materialization manifest hash differs from source run provenance")
    if preflight.get("state") != "passed":
        raise RuntimeError("Materialization audit is not marked passed")
    for key in ("label_bytes_identical", "non_target_train_bytes_identical", "clean_val_bytes_identical"):
        if preflight.get(key) is not True:
            raise RuntimeError(f"Materialization integrity flag failed: {key}")
    if preflight.get("changed_outside_pixels") != 0:
        raise RuntimeError("Materialization has pixel changes outside target support")
    if max(preflight.get("max_abs_delta_byte", {}).values()) > 8:
        raise RuntimeError("Materialization exceeds Full-BBox-8 pixel budget")

    checkpoints = {
        "last_primary": run_root / "victim" / "weights" / "last.pt",
        "best_secondary": run_root / "victim" / "weights" / "best.pt",
    }
    if not all(path.is_file() for path in checkpoints.values()):
        raise RuntimeError("Saved last.pt and best.pt are both required")
    data_yaml = Path(preflight["data_yaml"]).resolve()
    if not data_yaml.is_file():
        raise RuntimeError("Previously audited materialized dataset YAML is missing")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    clean_metrics = validate_clean_reference(lock)
    clean_reference = {
        "AP50_per_class": clean_metrics["AP50_per_class"],
        "mAP50_non_target": clean_metrics["mAP50_non_target"],
    }
    eval_parent = run_root / "evaluation_recovery"
    eval_parent.mkdir(exist_ok=True)
    output_root = eval_parent / args.recovery_id
    output_root.mkdir(exist_ok=False)

    status_path = output_root / "status.json"
    status = {
        "state": "running",
        "stage": "preflight_passed",
        "source_run_id": args.source_run_id,
        "recovery_id": args.recovery_id,
        "method": "em_det",
        "protocol_lock_sha256": sha256_file(lock_path),
        "source_materialization_audit_sha256": sha256_file(preflight_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_script_sha256": sha256_file(Path(__file__).resolve().parent / "run_fidelity_victim.py"),
        "checkpoint_sha256": {role: sha256_file(path) for role, path in checkpoints.items()},
        "initialization_or_training_performed": False,
        "started_at_unix": time.time(),
        "evaluations": {},
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train_args = run_spec["train_args"]
    try:
        for role, checkpoint in checkpoints.items():
            status.update(stage=f"evaluating_{role}", stage_started_at_unix=time.time())
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            started = time.time()
            result = YOLO(str(checkpoint)).val(
                data=str(data_yaml),
                imgsz=int(train_args["imgsz"]),
                batch=int(train_args["batch"]),
                workers=int(train_args["workers"]),
                device=train_args["device"],
                verbose=False,
                project=str(output_root),
                name=role,
                exist_ok=False,
            )
            values = summarize_metrics(result.box)
            values.update({
                "run_id": args.source_run_id,
                "evaluation_recovery_id": args.recovery_id,
                "method": "em_det",
                "checkpoint_role": role,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "clean_validation_images": int(lock["dataset"]["val_images"]),
                "protocol_lock_sha256": sha256_file(lock_path),
                "dataset_fingerprint": lock["dataset"]["fingerprint"],
                "materialization_preflight_sha256": sha256_file(preflight_path),
                "evaluation_seconds": time.time() - started,
                "evaluation_code_sha256": sha256_file(Path(__file__).resolve()),
            })
            values["selective_metrics_vs_clean_last"] = selective_metrics(clean_reference, values)
            metrics_path = output_root / f"{role}_metrics.json"
            write_json_new(metrics_path, values)
            write_class_csv(output_root / f"{role}_voc20_metrics.csv", role, values, clean_reference)
            status["evaluations"][role] = str(metrics_path)
            status.update(stage=f"completed_{role}", checkpoints_evaluated=len(status["evaluations"]))
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        status.update(state="complete", stage="complete", completed_at_unix=time.time())
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), completed_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
