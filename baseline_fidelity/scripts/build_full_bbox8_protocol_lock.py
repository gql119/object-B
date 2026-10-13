#!/usr/bin/env python3
"""Create a separate Full-BBox-8 / last-primary lock without mutating history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_LOCK = ROOT / "protocol_lock.json"
CLEAN_METRICS = ROOT / "docs" / "fidelity" / "e2_archive_evidence" / "clean_validator_metrics.json"
OUTPUT = ROOT / "docs" / "fidelity" / "protocol_lock_full_bbox8_last_primary_20261011.json"
CLEAN_METRICS_REMOTE = (
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/"
    "ue_project/ablation_safe_bg_20261004/candidate/utility_feasibility_20261009/"
    "artifacts/formal_e2_loc_retry1/detection_component_preflight/validator_replay/"
    "clean_validator_metrics.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    lock = json.loads(OLD_LOCK.read_text(encoding="utf-8"))
    clean = json.loads(CLEAN_METRICS.read_text(encoding="utf-8"))
    if lock.get("status") != "matched_clean_baseline_complete":
        raise RuntimeError("The parent lock lacks a complete matched clean victim")
    if clean.get("checkpoint_sha256") != "834c68977d31dedc859e6e905a7d2d4259b73fb0077fb587e5a5636aa93e3bad":
        raise RuntimeError("The clean-last checkpoint does not match verified E2 replay evidence")
    if clean.get("replay_max_abs_error", {}).get("AP50_per_class_max_abs") != 0.0:
        raise RuntimeError("Clean validator replay differs from the archived matched baseline")

    lock["schema_version"] = 2
    lock["protocol_id"] = "voc20_person_full_bbox8_last_primary_seed0_20261011"
    lock["source_lock_sha256"] = digest(OLD_LOCK)
    lock["poison"].update({
        "epsilon": "8/255",
        "overlap_scale": 1.0,
        "overlap_effective_epsilon": "8/255",
        "support": "person_box_union",
        "outside_target_epsilon": 0,
        "spatial_protocol": "new_full_bbox8",
        "labels": "byte-identical",
    })
    lock["victim"]["checkpoint_for_evaluation"] = "last.pt primary; best.pt secondary"
    lock["evaluation"].update({
        "primary_checkpoint": "last.pt",
        "secondary_checkpoint": "best.pt",
        "validation_images": 4952,
        "metric": "VOC20 AP50, AP75, Recall, Precision; last.pt is primary, best.pt is secondary",
    })

    clean_lock = lock["clean_baseline"]
    clean_lock.update({
        "checkpoint": clean["checkpoint"],
        "checkpoint_sha256": clean["checkpoint_sha256"],
        "metrics_json": CLEAN_METRICS_REMOTE,
        "metrics_sha256": digest(CLEAN_METRICS),
        "evaluation_checkpoint_role": "last.pt primary",
        "AP50_all": clean["mAP50_all"],
        "AP50_target": clean["mAP50_target"],
        "AP50_non_target": clean["mAP50_non_target"],
        "AP50_per_class": clean["AP50_per_class"],
        "AP75_per_class": clean["AP75_per_class"],
        "AP50_95_per_class": clean["AP50_95_per_class"],
        "Recall_per_class": clean["Recall50_official_per_class"],
    })
    lock["spatial_protocol_note"] = (
        "This new protocol lock applies only to EM/SEM/EntF fidelity baselines. "
        "It does not modify Safe-Soft05, E2, or historical locks/results."
    )

    OUTPUT.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Local assertions mirror the strict runtime lock validator.
    poison = lock["poison"]
    assert poison["overlap_scale"] == 1.0
    assert poison["overlap_effective_epsilon"] == "8/255"
    assert poison["support"] == "person_box_union"
    assert lock["victim"]["checkpoint_for_evaluation"].startswith("last.pt primary")
    print(json.dumps({
        "path": str(OUTPUT),
        "sha256": digest(OUTPUT),
        "clean_last_sha256": clean["checkpoint_sha256"],
        "clean_metrics_sha256": digest(CLEAN_METRICS),
        "clean_mAP50_target": clean["mAP50_target"],
        "clean_mAP50_non_target": clean["mAP50_non_target"],
        "clean_mAP50_all": clean["mAP50_all"],
        "spatial_protocol": poison["spatial_protocol"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
