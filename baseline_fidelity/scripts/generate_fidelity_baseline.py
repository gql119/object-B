"""Generate detection-adapted EM/SEM/EntF fidelity poison cycles.

The batch dimension only groups independent sample-wise updates. Official
method-internal loops, losses, radii, sampling and step sizes are fixed here.
This script writes only target-image poison PNGs and small run evidence; full
dataset materialization and victim training are separate audited stages.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression

from ue_framework.baselines_det.core import (
    EPS, TARGET_ID, _xyxy, file_sha256, letterbox, model_state_sha256,
)
from ue_framework.baselines_det.production import (
    BatchItem, BatchedPersonROIFeatures, _batch_dict, _loss_on_batch,
    delta_transition_stats, em_noise_batch, em_train_batch, entf_centroid_batch, entf_push_batch,
    make_item, new_counters, sem_defender_batch,
)
from ue_framework.baselines_det.convergence import (
    CYCLE_FIELDS, sem_refinement_audit, summarize_cycle, utc_iso,
    write_csv_snapshot, write_json_snapshot,
)
from ue_framework.baselines_det.locked_data import locked_source_inventory
from ue_framework.baselines_det.fidelity import (
    PersistentShuffledSampler, fidelity_support_mask, validate_spatial_protocol_lock,
)
from ue_framework.data_utils import (
    label_path_for_image, list_images, read_yolo_annotations,
)


METHODS = ("em_det", "entf_push_det", "sem_det")
ALLOWED_ROOT = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/"
    "ue_project/ablation_safe_bg_20261004/candidate/utility_feasibility_20261009/"
    "baseline_fidelity_20261011"
)
IMG_SIZE = 640
SURROGATE_LR = 1e-4
EM_TRAIN_BATCHES = 10
EM_NOISE_STEPS = 20
EM_NOISE_STEP = 0.8 / 255.0
EM_STOP_ERROR = 0.01
DEFENDER_STEPS = 10
DEFENDER_STEP = 1.6 / 255.0
EOT_SAMPLES = 5
INNER_RADIUS = 4.0 / 255.0
INNER_STEPS = 10
INNER_STEP = 0.8 / 255.0
ENTF_STEPS = 300
ENTF_STEP = EPS * 0.05
PERSON_COUNT = 6095
OBJECT_COUNT = 13256
BUDGET_REFERENCE_CYCLE_SECONDS = {
    # Conservative full-pass envelopes from the latest complete-inner-unit
    # batch128 probes. These multiply the unit duration by ceil(6095/128);
    # EM thereby overcounts its 10 surrogate updates once per target batch.
    "em_det": 23.05746798682958 * ((PERSON_COUNT + 127) // 128),
    # EntF excludes the full-object centroid pass; its representation gate
    # remains blocking until that stage has a valid object-level embedding.
    "entf_push_det": 479.55381646193564 * ((PERSON_COUNT + 127) // 128),
    "sem_det": 91.88375750789419 * ((PERSON_COUNT + 127) // 128),
}
BUDGET_CYCLE_FACTOR = 1.25
BUDGET_FINAL_RESERVE_SECONDS = 600.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SOURCE_FILES = (
    "scripts/generate_fidelity_baseline.py",
    "ue_framework/baselines_det/fidelity.py",
    "scripts/baseline_batch_probe.py",
    "ue_framework/baselines_det/production.py",
    "ue_framework/baselines_det/convergence.py",
    "ue_framework/baselines_det/core.py",
    "ue_framework/baselines_det/em.py",
    "ue_framework/baselines_det/sem.py",
    "ue_framework/baselines_det/entf.py",
    "ue_framework/baselines_det/robust_common.py",
    "ue_framework/baselines_det/locked_data.py",
    "ue_framework/data_utils.py",
)


def source_hashes(root: Path) -> dict:
    return {name: file_sha256(root / name) for name in SOURCE_FILES}


def update_status(path: Path, status: dict) -> None:
    now = time.time()
    status["status_updated_at_unix"] = now
    status["status_updated_at_utc"] = utc_iso(now)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_runtime_artifacts(out_root: Path, spec: dict) -> None:
    """Write compact, reproducible runtime and convergence views."""
    cycle_rows = list(spec.get("cycle_metrics", []))
    write_csv_snapshot(out_root / "cycle_metrics.csv", cycle_rows, CYCLE_FIELDS)
    write_csv_snapshot(out_root / "convergence_checkpoints.csv",
                       list(spec.get("convergence_checkpoints", [])), CYCLE_FIELDS)
    phase_fields = (
        "run_id", "session_id", "phase_id", "phase_state", "stage", "cycle",
        "batch_index", "batch_image_count", "started_at_utc", "completed_at_utc",
        "status_update_race_after_completion",
        "elapsed_seconds", "elapsed_seconds_lower_bound", "duration_is_lower_bound",
        "active_elapsed_seconds", "wall_elapsed_seconds", "surrogate_detection_loss",
        "robust_surrogate_loss", "loss_first", "loss_mean", "loss_last",
        "gradient_norm_mean", "gradient_norm_per_sqrt_image_mean",
        "projected_gradient_residual_mean", "delta_change_l2",
        "delta_drift_from_previous_cycle_l2", "epsilon_saturation_ratio",
    )
    phase_rows: dict[str, dict] = {}
    phase_path = Path(spec.get("runtime_events_path", out_root / "runtime_events.jsonl"))
    parse_errors = 0
    if phase_path.is_file():
        for line in phase_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if event.get("event") not in {"phase_start", "phase_complete", "phase_interrupted"}:
                continue
            phase_id = event.get("phase_id")
            if not phase_id:
                continue
            row = phase_rows.setdefault(phase_id, {})
            if event["event"] == "phase_interrupted" and row.get("phase_state") == "complete":
                row["status_update_race_after_completion"] = True
                continue
            row.update(event)
            if event["event"] == "phase_start":
                row["phase_state"] = "started_without_end_event"
            elif event["event"] == "phase_complete":
                row["phase_state"] = "complete"
            else:
                row["phase_state"] = "interrupted"
    phase_metrics = list(phase_rows.values())
    write_csv_snapshot(out_root / "phase_metrics.csv", phase_metrics, phase_fields)
    spec["phase_event_parse_errors"] = parse_errors
    write_json_snapshot(out_root / "runtime_summary.json", {
        "schema": "baseline_fidelity_runtime_summary.v1",
        "run_id": spec.get("run_id"), "method": spec.get("method"),
        "state": spec.get("state"), "started_at_unix": spec.get("started_at_unix"),
        "started_at_utc": spec.get("started_at_utc"),
        "completed_at_unix": spec.get("completed_at_unix"),
        "completed_at_utc": spec.get("completed_at_utc"),
        "active_duration_seconds": spec.get("active_duration_seconds"),
        "wall_elapsed_seconds": spec.get("wall_elapsed_seconds"),
        "completed_outer_cycles": spec.get("method_schedule", {}).get("complete_outer_cycles", 0),
        "requested_outer_cycles": spec.get("requested_max_cycles"),
        "schedule_complete": spec.get("schedule_complete", False),
        "completed_target_images": spec.get("completed_target_images", 0),
        "stop_reason": spec.get("stop_reason"),
        "last_complete_cycle": spec.get("last_complete_cycle",
                                         spec.get("method_schedule", {}).get("complete_outer_cycles", 0)),
        "last_complete_checkpoint": spec.get("last_complete_checkpoint"),
        "active_operation": spec.get("active_operation"),
        "generation_sessions": spec.get("generation_sessions", []),
        "sem_final_refinement": spec.get("sem_final_refinement"),
        "sem_final_refinement_attempts": spec.get("sem_final_refinement_attempts", []),
        "cycle_metrics_csv": str(out_root / "cycle_metrics.csv"),
        "convergence_checkpoints_csv": str(out_root / "convergence_checkpoints.csv"),
        "phase_metrics_csv": str(out_root / "phase_metrics.csv"),
        "phase_metric_rows": len(phase_metrics),
        "phase_event_parse_errors": parse_errors,
        "runtime_events_jsonl": spec.get("runtime_events_path"),
        "batch_diagnostics_jsonl": spec.get("convergence_log_path"),
        "convergence_status": spec.get("convergence_status", "not_established"),
        "convergence_claim": "descriptive metrics only; no numerical stopping threshold is registered",
    })


def _close_generation_session(spec: dict, *, completed_at_unix: float,
                              session_started_at_unix: float,
                              session_started_perf: float,
                              active_time_before_resume: float) -> None:
    session_duration = max(0.0, time.perf_counter() - session_started_perf)
    sessions = spec.setdefault("generation_sessions", [])
    if not sessions or sessions[-1].get("completed_at_unix") is not None:
        sessions.append({"started_at_unix": session_started_at_unix,
                         "started_at_utc": utc_iso(session_started_at_unix)})
    sessions[-1].update({"completed_at_unix": completed_at_unix,
                         "completed_at_utc": utc_iso(completed_at_unix),
                         "duration_seconds": session_duration,
                         "duration_source": "monotonic_clock"})
    spec["active_duration_seconds"] = active_time_before_resume + session_duration
    spec["wall_elapsed_seconds"] = max(0.0, completed_at_unix - float(spec["started_at_unix"]))


REFINEMENT_FIELDS = (
    "run_id", "method", "attempt", "event", "started_at_utc", "completed_at_utc",
    "duration_seconds", "expected_person_images", "processed_person_images",
    "image_id_set_matches", "full_image_coverage",
    "frozen_model_sha256_before", "frozen_model_sha256_after",
    "frozen_model_unchanged", "optimizer_steps_before", "optimizer_steps_after",
    "optimizer_steps_unchanged", "optimizer_sha256_before", "optimizer_sha256_after",
    "optimizer_state_unchanged",
    "duration_is_lower_bound", "complete", "convergence_status",
)
REFINEMENT_BATCH_FIELDS = (
    "run_id", "stage", "cycle", "batch_index", "batch_image_count",
    "started_at_utc", "completed_at_utc", "elapsed_seconds",
    "active_elapsed_seconds", "wall_elapsed_seconds",
    "loss_first", "loss_mean", "loss_last", "gradient_norm_mean",
    "gradient_norm_per_sqrt_image_mean", "projected_gradient_residual_mean",
    "delta_change_l2", "delta_drift_from_previous_cycle_l2",
    "delta_drift_from_previous_cycle_l2_per_image_mean",
    "delta_drift_from_previous_cycle_linf_max", "epsilon_saturation_ratio",
)


def _write_refinement_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REFINEMENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in REFINEMENT_FIELDS})
        stream.flush()
        temporary.replace(path)


def _write_refinement_batches_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REFINEMENT_BATCH_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in REFINEMENT_BATCH_FIELDS})
        stream.flush()
    temporary.replace(path)


def registered_config_hash(args: argparse.Namespace) -> str:
    """Hash every launch choice that must remain identical across resume."""
    fields = {name: str(getattr(args, name)) for name in (
        "method", "run_id", "dataset_root", "surrogate", "protocol_lock",
        "output_root", "batch_size", "time_cap_hours", "max_cycles",
        "seed", "code_commit", "imgsz", "spatial_protocol",
        "continue_after_time_cap")}
    fields["protocol_lock_sha256"] = file_sha256(args.protocol_lock)
    fields["source_sha256"] = source_hashes(Path(__file__).resolve().parent.parent)
    return sha256_bytes(json.dumps(fields, sort_keys=True).encode())


def optimizer_digest(optimizer: torch.optim.Optimizer | None) -> str | None:
    if optimizer is None:
        return None
    digest = hashlib.sha256()
    for group in optimizer.param_groups:
        for name in sorted(key for key in group if key != "params"):
            digest.update(f"{name}={group[name]!r};".encode())
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            for name in sorted(state):
                digest.update(name.encode())
                value = state[name]
                if isinstance(value, torch.Tensor):
                    array = value.detach().cpu().contiguous().numpy()
                    digest.update(str(array.dtype).encode())
                    digest.update(str(array.shape).encode())
                    digest.update(array.tobytes())
                else:
                    digest.update(repr(value).encode())
    return digest.hexdigest()


def save_complete_cycle_checkpoint(path: Path, *, model: torch.nn.Module,
                                   optimizer: torch.optim.Optimizer | None,
                                   spec: dict, deltas: dict[str, torch.Tensor],
                                   sampler: PersistentShuffledSampler | None,
                                   person_order: list[int], rng: np.random.Generator,
                                   centroid: torch.Tensor | None) -> float:
    """Atomically replace one complete-cycle checkpoint; never save partials."""
    completed_cycles = int(spec["method_schedule"]["complete_outer_cycles"])
    payload = {
        "schema": "detection_baseline_fidelity_cycle_boundary.v2",
        "checkpoint_boundary": "run_start" if completed_cycles == 0 else "complete_cycle",
        "registered_config_sha256": spec["registered_config_sha256"],
        "completed_cycles": completed_cycles,
        "spec": spec, "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "deltas": deltas,
        "sampler": None if sampler is None else sampler.state_dict(),
        "person_order": person_order, "numpy_generator_state": rng.bit_generator.state,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(), "torch_cuda_rng": torch.cuda.get_rng_state_all(),
        "centroid": None if centroid is None else centroid.detach().cpu(),
    }
    temporary = path.with_suffix(".incomplete")
    write_started = time.time()
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        torch.save(payload, stream)
    os.replace(temporary, path)
    return time.time() - write_started


def load_complete_cycle_checkpoint(path: Path, *, config_sha256: str,
                                   sample_ids: list[str], model: torch.nn.Module,
                                   optimizer: torch.optim.Optimizer | None,
                                   device: torch.device) -> dict:
    with gzip.open(path, "rb") as stream:
        # This file is created locally by save_complete_cycle_checkpoint and
        # contains NumPy/Python RNG objects beyond a tensor-only state dict.
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    if payload.get("schema") not in (
            "detection_baseline_fidelity_complete_cycle.v1",
            "detection_baseline_fidelity_cycle_boundary.v2"):
        raise RuntimeError("Unrecognized cycle-boundary checkpoint")
    if payload["registered_config_sha256"] != config_sha256:
        raise RuntimeError("Exact resume config/source/protocol hash mismatch")
    if payload["completed_cycles"] < 0:
        raise RuntimeError("Checkpoint has an invalid completed-cycle count")
    expected_delta_ids = set(sample_ids) if payload["completed_cycles"] > 0 else set()
    if set(payload["deltas"]) != expected_delta_ids:
        raise RuntimeError("Checkpoint perturbation IDs do not match its cycle boundary")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise RuntimeError("Missing optimizer state for exact resume")
        optimizer.load_state_dict(payload["optimizer"])
        # load_state_dict maps tensor moments to their parameter device and
        # retains scalar step tensors in the optimizer's required placement.
        # Moving every state tensor to CUDA changes Adam's next update.
    elif payload["optimizer"] is not None:
        raise RuntimeError("Unexpected optimizer state in frozen-reference method")
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["torch_cuda_rng"])
    if model_state_sha256(model) != payload["spec"]["checkpoint_model_sha256"]:
        raise RuntimeError("Resume model state digest mismatch")
    if optimizer_digest(optimizer) != payload["spec"]["checkpoint_optimizer_sha256"]:
        raise RuntimeError("Resume optimizer state digest mismatch")
    if not torch.equal(torch.get_rng_state(), payload["torch_rng"]):
        raise RuntimeError("Resume CPU RNG mismatch")
    if any(not torch.equal(a, b) for a, b in zip(
            torch.cuda.get_rng_state_all(), payload["torch_cuda_rng"])):
        raise RuntimeError("Resume CUDA RNG mismatch")
    return payload


def read_item(image_path: Path, labels: Path, device: torch.device,
              spatial_protocol: str = "new_full_bbox8") -> BatchItem:
    label_path = Path(label_path_for_image(str(image_path), str(labels)))
    anns = read_yolo_annotations(str(label_path))
    if not any(int(x["cls"]) == TARGET_ID for x in anns):
        raise ValueError(f"Target image set contains a person-free row: {image_path.stem}")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot decode train image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    clean = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device).float().div_(255.0)
    item = make_item(clean, anns, image_path.stem)
    support = fidelity_support_mask(clean.shape[-2], clean.shape[-1], anns, spatial_protocol)
    item.support = torch.from_numpy(support).to(clean.device, clean.dtype)[None, None]
    return item


def batches(indices: list[int], size: int):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def expected_cycle_counters(method: str, batch_count: int) -> dict[str, int]:
    counters = {"forward_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
                "pgd_steps": 0, "noise_steps": 0, "eot_samples": 0,
                "stability_samples": 0, "push_steps": 0}
    if method == "em_det":
        counters.update(forward_calls=EM_TRAIN_BATCHES + EM_NOISE_STEPS * batch_count + PERSON_COUNT,
                        backward_calls=EM_TRAIN_BATCHES + EM_NOISE_STEPS * batch_count,
                        optimizer_steps=EM_TRAIN_BATCHES,
                        noise_steps=PERSON_COUNT * EM_NOISE_STEPS)
    elif method == "sem_det":
        calls_per_batch = DEFENDER_STEPS * EOT_SAMPLES + INNER_STEPS + 1
        counters.update(forward_calls=calls_per_batch * batch_count,
                        backward_calls=calls_per_batch * batch_count,
                        optimizer_steps=batch_count,
                        pgd_steps=PERSON_COUNT * INNER_STEPS,
                        stability_samples=PERSON_COUNT * DEFENDER_STEPS * EOT_SAMPLES)
    else:
        counters.update(forward_calls=batch_count * (1 + ENTF_STEPS),
                        backward_calls=batch_count * ENTF_STEPS,
                        push_steps=PERSON_COUNT * ENTF_STEPS)
    return counters


def expected_total_counters(method: str, batch_count: int, cycles: int) -> dict[str, int]:
    """A frozen EntF centroid is computed once; all other work repeats."""
    one_cycle = expected_cycle_counters(method, batch_count)
    total = {name: count * cycles for name, count in one_cycle.items()}
    if method == "entf_push_det":
        total["forward_calls"] -= (cycles - 1) * batch_count
    return total


def budget_allows_next_cycle(method: str, deadline: float, now: float,
                             measured_cycle_seconds: list[float]) -> bool:
    reference = max([BUDGET_REFERENCE_CYCLE_SECONDS[method], *measured_cycle_seconds])
    projected_cycle = reference * BUDGET_CYCLE_FACTOR
    # SEM must still have time for its mandatory full-dataset frozen-model
    # refinement after the last training cycle. Conservatively reserve one
    # train-cycle envelope because no full refinement-only benchmark exists.
    final_refinement_reserve = projected_cycle if method == "sem_det" else 0.0
    return (now + projected_cycle + final_refinement_reserve
            + BUDGET_FINAL_RESERVE_SECONDS < deadline)


def generation_deadline(started_at_unix: float, warning_hours: float,
                        active_seconds_before_resume: float = 0.0, *,
                        continue_after_warning: bool = False) -> float:
    """Return a hard deadline, or infinity when the threshold is warning-only."""
    if warning_hours <= 0 or active_seconds_before_resume < 0:
        raise ValueError("warning_hours must be positive and prior active time non-negative")
    if continue_after_warning:
        return float("inf")
    return started_at_unix + warning_hours * 3600.0 - active_seconds_before_resume


def warning_threshold_exceeded(active_elapsed_seconds: float, warning_hours: float) -> bool:
    if active_elapsed_seconds < 0 or warning_hours <= 0:
        raise ValueError("elapsed time must be non-negative and warning_hours positive")
    return active_elapsed_seconds >= warning_hours * 3600.0


def load_batch(paths: list[Path], labels: Path, device: torch.device,
               spatial_protocol: str = "new_full_bbox8") -> list[BatchItem]:
    return [read_item(p, labels, device, spatial_protocol) for p in paths]


def write_poison(item: BatchItem, delta: torch.Tensor, path: Path) -> None:
    value = (item.clean + (delta.clamp(-EPS, EPS) * item.support)).clamp(0, 1)
    rgb = np.rint(value.detach().cpu()[0].permute(1, 2, 0).numpy() * 255.0)
    rgb = rgb.clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=1)


def _train_error_for_batch(model, items, deltas, imgsz: int = IMG_SIZE) -> tuple[int, int]:
    batch_images = [((x.clean + delta * x.support).clamp(0, 1)).detach() for x, delta in zip(items, deltas)]
    packed = _batch_dict(batch_images, [x.annotations for x in items], None, imgsz)
    with torch.no_grad():
        model.eval()
        prediction = model(packed["img"])
        detections = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.7,
                                         classes=[TARGET_ID], max_det=300)
    missed = total = 0
    for item, det in zip(items, detections):
        _, geometry = letterbox(item.clean, imgsz)
        predicted = det[:, :4]
        for ann in item.annotations:
            if int(ann["cls"]) != TARGET_ID:
                continue
            x1, y1, x2, y2 = _xyxy(ann["bbox"], geometry.original_height, geometry.original_width)
            gt = torch.tensor([[
                x1 * geometry.new_width / geometry.original_width + geometry.pad_left,
                y1 * geometry.new_height / geometry.original_height + geometry.pad_top,
                x2 * geometry.new_width / geometry.original_width + geometry.pad_left,
                y2 * geometry.new_height / geometry.original_height + geometry.pad_top,
            ]], device=item.clean.device, dtype=item.clean.dtype)
            hit = bool(predicted.numel() and (box_iou(gt, predicted) >= 0.5).any())
            total += 1
            missed += int(not hit)
    return missed, total


def _assert_lock(lock: dict, surrogate: Path, data_root: Path,
                 spatial_protocol: str = "new_full_bbox8") -> tuple[Path, Path, list[Path]]:
    validate_spatial_protocol_lock(lock, spatial_protocol)
    if lock.get("status") != "matched_clean_baseline_complete":
        raise RuntimeError("The protocol lock does not contain a completed matched clean baseline")
    if lock["poison"]["epsilon"] != "8/255" or lock["target"]["class_id"] != TARGET_ID:
        raise RuntimeError("The common epsilon/target protocol differs")
    if lock["clean_baseline"]["status"] != "matched_complete":
        raise RuntimeError("A matched complete clean baseline is required")
    if file_sha256(surrogate) != lock["surrogate"]["checkpoint_sha256"]:
        raise RuntimeError("The clean VOC surrogate checkpoint SHA256 differs from protocol lock")
    inventory, counts = locked_source_inventory(lock, data_root)
    if counts["train_fingerprint"] != lock["dataset"]["train_split_fingerprint"]:
        raise RuntimeError("Recomputed train fingerprint differs from the protocol lock")
    images = Path(data_root) / "images" / "train"
    labels = Path(data_root) / "labels" / "train"
    target_images = []
    labels_by_stem = {path.stem: path for path in inventory["train"]["labels"]}
    for image_path in inventory["train"]["images"]:
        anns = read_yolo_annotations(str(labels_by_stem[image_path.stem]))
        if any(int(x["cls"]) == TARGET_ID for x in anns):
            target_images.append(image_path)
    if len(target_images) != PERSON_COUNT or counts["target_objects"] != OBJECT_COUNT:
        raise RuntimeError(f"Train target index mismatch: images={len(target_images)}, objects={counts['target_objects']}")
    return images, labels, target_images


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--surrogate", type=Path, required=True)
    p.add_argument("--protocol-lock", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--time-cap-hours", type=float, default=8.0)
    p.add_argument("--continue-after-time-cap", action="store_true",
                   help="Treat the registered time threshold as a warning and finish the locked schedule")
    p.add_argument("--max-cycles", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--code-commit", default="unspecified")
    p.add_argument("--spatial-protocol", choices=("old_overlap_half", "new_full_bbox8"),
                   default="new_full_bbox8")
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    p.add_argument("--resume", action="store_true",
                   help="Resume only from the last complete-cycle checkpoint")
    args = p.parse_args()
    out_root = args.output_root.resolve()
    try:
        out_root.relative_to(ALLOWED_ROOT)
    except ValueError as exc:
        raise PermissionError(f"Output must remain under authorized root {ALLOWED_ROOT}") from exc
    if out_root.exists() != args.resume:
        raise FileExistsError("Fresh RunID must be absent; exact resume must already exist")
    if args.batch_size < 1 or args.time_cap_hours <= 0 or args.imgsz != IMG_SIZE or args.max_cycles < 1:
        raise ValueError("Invalid batch/time/imgsz setting")
    if args.spatial_protocol not in args.run_id:
        raise ValueError("RunID must name the preregistered spatial protocol")
    started = time.time()
    generation_session_id = f"{args.run_id}:{int(started * 1000)}"
    session_started_perf = time.perf_counter()
    active_time_before_resume = 0.0
    deadline = generation_deadline(
        started, args.time_cap_hours,
        continue_after_warning=args.continue_after_time_cap,
    )
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if args.continue_after_time_cap:
        time_policy = lock.get("method_generation_authorization", {}).get("noise_time_policy")
        if time_policy != "warning_only_continue_to_registered_schedule":
            raise PermissionError("Warning-only time policy is not authorized by the protocol lock")
    images_dir, labels_dir, target_paths = _assert_lock(
        lock, args.surrogate, args.dataset_root, args.spatial_protocol
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Official generation requires the authorized CUDA device")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = YOLO(str(args.surrogate)).model.to(device)
    if int(model.nc) != 20:
        raise RuntimeError(f"Expected a VOC20 detector, got nc={model.nc}")
    torch.cuda.reset_peak_memory_stats(device)
    if args.method == "entf_push_det":
        extractor = BatchedPersonROIFeatures(model, imgsz=args.imgsz)
        optimizer = None
    else:
        extractor = None
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=SURROGATE_LR)

    if not args.resume:
        out_root.mkdir(parents=True)
    outputs = out_root / "noise"
    if not args.resume:
        outputs.mkdir()
    elif not outputs.is_dir():
        raise RuntimeError("Exact resume is missing its noise output directory")
    status_path = out_root / "status.json"
    checkpoint_path = out_root / "last_complete_cycle.pt.gz"
    diagnostics_path = out_root / "convergence_batches.jsonl"

    runtime_events_path = out_root / "runtime_events.jsonl"

    def append_diagnostic(record: dict) -> None:
        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with diagnostics_path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
        if record.get("event") in {
                "phase_start", "phase_complete", "phase_interrupted",
                "generation_session_start", "generation_session_complete",
                "time_warning_exceeded",
                "cycle_complete", "convergence_checkpoint"}:
            with runtime_events_path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
        elif (record.get("stage") == "sem_final_refinement"
              and record.get("event") in {"start", "complete", "interrupted"}):
            with runtime_events_path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")

    def begin_phase(stage: str, cycle: int, batch_index: int,
                    batch_image_count: int) -> dict:
        started_at_unix = time.time()
        token = {
            "session_id": generation_session_id,
            "phase_id": f"{generation_session_id}:{cycle}:{stage}:{batch_index}",
            "stage": stage, "cycle": cycle, "batch_index": batch_index,
            "batch_image_count": batch_image_count,
            "started_at_unix": started_at_unix,
            "started_at_utc": utc_iso(started_at_unix),
            "_started_perf": time.perf_counter(),
        }
        start_event = {key: value for key, value in token.items() if not key.startswith("_")}
        start_event["event"] = "phase_start"
        append_diagnostic(start_event)
        spec["active_operation"] = start_event
        spec["active_duration_seconds"] = active_time_before_resume + max(
            0.0, time.perf_counter() - session_started_perf)
        spec["wall_elapsed_seconds"] = max(
            0.0, started_at_unix - float(spec.get("started_at_unix", started_at_unix)))
        update_status(status_path, spec)
        return token

    def complete_phase(token: dict, measurements: dict | None = None) -> dict:
        completed_at_unix = time.time()
        record = {key: value for key, value in token.items() if not key.startswith("_")}
        record.update({
            "event": "phase_complete",
            "completed_at_unix": completed_at_unix,
            "completed_at_utc": utc_iso(completed_at_unix),
            "elapsed_seconds": max(0.0, time.perf_counter() - token["_started_perf"]),
            "active_elapsed_seconds": active_time_before_resume + max(
                0.0, time.perf_counter() - session_started_perf),
            "wall_elapsed_seconds": max(
                0.0, completed_at_unix - float(spec.get("started_at_unix", completed_at_unix))),
        })
        if measurements:
            record.update(measurements)
        append_diagnostic(record)
        spec["active_operation"] = None
        spec["active_duration_seconds"] = record["active_elapsed_seconds"]
        spec["wall_elapsed_seconds"] = record["wall_elapsed_seconds"]
        return record
    counters = new_counters()
    spec = {
        "schema_version": 2, "run_id": args.run_id, "method": args.method,
        "official_run": False, "detection_adaptation": True,
        "state": "running", "stage": "generation",
        "seed": args.seed, "batch_size": args.batch_size, "imgsz": args.imgsz,
        "determinism": {"cudnn_benchmark": False, "cudnn_deterministic": True,
                        "torch_deterministic_algorithms": True, "warn_only": True},
        "spatial_protocol": args.spatial_protocol,
        "time_cap_hours": args.time_cap_hours,
        "time_cap_mode": ("warning_only_continue_to_registered_schedule"
                           if args.continue_after_time_cap else "hard_stop"),
        "continue_after_time_cap": args.continue_after_time_cap,
        "time_warning_emitted": False,
        "time_warning_exceeded": False,
        "requested_max_cycles": args.max_cycles,
        "cycle_guard": {"factor": BUDGET_CYCLE_FACTOR,
                        "final_reserve_seconds": BUDGET_FINAL_RESERVE_SECONDS,
                        "reference_cycle_seconds": BUDGET_REFERENCE_CYCLE_SECONDS[args.method]},
        "dataset_fingerprint": lock["dataset"]["fingerprint"],
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "surrogate_sha256": file_sha256(args.surrogate),
        "candidate_code_commit": args.code_commit,
        "source_file_sha256": source_hashes(Path(__file__).resolve().parent.parent),
        "registered_config_sha256": registered_config_hash(args),
        "convergence_log_path": str(diagnostics_path),
        "cycle_metrics_path": str(out_root / "cycle_metrics.csv"),
        "convergence_checkpoints_path": str(out_root / "convergence_checkpoints.csv"),
        "runtime_summary_path": str(out_root / "runtime_summary.json"),
        "runtime_events_path": str(runtime_events_path),
        "train_person_image_count": len(target_paths),
        "train_person_object_count": OBJECT_COUNT,
        "method_schedule": {}, "counters": counters, "cycle_history": [],
        "cycle_metrics": [], "convergence_checkpoints": [],
        "started_at_unix": started, "started_at_utc": utc_iso(started),
        "completed_at_unix": None, "completed_at_utc": None,
        "active_duration_seconds": 0.0, "wall_elapsed_seconds": 0.0,
        "generation_sessions": [{"started_at_unix": started, "started_at_utc": utc_iso(started)}],
        "active_operation": None,
        "convergence_status": "not_established",
        "convergence_claim": "descriptive metrics only; no numerical stopping threshold is registered",
    }

    time_warning_emitted = bool(spec.get("time_warning_emitted", False))

    def record_time_warning_if_due(stage: str) -> None:
        nonlocal time_warning_emitted
        if not args.continue_after_time_cap or time_warning_emitted:
            return
        active_elapsed = active_time_before_resume + max(
            0.0, time.perf_counter() - session_started_perf)
        if not warning_threshold_exceeded(active_elapsed, args.time_cap_hours):
            return
        observed_at = time.time()
        event = {
            "event": "time_warning_exceeded", "run_id": args.run_id,
            "method": args.method, "stage": stage,
            "warning_threshold_hours": args.time_cap_hours,
            "active_elapsed_seconds": active_elapsed,
            "action": "continue_to_registered_schedule_completion",
            "convergence_status": "not_established",
            "observed_at_unix": observed_at,
            "observed_at_utc": utc_iso(observed_at),
        }
        spec.update({"time_warning_emitted": True, "time_warning_exceeded": True,
                     "time_warning_exceeded_at_utc": event["observed_at_utc"],
                     "time_warning_active_elapsed_seconds": active_elapsed})
        append_diagnostic(event)
        update_status(status_path, spec)
        time_warning_emitted = True

    def finalize_runtime(state: str, *, reason: str | None = None) -> None:
        completed_at = time.time()
        active_operation = spec.get("active_operation")
        if active_operation:
            interrupted = {
                **active_operation, "event": "phase_interrupted",
                "observed_at_unix": completed_at, "observed_at_utc": utc_iso(completed_at),
                "elapsed_seconds_lower_bound": max(
                    0.0, completed_at - float(active_operation["started_at_unix"])),
                "duration_is_lower_bound": True,
                "reason": reason or state,
            }
            spec.setdefault("interrupted_operations", []).append(interrupted)
            append_diagnostic(interrupted)
            spec["active_operation"] = None
        refinement = spec.get("sem_final_refinement")
        if refinement and refinement.get("state") == "running":
            event = {
                "run_id": args.run_id, "method": args.method,
                "attempt": refinement.get("attempt"), "event": "interrupted",
                "completed_at_unix": completed_at, "completed_at_utc": utc_iso(completed_at),
                "expected_person_images": PERSON_COUNT,
                "processed_person_images": refinement.get("processed_person_images", 0),
                "duration_is_lower_bound": True,
                "convergence_status": "not_established",
                "reason": reason or state,
            }
            refinement.setdefault("events", []).append(event)
            refinement.update({"state": "interrupted", "interrupted_at_unix": completed_at,
                               "interrupted_at_utc": utc_iso(completed_at), "complete": False,
                               "convergence_status": "not_established"})
            attempts = spec.setdefault("sem_final_refinement_attempts", [])
            if attempts:
                attempts[-1] = refinement
            else:
                attempts.append(refinement)
            append_diagnostic({"stage": "sem_final_refinement", **event})
            _write_refinement_csv(
                out_root / "sem_final_refinement.csv",
                [item for attempt in attempts for item in attempt.get("events", [])])
            _write_refinement_batches_csv(
                out_root / "sem_final_refinement_batches.csv",
                [item for attempt in attempts for item in attempt.get("batch_records", [])])
            write_json_snapshot(out_root / "sem_final_refinement.json", refinement)
        spec["state"] = state
        spec["completed_at_unix"] = completed_at
        spec["completed_at_utc"] = utc_iso(completed_at)
        spec["duration_seconds"] = active_time_before_resume + max(
            0.0, time.perf_counter() - session_started_perf)
        if reason:
            spec["stop_reason"] = reason
        _close_generation_session(
            spec, completed_at_unix=completed_at, session_started_at_unix=started,
            session_started_perf=session_started_perf,
            active_time_before_resume=active_time_before_resume)
        append_diagnostic({
            "event": "generation_session_complete", "run_id": args.run_id,
            "method": args.method, "state": state,
            "completed_at_unix": completed_at, "completed_at_utc": utc_iso(completed_at),
            "session_active_seconds": max(0.0, time.perf_counter() - session_started_perf),
            "cumulative_active_seconds": spec["active_duration_seconds"],
            "wall_elapsed_seconds": spec["wall_elapsed_seconds"],
            "stop_reason": spec.get("stop_reason"),
            "completed_outer_cycles": cycles_completed,
            "completed_target_images": records_written,
        })
        _write_runtime_artifacts(out_root, spec)
        (out_root / "run_spec.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        update_status(status_path, spec)
    if not args.resume:
        (out_root / "run_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        update_status(status_path, spec)
    state_before = model_state_sha256(model)
    spec["initial_model_state_sha256"] = state_before
    max_memory = 0
    person_order = list(range(len(target_paths)))
    rng = np.random.default_rng(args.seed)
    em_sampler = PersistentShuffledSampler(tuple(path.stem for path in target_paths), args.seed) \
        if args.method == "em_det" else None
    if args.method != "em_det":
        rng.shuffle(person_order)
    if args.method == "em_det":
        spec["method_schedule"] = {
            "complete_outer_cycles": 0,
            "requested_max_complete_outer_cycles": args.max_cycles,
            "surrogate_batches_per_cycle": EM_TRAIN_BATCHES,
            "noise_descent_steps_per_image": EM_NOISE_STEPS,
            "noise_step_size": EM_NOISE_STEP,
            "stop_error_threshold_train_only": EM_STOP_ERROR,
            "stop_policy": "registered_full_cycles_or_resource_guard; train_error_record_only",
        }
    elif args.method == "sem_det":
        spec["method_schedule"] = {
            "complete_outer_cycles": 0,
            "requested_max_complete_outer_cycles": args.max_cycles,
            "defender_steps": DEFENDER_STEPS,
            "uniform_stability_samples_per_defender_step": EOT_SAMPLES,
            "uniform_stability_radius": INNER_RADIUS,
            "defender_adversarial_pgd_steps": 0,
            "surrogate_adversarial_pgd_steps": INNER_STEPS,
            "defender_step_size": DEFENDER_STEP, "adversarial_step_size": INNER_STEP,
        }
    else:
        spec["method_schedule"] = {
            "complete_outer_cycles": 0,
            "requested_max_complete_outer_cycles": args.max_cycles,
            "centroid_person_objects": OBJECT_COUNT,
            "push_steps_per_person_image": ENTF_STEPS, "push_step_size": ENTF_STEP,
            "push_init": "uniform[-8/255,8/255] once, then persist delta across cycles",
        }
    batch_count = (PERSON_COUNT + args.batch_size - 1) // args.batch_size
    spec["expected_full_cycle_counters"] = expected_cycle_counters(args.method, batch_count)
    spec["expected_target_batch_count"] = batch_count
    if not args.resume:
        # Persist a deterministic run-start anchor before the first training
        # or noise batch. If the resource cap is reached before outer cycle 1
        # finishes, exact resume replays that cycle from this state.
        spec["checkpoint_model_sha256"] = model_state_sha256(model)
        spec["checkpoint_optimizer_sha256"] = optimizer_digest(optimizer)
        spec["active_duration_seconds"] = 0.0
        spec["peak_gpu_memory_bytes_so_far"] = 0
        spec["checkpoint_boundary"] = "run_start"
        save_complete_cycle_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, spec=spec,
            deltas={}, sampler=em_sampler, person_order=person_order,
            rng=rng, centroid=None,
        )
        update_status(status_path, spec)
    records_written = 0
    cycles_completed = 0
    cycle_seconds: list[float] = []
    delta_state: dict[str, torch.Tensor] = {}
    train_error = None
    centroid = None
    visible_status = None
    resume_cap_exhausted = False
    if args.resume:
        if status_path.is_file():
            visible_status = json.loads(status_path.read_text(encoding="utf-8"))
            if visible_status.get("state") == "complete":
                raise RuntimeError("A completed RunID must not be resumed or regenerated")
        payload = load_complete_cycle_checkpoint(
            checkpoint_path, config_sha256=spec["registered_config_sha256"],
            sample_ids=[path.stem for path in target_paths], model=model,
            optimizer=optimizer, device=device)
        spec = payload["spec"]
        if spec["run_id"] != args.run_id or spec["state"] == "complete":
            raise RuntimeError("Completed or different RunID cannot be resumed")
        cycles_completed = payload["completed_cycles"]
        # A checkpoint is written only after all target images of its last
        # cycle were processed. This also permits replaying an interrupted
        # SEM final-refinement pass against that complete-cycle checkpoint.
        records_written = PERSON_COUNT if cycles_completed > 0 else 0
        cycle_seconds = [entry["duration_seconds"] for entry in spec["cycle_history"]]
        if len(cycle_seconds) != cycles_completed:
            raise RuntimeError("Checkpoint cycle-history mismatch")
        counters = dict(spec["counters"])
        delta_state = payload["deltas"]
        person_order = list(payload["person_order"])
        rng.bit_generator.state = payload["numpy_generator_state"]
        if em_sampler is not None:
            em_sampler = PersistentShuffledSampler.from_state_dict(
                payload["sampler"], [path.stem for path in target_paths])
        centroid = None if payload["centroid"] is None else payload["centroid"].to(device)
        train_error = spec.get("training_detection_error")
        state_before = spec["initial_model_state_sha256"]
        active_time_before_resume = float(spec["active_duration_seconds"])
        max_memory = int(spec.get("peak_gpu_memory_bytes_so_far", 0))
        if status_path.is_file():
            visible_status = json.loads(status_path.read_text(encoding="utf-8"))
            if (visible_status.get("registered_config_sha256") == spec["registered_config_sha256"]
                    and visible_status.get("method_schedule", {}).get("complete_outer_cycles") == cycles_completed
                    and visible_status.get("checkpoint_model_sha256") == spec["checkpoint_model_sha256"]):
                active_time_before_resume = max(
                    active_time_before_resume, float(visible_status.get("active_duration_seconds", 0)))
        if visible_status is not None:
            interrupted = visible_status.get("active_operation")
            if interrupted:
                observed_at = float(visible_status.get("status_updated_at_unix", started))
                lower_bound = max(0.0, observed_at - float(interrupted["started_at_unix"]))
                interrupted_record = {
                    **interrupted, "event": "phase_interrupted",
                    "observed_at_unix": observed_at,
                    "observed_at_utc": visible_status.get("status_updated_at_utc", utc_iso(observed_at)),
                    "elapsed_seconds_lower_bound": lower_bound,
                    "duration_is_lower_bound": True,
                    "reason": "process_exit_or_resume_from_previous_complete_cycle",
                }
                append_diagnostic(interrupted_record)
                spec.setdefault("interrupted_operations", []).append(interrupted_record)
            sessions = list(visible_status.get("generation_sessions", spec.get("generation_sessions", [])))
            if sessions and sessions[-1].get("completed_at_unix") is None:
                last_status_time = visible_status.get("status_updated_at_unix")
                if last_status_time is not None:
                    old_session = sessions[-1]
                    old_session.update({
                        "completed_at_unix": float(last_status_time),
                        "completed_at_utc": visible_status.get("status_updated_at_utc", utc_iso(last_status_time)),
                        "duration_seconds": max(0.0, float(last_status_time) - float(old_session["started_at_unix"])),
                        "duration_source": "last_status_update_lower_bound_after_unobserved_exit",
                        "duration_is_lower_bound": True,
                    })
            sessions.append({"started_at_unix": started, "started_at_utc": utc_iso(started)})
            spec["generation_sessions"] = sessions
            spec["sem_final_refinement_attempts"] = list(
                visible_status.get("sem_final_refinement_attempts",
                                  spec.get("sem_final_refinement_attempts", [])))
            previous_refinement = visible_status.get("sem_final_refinement")
            if (previous_refinement and previous_refinement.get("state") == "running"
                    and spec["sem_final_refinement_attempts"]
                    and spec["sem_final_refinement_attempts"][-1].get("attempt")
                    == previous_refinement.get("attempt")):
                spec["sem_final_refinement_attempts"].pop()
            if previous_refinement and previous_refinement.get("state") == "running":
                prior = dict(previous_refinement)
                interrupted_at = float(visible_status.get("status_updated_at_unix", started))
                lower_bound_duration = max(
                    0.0, interrupted_at - float(previous_refinement["started_at_unix"]))
                prior.update({
                    "state": "interrupted_before_resume",
                    "interrupted_at_unix": interrupted_at,
                    "interrupted_at_utc": visible_status.get("status_updated_at_utc", utc_iso(started)),
                    "elapsed_seconds_lower_bound": lower_bound_duration,
                    "duration_is_lower_bound": True,
                })
                interruption_event = {
                    "run_id": args.run_id, "method": args.method,
                    "attempt": prior.get("attempt"), "event": "interrupted",
                    "completed_at_unix": interrupted_at,
                    "completed_at_utc": visible_status.get("status_updated_at_utc", utc_iso(started)),
                    "expected_person_images": PERSON_COUNT,
                    "processed_person_images": prior.get("processed_person_images", 0),
                    "duration_seconds": lower_bound_duration,
                    "duration_is_lower_bound": True,
                    "convergence_status": "not_established",
                }
                prior.setdefault("events", []).append(interruption_event)
                spec.setdefault("sem_final_refinement_attempts", []).append(prior)
                append_diagnostic({"stage": "sem_final_refinement", **interruption_event})
        else:
            spec.setdefault("generation_sessions", []).append(
                {"started_at_unix": started, "started_at_utc": utc_iso(started)})
        write_csv_snapshot(out_root / "cycle_metrics.csv",
                           list(spec.get("cycle_metrics", [])), CYCLE_FIELDS)
        write_csv_snapshot(out_root / "convergence_checkpoints.csv",
                           list(spec.get("convergence_checkpoints", [])), CYCLE_FIELDS)
        deadline = generation_deadline(
            started, args.time_cap_hours, active_time_before_resume,
            continue_after_warning=args.continue_after_time_cap,
        )
        if not args.continue_after_time_cap and deadline <= started:
            resume_cap_exhausted = True
            deadline = started
        spec.update({"state": "running", "stage": "generation",
                     "resumed_from_complete_cycle": cycles_completed,
                     "resume_at_unix": started,
                     "active_operation": None})
        update_status(status_path, spec)
        append_diagnostic({"event": "generation_session_start", "run_id": args.run_id,
                           "started_at_unix": started, "started_at_utc": utc_iso(started),
                           "resumed_from_complete_cycle": cycles_completed})
    else:
        append_diagnostic({"event": "generation_session_start", "run_id": args.run_id,
                           "started_at_unix": started, "started_at_utc": utc_iso(started),
                           "resumed_from_complete_cycle": 0})

    def load_cycle_items(index_batch: list[int], *, continuing: bool) -> list[BatchItem]:
        items = load_batch([target_paths[i] for i in index_batch], labels_dir, device,
                           args.spatial_protocol)
        if continuing:
            for item in items:
                if item.stem not in delta_state:
                    raise RuntimeError(f"Missing persistent perturbation for {item.stem}")
                item.raw_delta = delta_state[item.stem].to(device)
        return items

    def retain_deltas(items: list[BatchItem], deltas: list[torch.Tensor]) -> None:
        for item, delta in zip(items, deltas):
            delta_state[item.stem] = delta.detach().cpu().clone()

    try:
        if resume_cap_exhausted:
            raise TimeoutError("Registered compute cap exhausted at checkpoint")
        if args.method == "entf_push_det" and centroid is None:
            assert extractor is not None
            def centroid_batches():
                for index_batch in batches(person_order, args.batch_size):
                    if time.time() + BUDGET_FINAL_RESERVE_SECONDS >= deadline:
                        raise TimeoutError("Registered cap reserve reached during full-train EntF centroid pass")
                    yield load_batch([target_paths[i] for i in index_batch], labels_dir, device,
                                     args.spatial_protocol)
            centroid, object_count, centroid_batch_count = entf_centroid_batch(extractor, centroid_batches())
            counters["forward_calls"] += centroid_batch_count
            if object_count != OBJECT_COUNT:
                raise RuntimeError(f"EntF centroid used {object_count}, expected {OBJECT_COUNT} person objects")
            spec["centroid_objects_seen"] = object_count
            spec["centroid_batches"] = centroid_batch_count
            update_status(status_path, spec)

        for cycle_index in range(cycles_completed, args.max_cycles):
            record_time_warning_if_due("cycle_start")
            if not budget_allows_next_cycle(args.method, deadline, time.time(), cycle_seconds):
                raise TimeoutError("Budget guard refused to begin a full outer cycle")
            if cycle_index > 0 and args.method != "em_det":
                rng.shuffle(person_order)
            cycle_started = time.time()
            cycle_started_perf = time.perf_counter()
            cycle_records: list[dict] = []
            cycle_surrogate_indices: set[int] = set()
            cycle_surrogate_exposures = 0
            cycle_counters_before = dict(counters)
            records_written = 0
            spec.update({"current_cycle": cycle_index + 1,
                         "current_cycle_started_at_unix": cycle_started,
                         "current_cycle_started_at_utc": utc_iso(cycle_started),
                         "completed_target_images": 0,
                         "counters": counters})
            update_status(status_path, spec)

            if args.method == "em_det":
                # Preserve train -> noise -> training-error evaluation.
                # The budget variant records the original stop criterion but
                # follows the user-authorized full-cycle wall-clock budget.
                assert em_sampler is not None
                for surrogate_batch_index in range(EM_TRAIN_BATCHES):
                    index_batch = em_sampler.next_batch(args.batch_size)
                    if time.time() + BUDGET_FINAL_RESERVE_SECONDS >= deadline:
                        raise TimeoutError("Registered cap reserve reached inside an EM cycle")
                    phase = begin_phase("em_surrogate", cycle_index + 1,
                                        surrogate_batch_index, len(index_batch))
                    cycle_surrogate_exposures += len(index_batch)
                    cycle_surrogate_indices.update(index_batch)
                    items = load_cycle_items(index_batch, continuing=cycle_index > 0)
                    surrogate_loss = em_train_batch(
                        model, optimizer, items, imgsz=args.imgsz, counters=counters)
                    record = complete_phase(phase, {
                        "optimizer_step": counters["optimizer_steps"],
                        "surrogate_detection_loss": surrogate_loss,
                        "coverage": em_sampler.coverage(),
                    })
                    cycle_records.append(record)
                    spec["surrogate_batch_count_completed"] = counters["optimizer_steps"]
                    spec.update(em_sampler.coverage())
                    spec["sampler_state"] = em_sampler.state_dict()
                    spec["counters"] = counters
                    update_status(status_path, spec)
                total_train_missed = total_train_person = 0
                for noise_batch_index, index_batch in enumerate(batches(person_order, args.batch_size)):
                    if time.time() + BUDGET_FINAL_RESERVE_SECONDS >= deadline:
                        raise TimeoutError("Registered cap reserve reached inside an EM noise cycle")
                    phase = begin_phase("em_noise", cycle_index + 1,
                                        noise_batch_index, len(index_batch))
                    items = load_cycle_items(index_batch, continuing=cycle_index > 0)
                    noise_stats: dict = {}
                    deltas = em_noise_batch(model, items, steps=EM_NOISE_STEPS,
                                            step_size=EM_NOISE_STEP, imgsz=args.imgsz,
                                            counters=counters, stats=noise_stats)
                    noise_stats["delta_drift_from_previous_cycle_l2"] = (
                        noise_stats["delta_change_l2"] if cycle_index > 0 else None)
                    retain_deltas(items, deltas)
                    batch_missed = batch_person = 0
                    for item, delta in zip(items, deltas):
                        write_poison(item, delta, outputs / f"{item.stem}.png")
                        missed, total = _train_error_for_batch(model, [item], [delta], args.imgsz)
                        counters["forward_calls"] += 1
                        total_train_missed += missed
                        total_train_person += total
                        batch_missed += missed
                        batch_person += total
                        records_written += 1
                    record = complete_phase(phase, {
                        **noise_stats,
                        "batch_index": noise_batch_index,
                        "training_missed_person_count": batch_missed,
                        "training_person_count": batch_person,
                        "training_detection_error_so_far": total_train_missed / max(1, total_train_person),
                    })
                    cycle_records.append(record)
                    spec.update({"completed_target_images": records_written,
                                 "train_error_person_miss_rate_so_far":
                                     total_train_missed / max(1, total_train_person),
                                 "counters": counters})
                    update_status(status_path, spec)
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(device))
                train_error = total_train_missed / max(1, total_train_person)
                spec["training_detection_error"] = train_error
                spec["training_error_threshold_met"] = train_error <= EM_STOP_ERROR
                if train_error <= EM_STOP_ERROR:
                    spec.setdefault("training_error_threshold_first_met_cycle", cycle_index + 1)

            elif args.method == "sem_det":
                for batch_index, index_batch in enumerate(batches(person_order, args.batch_size)):
                    if time.time() + BUDGET_FINAL_RESERVE_SECONDS >= deadline:
                        raise TimeoutError("Registered cap reserve reached inside a defender cycle")
                    phase = begin_phase("sem_train", cycle_index + 1, batch_index, len(index_batch))
                    cycle_surrogate_exposures += len(index_batch)
                    cycle_surrogate_indices.update(index_batch)
                    items = load_cycle_items(index_batch, continuing=cycle_index > 0)
                    method_seed = args.seed + cycle_index * batch_count + batch_index
                    sem_stats: dict = {}
                    deltas, surrogate_loss = sem_defender_batch(
                        model, optimizer, items, imgsz=args.imgsz,
                        seed=method_seed, counters=counters, random_start=True,
                        stats=sem_stats)
                    if cycle_index > 0:
                        sem_stats.update(delta_transition_stats(items, deltas))
                    retain_deltas(items, deltas)
                    for item, delta in zip(items, deltas):
                        write_poison(item, delta, outputs / f"{item.stem}.png")
                        records_written += 1
                    record = complete_phase(phase, {
                        "robust_surrogate_loss": surrogate_loss, **sem_stats,
                    })
                    cycle_records.append(record)
                    spec.update({"completed_target_images": records_written, "counters": counters})
                    update_status(status_path, spec)
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(device))

            else:
                assert extractor is not None
                for batch_index, index_batch in enumerate(batches(person_order, args.batch_size)):
                    if time.time() + BUDGET_FINAL_RESERVE_SECONDS >= deadline:
                        raise TimeoutError("Registered cap reserve reached inside an EntF Push cycle")
                    phase = begin_phase("entf_push", cycle_index + 1, batch_index, len(index_batch))
                    items = load_cycle_items(index_batch, continuing=cycle_index > 0)
                    initial_deltas = [item.raw_delta for item in items] if cycle_index > 0 else None
                    push_stats: dict = {}
                    deltas = entf_push_batch(
                        extractor, items, centroid, steps=ENTF_STEPS,
                        step_size=ENTF_STEP, seed=args.seed + batch_index,
                        counters=counters, initial_deltas=initial_deltas, stats=push_stats)
                    if cycle_index > 0:
                        push_stats.update(delta_transition_stats(items, deltas))
                    retain_deltas(items, deltas)
                    for item, delta in zip(items, deltas):
                        write_poison(item, delta, outputs / f"{item.stem}.png")
                        records_written += 1
                    record = complete_phase(phase, push_stats)
                    cycle_records.append(record)
                    spec.update({"completed_target_images": records_written, "counters": counters})
                    update_status(status_path, spec)
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(device))

            if records_written != PERSON_COUNT or len(delta_state) != PERSON_COUNT:
                raise RuntimeError(
                    f"Incomplete cycle {cycle_index + 1}: images={records_written}, deltas={len(delta_state)}"
                )
            cycle_elapsed_seconds = max(0.0, time.perf_counter() - cycle_started_perf)
            cycle_completed_at_unix = time.time()
            cycles_completed += 1
            # Keep the existing resource guard's cycle timing semantics. The
            # monotonic measurement is recorded separately as evidence.
            cycle_seconds.append(time.time() - cycle_started)
            spec["method_schedule"]["complete_outer_cycles"] = cycles_completed
            if args.method == "em_det":
                coverage = em_sampler.coverage()
            elif args.method == "sem_det":
                coverage = {
                    "surrogate_seen_images": PERSON_COUNT * cycles_completed,
                    "surrogate_unique_images": len(cycle_surrogate_indices),
                    "surrogate_coverage_ratio": len(cycle_surrogate_indices) / PERSON_COUNT,
                    "surrogate_full_passes": cycles_completed,
                }
            else:
                coverage = {"surrogate_seen_images": 0, "surrogate_unique_images": 0,
                            "surrogate_coverage_ratio": 0.0, "surrogate_full_passes": 0}
            current_model_sha = model_state_sha256(model)
            current_optimizer_sha = optimizer_digest(optimizer)
            cycle_summary = summarize_cycle(
                run_id=args.run_id, method=args.method, cycle=cycles_completed,
                records=cycle_records, started_at_unix=cycle_started,
                completed_at_unix=cycle_completed_at_unix,
                cycle_elapsed_seconds=cycle_elapsed_seconds,
                active_elapsed_seconds=active_time_before_resume + max(
                    0.0, time.perf_counter() - session_started_perf),
                wall_elapsed_seconds=max(0.0, cycle_completed_at_unix - float(spec["started_at_unix"])),
                target_images_completed=records_written,
                surrogate_images_exposed=cycle_surrogate_exposures,
                surrogate_unique_images=len(cycle_surrogate_indices), coverage=coverage,
                counters_before=cycle_counters_before, counters_after=counters,
                training_detection_error=train_error,
                model_sha256=current_model_sha,
                optimizer_sha256=current_optimizer_sha,
                session_id=generation_session_id,
            )
            spec.setdefault("cycle_metrics", []).append(cycle_summary)
            append_diagnostic({"event": "cycle_complete", **cycle_summary})
            if cycle_summary["checkpoint_cycle"]:
                spec.setdefault("convergence_checkpoints", []).append(cycle_summary)
                append_diagnostic({"event": "convergence_checkpoint", **cycle_summary})
            spec["cycle_history"].append({
                "cycle": cycles_completed, "duration_seconds": cycle_seconds[-1],
                "completed_target_images": records_written,
                "training_detection_error": train_error,
                "counters": dict(counters),
                "convergence_summary": cycle_summary,
            })
            spec["counters"] = counters
            spec["peak_gpu_memory_bytes_so_far"] = max_memory
            spec["checkpoint_model_sha256"] = current_model_sha
            spec["checkpoint_optimizer_sha256"] = current_optimizer_sha
            spec["checkpoint_boundary"] = "complete_cycle"
            spec["active_duration_seconds"] = cycle_summary["active_elapsed_seconds"]
            spec["wall_elapsed_seconds"] = cycle_summary["wall_elapsed_seconds"]
            checkpoint_seconds = save_complete_cycle_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, spec=spec,
                deltas=delta_state, sampler=em_sampler, person_order=person_order,
                rng=rng, centroid=centroid)
            spec["checkpoint_write_seconds_last"] = checkpoint_seconds
            spec["active_duration_seconds"] = active_time_before_resume + max(
                0.0, time.perf_counter() - session_started_perf)
            spec["wall_elapsed_seconds"] = max(0.0, time.time() - float(spec["started_at_unix"]))
            update_status(status_path, spec)
            _write_runtime_artifacts(out_root, spec)
            record_time_warning_if_due("cycle_boundary")
            if cycles_completed >= args.max_cycles:
                spec["stop_reason"] = "registered_max_cycles"
                break
            if not budget_allows_next_cycle(args.method, deadline, time.time(), cycle_seconds):
                raise TimeoutError("Budget guard refused another full outer cycle")

        if args.method == "sem_det":
            # Official SEM regenerates every sample against the *final* model.
            # The historical adapter exported stage-mixed deltas directly.
            frozen_sha = model_state_sha256(model)
            frozen_optimizer_sha = optimizer_digest(optimizer)
            frozen_optimizer_steps = counters["optimizer_steps"]
            refinement_started_at_unix = time.time()
            refinement_started_perf = time.perf_counter()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()
            spec["stage"] = "final_refinement"
            spec["final_refinement_images"] = 0
            refinement_attempt = len(spec.get("sem_final_refinement_attempts", [])) + 1
            refinement_start_event = {
                "run_id": args.run_id, "method": args.method, "event": "start",
                "attempt": refinement_attempt,
                "started_at_unix": refinement_started_at_unix,
                "started_at_utc": utc_iso(refinement_started_at_unix),
                "expected_person_images": PERSON_COUNT, "processed_person_images": 0,
                "frozen_model_sha256_before": frozen_sha,
                "optimizer_steps_before": frozen_optimizer_steps,
                "optimizer_sha256_before": frozen_optimizer_sha,
                "complete": False, "convergence_status": "in_progress_not_established",
            }
            spec["sem_final_refinement"] = {
                "attempt": refinement_attempt,
                "state": "running", "started_at_unix": refinement_started_at_unix,
                "started_at_utc": utc_iso(refinement_started_at_unix),
                "expected_person_images": PERSON_COUNT, "processed_person_images": 0,
                "frozen_model_sha256_before": frozen_sha,
                "optimizer_steps_before": frozen_optimizer_steps,
                "optimizer_sha256_before": frozen_optimizer_sha,
                "events": [refinement_start_event], "batch_records": [],
            }
            spec.setdefault("sem_final_refinement_attempts", []).append(
                spec["sem_final_refinement"])
            append_diagnostic({"stage": "sem_final_refinement", **refinement_start_event})
            update_status(status_path, spec)
            refined = 0
            for batch_index, index_batch in enumerate(batches(person_order, args.batch_size)):
                if time.time() + BUDGET_FINAL_RESERVE_SECONDS >= deadline:
                    raise TimeoutError("Generation cap reserve reached during SEM final refinement")
                phase = begin_phase("sem_final_refinement", cycles_completed,
                                    batch_index, len(index_batch))
                items = load_cycle_items(index_batch, continuing=True)
                refine_stats: dict = {}
                deltas, surrogate_loss = sem_defender_batch(
                    model, optimizer, items, imgsz=args.imgsz,
                    seed=args.seed + 10_000_000 + batch_index,
                    counters=counters, update_surrogate=False,
                    random_start=True, stats=refine_stats)
                if surrogate_loss is not None:
                    raise RuntimeError("Final SEM refinement unexpectedly updated surrogate")
                refine_stats.update(delta_transition_stats(items, deltas))
                retain_deltas(items, deltas)
                for item, delta in zip(items, deltas):
                    write_poison(item, delta, outputs / f"{item.stem}.png")
                    refined += 1
                record = complete_phase(phase, refine_stats)
                spec["sem_final_refinement"].setdefault("batch_records", []).append(record)
                spec["sem_final_refinement_attempts"][-1] = spec["sem_final_refinement"]
                spec["final_refinement_images"] = refined
                spec["sem_final_refinement"]["processed_person_images"] = refined
                spec["sem_final_refinement"]["elapsed_seconds_so_far"] = max(
                    0.0, time.perf_counter() - refinement_started_perf)
                spec["counters"] = counters
                update_status(status_path, spec)
                _write_refinement_batches_csv(
                    out_root / "sem_final_refinement_batches.csv",
                    spec["sem_final_refinement"].get("batch_records", []))
            frozen_sha_after = model_state_sha256(model)
            frozen_optimizer_sha_after = optimizer_digest(optimizer)
            optimizer_steps_after = counters["optimizer_steps"]
            expected_stems = {path.stem for path in target_paths}
            actual_stems = {path.stem for path in outputs.glob("*.png")}
            image_ids_match = actual_stems == expected_stems
            refinement_completed_at = time.time()
            refinement_duration = max(0.0, time.perf_counter() - refinement_started_perf)
            refinement_audit = sem_refinement_audit(
                expected_person_images=PERSON_COUNT, processed_person_images=refined,
                image_id_set_matches=image_ids_match,
                frozen_model_sha256_before=frozen_sha,
                frozen_model_sha256_after=frozen_sha_after,
                optimizer_steps_before=frozen_optimizer_steps,
                optimizer_steps_after=optimizer_steps_after,
                optimizer_sha256_before=frozen_optimizer_sha,
                optimizer_sha256_after=frozen_optimizer_sha_after,
                duration_seconds=refinement_duration,
            )
            refinement_end_event = {
                "run_id": args.run_id, "method": args.method, "event": "complete",
                "started_at_utc": utc_iso(refinement_started_at_unix),
                "completed_at_unix": refinement_completed_at,
                "completed_at_utc": utc_iso(refinement_completed_at),
                **refinement_audit,
                "attempt": spec["sem_final_refinement"].get("attempt"),
            }
            spec["sem_final_refinement"].update({
                **refinement_audit,
                "state": "complete" if refinement_audit["complete"] else "failed",
                "completed_at_unix": refinement_completed_at,
                "completed_at_utc": utc_iso(refinement_completed_at),
                "attempt": spec["sem_final_refinement"].get("attempt"),
                "events": [refinement_start_event, refinement_end_event],
                "batch_records": spec["sem_final_refinement"].get("batch_records", []),
            })
            spec["sem_final_refinement_attempts"][-1] = spec["sem_final_refinement"]
            append_diagnostic({"stage": "sem_final_refinement", **refinement_end_event})
            _write_refinement_csv(out_root / "sem_final_refinement.csv",
                                  [event for attempt in spec["sem_final_refinement_attempts"]
                                   for event in attempt.get("events", [])])
            _write_refinement_batches_csv(
                out_root / "sem_final_refinement_batches.csv",
                [record for attempt in spec["sem_final_refinement_attempts"]
                 for record in attempt.get("batch_records", [])])
            write_json_snapshot(out_root / "sem_final_refinement.json", spec["sem_final_refinement"])
            if not refinement_audit["complete"]:
                raise RuntimeError("SEM final refinement changed model or missed target images")
            spec["final_refinement_final_model_sha256"] = frozen_sha

        if records_written != PERSON_COUNT:
            raise RuntimeError(f"Incomplete target coverage: {records_written}/{PERSON_COUNT}")
        if not args.continue_after_time_cap and time.time() > deadline:
            raise TimeoutError("Completed generation exceeded the registered cap")
        expected_counters = expected_total_counters(args.method, batch_count, cycles_completed)
        spec["expected_total_counters"] = expected_counters
        if args.method == "sem_det":
            extra_calls = DEFENDER_STEPS * EOT_SAMPLES * batch_count
            expected_counters["forward_calls"] += extra_calls
            expected_counters["backward_calls"] += extra_calls
            expected_counters["stability_samples"] += PERSON_COUNT * DEFENDER_STEPS * EOT_SAMPLES
        if counters != expected_counters:
            raise RuntimeError(
                f"Full-cycle counters differ from registered schedule: expected={expected_counters}, actual={counters}"
            )
        state_after = model_state_sha256(model)
        spec.update({"state": "complete", "stage": "generation", "schedule_complete": True,
                     "convergence_status": "not_established",
                     "completed_target_images": records_written,
                     "training_detection_error": train_error,
                     "surrogate_updated": state_before != state_after,
                     "surrogate_update_expected": args.method != "entf_push_det",
                     "model_state_before": state_before, "model_state_after": state_after,
                     "peak_gpu_memory_bytes": max_memory,
                     "duration_seconds": active_time_before_resume + time.time() - started,
                     "time_warning_exceeded": bool(spec.get("time_warning_exceeded", False)),
                     "time_warning_overrun_seconds": max(
                         0.0, active_time_before_resume + time.time() - started
                         - args.time_cap_hours * 3600.0),
                     "completed_at_unix": time.time(), "counters": counters,
                     "epsilon": EPS, "overlap_epsilon": EPS if args.spatial_protocol == "new_full_bbox8" else EPS / 2,
                     "outside_target_epsilon": 0})
        if spec["surrogate_updated"] != spec["surrogate_update_expected"]:
            raise RuntimeError("Surrogate update behavior differs from the registered method")
        finalize_runtime("complete", reason="registered_schedule_complete")
    except TimeoutError as exc:
        spec.update({"state": "time_cap_reached", "stage": "generation",
                     "stop_reason": "registered_time_cap_or_checkpoint_reserve",
                     "schedule_complete": False, "convergence_status": "not_established",
                     "resume_available": checkpoint_path.is_file(),
                     "last_complete_checkpoint": str(checkpoint_path) if checkpoint_path.is_file() else None,
                     "last_complete_cycle": cycles_completed,
                     "error": repr(exc), "completed_target_images": records_written,
                     "counters": counters})
        finalize_runtime("time_cap_reached", reason="registered_time_cap_or_checkpoint_reserve")
        raise
    except Exception as exc:
        spec.update({"state": "failed", "stage": "generation", "error": repr(exc),
                     "schedule_complete": False, "convergence_status": "not_established",
                     "resume_available": checkpoint_path.is_file(),
                     "last_complete_checkpoint": str(checkpoint_path) if checkpoint_path.is_file() else None,
                     "last_complete_cycle": cycles_completed,
                     "completed_target_images": records_written,
                     "counters": counters})
        finalize_runtime("failed", reason=repr(exc))
        raise
    finally:
        if extractor is not None:
            extractor.close()
    print(json.dumps({"state": "complete", "run_id": args.run_id,
                      "method": args.method, "target_images": records_written,
                      "duration_seconds": spec["duration_seconds"],
                      "training_detection_error": train_error,
                      "peak_gpu_memory_bytes": max_memory,
                      "counters": counters}, ensure_ascii=False))


if __name__ == "__main__":
    main()
