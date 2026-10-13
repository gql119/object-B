"""Descriptive runtime and convergence evidence for fidelity baseline runs.

These summaries are observational only: they never decide when an optimizer
stops and never modify a method's update rule.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EM_CHECKPOINT_CYCLES = (1, 5, 10, 20, 30)
SEM_CHECKPOINT_PASSES = (1, 3, 5, 7, 9)

CYCLE_FIELDS = (
    "run_id", "method", "session_id", "cycle", "checkpoint_cycle", "cycle_complete",
    "started_at_utc", "completed_at_utc", "cycle_elapsed_seconds",
    "active_elapsed_seconds", "wall_elapsed_seconds", "target_images_completed",
    "surrogate_images_exposed_this_cycle", "surrogate_unique_images_this_cycle",
    "surrogate_cycle_coverage_ratio", "surrogate_seen_images_total",
    "surrogate_unique_images_total", "surrogate_full_passes_total",
    "proxy_loss_kind", "proxy_loss_start", "proxy_loss_mean", "proxy_loss_end",
    "proxy_loss_abs_change_start_to_end", "proxy_loss_relative_change_start_to_end",
    "optimization_loss_kind", "optimization_loss_start", "optimization_loss_mean",
    "optimization_loss_end", "optimization_loss_abs_change_start_to_end",
    "optimization_loss_relative_change_start_to_end", "gradient_norm_mean",
    "gradient_norm_max", "gradient_norm_per_sqrt_image_mean",
    "gradient_norm_per_sqrt_image_max",
    "projected_gradient_residual_mean",
    "projected_gradient_residual_per_sqrt_image_mean",
    "projected_gradient_residual_per_sqrt_image_max", "epsilon_saturation_ratio",
    "delta_update_l2", "delta_update_l2_per_image_rms", "delta_update_linf_max",
    "delta_drift_from_previous_cycle_l2", "delta_drift_from_previous_cycle_l2_per_image_rms",
    "delta_drift_from_previous_cycle_linf_max",
    "optimizer_updates_this_cycle", "noise_image_steps_this_cycle",
    "stability_sample_image_exposures_this_cycle", "training_detection_error",
    "model_sha256", "optimizer_sha256", "coverage_json", "convergence_claim",
)

CHECKPOINT_FIELDS = CYCLE_FIELDS


def utc_iso(unix_seconds: float | None = None) -> str:
    value = datetime.now(timezone.utc).timestamp() if unix_seconds is None else unix_seconds
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _finite_values(records: Iterable[dict], key: str) -> list[tuple[float, float]]:
    values = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        weight = max(1.0, float(record.get("batch_image_count", 1)))
        values.append((value, weight))
    return values


def _weighted_mean(records: list[dict], key: str) -> float | None:
    values = _finite_values(records, key)
    if not values:
        return None
    return sum(value * weight for value, weight in values) / sum(weight for _, weight in values)


def _edge_value(records: list[dict], key: str, *, last: bool = False) -> float | None:
    values = [record.get(key) for record in records]
    values = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    if not values:
        return None
    return values[-1] if last else values[0]


def _loss_change(start: float | None, end: float | None) -> tuple[float | None, float | None]:
    if start is None or end is None:
        return None, None
    absolute = end - start
    relative = absolute / abs(start) if abs(start) > 1e-12 else None
    return absolute, relative


def _combine_l2(records: list[dict], key: str) -> float | None:
    values = [float(record[key]) for record in records
              if record.get(key) is not None and math.isfinite(float(record[key]))]
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values))


def _exact_saturation(records: list[dict]) -> float | None:
    saturated = sum(int(record.get("epsilon_saturated_channels", 0)) for record in records)
    support = sum(int(record.get("support_channels", 0)) for record in records)
    if support:
        return saturated / support
    return _weighted_mean(records, "epsilon_saturation_ratio")


def _counter_delta(before: dict, after: dict, key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def sem_refinement_audit(*, expected_person_images: int, processed_person_images: int,
                         image_id_set_matches: bool, frozen_model_sha256_before: str,
                         frozen_model_sha256_after: str, optimizer_steps_before: int,
                         optimizer_steps_after: int, optimizer_sha256_before: str | None,
                         optimizer_sha256_after: str | None,
                         duration_seconds: float) -> dict:
    """Return evidence checks for final SEM refinement; this is not a stop rule."""
    frozen_model_unchanged = frozen_model_sha256_before == frozen_model_sha256_after
    optimizer_steps_unchanged = optimizer_steps_before == optimizer_steps_after
    optimizer_state_unchanged = optimizer_sha256_before == optimizer_sha256_after
    full_image_coverage = (
        processed_person_images == expected_person_images and image_id_set_matches)
    complete = (full_image_coverage and frozen_model_unchanged
                and optimizer_steps_unchanged and optimizer_state_unchanged)
    return {
        "duration_seconds": max(0.0, float(duration_seconds)),
        "expected_person_images": int(expected_person_images),
        "processed_person_images": int(processed_person_images),
        "image_id_set_matches": bool(image_id_set_matches),
        "full_image_coverage": full_image_coverage,
        "frozen_model_sha256_before": frozen_model_sha256_before,
        "frozen_model_sha256_after": frozen_model_sha256_after,
        "frozen_model_unchanged": frozen_model_unchanged,
        "optimizer_steps_before": int(optimizer_steps_before),
        "optimizer_steps_after": int(optimizer_steps_after),
        "optimizer_steps_unchanged": optimizer_steps_unchanged,
        "optimizer_sha256_before": optimizer_sha256_before,
        "optimizer_sha256_after": optimizer_sha256_after,
        "optimizer_state_unchanged": optimizer_state_unchanged,
        "complete": complete,
        "convergence_status": ("refinement_complete_convergence_not_asserted"
                               if complete else "refinement_incomplete_not_established"),
    }


def summarize_cycle(
    *, run_id: str, method: str, cycle: int, records: list[dict],
    started_at_unix: float, completed_at_unix: float, cycle_elapsed_seconds: float,
    active_elapsed_seconds: float, wall_elapsed_seconds: float,
    target_images_completed: int, surrogate_images_exposed: int,
    surrogate_unique_images: int, coverage: dict, counters_before: dict,
    counters_after: dict, training_detection_error: float | None,
    model_sha256: str, optimizer_sha256: str | None, session_id: str | None = None,
) -> dict:
    """Summarize one completed outer cycle without creating a stop decision."""
    is_em = method == "em_det"
    is_sem = method == "sem_det"
    proxy_stage = "em_surrogate" if is_em else "sem_train" if is_sem else "entf_push"
    proxy_key = "surrogate_detection_loss" if is_em else "robust_surrogate_loss" if is_sem else "push_objective"
    proxy_records = [record for record in records if record.get("stage") == proxy_stage]
    optimization_stage = "em_noise" if is_em else "sem_train" if is_sem else "entf_push"
    optimization_records = [record for record in records
                            if record.get("stage") == optimization_stage
                            and record.get("loss_mean") is not None]
    proxy_values = _finite_values(proxy_records, proxy_key)
    proxy_start = proxy_values[0][0] if proxy_values else None
    proxy_end = proxy_values[-1][0] if proxy_values else None
    proxy_abs_change, proxy_relative_change = _loss_change(proxy_start, proxy_end)
    optimization_start = _edge_value(optimization_records, "loss_first")
    optimization_end = _edge_value(optimization_records, "loss_last", last=True)
    optimization_abs_change, optimization_relative_change = _loss_change(
        optimization_start, optimization_end)
    delta_update_l2 = _combine_l2(optimization_records, "delta_change_l2")
    delta_drift_l2 = _combine_l2(optimization_records, "delta_drift_from_previous_cycle_l2")
    delta_update_linf = max(
        (float(record["delta_change_linf_max"]) for record in optimization_records
         if record.get("delta_change_linf_max") is not None), default=None)
    delta_drift_linf = max(
        (float(record["delta_drift_from_previous_cycle_linf_max"])
         for record in optimization_records
         if record.get("delta_drift_from_previous_cycle_linf_max") is not None), default=None)
    row = {
        "run_id": run_id,
        "method": method,
        "session_id": session_id,
        "cycle": cycle,
        "checkpoint_cycle": cycle in (EM_CHECKPOINT_CYCLES if is_em else SEM_CHECKPOINT_PASSES if is_sem else ()),
        "cycle_complete": True,
        "started_at_utc": utc_iso(started_at_unix),
        "completed_at_utc": utc_iso(completed_at_unix),
        "cycle_elapsed_seconds": float(cycle_elapsed_seconds),
        "active_elapsed_seconds": float(active_elapsed_seconds),
        "wall_elapsed_seconds": float(wall_elapsed_seconds),
        "target_images_completed": int(target_images_completed),
        "surrogate_images_exposed_this_cycle": int(surrogate_images_exposed),
        "surrogate_unique_images_this_cycle": int(surrogate_unique_images),
        "surrogate_cycle_coverage_ratio": surrogate_unique_images / max(1, target_images_completed),
        "surrogate_seen_images_total": coverage.get("surrogate_seen_images"),
        "surrogate_unique_images_total": coverage.get("surrogate_unique_images"),
        "surrogate_full_passes_total": coverage.get("surrogate_full_passes"),
        "proxy_loss_kind": proxy_key,
        "proxy_loss_start": proxy_start,
        "proxy_loss_mean": _weighted_mean(proxy_records, proxy_key),
        "proxy_loss_end": proxy_end,
        "proxy_loss_abs_change_start_to_end": proxy_abs_change,
        "proxy_loss_relative_change_start_to_end": proxy_relative_change,
        "optimization_loss_kind": "target_detection_loss" if is_em else "stability_sampling_loss" if is_sem else "push_objective",
        "optimization_loss_start": optimization_start,
        "optimization_loss_mean": _weighted_mean(optimization_records, "loss_mean"),
        "optimization_loss_end": optimization_end,
        "optimization_loss_abs_change_start_to_end": optimization_abs_change,
        "optimization_loss_relative_change_start_to_end": optimization_relative_change,
        "gradient_norm_mean": _weighted_mean(optimization_records, "gradient_norm_mean"),
        "gradient_norm_max": max(
            (float(record["gradient_norm_max"]) for record in optimization_records
             if record.get("gradient_norm_max") is not None), default=None),
        "gradient_norm_per_sqrt_image_mean": _weighted_mean(optimization_records, "gradient_norm_per_sqrt_image_mean"),
        "gradient_norm_per_sqrt_image_max": max(
            (float(record["gradient_norm_per_sqrt_image_max"]) for record in optimization_records
             if record.get("gradient_norm_per_sqrt_image_max") is not None), default=None),
        "projected_gradient_residual_per_sqrt_image_mean": _weighted_mean(
            optimization_records, "projected_gradient_residual_per_sqrt_image_mean"),
        "projected_gradient_residual_mean": _weighted_mean(
            optimization_records, "projected_gradient_residual_mean"),
        "projected_gradient_residual_per_sqrt_image_max": max(
            (float(record["projected_gradient_residual_per_sqrt_image_max"]) for record in optimization_records
             if record.get("projected_gradient_residual_per_sqrt_image_max") is not None), default=None),
        "epsilon_saturation_ratio": _exact_saturation(optimization_records),
        "delta_update_l2": delta_update_l2,
        "delta_update_l2_per_image_rms": (
            delta_update_l2 / math.sqrt(max(1, target_images_completed))
            if delta_update_l2 is not None else None),
        "delta_update_linf_max": delta_update_linf,
        "delta_drift_from_previous_cycle_l2": delta_drift_l2,
        "delta_drift_from_previous_cycle_l2_per_image_rms": (
            delta_drift_l2 / math.sqrt(max(1, target_images_completed))
            if delta_drift_l2 is not None else None),
        "delta_drift_from_previous_cycle_linf_max": delta_drift_linf,
        "optimizer_updates_this_cycle": _counter_delta(counters_before, counters_after, "optimizer_steps"),
        "noise_image_steps_this_cycle": _counter_delta(counters_before, counters_after, "noise_steps"),
        "stability_sample_image_exposures_this_cycle": _counter_delta(counters_before, counters_after, "stability_samples"),
        "training_detection_error": training_detection_error,
        "model_sha256": model_sha256,
        "optimizer_sha256": optimizer_sha256,
        "coverage_json": json.dumps(coverage, sort_keys=True, separators=(",", ":")),
        "convergence_claim": "diagnostics_only; no numerical stopping threshold registered",
    }
    return row


def write_csv_snapshot(path: Path, rows: list[dict], fields: tuple[str, ...] = CYCLE_FIELDS) -> None:
    """Atomically materialize a small CSV view from checkpointed JSON state."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
        stream.flush()
    temporary.replace(path)


def write_json_snapshot(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
