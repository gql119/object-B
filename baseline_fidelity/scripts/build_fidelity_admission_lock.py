"""Create an immutable provisional schedule lock for the fidelity adaptations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_LOCK = ROOT / "docs" / "fidelity" / "protocol_lock_full_bbox8_last_primary_20261011.json"
OUTPUT_LOCK = ROOT / "docs" / "fidelity" / "protocol_lock_full_bbox8_last_primary_20261011_admission_v2.json"
OUTPUT_SCHEDULE = ROOT / "docs" / "fidelity" / "fidelity_admission_schedule_20261011_v2.json"
REPORT = ROOT / "docs" / "fidelity" / "fidelity_admission_schedule_20261011_v2.md"
CAPS_HOURS = {"em_det": 12.0, "sem_det": 16.0, "entf_push_det": 12.0}
BATCH_COUNT = math.ceil(6095 / 128)
FACTOR = 1.25
FINALIZE_RESERVE_SECONDS = 600.0
FULL_INNER_BATCH_SECONDS = {
    "em_det": 23.05746798682958,
    "sem_det": 91.88375750789419,
    "entf_push_det": 479.55381646193564,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path in (OUTPUT_LOCK, OUTPUT_SCHEDULE, REPORT):
        if path.exists():
            raise FileExistsError(path)
    lock = json.loads(BASE_LOCK.read_text(encoding="utf-8"))
    if lock["protocol_id"] != "voc20_person_full_bbox8_last_primary_seed0_20261011":
        raise RuntimeError("Base Full-BBox-8 protocol lock identity changed")
    lock["protocol_id"] = "voc20_person_full_bbox8_last_primary_seed0_20261011_admission_v2"
    lock["official_full_experiments_allowed"] = False
    lock["allow_condition"] = (
        "Admission only. Do not start full generation until the full-cycle benchmark, "
        "checkpoint-resume gate, implementation review, and final user-facing schedule report pass."
    )
    lock["method_generation_authorization"] = {
        "authorized_by": "user instruction on 2026-10-11 for EM, SEM, EntF follow-up admission",
        "order": ["EM-Det-Fidelity", "SEM-Det-Fidelity", "EntF-Push-Det-Fidelity"],
        "noise_generation_wall_clock_caps_hours": CAPS_HOURS,
        "victim_training_excluded_from_noise_caps": True,
        "permitted_schedule_changes": ["complete outer-cycle count only", "memory-safe batch size"],
        "forbidden_changes": ["method inner-loop definitions", "losses", "sampling", "step sizes", "final refinement"],
        "deletion_authorized": False,
        "github_publication": "only after full experiments, on a new branch",
    }
    OUTPUT_LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    em_cycle = FULL_INNER_BATCH_SECONDS["em_det"] * BATCH_COUNT
    sem_cycle = FULL_INNER_BATCH_SECONDS["sem_det"] * BATCH_COUNT
    em_planned_cycles = math.floor((CAPS_HOURS["em_det"] * 3600 - FINALIZE_RESERVE_SECONDS)
                                   / (em_cycle * FACTOR))
    sem_total_passes = math.floor((CAPS_HOURS["sem_det"] * 3600 - FINALIZE_RESERVE_SECONDS)
                                  / (sem_cycle * FACTOR))
    sem_planned_cycles = sem_total_passes - 1  # reserve one full pass for mandatory final refinement
    schedule = {
        "schema": "fidelity_method_admission_schedule.v2",
        "status": "provisional_admission_blocked",
        "protocol_lock": str(OUTPUT_LOCK),
        "protocol_lock_sha256": sha(OUTPUT_LOCK),
        "source_full_bbox8_lock_sha256": sha(BASE_LOCK),
        "formal_generation_enabled": False,
        "method_order": ["em_det", "sem_det", "entf_push_det"],
        "shared": {
            "target_images": 6095, "target_objects": 13256, "batch_size": 128,
            "spatial_protocol": "Full Person bbox union, 8/255 including overlaps, zero outside",
            "training_epochs": 200, "noise_caps_hours": CAPS_HOURS,
            "projection_factor": FACTOR, "checkpoint_finalization_reserve_seconds": FINALIZE_RESERVE_SECONDS,
            "victim_noise_cap_included": False,
            "convergence_policy": "A wall-clock cap or a fixed cycle count is not evidence of convergence. Preserve and report per-cycle objective, gradient, delta-drift, projection and saturation curves; do not claim convergence absent a pre-registered method-specific criterion and measured evidence.",
        },
        "methods": {
            "em_det": {
                "status": "blocked_pending_full_cycle_and_resume_validation",
                "time_cap_hours": CAPS_HOURS["em_det"],
                "planned_complete_outer_cycles": em_planned_cycles,
                "per_cycle": {
                    "surrogate_optimizer_steps": 10,
                    "surrogate_sample_exposures": 1280,
                    "full_dataset_surrogate_exposure_equivalent": 1280 / 6095,
                    "noise_updates_per_target_image": 20,
                    "noise_delta_steps": 6095 * 20,
                    "train_only_detection_error_passes": 1,
                },
                "planned_totals": {
                    "surrogate_optimizer_steps": 10 * em_planned_cycles,
                    "surrogate_sample_exposures": 1280 * em_planned_cycles,
                    "surrogate_full_dataset_exposure_equivalent": 1280 * em_planned_cycles / 6095,
                    "noise_full_target_traversals": em_planned_cycles,
                    "train_error_full_target_traversals": em_planned_cycles,
                    "noise_delta_steps": 6095 * 20 * em_planned_cycles,
                },
                "benchmark": {
                    "measured_unit_seconds": FULL_INNER_BATCH_SECONDS["em_det"],
                    "extrapolated_full_cycle_seconds_before_factor": em_cycle,
                    "projected_seconds_with_factor_and_reserve": em_planned_cycles * em_cycle * FACTOR + FINALIZE_RESERVE_SECONDS,
                    "projection_is_full_cycle_measurement": False,
                },
                "blocking_gates": ["one actual 6095-image complete cycle timing", "full-cycle checkpoint interruption and exact resume test", "complete target-dataset noise counter audit"],
            },
            "sem_det": {
                "status": "blocked_pending_full_cycle_and_resume_validation",
                "time_cap_hours": CAPS_HOURS["sem_det"],
                "planned_complete_outer_cycles": sem_planned_cycles,
                "per_cycle": {
                    "surrogate_optimizer_steps": BATCH_COUNT,
                    "stability_samples": 6095 * 10 * 5,
                    "attacker_pgd_steps": 6095 * 10,
                    "full_target_traversals": 1,
                },
                "final_refinement": {
                    "required": True, "target_images": 6095,
                    "stability_samples": 6095 * 10 * 5,
                    "attacker_pgd_steps": 0, "surrogate_optimizer_steps": 0,
                    "surrogate_model_must_remain_frozen": True,
                },
                "planned_totals": {
                    "surrogate_optimizer_steps": BATCH_COUNT * sem_planned_cycles,
                    "training_stability_samples": 6095 * 10 * 5 * sem_planned_cycles,
                    "final_refinement_stability_samples": 6095 * 10 * 5,
                    "attacker_pgd_steps": 6095 * 10 * sem_planned_cycles,
                    "training_full_target_traversals": sem_planned_cycles,
                    "final_refinement_full_target_traversals": 1,
                },
                "benchmark": {
                    "measured_unit_seconds": FULL_INNER_BATCH_SECONDS["sem_det"],
                    "extrapolated_full_cycle_seconds_before_factor": sem_cycle,
                    "projected_seconds_with_factor_and_reserve": (sem_planned_cycles + 1) * sem_cycle * FACTOR + FINALIZE_RESERVE_SECONDS,
                    "projection_is_full_cycle_measurement": False,
                },
                "blocking_gates": ["one actual 6095-image complete cycle timing", "exact cycle-boundary checkpoint/resume validation", "full final-refinement resume and model-freeze audit"],
            },
            "entf_push_det": {
                "status": "blocked_representation_gate_failed",
                "time_cap_hours": CAPS_HOURS["entf_push_det"],
                "planned_complete_outer_cycles": None,
                "benchmark": {
                    "measured_unit_seconds": FULL_INNER_BATCH_SECONDS["entf_push_det"],
                    "push_only_full_pass_seconds": FULL_INNER_BATCH_SECONDS["entf_push_det"] * BATCH_COUNT,
                    "push_only_full_pass_hours": FULL_INNER_BATCH_SECONDS["entf_push_det"] * BATCH_COUNT / 3600,
                    "full_centroid_included": False,
                    "representation_gate_passed": False,
                },
                "blocking_gates": ["valid GT-matched object-level ROI embedding", "clean centroid construction with explicit non-target overlap policy", "reference model policy", "full centroid/push runtime and convergence benchmark"],
            },
        },
    }
    OUTPUT_SCHEDULE.write_text(json.dumps(schedule, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# Fidelity baseline admission schedule (provisional)

Status: **BLOCKED pending the listed method gates.** This file does not authorize or start full poison generation. It supersedes the older 8-hour/REM schedule draft only for the new EM/SEM/EntF fidelity candidate. Safe-Soft05 and historical outputs are unchanged.

The separate Full-BBox-8 lock uses 8/255 throughout the union of Person boxes (overlap included) and zero outside. Victims are random-init YOLOv8n, 200 epochs; `last.pt` is primary and `best.pt` secondary.

| Method | Cap | Provisional complete cycles | Full-target work | Projected noise wall time* | Admission |
|---|---:|---:|---|---:|---|
| EM-Det-Fidelity | 12 h | {em_planned_cycles} | {em_planned_cycles} noise passes; {em_planned_cycles} train-error passes; {1280*em_planned_cycles:,} surrogate sample exposures ({1280*em_planned_cycles/6095:.2f} full-dataset equivalents); {10*em_planned_cycles:,} optimizer updates; {6095*20*em_planned_cycles:,} per-image noise descent steps | {(em_planned_cycles*em_cycle*FACTOR+FINALIZE_RESERVE_SECONDS)/3600:.2f} h | blocked: no real full-cycle resume yet |
| SEM-Det-Fidelity | 16 h | {sem_planned_cycles} | {sem_planned_cycles} defender/attacker training passes + 1 mandatory frozen-model final-refinement pass; {BATCH_COUNT*sem_planned_cycles:,} surrogate updates; {6095*10*5*(sem_planned_cycles+1):,} EOT samples; {6095*10*sem_planned_cycles:,} attacker PGD steps | {((sem_planned_cycles+1)*sem_cycle*FACTOR+FINALIZE_RESERVE_SECONDS)/3600:.2f} h | blocked: full-cycle resume/refinement audit pending |
| EntF-Push-Det-Fidelity | 12 h | — | 300 Push steps/image must remain intact; full object centroid has not passed representation audit | Push-only pass estimate {(FULL_INNER_BATCH_SECONDS['entf_push_det']*BATCH_COUNT/3600):.2f} h, centroid excluded | blocked: ROI/centroid representation invalid/unverified |

\\*Projection is derived from a complete-inner-unit batch-128 GPU probe multiplied by {BATCH_COUNT} batches/cycle, a {FACTOR:.2f} safety factor and {FINALIZE_RESERVE_SECONDS/60:.0f}-minute finalization reserve. It is **not** a measured full-cycle runtime. The full-cycle benchmark is a pre-run gate; if it invalidates the cap estimate, the run remains blocked rather than silently reducing inner work.

The cycle schedule is not a convergence claim. Each cap is a resource ceiling. The current logs support descriptive tracking of loss, train-only detection error (EM), gradient norm, delta drift, projection residual, epsilon saturation and cycle-to-cycle changes. A numeric convergence criterion has not yet been established for these adaptations; until pre-registered and measured, completion must be reported as `schedule_complete` with `convergence_status=not_established`, never as convergence.

Before generation, EM and SEM must pass full-dataset cycle timing and exact checkpoint recovery at the cycle boundary. On cap, only the last complete-cycle checkpoint is recoverable; any partial cycle is discarded on exact resume. SEM's final refinement must be a complete 6,095-image pass with a frozen model. EntF cannot be scheduled until the object-level representation and centroid gates pass.

The matched clean reference and E2 archive are read-only context. E2's overlap epsilon was 4/255 and is not used as the spatial protocol for these candidates.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"protocol_lock": str(OUTPUT_LOCK), "protocol_lock_sha256": sha(OUTPUT_LOCK),
                      "schedule": str(OUTPUT_SCHEDULE), "schedule_sha256": sha(OUTPUT_SCHEDULE),
                      "report": str(REPORT), "em_cycles": em_planned_cycles,
                      "sem_cycles": sem_planned_cycles, "formal_generation_enabled": False}, indent=2))


if __name__ == "__main__":
    main()
