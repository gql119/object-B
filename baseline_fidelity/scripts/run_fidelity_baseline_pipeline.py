"""Run one locked EM/SEM/EntF generation, audit, victim, and evaluation chain.

This runner is intentionally single-method. A controller may launch the next
method only after the previous RunID is fully audited and archived.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ALLOWED_ROOT = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/"
    "ue_project/ablation_safe_bg_20261004/candidate/utility_feasibility_20261009/"
    "baseline_fidelity_20261011"
)
METHODS = ("em_det", "sem_det", "entf_push_det")
CAP_HOURS = {"em_det": 12.0, "sem_det": 16.0, "entf_push_det": 12.0}
EM_CHECKPOINT_CYCLES = (1, 5, 10, 20, 30)
SEM_CHECKPOINT_PASSES = (1, 3, 5, 7, 9)


class BudgetIncomplete(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def registered_identity(args: argparse.Namespace, lock_path: Path) -> dict:
    code_files = (
        "scripts/run_fidelity_baseline_pipeline.py",
        "scripts/run_fidelity_victim.py",
        "scripts/generate_fidelity_baseline.py",
        "scripts/materialize_fidelity_baseline.py",
        "scripts/run_locked_clean_baseline.py",
        "ue_framework/baselines_det/fidelity.py",
        "ue_framework/baselines_det/convergence.py",
        "ue_framework/baselines_det/production.py",
        "ue_framework/baselines_det/core.py",
        "ue_framework/baselines_det/locked_data.py",
    )
    return {
        "method": args.method, "run_id": args.run_id,
        "protocol_lock_sha256": sha256_file(lock_path),
        "schedule_lock_sha256": sha256_file(args.schedule_lock.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "surrogate_sha256": sha256_file(args.surrogate),
        "batch_size": args.batch_size, "time_cap_hours": args.time_cap_hours,
        "continue_after_time_cap": args.continue_after_time_cap,
        "max_cycles": args.max_cycles, "seed": args.seed,
        "code_commit": args.code_commit,
        "source_sha256": {name: sha256_file(ROOT / name) for name in code_files},
    }


def validate_admission(lock: dict, schedule: dict, *, lock_sha256: str,
                       method: str, max_cycles: int, time_cap_hours: float,
                       continue_after_time_cap: bool = False) -> None:
    if lock.get("protocol_id") != "voc20_person_full_bbox8_last_primary_seed0_20261012_em_formal_v3":
        raise RuntimeError("This pipeline accepts only the current EM formal Full-BBox-8 lock")
    if lock.get("official_full_experiments_allowed") is not True:
        raise RuntimeError("Protocol lock is admission-only; formal generation is not authorized in this snapshot")
    if schedule.get("protocol_lock_sha256") != lock_sha256:
        raise RuntimeError("Admission schedule is not bound to this protocol lock")
    method_schedule = schedule.get("methods", {}).get(method, {})
    authorization = lock.get("method_generation_authorization", {})
    if method == "entf_push_det":
        raise RuntimeError("EntF remains blocked: object-level ROI/centroid representation gate is unresolved")
    if method not in authorization.get("allowed_methods", []):
        raise RuntimeError(f"Method is outside the current formal authorization: {method}")
    if method not in schedule.get("authorized_methods", []):
        raise RuntimeError(f"Method is outside the admitted schedule: {method}")
    if method_schedule.get("status") not in ("admitted", "ready_to_launch"):
        raise RuntimeError(f"Method is not admitted: {method_schedule.get('status')}")
    if schedule.get("formal_generation_enabled") is not True:
        raise RuntimeError("Formal generation remains locked pending admission report and gate completion")
    if method_schedule.get("planned_complete_outer_cycles") != max_cycles:
        raise RuntimeError("Requested cycle count differs from the frozen admission schedule")
    if method_schedule.get("time_cap_hours") != time_cap_hours:
        raise RuntimeError("Requested time cap differs from the frozen admission schedule")
    if bool(method_schedule.get("continue_after_time_cap", False)) != continue_after_time_cap:
        raise RuntimeError("Requested time-limit behavior differs from the frozen schedule")
    if continue_after_time_cap:
        policy = "warning_only_continue_to_registered_schedule"
        if authorization.get("noise_time_policy") != policy or method_schedule.get("noise_time_policy") != policy:
            raise RuntimeError("Warning-only time policy is not bound by both locks")
    if lock.get("poison", {}).get("spatial_protocol") != "new_full_bbox8":
        raise RuntimeError("Locked perturbation support is not Full-BBox-8")
    if lock.get("victim", {}).get("checkpoint_for_evaluation") != "last.pt primary; best.pt secondary":
        raise RuntimeError("Lock does not assign last.pt primary evaluation")


def launch(command: list[str], cwd: Path, log_path: Path) -> int:
    environment = dict(os.environ)
    old = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(cwd) if not old else str(cwd) + os.pathsep + old
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                env=environment, check=False)
    return int(result.returncode)


def assert_idle_gpu() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    pids = [line.strip() for line in apps.stdout.splitlines()
            if line.strip() and line.strip() != "[Not Found]"]
    if pids:
        raise RuntimeError(f"GPU is occupied by compute process(es): {pids}")
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"Unexpected nvidia-smi response: {result.stdout!r}")
    return {"name": fields[0], "memory_total_mib": int(fields[1]),
            "memory_used_mib": int(fields[2]), "utilization_percent": int(fields[3])}


def _csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def generation_evidence_is_ready(status: dict, args: argparse.Namespace) -> None:
    """Require the registered runtime/convergence evidence before materializing data."""
    cycle_path = Path(status.get("cycle_metrics_path", ""))
    if not cycle_path.is_file():
        raise RuntimeError("Generation is missing cycle_metrics.csv")
    generation_root = cycle_path.parent
    expected_paths = (
        generation_root / "runtime_summary.json",
        generation_root / "runtime_events.jsonl",
        generation_root / "phase_metrics.csv",
        generation_root / "convergence_checkpoints.csv",
        generation_root / "convergence_batches.jsonl",
    )
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Generation is missing runtime/convergence evidence: {missing}")

    runtime = json.loads(expected_paths[0].read_text(encoding="utf-8"))
    if runtime.get("run_id") != args.run_id or runtime.get("method") != args.method:
        raise RuntimeError("Runtime summary belongs to a different run or method")
    if runtime.get("state") != "complete" or runtime.get("schedule_complete") is not True:
        raise RuntimeError("Runtime summary does not confirm schedule completion")
    active_seconds = float(runtime.get("active_duration_seconds") or 0.0)
    wall_seconds = float(runtime.get("wall_elapsed_seconds") or 0.0)
    if not math.isfinite(active_seconds) or not math.isfinite(wall_seconds) \
            or active_seconds <= 0 or wall_seconds + 0.05 < active_seconds:
        raise RuntimeError("Runtime summary has invalid active/wall duration evidence")
    if runtime.get("convergence_status") != "not_established":
        raise RuntimeError("Run must not infer convergence from a schedule or time cap")

    cycle_rows = _csv_rows(cycle_path)
    if len(cycle_rows) != args.max_cycles:
        raise RuntimeError("Cycle metrics do not cover every registered outer cycle")
    if [int(row["cycle"]) for row in cycle_rows] != list(range(1, args.max_cycles + 1)):
        raise RuntimeError("Cycle metric sequence has a gap or duplicate")
    if any(row.get("cycle_complete", "").lower() != "true" for row in cycle_rows):
        raise RuntimeError("Cycle metrics contain an incomplete cycle")

    required_checkpoints = (EM_CHECKPOINT_CYCLES if args.method == "em_det"
                            else SEM_CHECKPOINT_PASSES if args.method == "sem_det" else ())
    checkpoint_rows = _csv_rows(expected_paths[3])
    observed_checkpoints = [int(row["cycle"]) for row in checkpoint_rows]
    if observed_checkpoints != [cycle for cycle in required_checkpoints if cycle <= args.max_cycles]:
        raise RuntimeError("Convergence checkpoint CSV is missing a registered milestone")
    for row in checkpoint_rows:
        for field in ("proxy_loss_start", "proxy_loss_mean", "proxy_loss_end",
                      "optimization_loss_start", "optimization_loss_mean", "optimization_loss_end",
                      "gradient_norm_mean", "projected_gradient_residual_mean",
                      "epsilon_saturation_ratio", "delta_update_l2"):
            value = row.get(field)
            if value in (None, "") or not math.isfinite(float(value)):
                raise RuntimeError(f"Convergence checkpoint {row.get('cycle')} lacks finite {field}")
        if row.get("convergence_claim", "").find("no numerical stopping threshold") < 0:
            raise RuntimeError("A checkpoint contains an unregistered convergence claim")

    phases = _csv_rows(expected_paths[2])
    if runtime.get("phase_metric_rows") != len(phases) or runtime.get("phase_event_parse_errors") != 0:
        raise RuntimeError("Phase runtime CSV does not reconcile with its event log")
    if any(row.get("phase_state") == "started_without_end_event" for row in phases):
        raise RuntimeError("A timed batch phase has no completion/interruption event")
    batch_count = (6095 + args.batch_size - 1) // args.batch_size
    stage_counts_per_cycle = ({"em_surrogate": 10, "em_noise": batch_count}
                              if args.method == "em_det" else
                              {"sem_train": batch_count}
                              if args.method == "sem_det" else {})
    for cycle in cycle_rows:
        cycle_number = int(cycle["cycle"])
        cycle_session = cycle.get("session_id")
        if not cycle_session:
            raise RuntimeError(f"Cycle {cycle_number} lacks its generation-session identity")
        for stage, expected_count in stage_counts_per_cycle.items():
            completed = [row for row in phases
                         if row.get("session_id") == cycle_session
                         and int(row.get("cycle", -1)) == cycle_number
                         and row.get("stage") == stage and row.get("phase_state") == "complete"]
            if len(completed) != expected_count:
                raise RuntimeError(
                    f"Cycle {cycle_number} timed {stage} evidence has {len(completed)} completed batches; "
                    f"expected {expected_count}")
            for row in completed:
                elapsed = row.get("elapsed_seconds")
                if elapsed in (None, "") or not math.isfinite(float(elapsed)) or float(elapsed) < 0:
                    raise RuntimeError(f"Timed {stage} batch has invalid elapsed time")

    if args.method == "sem_det":
        refinement_path = generation_root / "sem_final_refinement.json"
        refinement_csv = generation_root / "sem_final_refinement.csv"
        batch_csv = generation_root / "sem_final_refinement_batches.csv"
        if not refinement_path.is_file() or not refinement_csv.is_file() or not batch_csv.is_file():
            raise RuntimeError("SEM is missing final-refinement evidence files")
        refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
        if (refinement.get("state") != "complete" or refinement.get("complete") is not True
                or refinement.get("processed_person_images") != 6095
                or refinement.get("full_image_coverage") is not True
                or refinement.get("frozen_model_unchanged") is not True
                or refinement.get("optimizer_steps_unchanged") is not True
                or refinement.get("optimizer_state_unchanged") is not True
                or refinement.get("frozen_model_sha256_after") != status.get("model_state_after")):
            raise RuntimeError("SEM final refinement did not prove full coverage and frozen-state integrity")
        refinement_batches = refinement.get("batch_records", [])
        if (len(refinement_batches) != batch_count
                or sum(int(row.get("batch_image_count", 0)) for row in refinement_batches) != 6095
                or float(refinement.get("duration_seconds", 0.0)) <= 0):
            raise RuntimeError("SEM final-refinement duration or per-image batch record is incomplete")
        phase_by_id = {row.get("phase_id"): row for row in phases}
        for record in refinement_batches:
            phase = phase_by_id.get(record.get("phase_id"))
            if (phase is None or phase.get("phase_state") != "complete"
                    or phase.get("session_id") != record.get("session_id")):
                raise RuntimeError("SEM final-refinement batch timing does not reconcile with phase log")


def generation_is_ready(status: dict, args: argparse.Namespace) -> None:
    schedule = status.get("method_schedule", {})
    if status.get("state") == "time_cap_reached":
        raise BudgetIncomplete(
            f"Generation reached its cap after {schedule.get('complete_outer_cycles', 0)} "
            "complete cycle(s); last checkpoint retained, convergence not established"
        )
    if status.get("state") != "complete" or status.get("schedule_complete") is not True:
        raise RuntimeError("Generation did not complete its registered schedule")
    if status.get("completed_target_images") != 6095:
        raise RuntimeError("Generation did not produce a complete 6,095-image target pass")
    if int(schedule.get("complete_outer_cycles", 0)) != args.max_cycles:
        raise RuntimeError("Completed cycle count differs from the registered schedule")
    if args.method == "em_det" and status.get("surrogate_coverage_ratio") != 1.0:
        raise RuntimeError("EM sampler did not cover all locked Person images")
    if args.method == "sem_det" and status.get("final_refinement_images") != 6095:
        raise RuntimeError("SEM did not complete its final full-dataset refinement")
    generation_evidence_is_ready(status, args)


def classify_generation_result(status_code: int, status: dict, *, launched_at_unix: float) -> str:
    """Distinguish a resumable time cap from crashes and incomplete exits."""
    if status.get("state") == "time_cap_reached":
        checkpoint = status.get("last_complete_checkpoint")
        if (status.get("resume_available") is not True or not checkpoint
                or not Path(checkpoint).is_file()):
            raise RuntimeError("Time-cap status has no readable complete-cycle checkpoint")
        if float(status.get("completed_at_unix", 0)) < launched_at_unix:
            raise RuntimeError("Time-cap status predates this generation attempt; refusing stale status")
        return "time_cap_reached"
    if status_code != 0:
        raise RuntimeError(f"Generation failed with exit code {status_code}; artifacts retained")
    if status.get("state") != "complete":
        raise RuntimeError("Generation exited successfully without a complete status")
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol-lock", required=True, type=Path)
    parser.add_argument("--schedule-lock", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--surrogate", required=True, type=Path)
    parser.add_argument("--max-cycles", required=True, type=int)
    parser.add_argument("--time-cap-hours", required=True, type=float)
    parser.add_argument("--continue-after-time-cap", action="store_true",
                        help="Treat the registered threshold as a warning; finish the locked schedule")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Resume an incomplete generation only from its exact complete-cycle checkpoint")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,95}", args.run_id):
        raise ValueError("RunID must be a unique simple path component")
    if not re.fullmatch(r"[0-9a-f]{40,64}", args.code_commit):
        raise ValueError("code-commit must identify the immutable reviewed code snapshot")
    if args.batch_size != 128:
        raise ValueError("This admission schedule is benchmarked only at batch size 128")
    if not 0 < args.time_cap_hours <= CAP_HOURS[args.method]:
        raise ValueError(f"{args.method} cap must be in (0, {CAP_HOURS[args.method]}] hours")
    if args.max_cycles < 1:
        raise ValueError("At least one complete outer cycle is required")

    lock_path = args.protocol_lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    schedule_path = args.schedule_lock.resolve()
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    validate_admission(lock, schedule, lock_sha256=sha256_file(lock_path),
                       method=args.method, max_cycles=args.max_cycles,
                       time_cap_hours=args.time_cap_hours,
                       continue_after_time_cap=args.continue_after_time_cap)

    output_root = (ALLOWED_ROOT / args.run_id).resolve()
    output_root.relative_to(ALLOWED_ROOT.resolve())
    state_path = output_root / "pipeline_status.json"
    identity = registered_identity(args, lock_path)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if output_root.exists() != args.resume:
        raise FileExistsError("Fresh run requires a new RunID; resume requires the existing isolated RunID")
    if not args.resume:
        if not ALLOWED_ROOT.is_dir():
            raise FileNotFoundError(f"Authorized isolated root is missing: {ALLOWED_ROOT}")
        free_bytes = shutil.disk_usage(ALLOWED_ROOT).free
        if free_bytes < 55 * 1024**3:
            raise RuntimeError("Less than the locked 55 GiB free-space reserve; no training started")
        gpu = assert_idle_gpu()
        output_root.mkdir(parents=True, exist_ok=False)
        status = {"state": "running", "stage": "generation_preflight", "run_id": args.run_id,
                  "method": args.method, "registered_identity": identity,
                  "registered_identity_sha256": identity_sha, "gpu_preflight": gpu,
                  "free_bytes_pre_generation": free_bytes,
                  "started_at_unix": time.time(),
                  "noise_time_cap_hours": (None if args.continue_after_time_cap else args.time_cap_hours),
                  "noise_time_warning_hours": (args.time_cap_hours if args.continue_after_time_cap else None),
                  "noise_time_policy": ("warning_only_continue_to_registered_schedule"
                                        if args.continue_after_time_cap else "hard_stop"),
                  "max_complete_outer_cycles": args.max_cycles,
                  "official_metrics_disabled_until_victim": True,
                  "artifact_ingest_disabled": True}
        atomic_json(state_path, status)
    else:
        if not state_path.is_file():
            raise RuntimeError("Cannot resume without the original pipeline status")
        status = json.loads(state_path.read_text(encoding="utf-8"))
        if status.get("registered_identity_sha256") != identity_sha:
            raise RuntimeError("Resume identity differs; exact resume is unsafe")
        if status.get("state") not in ("time_cap_reached", "incomplete", "failed"):
            raise RuntimeError("Only an incomplete or failed isolated run may be resumed")
        assert_idle_gpu()
        if shutil.disk_usage(output_root).free < 55 * 1024**3:
            raise RuntimeError("Free disk reserve fell below 55 GiB; resume not started")
        status.update(state="running", stage="resume_preflight", resumed_at_unix=time.time())
        atomic_json(state_path, status)

    generation = output_root / "generation"
    materialization = output_root / "materialization"
    try:
        gen_status_path = generation / "status.json"
        gen_status = json.loads(gen_status_path.read_text(encoding="utf-8")) if gen_status_path.exists() else None
        if gen_status is None or gen_status.get("state") != "complete":
            if args.resume:
                if gen_status is None or gen_status.get("state") not in ("time_cap_reached", "failed"):
                    raise RuntimeError("No resumable generation checkpoint state exists")
                resume = True
            else:
                if generation.exists():
                    raise FileExistsError("Generation output already exists; refusing duplicate generation")
                resume = False
            gpu = assert_idle_gpu()
            status.update(stage="generation", gpu_preflight=gpu, stage_started_unix=time.time())
            atomic_json(state_path, status)
            command = [
                sys.executable, str(ROOT / "scripts" / "generate_fidelity_baseline.py"),
                "--method", args.method, "--run-id", args.run_id,
                "--dataset-root", str(args.dataset_root.resolve()),
                "--surrogate", str(args.surrogate.resolve()),
                "--protocol-lock", str(lock_path), "--output-root", str(generation),
                "--batch-size", str(args.batch_size), "--time-cap-hours", str(args.time_cap_hours),
                "--max-cycles", str(args.max_cycles), "--seed", str(args.seed),
                "--code-commit", args.code_commit, "--spatial-protocol", "new_full_bbox8",
            ]
            if args.continue_after_time_cap:
                command.append("--continue-after-time-cap")
            if resume:
                command.append("--resume")
            generation_started_at = time.time()
            status_code = launch(command, ROOT, output_root / (
                "generation_resume.console.log" if resume else "generation.console.log"))
            gen_status = json.loads(gen_status_path.read_text(encoding="utf-8"))
            outcome = classify_generation_result(status_code, gen_status,
                                                 launched_at_unix=generation_started_at)
            if outcome == "time_cap_reached":
                status.update(state="time_cap_reached", stage="generation",
                              generation_state=gen_status.get("state"),
                              completed_cycles=gen_status.get("last_complete_cycle",
                                  gen_status.get("method_schedule", {}).get("complete_outer_cycles", 0)),
                              checkpoint=gen_status.get("last_complete_checkpoint"),
                              convergence_status="not_established", resume_available=bool(
                                  gen_status.get("resume_available")), failed_at_unix=time.time())
                atomic_json(state_path, status)
                raise BudgetIncomplete("Noise generation reached its cap; preserved the last complete checkpoint")
            generation_is_ready(gen_status, args)
        else:
            generation_is_ready(gen_status, args)

        audit_path = materialization / "full_preflight.json"
        if not audit_path.is_file():
            if materialization.exists():
                raise FileExistsError("Partial materialization exists; preserving it for diagnosis")
            status.update(stage="materialization", stage_started_unix=time.time())
            atomic_json(state_path, status)
            code = launch([
                sys.executable, str(ROOT / "scripts" / "materialize_fidelity_baseline.py"),
                "--method", args.method, "--run-id", args.run_id,
                "--dataset-root", str(args.dataset_root.resolve()), "--protocol-lock", str(lock_path),
                "--noise-run", str(generation), "--output", str(materialization),
            ], ROOT, output_root / "materialization.console.log")
            if code != 0:
                raise RuntimeError("Materialization/audit command failed; artifacts retained")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("state") != "passed":
            raise RuntimeError("Full materialization audit did not pass")

        victim_status_path = output_root / "victim_status.json"
        if not victim_status_path.exists():
            assert_idle_gpu()
            status.update(stage="victim", materialization_audit_sha256=sha256_file(audit_path),
                          stage_started_unix=time.time())
            atomic_json(state_path, status)
            code = launch([
                sys.executable, str(ROOT / "scripts" / "run_fidelity_victim.py"),
                "--run-root", str(output_root), "--dataset-root", str(args.dataset_root.resolve()),
                "--protocol-lock", str(lock_path), "--method", args.method,
                "--run-id", args.run_id, "--code-commit", args.code_commit,
            ], ROOT, output_root / "victim.console.log")
            if code != 0:
                raise RuntimeError("Victim stage failed; checkpoints/logs retained")
        victim_status = json.loads(victim_status_path.read_text(encoding="utf-8"))
        if victim_status.get("state") != "complete" or victim_status.get("epochs_completed") != 200:
            raise RuntimeError("Victim/evaluation stage is not complete")
        status.update(state="complete", stage="complete", victim_status_sha256=sha256_file(victim_status_path),
                      elapsed_seconds=time.time() - status["started_at_unix"],
                      convergence_status="reported_from_cycle_diagnostics_not_from_cap",
                      completed_at_unix=time.time())
        atomic_json(state_path, status)
    except BudgetIncomplete as exc:
        status.setdefault("error", str(exc))
        atomic_json(state_path, status)
        raise
    except BaseException as exc:
        status.update(state="failed", failed_stage=status.get("stage"), error=repr(exc),
                      failed_at_unix=time.time())
        atomic_json(state_path, status)
        raise
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
