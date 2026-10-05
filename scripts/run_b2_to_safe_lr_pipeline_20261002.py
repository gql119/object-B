"""Run the already-admitted B2-TO-Safe-LR experiment through final reporting."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "b2_to_safe_lr_eps8_person_seed0_20261002"
NOISE_ID = "noise_full_" + RUN_ID
VICTIM_ID = "victim_" + RUN_ID
EXPECTED_INDICES = [0, 6, 10, 29, 41, 68, 77, 86, 89, 112, 116, 125,
                    128, 133, 140, 167, 173, 184, 220, 229, 237, 238, 239, 240]
REVIEWED_FILES = {
    "ue_framework/configs/exp_voc_person_b2_to_safe_lr_p5_rank8_eps8_seed0_20261002.yaml",
    "ue_framework/methods/b2_semantic.py",
    "ue_framework/methods/safebg.py",
    "scripts/check_b2_to_safe_lr_sanity_20261002.py",
    "experiments/b2_to_safebg_lr_20261002/lowrank_sanity_gate.json",
    "scripts/audit_b2_to_safe_lr_materialization_20261002.py",
    "scripts/run_b2_to_safe_lr_pipeline_20261002.py",
    "scripts/run_isolated_victim.py",
    "scripts/evaluate_b2_to_safebg_checkpoints.py",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Output/input path escapes isolated Safe-LR tree: {resolved}")
    return resolved


def run_stage(name, command, report_dir, status_path, status, env, timeout=None):
    status.update(state="running", stage=name, updated_at_unix=time.time())
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    log_path = report_dir / f"{name}.console.log"
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        peak = 0
        try:
            while child.poll() is None:
                try:
                    out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=used_memory",
                                                   "--format=csv,noheader,nounits"],
                                                  text=True, stderr=subprocess.DEVNULL)
                    peak = max(peak, sum(int(x.strip()) for x in out.splitlines() if x.strip()))
                except Exception:
                    pass
                if timeout is not None and time.monotonic() - started > timeout:
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(20)
            code = child.wait()
        except KeyboardInterrupt:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 130
    (report_dir / f"{name}.exitcode").write_text(str(code) + "\n", encoding="utf-8")
    status.setdefault("stages", {})[name] = {"exitcode": code, "peak_gpu_memory_mib": peak,
                                               "completed_at_unix": time.time()}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if code != 0:
        raise RuntimeError(f"{name} failed with exit code {code}; all artifacts retained")
    return peak


def wait_gpu():
    while True:
        pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                        "--format=csv,noheader"], text=True).strip()
        if not pids:
            return
        time.sleep(30)


def validate_locked_inputs(config, sanity_gate, admission):
    from ue_framework.config import load_config
    cfg = load_config(str(config))
    gate = json.loads(sanity_gate.read_text(encoding="utf-8"))
    proof = json.loads(admission.read_text(encoding="utf-8"))
    sem = cfg["methods"]["legacy_kproto_ret"]["semantic"]
    safe = sem["safe_bg"]
    strict = sem["strict_tsfl"]
    if gate.get("state") != "passed" or gate.get("config_sha256") != sha256(config):
        raise RuntimeError("Five-step correctness receipt missing or bound to another config")
    if (proof.get("verdict") != "scientifically_correct" or proof.get("decision") != "allow_run"
            or proof.get("config_sha256") != sha256(config)):
        raise RuntimeError("Independent Safe-LR scientific review does not admit this config")
    reviewed_hashes = proof.get("file_sha256", {})
    if set(reviewed_hashes) != REVIEWED_FILES:
        raise RuntimeError("Admission receipt does not bind the complete reviewed file set")
    for relative, digest in reviewed_hashes.items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Reviewed source changed after admission: {relative}")
    if (sem["artifact_dir"] != str(ROOT / "artifacts_b2_to_safe_ls_p5_eps8_seed0_20261002")
            or safe.get("mode") != "low_rank" or safe.get("rank") != 8 or safe.get("active_k") != 3
            or abs(float(cfg["experiment"]["eps"]) - 8 / 255) > 1e-9
            or cfg["experiment"]["steps"] != [24]
            or cfg["methods"]["legacy_kproto_ret"]["universal_epochs"] != 24
            or cfg["methods"]["legacy_kproto_ret"]["universal_batch_size"] != 16
            or cfg["methods"]["legacy_kproto_ret"]["kproto"]["feature_layer"] != "model.21"
            or strict.get("strict_tsfl_feature_level") != "P5"
            or strict.get("target_candidate_fraction") != .4 or strict.get("nt_safety_quantile") != .6
            or strict.get("sensitivity_worst_k") != 3 or strict.get("target_channel_fraction") != .15
            or sem["lambda_orth"] != .0005 or sem["orth_worst_k"] != 3
            or cfg["methods"]["legacy_kproto_ret"]["adaptive"]["feature_layer"] != "model.15"):
        raise RuntimeError("Frozen B2-TO-Safe-LR method/config changed")
    if gate.get("selected_channel_indices") != EXPECTED_INDICES or gate.get("safe_bg_rank") != 8:
        raise RuntimeError("Frozen P5 selected indices or rank differs from passed sanity")
    return cfg, gate, proof


def verify_noise(noise, config, cfg, report_dir):
    status = json.loads((noise / "status.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((noise / "diagnostics.json").read_text(encoding="utf-8"))
    method = cfg["methods"]["legacy_kproto_ret"]
    epochs, batch = int(method["universal_epochs"]), int(method["universal_batch_size"])
    target_images = 6095
    expected = math.ceil(target_images / batch) * epochs
    if (status.get("state") != "complete" or status.get("smoke")
            or status.get("steps") != expected or status.get("epochs_completed") != epochs):
        raise RuntimeError(f"Formal Safe-LR noise incomplete: expected {expected} steps/{epochs} epochs")
    if status.get("config_sha256") != sha256(config):
        raise RuntimeError("Noise config hash mismatch")
    for key in ("frozen_state_sha256_before", "adaptive_state_sha256_before"):
        after_key = key.replace("_before", "_after")
        if status.get(key) != status.get(after_key):
            raise RuntimeError(f"Frozen model changed: {key}")
    rows = list(csv.DictReader((noise / "diagnostics.csv").open(newline="", encoding="utf-8")))
    if len(rows) != expected or not diagnostics.get("latest"):
        raise RuntimeError("Noise step diagnostics count mismatch")
    retention = []
    for row in rows:
        for key, value in row.items():
            if key == "selected_proto_ids":
                continue
            if not math.isfinite(float(value)):
                raise RuntimeError(f"Nonfinite noise diagnostic {key}")
        if int(round(float(row["safe_projector_rank"]))) != 8:
            raise RuntimeError("Safe-BG projector left the fixed rank-8 computation")
        if int(round(float(row["safe_projector_basis_columns"]))) != 8:
            raise RuntimeError("Safe-BG basis columns are not rank eight")
        if float(row["decomposition_error"]) > 1e-5 or float(row["safe_projector_orthonormal_error"]) > 1e-5:
            raise RuntimeError("Systematic low-rank projection numerical error")
        retention.append(float(row["safe_direction_retention"]))
    latest = diagnostics["latest"]
    support_sources = diagnostics.get("optimization_support_sources", {})
    if support_sources != {"bbox_half_overlap": target_images * epochs}:
        raise RuntimeError(
            "Support coverage mismatch: expected exactly one bbox_half_overlap source "
            f"for each target image and epoch, got {support_sources}"
        )
    summary = {"run_id": RUN_ID, "noise_run": str(noise), "state": "complete",
        "total_train_images": 16551, "target_images": target_images, "poisoned_images": target_images,
        "target_object_count": None, "epochs": epochs, "outer_steps": expected,
        "batch_size": batch, "wall_clock_seconds": status.get("duration_seconds"),
        "peak_gpu_memory_mib": status.get("peak_gpu_memory_mib"),
        "configured_epsilon": float(cfg["experiment"]["eps"]),
        "effective_target_epsilon": float(cfg["experiment"]["eps"]),
        "effective_overlap_epsilon": float(cfg["experiment"]["eps"]) * .5,
        "safe_direction_retention_mean_of_step_medians": statistics.mean(retention),
        "safe_direction_retention_min_step_median": min(retention),
        "safe_direction_retention_max_step_median": max(retention),
        "removed_component_ratio_mean": statistics.mean(float(row["removed_component_ratio"]) for row in rows),
        "L_BG_safe_mean": statistics.mean(float(row["L_BG_safe_lr"]) for row in rows),
        "L_rel_safe_mean": statistics.mean(float(row["L_rel_safe_lr"]) for row in rows),
        "L_orth_mean": statistics.mean(float(row["L_orth_objective"]) for row in rows),
        "L_adaptive_mean": statistics.mean(float(row["L_adaptive"]) for row in rows),
        "gradient_means": {key: statistics.mean(float(row[key]) for row in rows) for key in
                           ("grad_BG_safe_norm", "grad_rel_safe_norm", "grad_ret_norm",
                            "grad_orth_norm", "grad_adapt_norm", "grad_total_norm",
                            "cos_grad_attack_orth")},
        "latest_step": latest, "optimization_support_sources": support_sources,
        "diagnostics_csv_sha256": sha256(noise / "diagnostics.csv"),
        "noise_status_sha256": sha256(noise / "status.json"),
        "retention_gate": "No per-instance 0.10 hard stop in full run; retain full distribution and stop only on nonfinite/projector/data errors."}
    (report_dir / "poison_generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sanity-gate", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--victim", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--clean-victim", required=True)
    parser.add_argument("--b2to-victim", required=True)
    parser.add_argument("--formal-b2to-noise-csv", required=True)
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--allow-missing-lpips", action="store_true")
    args = parser.parse_args()
    config, gate, proof, noise, victim, report_dir = map(inside,
        (args.config, args.sanity_gate, args.admission, args.noise, args.victim, args.report_dir))
    clean_victim, b2to_victim, b2to_noise_csv = map(Path,
        (args.clean_victim, args.b2to_victim, args.formal_b2to_noise_csv))
    report_dir.mkdir(parents=True, exist_ok=False)
    status_path = report_dir / "pipeline_status.json"
    status = {"run_id": RUN_ID, "state": "running", "stage": "preflight",
              "started_at_unix": time.time(), "noise": str(noise), "victim": str(victim),
              "cleanup_policy": "retain_all", "admission_receipt_sha256": sha256(proof)}
    with status_path.open("x", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
    try:
        cfg, gate_data, proof_data = validate_locked_inputs(config, gate, proof)
        status["code_snapshot"] = proof_data["file_sha256"]
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        method = cfg["methods"]["legacy_kproto_ret"]
        for p in (noise, victim):
            if p.exists():
                raise FileExistsError(f"Output already exists; refusing overwrite: {p}")
        if noise == victim:
            raise ValueError("Noise and victim roots must be distinct")
        clean_protocol = json.loads((clean_victim / "run_spec.json").read_text(encoding="utf-8"))
        formal_protocol = json.loads((b2to_victim / "run_spec.json").read_text(encoding="utf-8"))
        if clean_protocol.get("protocol_lock_sha256") is None:
            raise RuntimeError("Clean victim has no locked protocol identity")
        (report_dir / "reference_protocols.json").write_text(json.dumps({
            "clean_protocol_lock_sha256": clean_protocol["protocol_lock_sha256"],
            "clean_run_spec_sha256": sha256(clean_victim / "run_spec.json"),
            "B2_TO_run_spec_sha256": sha256(b2to_victim / "run_spec.json"),
            "formal_B2_TO_last_checkpoint_sha256": sha256(b2to_victim / "victim/weights/last.pt")}, indent=2), encoding="utf-8")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT),
            YOLO_CONFIG_DIR=str(ROOT / ".ultralytics"), MPLCONFIGDIR=str(ROOT / ".mpl"),
            XDG_CACHE_HOME=str(ROOT / ".cache"), TORCH_HOME=str(ROOT / ".torch"), WANDB_MODE="disabled")
        wait_gpu()
        noise_peak = run_stage("noise", [sys.executable, "scripts/train_noise_once.py", "--config",
            str(config), "--seed", "0", "--output", str(noise)], report_dir, status_path, status,
            env, timeout=6 * 3600)
        status["noise_peak_gpu_memory_mib"] = noise_peak
        noise_summary = verify_noise(noise, config, cfg, report_dir)
        noise_summary["peak_gpu_memory_mib"] = noise_peak
        labels = Path(cfg["data"]["dataset_root"]) / cfg["data"]["train_labels"]
        from ue_framework.data_utils import list_images, read_yolo_annotations
        image_dir = Path(cfg["data"]["dataset_root"]) / cfg["data"]["train_images"]
        noise_summary["target_object_count"] = sum(
            sum(1 for a in read_yolo_annotations(str(labels / f"{Path(p).stem}.txt"))
                if int(a["cls"]) == 14) for p in list_images(str(image_dir)))
        (report_dir / "poison_generation_summary.json").write_text(json.dumps(noise_summary, indent=2), encoding="utf-8")
        run_stage("materialization", [sys.executable, "scripts/materialize_noise_once.py", "--config",
            str(config), "--seed", "0", "--noise", str(noise)], report_dir, status_path, status, env)
        audit_command = [sys.executable, "scripts/audit_b2_to_safe_lr_materialization_20261002.py",
            "--config", str(config), "--noise", str(noise), "--output", str(report_dir),
            "--sample-count", "256"]
        if args.compute_lpips:
            audit_command.append("--compute-lpips")
        run_stage("dataset_integrity_audit", audit_command, report_dir, status_path, status, env, timeout=4 * 3600)
        audit = json.loads((report_dir / "dataset_integrity.json").read_text(encoding="utf-8"))
        if audit.get("state") != "passed":
            raise RuntimeError("Dataset integrity/budget audit did not pass")
        if audit.get("LPIPS_status") != "computed" and not args.allow_missing_lpips:
            status.update(state="blocked", stage="LPIPS_audit_dependency",
                blocker="Official LPIPS implementation/weights are unavailable locally; no metric weight download was authorized.",
                updated_at_unix=time.time())
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            return
        wait_gpu()
        run_stage("victim", [sys.executable, "scripts/run_isolated_victim.py", "--config",
            str(config), "--seed", "0", "--noise", str(noise), "--output", str(victim)],
            report_dir, status_path, status, env)
        victim_status = json.loads((victim / "status.json").read_text(encoding="utf-8"))
        run_spec = json.loads((victim / "run_spec.json").read_text(encoding="utf-8"))
        result_csv = victim / "victim/results.csv"
        rows = list(csv.DictReader(result_csv.open(newline="", encoding="utf-8")))
        weights = victim / "victim/weights"
        if (victim_status.get("state") != "complete" or int(run_spec["train_args"]["epochs"]) != 200
                or len(rows) != 200 or not (weights / "last.pt").is_file()
                or not (weights / "best.pt").is_file()):
            raise RuntimeError("Victim did not finish the locked 200 epochs with both checkpoints")
        eval_output = report_dir / "checkpoint_evaluation"
        run_stage("checkpoint_evaluation", [sys.executable,
            "scripts/evaluate_b2_to_safebg_checkpoints.py", "--config", str(config),
            "--victim", str(victim), "--output", str(eval_output)], report_dir, status_path, status, env)
        # Re-evaluate the existing clean and B2-TO last checkpoints on the same
        # clean val copy created by the isolated Safe-LR victim run.
        run_stage("reference_last_evaluation", [sys.executable,
            "scripts/evaluate_safe_lr_reference_last_20261002.py", "--config", str(config),
            "--clean-victim", str(clean_victim), "--b2to-victim", str(b2to_victim),
            "--data-yaml", str(victim / "train_data.yaml"), "--output",
            str(report_dir / "reference_last_eval")], report_dir, status_path, status, env)
        run_stage("result_analysis", [sys.executable,
            "scripts/analyze_b2_to_safe_lr_results_20261002.py", "--clean-last",
            str(report_dir / "reference_last_eval/clean_last_metrics.json"), "--b2to-last",
            str(report_dir / "reference_last_eval/b2to_last_metrics.json"), "--safe-last",
            str(report_dir / "checkpoint_evaluation/primary_last_metrics.json"), "--safe-best",
            str(report_dir / "checkpoint_evaluation/secondary_best_metrics.json"), "--output", str(report_dir)],
            report_dir, status_path, status, env)
        run_stage("mechanism_comparison", [sys.executable,
            "scripts/compare_b2_to_safebg_mechanisms.py", "--formal-b2-to-csv",
            str(b2to_noise_csv), "--safe-csv", str(noise / "diagnostics.csv"),
            "--output", str(report_dir / "mechanism_comparison.json")],
            report_dir, status_path, status, env)
        status.update(state="complete", completed_at_unix=time.time(),
                      primary_checkpoint="last.pt", secondary_checkpoint="best.pt",
                      final_report=str(report_dir / "ablation_metrics.json"))
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), completed_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
