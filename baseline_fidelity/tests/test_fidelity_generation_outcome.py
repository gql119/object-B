import argparse
import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

from scripts.run_fidelity_baseline_pipeline import (
    classify_generation_result,
    generation_evidence_is_ready,
)


def persistent_checkpoint() -> Path:
    root = Path(__file__).resolve().parent / ".test_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="generation-outcome-", dir=root))
    checkpoint = directory / "last_complete_cycle.pt.gz"
    checkpoint.write_bytes(b"test checkpoint marker")
    return checkpoint


class GenerationOutcomeTest(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _generation_evidence_fixture(self, method: str, cycles: int) -> tuple[dict, argparse.Namespace]:
        root = (Path(__file__).parent / ".test_artifacts" /
                f"generation-evidence-{method}-{cycles}")
        root.mkdir(parents=True, exist_ok=True)
        run_id = f"evidence-{method}"
        batch_count = 48
        milestones = ([1, 5, 10, 20, 30] if method == "em_det" else [1, 3, 5, 7, 9])
        cycle_rows = []
        checkpoint_rows = []
        for cycle in range(1, cycles + 1):
            row = {
                "run_id": run_id, "method": method, "session_id": f"session-{cycle}",
                "cycle": cycle, "cycle_complete": True,
                "proxy_loss_start": 3.0, "proxy_loss_mean": 2.0, "proxy_loss_end": 1.0,
                "optimization_loss_start": 4.0, "optimization_loss_mean": 3.0,
                "optimization_loss_end": 2.0, "gradient_norm_mean": 1.0,
                "projected_gradient_residual_mean": 0.1,
                "epsilon_saturation_ratio": 0.2, "delta_update_l2": 5.0,
                "convergence_claim": "diagnostics_only; no numerical stopping threshold registered",
            }
            cycle_rows.append(row)
            if cycle in milestones:
                checkpoint_rows.append(row)
        self._write_csv(root / "cycle_metrics.csv", cycle_rows)
        self._write_csv(root / "convergence_checkpoints.csv", checkpoint_rows)

        phases = []
        refinement_records = []
        for cycle in range(1, cycles + 1):
            stages = ({"em_surrogate": 10, "em_noise": batch_count}
                      if method == "em_det" else {"sem_train": batch_count})
            for stage, count in stages.items():
                for index in range(count):
                    phases.append({
                        "run_id": run_id, "session_id": f"session-{cycle}",
                        "phase_id": f"{cycle}:{stage}:{index}", "phase_state": "complete",
                        "stage": stage, "cycle": cycle, "batch_index": index,
                        "batch_image_count": 128 if index < 47 else 79,
                        "elapsed_seconds": 1.0,
                    })
        if method == "sem_det":
            for index in range(batch_count):
                count = 128 if index < 47 else 79
                record = {
                    "run_id": run_id, "session_id": "refinement-session",
                    "phase_id": f"refine:{index}", "stage": "sem_final_refinement",
                    "cycle": cycles, "batch_index": index,
                    "batch_image_count": count, "elapsed_seconds": 1.0,
                }
                refinement_records.append(record)
                phases.append({**record, "phase_state": "complete"})
            self._write_csv(root / "sem_final_refinement.csv", [
                {"run_id": run_id, "method": method, "attempt": 1,
                 "event": "start", "complete": False},
                {"run_id": run_id, "method": method, "attempt": 1,
                 "event": "complete", "complete": True},
            ])
            self._write_csv(root / "sem_final_refinement_batches.csv", refinement_records)
            (root / "sem_final_refinement.json").write_text(json.dumps({
                "state": "complete", "complete": True, "processed_person_images": 6095,
                "full_image_coverage": True, "frozen_model_unchanged": True,
                "optimizer_steps_unchanged": True, "optimizer_state_unchanged": True,
                "frozen_model_sha256_after": "final-model", "duration_seconds": 5.0,
                "batch_records": refinement_records,
            }), encoding="utf-8")
        # Resume may leave an older interrupted attempt in the event CSV; the
        # admission check must bind successful phases to each cycle's session.
        if method == "em_det":
            phases.append({"run_id": run_id, "session_id": "old-replayed-session",
                           "phase_id": "old-phase", "phase_state": "complete",
                           "stage": "em_surrogate", "cycle": 2, "batch_index": 0,
                           "batch_image_count": 128, "elapsed_seconds": 1.0})
        self._write_csv(root / "phase_metrics.csv", phases)
        (root / "runtime_summary.json").write_text(json.dumps({
            "run_id": run_id, "method": method, "state": "complete",
            "schedule_complete": True, "active_duration_seconds": 100.0,
            "wall_elapsed_seconds": 105.0, "convergence_status": "not_established",
            "phase_metric_rows": len(phases), "phase_event_parse_errors": 0,
        }), encoding="utf-8")
        (root / "runtime_events.jsonl").write_text('{"event":"test"}\n', encoding="utf-8")
        (root / "convergence_batches.jsonl").write_text('{"event":"test"}\n', encoding="utf-8")
        status = {"cycle_metrics_path": str(root / "cycle_metrics.csv"),
                  "model_state_after": "final-model"}
        args = argparse.Namespace(method=method, run_id=run_id,
                                  max_cycles=cycles, batch_size=128)
        return status, args

    def test_only_explicit_cap_with_checkpoint_is_budget_stop(self):
        checkpoint = persistent_checkpoint()
        started = time.time()
        status = {"state": "time_cap_reached", "resume_available": True,
                  "last_complete_checkpoint": str(checkpoint), "completed_at_unix": started}
        self.assertEqual(classify_generation_result(-2, status, launched_at_unix=started),
                         "time_cap_reached")

    def test_time_cap_without_checkpoint_is_not_a_valid_budget_stop(self):
        status = {"state": "time_cap_reached", "resume_available": False,
                  "last_complete_checkpoint": None}
        with self.assertRaisesRegex(RuntimeError, "no readable complete-cycle checkpoint"):
            classify_generation_result(-2, status, launched_at_unix=time.time())

    def test_stale_previous_cap_cannot_hide_a_failed_resume(self):
        checkpoint = persistent_checkpoint()
        status = {"state": "time_cap_reached", "resume_available": True,
                  "last_complete_checkpoint": str(checkpoint),
                  "completed_at_unix": time.time() - 10}
        with self.assertRaisesRegex(RuntimeError, "predates this generation attempt"):
            classify_generation_result(-2, status, launched_at_unix=time.time())

    def test_crash_cannot_be_misreported_as_time_cap(self):
        status = {"state": "failed", "error": "KeyboardInterrupt"}
        with self.assertRaisesRegex(RuntimeError, "exit code -2"):
            classify_generation_result(-2, status, launched_at_unix=time.time())

    def test_zero_exit_requires_complete_generation(self):
        status = {"state": "failed", "error": "missing outputs"}
        with self.assertRaisesRegex(RuntimeError, "without a complete status"):
            classify_generation_result(0, status, launched_at_unix=time.time())

    def test_generation_evidence_reconciles_em_checkpoints_and_replayed_phases(self):
        status, args = self._generation_evidence_fixture("em_det", 5)
        generation_evidence_is_ready(status, args)

    def test_generation_evidence_requires_sem_full_frozen_refinement(self):
        status, args = self._generation_evidence_fixture("sem_det", 3)
        generation_evidence_is_ready(status, args)
        root = Path(status["cycle_metrics_path"]).parent
        refinement_path = root / "sem_final_refinement.json"
        refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
        refinement["optimizer_state_unchanged"] = False
        refinement_path.write_text(json.dumps(refinement), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "final refinement"):
            generation_evidence_is_ready(status, args)


if __name__ == "__main__":
    unittest.main()
