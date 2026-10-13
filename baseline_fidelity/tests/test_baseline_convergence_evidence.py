import csv
import importlib.util
import unittest
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[1] / "ue_framework" / "baselines_det" / "convergence.py"
_MODULE_SPEC = importlib.util.spec_from_file_location("baseline_convergence", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE_SPEC.loader.exec_module(_MODULE)
CYCLE_FIELDS = _MODULE.CYCLE_FIELDS
EM_CHECKPOINT_CYCLES = _MODULE.EM_CHECKPOINT_CYCLES
SEM_CHECKPOINT_PASSES = _MODULE.SEM_CHECKPOINT_PASSES
summarize_cycle = _MODULE.summarize_cycle
write_csv_snapshot = _MODULE.write_csv_snapshot
sem_refinement_audit = _MODULE.sem_refinement_audit


class ConvergenceEvidenceTests(unittest.TestCase):
    def test_em_checkpoint_contains_losses_runtime_gradient_and_delta_drift(self):
        records = [
            {"stage": "em_surrogate", "batch_image_count": 2,
             "surrogate_detection_loss": 10.0},
            {"stage": "em_surrogate", "batch_image_count": 2,
             "surrogate_detection_loss": 8.0},
            {"stage": "em_noise", "batch_image_count": 2,
             "loss_first": 12.0, "loss_mean": 9.0, "loss_last": 6.0,
             "gradient_norm_mean": 4.0, "gradient_norm_max": 6.0,
             "gradient_norm_per_sqrt_image_mean": 2.0,
             "gradient_norm_per_sqrt_image_max": 3.0,
             "projected_gradient_residual_mean": 0.5,
             "projected_gradient_residual_per_sqrt_image_mean": 0.25,
             "projected_gradient_residual_per_sqrt_image_max": 0.5,
             "delta_change_l2": 3.0,
             "delta_drift_from_previous_cycle_l2": None,
             "epsilon_saturated_channels": 2, "support_channels": 10},
            {"stage": "em_noise", "batch_image_count": 2,
             "loss_first": 11.0, "loss_mean": 8.0, "loss_last": 5.0,
             "gradient_norm_mean": 6.0, "gradient_norm_max": 8.0,
             "gradient_norm_per_sqrt_image_mean": 3.0,
             "gradient_norm_per_sqrt_image_max": 4.0,
             "projected_gradient_residual_mean": 1.5,
             "projected_gradient_residual_per_sqrt_image_mean": 0.75,
             "projected_gradient_residual_per_sqrt_image_max": 1.0,
             "delta_change_l2": 4.0,
             "delta_drift_from_previous_cycle_l2": 4.0,
             "epsilon_saturated_channels": 3, "support_channels": 10},
        ]
        row = summarize_cycle(
            run_id="em-test", method="em_det", cycle=5, records=records,
            started_at_unix=100.0, completed_at_unix=160.0,
            cycle_elapsed_seconds=60.0, active_elapsed_seconds=160.0,
            wall_elapsed_seconds=160.0, target_images_completed=6095,
            surrogate_images_exposed=1280, surrogate_unique_images=1180,
            coverage={"surrogate_full_passes": 0.4, "surrogate_seen_images": 6400,
                      "surrogate_unique_images": 2600},
            counters_before={"optimizer_steps": 0, "noise_steps": 0, "stability_samples": 0},
            counters_after={"optimizer_steps": 10, "noise_steps": 121900,
                            "stability_samples": 0},
            training_detection_error=0.2, model_sha256="model", optimizer_sha256="optimizer",
        )

        self.assertEqual(EM_CHECKPOINT_CYCLES, (1, 5, 10, 20, 30))
        self.assertTrue(row["checkpoint_cycle"])
        self.assertEqual(row["proxy_loss_start"], 10.0)
        self.assertEqual(row["proxy_loss_end"], 8.0)
        self.assertEqual(row["optimization_loss_start"], 12.0)
        self.assertEqual(row["optimization_loss_end"], 5.0)
        self.assertAlmostEqual(row["delta_update_l2"], 5.0)
        self.assertAlmostEqual(row["delta_update_l2_per_image_rms"], 5.0 / (6095 ** 0.5))
        self.assertAlmostEqual(row["delta_drift_from_previous_cycle_l2"], 4.0)
        self.assertAlmostEqual(row["epsilon_saturation_ratio"], 0.25)
        self.assertEqual(row["cycle_elapsed_seconds"], 60.0)
        self.assertIn("no numerical stopping threshold", row["convergence_claim"])
        self.assertEqual(set(row), set(CYCLE_FIELDS))

    def test_sem_checkpoint_and_snapshot_keep_stability_diagnostics(self):
        record = {
            "stage": "sem_train", "batch_image_count": 3,
            "robust_surrogate_loss": 4.0, "loss_first": 5.0,
            "loss_mean": 4.0, "loss_last": 3.0,
            "gradient_norm_mean": 2.0, "gradient_norm_max": 3.0,
            "gradient_norm_per_sqrt_image_mean": 1.15,
            "gradient_norm_per_sqrt_image_max": 1.73,
            "projected_gradient_residual_mean": 0.25,
            "projected_gradient_residual_per_sqrt_image_mean": 0.14,
            "projected_gradient_residual_per_sqrt_image_max": 0.2,
            "delta_change_l2": 0.5,
            "delta_drift_from_previous_cycle_l2": 0.5,
            "epsilon_saturated_channels": 6, "support_channels": 30,
            "stability_samples": 15,
        }
        row = summarize_cycle(
            run_id="sem-test", method="sem_det", cycle=7, records=[record],
            started_at_unix=10.0, completed_at_unix=20.0,
            cycle_elapsed_seconds=10.0, active_elapsed_seconds=20.0,
            wall_elapsed_seconds=20.0, target_images_completed=6095,
            surrogate_images_exposed=6095, surrogate_unique_images=6095,
            coverage={"surrogate_full_passes": 7, "surrogate_seen_images": 42665,
                      "surrogate_unique_images": 6095},
            counters_before={"optimizer_steps": 0, "noise_steps": 0, "stability_samples": 0},
            counters_after={"optimizer_steps": 1, "noise_steps": 0, "stability_samples": 15},
            training_detection_error=None, model_sha256="model", optimizer_sha256="optimizer",
        )
        self.assertEqual(SEM_CHECKPOINT_PASSES, (1, 3, 5, 7, 9))
        self.assertTrue(row["checkpoint_cycle"])
        self.assertEqual(row["proxy_loss_kind"], "robust_surrogate_loss")
        self.assertEqual(row["optimization_loss_kind"], "stability_sampling_loss")
        self.assertEqual(row["surrogate_full_passes_total"], 7)
        self.assertEqual(row["stability_sample_image_exposures_this_cycle"], 15)
        self.assertEqual(row["epsilon_saturation_ratio"], 0.2)

        artifacts = Path(__file__).parent / ".test_artifacts" / "convergence_evidence"
        artifacts.mkdir(parents=True, exist_ok=True)
        path = artifacts / "checkpoints.csv"
        write_csv_snapshot(path, [row])
        with path.open(encoding="utf-8", newline="") as stream:
            loaded = next(csv.DictReader(stream))
        self.assertEqual(loaded["cycle"], "7")
        self.assertEqual(loaded["checkpoint_cycle"], "True")
        self.assertEqual(loaded["convergence_claim"], row["convergence_claim"])

    def test_sem_final_refinement_requires_full_image_set_and_frozen_state(self):
        complete = sem_refinement_audit(
            expected_person_images=6095, processed_person_images=6095,
            image_id_set_matches=True, frozen_model_sha256_before="model-a",
            frozen_model_sha256_after="model-a", optimizer_steps_before=900,
            optimizer_steps_after=900, optimizer_sha256_before="optim-a",
            optimizer_sha256_after="optim-a", duration_seconds=3600.0,
        )
        self.assertTrue(complete["complete"])
        self.assertTrue(complete["full_image_coverage"])
        self.assertTrue(complete["frozen_model_unchanged"])
        self.assertEqual(complete["convergence_status"],
                         "refinement_complete_convergence_not_asserted")

        incomplete = sem_refinement_audit(
            expected_person_images=6095, processed_person_images=6000,
            image_id_set_matches=False, frozen_model_sha256_before="model-a",
            frozen_model_sha256_after="model-b", optimizer_steps_before=900,
            optimizer_steps_after=901, optimizer_sha256_before="optim-a",
            optimizer_sha256_after="optim-b", duration_seconds=3600.0,
        )
        self.assertFalse(incomplete["complete"])
        self.assertFalse(incomplete["full_image_coverage"])
        self.assertFalse(incomplete["frozen_model_unchanged"])
        self.assertEqual(incomplete["convergence_status"], "refinement_incomplete_not_established")


if __name__ == "__main__":
    unittest.main()
