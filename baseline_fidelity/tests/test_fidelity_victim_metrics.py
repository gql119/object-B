import unittest
import hashlib
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from scripts.run_fidelity_victim import (
    selective_metrics,
    summarize_metrics,
    validate_clean_reference,
)


def persistent_test_dir(prefix: str) -> Path:
    """Keep tiny test artifacts; avoid recursive temp-directory cleanup."""
    root = Path(__file__).resolve().parent / ".test_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


class FidelityVictimMetricsTest(unittest.TestCase):
    def test_exports_all_voc_iou_and_pr_metrics(self):
        ap = np.tile(np.linspace(0.8, 0.2, 10), (20, 1))
        precision = np.linspace(0.5, 0.9, 20)
        recall = np.linspace(0.4, 0.8, 20)
        # Match Ultralytics Metric: ``all_ap`` is [nc, 10], while ``ap`` is
        # the per-class mean over IoU thresholds and therefore only [nc].
        box = SimpleNamespace(ap_class_index=np.arange(20), all_ap=ap,
                              ap=ap.mean(axis=1), p=precision, r=recall,
                              map50=float(ap[:, 0].mean()))
        result = summarize_metrics(box)
        self.assertEqual(len(result["AP50_per_class"]), 20)
        self.assertEqual(len(result["AP75_per_class"]), 20)
        self.assertEqual(len(result["Precision_per_class_best_f1"]), 20)
        self.assertEqual(len(result["Recall_per_class"]), 20)
        self.assertAlmostEqual(result["AP75_per_class"][0], ap[0, 5])
        self.assertAlmostEqual(result["Precision_per_class_best_f1"][14], precision[14])

    def test_selective_metrics_use_target_and_nontarget_class_sets(self):
        clean_ap = np.full(20, 0.8)
        poison_ap = np.full(20, 0.72)
        poison_ap[14] = 0.04
        clean = {"AP50_per_class": clean_ap.tolist(), "mAP50_non_target": 0.8}
        poison = {"AP50_per_class": poison_ap.tolist(), "mAP50_non_target": 0.72}
        metrics = selective_metrics(clean, poison)
        self.assertAlmostEqual(metrics["P_target_AP50_damage"], 0.95)
        self.assertAlmostEqual(metrics["U_non_target_AP50_retention"], 0.9)
        self.assertEqual(metrics["WCD_class"], "aeroplane")
        self.assertEqual(len(metrics["TCD3_classes"]), 3)

    def test_summary_rejects_missing_or_nonfinite_class_metrics(self):
        all_ap = np.ones((19, 10))
        box = SimpleNamespace(ap_class_index=np.arange(19), all_ap=all_ap,
                              ap=all_ap.mean(axis=1), p=np.ones(19),
                              r=np.ones(19), map50=1.0)
        with self.assertRaisesRegex(RuntimeError, "VOC20"):
            summarize_metrics(box)
        box.ap_class_index = np.arange(20)
        box.all_ap = np.ones((20, 10))
        box.ap = box.all_ap.mean(axis=1)
        box.all_ap[0, 0] = np.nan
        box.p = np.ones(20)
        box.r = np.ones(20)
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            summarize_metrics(box)

    def test_clean_reference_requires_checkpoint_and_metrics_hashes(self):
        ap = np.linspace(0.3, 0.9, 20).tolist()
        metrics = {"mAP50_all": float(np.mean(ap)), "mAP50_target": ap[14],
                   "mAP50_non_target": float(np.mean(np.delete(ap, 14))),
                   "AP50_per_class": ap}
        root = persistent_test_dir("victim-clean-ref-")
        checkpoint = root / "last.pt"
        checkpoint.write_bytes(b"locked-clean-checkpoint")
        metrics_path = root / "clean.json"
        metrics_path.write_text(json.dumps({"checkpoint": str(checkpoint),
                                            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                            **metrics}), encoding="utf-8")
        metrics_hash = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        clean = {"status": "matched_complete", "evaluation_checkpoint_role": "last.pt primary",
                 "checkpoint": str(checkpoint),
                 "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 "metrics_json": str(metrics_path), "metrics_sha256": metrics_hash,
                 "AP50_all": metrics["mAP50_all"], "AP50_target": metrics["mAP50_target"],
                 "AP50_non_target": metrics["mAP50_non_target"], "AP50_per_class": ap}
        result = validate_clean_reference({"clean_baseline": clean})
        self.assertEqual(result["AP50_per_class"], ap)
        clean["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "last.pt"):
            validate_clean_reference({"clean_baseline": clean})


if __name__ == "__main__":
    unittest.main()
