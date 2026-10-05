"""Verify the downloaded small-file evidence for the two formal ablations."""
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / "formal"
NAMES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
         "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
         "pottedplant", "sheep", "sofa", "train", "tvmonitor")
STAGES = ("locked_source_fingerprint", "r4_noise", "soft05_noise",
          "r4_materialization", "soft05_materialization", "r4_integrity",
          "soft05_integrity", "locked_source_fingerprint_before_victims",
          "r4_victim", "soft05_victim", "r4_checkpoint_evaluation",
          "soft05_checkpoint_evaluation", "reference_last_evaluation", "four_method_analysis")


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def same(actual, expected, name, tol=1e-10):
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{name}: {actual} != {expected}")


def main():
    pipeline = read("formal/pipeline_status.json")
    assert pipeline["state"] == pipeline["stage"] == "complete"
    assert set(pipeline["stages"]) == set(STAGES)
    assert all(pipeline["stages"][stage]["exitcode"] == 0 for stage in STAGES)
    before, after = (read("formal/" + suffix) for suffix in
                     ("source_fingerprint.json", "source_fingerprint_before_victims.json"))
    assert before == after and before["state"] == "passed"
    assert before["train_count"] == 16551 and before["val_count"] == 4952
    assert before["dataset_fingerprint"] == "29b131dfca2e3636e9c47502961caa6297acf0450f646ff6409b7b3bfb21b3fa"
    checks = {}
    for variant, rank, alpha in (("r4", 4, 1.0), ("soft05", 8, 0.5)):
        noise_summary = read(f"formal/{variant}_noise_summary.json")
        noise_status = read(f"{variant}_noise/status.json")
        integrity = read(f"formal/{variant}_integrity/dataset_integrity.json")
        audit = integrity["all_image_pixel_audit"]
        victim_status = read(f"{variant}_victim/status.json")
        spec = read(f"{variant}_victim/run_spec.json")
        primary = read(f"formal/{variant}_checkpoint_evaluation/primary_last_metrics.json")
        secondary = read(f"formal/{variant}_checkpoint_evaluation/secondary_best_metrics.json")
        with (ROOT / f"{variant}_victim/victim/results.csv").open(newline="", encoding="utf-8") as stream:
            epochs = list(csv.DictReader(stream))
        assert noise_summary["state"] == noise_status["state"] == "complete"
        assert noise_summary["epochs"] == noise_status["epochs_completed"] == 24
        assert noise_summary["outer_steps"] == noise_status["steps"] == 9144
        assert noise_summary["rank"] == rank and noise_summary["alpha"] == alpha
        assert noise_summary["wall_clock_seconds"] <= 8 * 3600
        assert noise_status["smoke"] is False
        assert noise_status["frozen_state_sha256_before"] == noise_status["frozen_state_sha256_after"]
        assert noise_status["adaptive_state_sha256_before"] == noise_status["adaptive_state_sha256_after"]
        assert integrity["state"] == audit["state"] == "passed"
        assert integrity["source_image_count"] == integrity["poisoned_image_count"] == 16551
        assert integrity["source_label_count"] == integrity["poisoned_label_count"] == 16551
        assert integrity["labels_byte_identical"] and integrity["same_stems"]
        assert audit["person_boxes_with_changed_pixels"] == 13256
        assert audit["pixel_maxima"] == {"global": 8, "target_only": 8, "overlap": 4, "outside": 0}
        assert victim_status["state"] == "complete" and spec["train_count"] == 16551 and spec["val_count"] == 4952
        args = spec["train_args"]
        assert (args["epochs"], args["seed"], args["batch"], args["imgsz"], args["optimizer"], args["pretrained"]) == (200, 0, 36, 640, "SGD", False)
        assert len(epochs) == 200 and [int(row["epoch"]) for row in epochs] == list(range(1, 201))
        assert primary["checkpoint_role"] == "primary_last" and secondary["checkpoint_role"] == "secondary_best"
        for result in (primary, secondary):
            assert result["class_names"] == list(NAMES)
            assert len(result["AP50_per_class"]) == len(result["Recall_per_class"]) == 20
            assert all(math.isfinite(value) for value in result["AP50_per_class"] + result["Recall_per_class"])
            same(result["mAP50_all"], sum(result["AP50_per_class"]) / 20, f"{variant} all")
            same(result["mAP50_non_target"], sum(value for i, value in enumerate(result["AP50_per_class"]) if i != 14) / 19, f"{variant} NT")
            same(result["mAP50_target"], result["AP50_per_class"][14], f"{variant} target")
        checks[variant] = {"noise_hours": noise_summary["wall_clock_seconds"] / 3600,
                           "noise_steps": 9144, "victim_epochs": 200,
                           "person_AP50_last": primary["mAP50_target"],
                           "non_target_mAP50_last": primary["mAP50_non_target"]}
    report = read("formal/comparison/ablation_metrics.json")
    assert report["primary_checkpoint"] == "last.pt" and report["secondary_checkpoint"] == "best.pt"
    clean = read("formal/reference_last_eval/clean_last_metrics.json")
    assert clean["class_names"] == list(NAMES)
    for row in report["rows"]:
        key = {"B2-TO": "formal/reference_last_eval/b2to_last_metrics.json",
               "Safe-LR-R8": "formal/reference_last_eval/safe_lr_last_metrics.json",
               "Safe-R4": "formal/r4_checkpoint_evaluation/primary_last_metrics.json",
               "Safe-Soft05": "formal/soft05_checkpoint_evaluation/primary_last_metrics.json"}[row["Method"]]
        result = read(key)
        same(row["Target AP50"], result["mAP50_target"], "target AP50")
        same(row["NT mAP50"], result["mAP50_non_target"], "NT AP50")
        same(row["all mAP50"], result["mAP50_all"], "all AP50")
        same(row["person Recall"], result["Recall_target"], "person Recall")
        p = (clean["mAP50_target"] - result["mAP50_target"]) / clean["mAP50_target"]
        u = result["mAP50_non_target"] / clean["mAP50_non_target"]
        same(row["P"], p, "P")
        same(row["U"], u, "U")
        same(row["S"], 2 * p * u / (p + u), "S")
        damage = sorted(((clean["AP50_per_class"][i] - result["AP50_per_class"][i]) / clean["AP50_per_class"][i], NAMES[i])
                        for i in range(20) if i != 14)
        damage.reverse()
        same(row["WCD"], damage[0][0], "WCD")
        same(row["TCD3"], sum(value for value, _ in damage[:3]) / 3, "TCD3")
    receipt = {"state": "passed", "pipeline_completed_at_unix": pipeline["completed_at_unix"],
               "dataset_fingerprint": before["dataset_fingerprint"], "checks": checks,
               "clean_last_AP50": clean["mAP50_target"],
               "comparison_sha256": hashlib.sha256((FORMAL / "comparison/ablation_metrics.json").read_bytes()).hexdigest(),
               "checkpoint_sha256": read("checkpoint_sha256.json")}
    (ROOT / "verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
