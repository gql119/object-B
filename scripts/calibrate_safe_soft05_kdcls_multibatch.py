"""Pre-registered train-only KD scale calibration; never updates perturbations."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from ue_framework.config import load_config
from ue_framework.data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.b2_semantic import B2SemanticTrainer
from ue_framework.methods.kd_utility import KDAdaptiveLearner


PARENT_CACHE = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/"
    "ue_project/ablation_safe_bg_20261004/candidate/"
    "noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt"
)
PARENT_CACHE_SHA256 = "12889ba0f80b29c194a59bd9eb2a6d8b137e9aa4977fdb94ad84444628b316a7"
IMAGE_COUNT = 64
BATCH_SIZE = 16
QUERY_INDICES = (0, 4, 8, 12)
SEED = 0


class RawKDCalibrationTrainer(B2SemanticTrainer):
    """Parent objective, with KD read only as a diagnostic on the adaptive path."""

    def __init__(self, cfg, method, device, frozen, adaptive):
        super().__init__(cfg, method, device, frozen, adaptive)
        self.adaptive = KDAdaptiveLearner(
            adaptive, self.adaptive_feature_layer, self.target_class_id,
            self.imgsz, float(method["adaptive"]["inner_lr"]),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cfg = load_config(args.config)
    method = cfg["methods"]["legacy_kproto_ret"]
    if (cfg["experiment"]["target_class_id"] != 14
            or method["adaptive"]["batch_size"] != 1
            or method["kproto"]["max_images"] != 0):
        raise ValueError("Parent Safe-Soft05 sampling settings changed")
    if hashlib.sha256(PARENT_CACHE.read_bytes()).hexdigest() != PARENT_CACHE_SHA256:
        raise ValueError("Completed parent prototype cache changed")

    device = torch.device("cuda")
    frozen = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    adaptive = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    trainer = RawKDCalibrationTrainer(cfg, method, device, frozen, adaptive)
    data = Path(cfg["data"]["dataset_root"])
    image_dir = data / cfg["data"]["train_images"]
    label_dir = data / cfg["data"]["train_labels"]
    cache = torch.load(PARENT_CACHE, map_location="cpu", weights_only=True)
    if cache["cache_key"] != trainer._cache_key(str(image_dir), str(label_dir)):
        raise ValueError("Parent prototype cache key differs from current data/model protocol")
    prototypes = cache["prototypes"].to(device)

    mixed = []
    for path in trainer._collect_target_images(str(image_dir), str(label_dir)):
        annotations = read_yolo_annotations(label_path_for_image(path, str(label_dir)))
        if any(int(ann["cls"]) != 14 for ann in annotations):
            mixed.append(path)
        if len(mixed) == IMAGE_COUNT:
            break
    if len(mixed) != IMAGE_COUNT:
        raise RuntimeError("Insufficient mixed target/non-target training images")
    trainer._epoch = 0
    trainer._active_idx = trainer._get_active_freq_indices(0, 0)
    trainer._active_coords = [tuple(trainer.freq_candidate_coords[int(i)])
                              for i in trainer._active_idx.cpu().tolist()]
    initial_coeff = trainer.fourier_coeff.detach().clone()
    initial_small = trainer.suppress_small.detach().clone()
    rows = []
    for batch_number in range(IMAGE_COUNT // BATCH_SIZE):
        paths = mixed[batch_number * BATCH_SIZE:(batch_number + 1) * BATCH_SIZE]
        items = []
        for path in paths:
            image = load_image_rgb_float(path)
            annotations = read_yolo_annotations(label_path_for_image(path, str(label_dir)))
            support, _, _ = trainer._build_support(image.shape, annotations, image_path=path)
            if support.sum() <= 10:
                raise ValueError("Pre-registered image has insufficient target support")
            items.append((image, annotations, support))
        for query_index in QUERY_INDICES:
            trainer._adaptive_calls = query_index
            result = trainer._batch_loss(items, prototypes)
            if result is None:
                raise RuntimeError("Calibration batch did not produce a loss")
            loss, metrics = result
            row = {
                "batch": batch_number,
                "query_index": query_index,
                "query_stem": Path(paths[query_index]).stem,
                "L_ret_raw": float(metrics["L_ret"].detach()),
                "L_KD_raw": float(metrics["L_KD_cls_nt_raw"].detach()),
                "kd_positive_count": float(metrics["kd_positive_count"].detach()),
                "kd_gt_object_count": float(metrics["kd_gt_object_count"].detach()),
                "target_gt_count": float(metrics["kd_target_gt_count"].detach()),
                "target_logit_count": float(metrics["kd_person_logit_in_distribution"].detach()),
                "shared_index_flag": float(metrics["kd_teacher_student_index_identical"].detach()),
                "finite_loss": bool(torch.isfinite(loss)),
            }
            rows.append(row)
            del result, loss, metrics
    ret_mean = float(np.mean([row["L_ret_raw"] for row in rows]))
    kd_mean = float(np.mean([row["L_KD_raw"] for row in rows]))
    checks = {
        "all_16_predeclared_queries": len(rows) == 16,
        "all_finite": bool(all(row["finite_loss"] and np.isfinite(row["L_ret_raw"])
                               and np.isfinite(row["L_KD_raw"]) for row in rows)),
        "positive_total": sum(row["kd_positive_count"] for row in rows) > 0,
        "person_gt_excluded": all(row["target_gt_count"] == 0 for row in rows),
        "person_logit_excluded": all(row["target_logit_count"] == 0 for row in rows),
        "teacher_student_shared_indices": all(row["shared_index_flag"] == 1
                                              or row["kd_positive_count"] == 0 for row in rows),
        "teacher_parameters_frozen": all(not p.requires_grad for p in frozen.parameters()),
        "adaptive_base_parameters_frozen": all(not p.requires_grad for p in adaptive.parameters()),
        "noise_unchanged": bool(torch.equal(initial_coeff, trainer.fourier_coeff)
                                and torch.equal(initial_small, trainer.suppress_small)),
        "positive_reference_scales": ret_mean > 1e-12 and kd_mean > 1e-12,
    }
    artifact = {
        "schema": "safe-soft05-kdcls-multibatch-calibration-v1",
        "seed": SEED,
        "optimizer_step_count": 0,
        "selection": "first 64 sorted train target images with at least one non-target GT",
        "batch_size": BATCH_SIZE,
        "query_indices_per_batch": list(QUERY_INDICES),
        "eot_per_query": 1,
        "parent_prototype_cache_sha256": PARENT_CACHE_SHA256,
        "selected_stems": [Path(path).stem for path in mixed],
        "rows": rows,
        "metrics": {
            "L_ret_raw": ret_mean,
            "L_KD_raw": kd_mean,
            "kd_fixed_scale": ret_mean / kd_mean if kd_mean > 0 else None,
            "kd_positive_count": sum(row["kd_positive_count"] for row in rows),
        },
        "checks": checks,
    }
    (output / "calibration.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError("Multi-batch KD calibration failed: " + str(checks))
    print(json.dumps({"checks": checks, "metrics": artifact["metrics"]}))


if __name__ == "__main__":
    main()
