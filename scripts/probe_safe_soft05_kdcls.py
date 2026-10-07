"""Fixed training-batch KD diagnosis without a perturbation optimizer step."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from ue_framework.config import load_config
from ue_framework.data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.b2_safe_kd import SafeSoft05KDTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    method = cfg["methods"]["legacy_kproto_ret"]
    device = torch.device("cuda")
    frozen = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    adaptive = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    trainer = SafeSoft05KDTrainer(cfg, method, device, frozen, adaptive)
    image_dir = Path(cfg["data"]["dataset_root"]) / cfg["data"]["train_images"]
    label_dir = Path(cfg["data"]["dataset_root"]) / cfg["data"]["train_labels"]
    images = trainer._collect_target_images(str(image_dir), str(label_dir))
    mixed = []
    for path in images:
        annotations = read_yolo_annotations(label_path_for_image(path, str(label_dir)))
        if any(int(ann["cls"]) != 14 for ann in annotations):
            mixed.append(path)
        if len(mixed) == 16:
            break
    if len(mixed) != 16:
        raise RuntimeError("Cannot form a fixed mixed target/non-target training batch")
    prototypes = trainer._build_bank(
        mixed, str(label_dir), str(output / "background_prototypes.pt"),
        trainer._cache_key(str(image_dir), str(label_dir)),
    )
    trainer._epoch = 0
    trainer._active_idx = trainer._get_active_freq_indices(0, 0)
    trainer._active_coords = [tuple(trainer.freq_candidate_coords[int(i)])
                              for i in trainer._active_idx.cpu().tolist()]
    items = []
    for path in mixed:
        image = load_image_rgb_float(path)
        annotations = read_yolo_annotations(label_path_for_image(path, str(label_dir)))
        support, _, _ = trainer._build_support(image.shape, annotations, image_path=path)
        items.append((image, annotations, support))
    before = trainer.fourier_coeff.detach().clone()
    before_small = trainer.suppress_small.detach().clone()
    result = trainer._batch_loss(items, prototypes)
    if result is None:
        raise RuntimeError("Fixed probe did not produce a loss")
    loss, metrics = result
    scalars = {key: float(value.detach()) for key, value in metrics.items()
               if torch.is_tensor(value) and value.numel() == 1}
    checks = {
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_diagnostics": all(np.isfinite(value) for value in scalars.values()),
        "teacher_parameters_frozen": all(not p.requires_grad for p in frozen.parameters()),
        "adaptive_base_parameters_frozen": all(not p.requires_grad for p in adaptive.parameters()),
        "noise_unchanged": bool(torch.equal(before, trainer.fourier_coeff)
                                and torch.equal(before_small, trainer.suppress_small)),
        "non_target_positive_units": scalars["kd_positive_count"] > 0,
        "target_gt_excluded": scalars["kd_target_gt_count"] == 0,
        "target_logit_excluded": scalars["kd_person_logit_in_distribution"] == 0,
        "shared_teacher_indices": scalars["kd_teacher_student_index_identical"] == 1,
        "kd_scale_calibrated": scalars["kd_scale_calibrated"] == 1,
    }
    evidence = {"seed": args.seed, "images": [Path(path).name for path in mixed],
                "optimizer_step_count": 0, "checks": checks, "metrics": scalars}
    (output / "fixed_batch_probe.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError("Fixed-batch KD correctness check failed: " + str(checks))
    print(json.dumps({"checks": checks, "L_ret": scalars["L_ret_raw"],
                      "L_KD": scalars["L_KD_raw"],
                      "positive_count": scalars["kd_positive_count"]}))


if __name__ == "__main__":
    main()
