"""No-update fixed-train-input equivalence probe for parent Safe-Soft05 terms."""

import argparse
import json
import os
import random
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from ultralytics import YOLO

from ue_framework.config import load_config
from ue_framework.data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.b2_semantic import B2SemanticTrainer
from ue_framework.methods.b2_safe_kd import SafeSoft05KDTrainer


def make_trainer(cls, cfg):
    torch.manual_seed(0)
    device = torch.device("cuda")
    frozen = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    adaptive = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    trainer = cls(cfg, cfg["methods"]["legacy_kproto_ret"], device, frozen, adaptive)
    trainer._epoch = 0
    trainer._active_idx = trainer._get_active_freq_indices(0, 0)
    trainer._active_coords = [tuple(trainer.freq_candidate_coords[int(i)])
                              for i in trainer._active_idx.cpu().tolist()]
    trainer._adaptive_calls = 0
    trainer._semantic_calls = 1
    return trainer


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--fixed-set", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parent_cfg, candidate_cfg = (load_config(args.parent_config),
                                 load_config(args.candidate_config))
    fixed = json.loads(Path(args.fixed_set).read_text(encoding="utf-8"))
    if fixed["query_indices_per_batch"] != [0, 4, 8, 12]:
        raise ValueError("Locked fixed query selection changed")
    reference = make_trainer(B2SemanticTrainer, parent_cfg)
    candidate = make_trainer(SafeSoft05KDTrainer, candidate_cfg)
    data = Path(parent_cfg["data"]["dataset_root"])
    images = data / parent_cfg["data"]["train_images"]
    labels = data / parent_cfg["data"]["train_labels"]
    paths = []
    for path in reference._collect_target_images(str(images), str(labels)):
        anns = read_yolo_annotations(label_path_for_image(path, str(labels)))
        if any(int(a["cls"]) != 14 for a in anns):
            paths.append(path)
        if len(paths) == 16:
            break
    if [Path(p).stem for p in paths] != fixed["selected_stems"][:16]:
        raise ValueError("Fixed train sample changed")
    items = []
    for path in paths:
        image = load_image_rgb_float(path)
        anns = read_yolo_annotations(label_path_for_image(path, str(labels)))
        support, _, _ = reference._build_support(image.shape, anns, image_path=path)
        items.append((image, anns, support))
    cache_path = Path(
        "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
        "b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/"
        "ue_project/ablation_safe_bg_20261004/candidate/"
        "noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt"
    )
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    prototypes = cache["prototypes"].to("cuda")
    outputs = []
    for trainer in (reference, candidate):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        _, metrics = trainer._batch_loss(items, prototypes)
        outputs.append({key: float(metrics[key].detach()) for key in
                        ("L_BG", "L_rel", "L_ret", "weighted_L_ret",
                         "L_orth_objective", "L_adaptive")})
    differences = {key: abs(outputs[0][key] - outputs[1][key]) for key in outputs[0]}
    tolerance = 1e-6
    result = {"reference_id": "formal Safe-Soft05 parent implementation",
              "reference_weights_path": parent_cfg["surrogate"]["ckpt"],
              "candidate_weights_path": candidate_cfg["surrogate"]["ckpt"],
              "evaluation_setting": {"dataset": "same locked first 16 mixed training images",
                                     "seed": 0, "sample_count": 16,
                                     "query_index": 0, "optimizer_steps": 0},
              "reference": outputs[0], "candidate": outputs[1],
              "absolute_differences": differences, "tolerance": tolerance,
              "pass": all(value <= tolerance for value in differences.values())}
    output = Path(args.output)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))
    if not result["pass"]:
        raise RuntimeError("Safe-Soft05 parent terms changed in KD candidate")


if __name__ == "__main__":
    main()
