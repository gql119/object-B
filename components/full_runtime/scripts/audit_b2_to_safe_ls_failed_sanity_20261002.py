"""Read-only diagnostics for a Safe-LS sanity run that failed admission."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from ue_framework.config import load_config
from ue_framework.data_utils import label_path_for_image, list_images, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import _rectangle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cfg = load_config(args.config)
    noise = Path(args.noise)
    data = Path(cfg["data"]["dataset_root"])
    train = data / cfg["data"]["train_images"]
    labels = data / cfg["data"]["train_labels"]
    clean_by_stem = {Path(path).stem: Path(path) for path in list_images(str(train))}
    preview = json.loads((noise / "preview_manifest.json").read_text(encoding="utf-8"))
    maxima = {"global": 0, "target": 0, "overlap": 0, "outside": 0}
    changed_target_pixels = 0
    target_pixels = 0
    for item in preview:
        poison_path = Path(item["path"])
        clean_path = clean_by_stem[poison_path.stem]
        clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        poisoned = cv2.imread(str(poison_path), cv2.IMREAD_COLOR)
        if clean is None or poisoned is None or clean.shape != poisoned.shape:
            raise RuntimeError("Missing or mismatched preview image")
        anns = read_yolo_annotations(label_path_for_image(str(clean_path), str(labels)))
        difference = np.abs(clean.astype(np.int16) - poisoned.astype(np.int16)).max(axis=2)
        target = np.zeros(difference.shape, dtype=bool)
        non_target = np.zeros(difference.shape, dtype=bool)
        for ann in anns:
            mask = _rectangle(difference.shape, ann["bbox"])[0, 0].numpy().astype(bool)
            if int(ann["cls"]) == 14:
                target |= mask
            else:
                non_target |= mask
        for key, mask in (("global", np.ones_like(target)), ("target", target),
                          ("overlap", target & non_target), ("outside", ~target)):
            maxima[key] = max(maxima[key], int(difference[mask].max(initial=0)))
        changed_target_pixels += int((difference[target] > 0).sum())
        target_pixels += int(target.sum())
    if (noise / "labels").exists() or list((noise / "preview").glob("*.txt")):
        raise RuntimeError("Sanity generated labels unexpectedly")
    with (noise / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5 or any(not math.isfinite(float(value)) for row in rows
                            for key, value in row.items() if key != "selected_proto_ids"):
        raise RuntimeError("Sanity diagnostics are incomplete or nonfinite")
    artifact = Path(cfg["methods"]["legacy_kproto_ret"]["semantic"]["artifact_dir"])
    sub = torch.load(artifact / "safe_bg_selected_subspaces.pt", map_location="cpu", weights_only=True)
    ranks = {int(cls): int(item["rank"]) for cls, item in sub["classes"].items()}
    span_ranks = []
    for triple in itertools.combinations(sorted(ranks), 3):
        bases = torch.cat([sub["classes"][cls]["basis"].double() for cls in triple], dim=1)
        span_ranks.append(int(torch.linalg.matrix_rank(bases, rtol=1e-6)))
    fixed = json.loads((noise / "fixed_pair.json").read_text(encoding="utf-8"))
    keys = ("safe_direction_retention", "removed_nt_component_ratio",
            "cos_d_safe_projected_nt", "safe_direction_norm", "collapsed_safe_direction_ratio",
            "grad_BG_safe_norm", "grad_rel_safe_norm", "cos_grad_attack_orth",
            "L_BG", "L_rel", "L_orth_worst", "L_total")
    receipt = {
        "state": "failed_admission", "reason": "safe_direction_retention_below_0.10",
        "five_steps_complete_and_finite": True,
        "fixed_probe": {phase: {key: fixed[phase][key] for key in keys}
                        for phase in ("initial", "after_5")},
        "preview_images": len(preview), "decoded_pixel_maxima_255": maxima,
        "target_pixel_change_fraction": changed_target_pixels / target_pixels,
        "preview_labels_created": False,
        "selected_space_class_ranks": {str(k): v for k, v in ranks.items()},
        "three_class_projector_span_rank": {
            "triples": len(span_ranks), "min": min(span_ranks),
            "median": sorted(span_ranks)[len(span_ranks)//2], "max": max(span_ranks),
            "full_rank_fraction": sum(rank == 24 for rank in span_ranks) / len(span_ranks)},
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
