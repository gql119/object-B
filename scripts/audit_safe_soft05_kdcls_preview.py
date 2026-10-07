"""Audit decoded smoke previews without reading or exporting weight artifacts."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

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
    config = load_config(args.config)
    dataset = Path(config["data"]["dataset_root"])
    train = dataset / config["data"]["train_images"]
    labels = dataset / config["data"]["train_labels"]
    by_stem = {Path(path).stem: Path(path) for path in list_images(str(train))}
    noise = Path(args.noise)
    manifest = json.loads((noise / "preview_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        poison_path = Path(item["path"])
        clean_path = by_stem[poison_path.stem]
        clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        poison = cv2.imread(str(poison_path), cv2.IMREAD_COLOR)
        if clean is None or poison is None or clean.shape != poison.shape:
            raise RuntimeError(f"Missing or mismatched image: {poison_path}")
        difference = np.abs(clean.astype(np.int16) - poison.astype(np.int16)).max(axis=2)
        target = np.zeros(difference.shape, dtype=bool)
        non_target = np.zeros(difference.shape, dtype=bool)
        for ann in read_yolo_annotations(label_path_for_image(str(clean_path), str(labels))):
            mask = _rectangle(difference.shape, ann["bbox"])[0, 0].numpy().astype(bool)
            if int(ann["cls"]) == 14:
                target |= mask
            else:
                non_target |= mask
        overlap = target & non_target
        nonoverlap = target & ~non_target
        row = {
            "stem": poison_path.stem,
            "target_pixels": int(target.sum()),
            "changed_target_pixels": int((difference[target] > 0).sum()),
            "nonoverlap_max_uint8": int(difference[nonoverlap].max(initial=0)),
            "overlap_max_uint8": int(difference[overlap].max(initial=0)),
            "outside_target_max_uint8": int(difference[~target].max(initial=0)),
            "global_max_uint8": int(difference.max(initial=0)),
        }
        row["pass"] = (
            row["target_pixels"] > 0
            and row["changed_target_pixels"] == row["target_pixels"]
            and row["nonoverlap_max_uint8"] <= 8
            and row["overlap_max_uint8"] <= 4
            and row["outside_target_max_uint8"] == 0
        )
        rows.append(row)
    result = {"sample_count": len(rows), "all_pass": len(rows) == 8 and all(r["pass"] for r in rows), "rows": rows}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"sample_count": result["sample_count"], "all_pass": result["all_pass"]}))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
