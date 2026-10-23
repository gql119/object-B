"""Full 8/4/0 decoded-pixel audit plus deterministic 256-person perceptual sample."""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from ue_framework.config import load_config
from ue_framework.data_utils import list_images, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import _rectangle
from ue_framework.paths import build_run_paths


ROOT = Path(__file__).resolve().parents[1]
VOC_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
             "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
             "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ssim_global(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    var_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    var_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2) /
             ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)))
    return float(score.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--compute-lpips", action="store_true",
                        help="Use an already-installed LPIPS package and already-cached official weights only.")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError("Audit output must remain inside the isolated Safe-LR project")
    cfg = load_config(args.config)
    dataset = Path(cfg["data"]["dataset_root"])
    paths = build_run_paths(cfg["platform"]["run_root"], "legacy_kproto_ret",
                            int(cfg["methods"]["legacy_kproto_ret"]["universal_epochs"]), 0)
    poisoned = Path(paths.poisoned_root)
    manifest = Path(paths.manifest_csv)
    # Reuse the established exhaustive parser/decoded-pixel audit (all images,
    # byte-identical labels, stems, changed person boxes, exact 8/4/0 gates).
    inherited = subprocess.run([sys.executable, str(ROOT / "scripts/audit_b2_to_safebg_materialization.py"),
        "--config", args.config, "--noise", args.noise, "--output", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    if inherited.returncode:
        raise RuntimeError("Exhaustive 8/4/0 materialization gate failed: " +
                           (inherited.stderr or inherited.stdout)[-3000:])
    pixel_receipt = json.loads((out / "materialization_audit.json").read_text(encoding="utf-8"))
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clean_dir = dataset / cfg["data"]["train_images"]
    clean_by_stem = {Path(p).stem: Path(p) for p in list_images(str(clean_dir))}
    poison_by_stem = {Path(r["image_path"]).stem: Path(r["image_path"]) for r in rows}
    clean_label_dir = dataset / cfg["data"]["train_labels"]
    poison_label_dir = poisoned / "labels/train"
    if (len(clean_by_stem) != 16551 or len(rows) != 16551 or
            set(clean_by_stem) != set(poison_by_stem) or
            {p.stem for p in poison_label_dir.glob("*.txt")} != set(clean_by_stem)):
        raise RuntimeError("Train image/label count or stem identity mismatch")
    random.seed(0)
    target_stems = [r["stem"] for r in rows if r["has_target"] == "1"]
    if len(target_stems) < args.sample_count:
        raise RuntimeError("Too few person images for the registered decoded audit sample")
    selected = sorted(random.sample(sorted(target_stems), args.sample_count))
    accum = {key: {"sum_abs": 0.0, "sum_sq": 0.0, "count": 0, "max": 0,
                   "saturated": 0, "perturbed": 0} for key in ("target_only", "overlap", "outside", "global")}
    image_summaries = []
    lpips_model = None
    lpips_scores = []
    if args.compute_lpips:
        import torch
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
        if not checkpoint.is_file():
            raise RuntimeError(
                "Canonical AlexNet LPIPS backbone weights are not already cached; "
                "refusing any network download."
            )
        import lpips
        package_weights = Path(lpips.__file__).resolve().parent / "weights" / "v0.1" / "alex.pth"
        if not package_weights.is_file():
            raise RuntimeError(
                "Canonical LPIPS AlexNet calibration weights are unavailable locally; "
                "refusing any network download."
            )
        lpips_model = lpips.LPIPS(net="alex", verbose=False)
        lpips_model = lpips_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    for stem in selected:
        source, noisy = clean_by_stem[stem], poison_by_stem[stem]
        clean = cv2.imread(str(source), cv2.IMREAD_COLOR)
        adv = cv2.imread(str(noisy), cv2.IMREAD_COLOR)
        if clean is None or adv is None or clean.shape != adv.shape:
            raise RuntimeError(f"Could not decode sample image {stem}")
        diff = np.abs(clean.astype(np.int16) - adv.astype(np.int16))
        annotations = read_yolo_annotations(str(clean_label_dir / f"{stem}.txt"))
        target = np.zeros(diff.shape[:2], dtype=bool)
        non_target = np.zeros(diff.shape[:2], dtype=bool)
        for ann in annotations:
            box = _rectangle(diff.shape[:2], ann["bbox"])[0, 0].numpy().astype(bool)
            if int(ann["cls"]) == 14:
                target |= box
            else:
                non_target |= box
        masks = {"target_only": target & ~non_target, "overlap": target & non_target,
                 "outside": ~target, "global": np.ones(target.shape, dtype=bool)}
        for name, mask in masks.items():
            vals = diff[mask].astype(np.float64)
            item = accum[name]
            if vals.size:
                item["sum_abs"] += float(vals.sum())
                item["sum_sq"] += float(np.square(vals).sum())
                item["count"] += int(vals.size)
                item["max"] = max(item["max"], int(vals.max()))
                cap = 8 if name in ("target_only", "global") else 4 if name == "overlap" else 0
                item["saturated"] += int((vals >= cap).sum()) if cap else 0
                item["perturbed"] += int((vals > 0).sum())
        mse = float(np.square(diff.astype(np.float64)).mean())
        psnr = float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
        record = {"stem": stem, "PSNR_dB": psnr, "SSIM_global": ssim_global(clean, adv),
                  "global_Linf": int(diff.max()), "mean_abs_delta_uint8": float(diff.mean()),
                  "global_L2": float(np.linalg.norm(diff.astype(np.float64)))}
        if lpips_model is not None:
            import torch
            def tensor(im):
                rgb = cv2.cvtColor(cv2.resize(im, (256, 256)), cv2.COLOR_BGR2RGB)
                return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(lpips_model.net.slice1[0].weight.device) / 127.5 - 1
            with torch.no_grad():
                score = float(lpips_model(tensor(clean), tensor(adv)).item())
            lpips_scores.append(score)
            record["LPIPS_alex_256"] = score
        image_summaries.append(record)
    stats = {}
    for name, item in accum.items():
        count = max(1, item["count"])
        stats[name] = {"max_abs_delta_uint8": item["max"],
            "mean_abs_delta_uint8": item["sum_abs"] / count,
            "L2_raw_uint8": math.sqrt(item["sum_sq"]),
            "saturation_ratio_at_budget": item["saturated"] / count,
            "perturbed_pixel_ratio": item["perturbed"] / count,
            "sampled_scalar_count": item["count"]}
    label_hashes = {stem: sha256(clean_label_dir / f"{stem}.txt") for stem in sorted(clean_by_stem)}
    poison_label_hashes = {stem: sha256(poison_label_dir / f"{stem}.txt") for stem in sorted(clean_by_stem)}
    if label_hashes != poison_label_hashes:
        raise RuntimeError("Annotation bytes differ in full train split")
    receipt = {"state": "passed" if pixel_receipt.get("state") == "passed" else "failed",
        "run_id": Path(args.noise).name.removeprefix("noise_full_"), "noise_run": str(args.noise),
        "source_image_count": len(clean_by_stem), "poisoned_image_count": len(poison_by_stem),
        "source_label_count": len(label_hashes), "poisoned_label_count": len(poison_label_hashes),
        "same_stems": True, "same_class_mapping": cfg["surrogate"]["num_classes"] == 20,
        "labels_byte_identical": True, "all_image_pixel_audit": pixel_receipt,
        "decoded_target_sample_count": len(selected), "decoded_target_sample_seed": 0,
        "decoded_pixel_statistics": stats, "sample_image_metrics": image_summaries,
        "LPIPS_status": "computed" if lpips_scores else "not_available_or_not_authorized",
        "LPIPS_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
        "LPIPS_note": "LPIPS requires a locally present official package/backbone; no download is attempted.",
        "class_mapping": VOC_NAMES, "epsilon_configured": float(cfg["experiment"]["eps"]),
        "epsilon_target": float(cfg["experiment"]["eps"]),
        "epsilon_overlap": float(cfg["experiment"]["eps"]) * 0.5,
        "outside_target_max_uint8": pixel_receipt["pixel_maxima"]["outside"]}
    (out / "dataset_integrity.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("state", "source_image_count", "poisoned_image_count",
        "decoded_target_sample_count", "decoded_pixel_statistics", "LPIPS_status", "LPIPS_mean")}, indent=2), flush=True)
    if args.compute_lpips and not lpips_scores:
        raise RuntimeError("LPIPS was requested but no scores were produced")


if __name__ == "__main__":
    main()
