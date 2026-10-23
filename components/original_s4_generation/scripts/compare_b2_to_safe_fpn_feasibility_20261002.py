"""Read-only P3/P4/P5 sensitivity separability comparison on clean train."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from scan_b2_to_safe_feasibility_20261002 import feasibility_scan, state_hash, write_json
from ue_framework.config import load_config
from ue_framework.data_utils import list_images, label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import LegacyKProtoTrainer
from ue_framework.methods.semantic_coordinates import gt_evidence, sha256


LAYERS = {"P3": "model.15", "P4": "model.18", "P5": "model.21"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-artifact-dir", required=True)
    parser.add_argument("--previous-p3-scan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Layer comparison output must stay in the isolated project")
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    method = cfg["methods"]["legacy_kproto_ret"]
    strict = method["semantic"]["strict_tsfl"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    model = YOLO(cfg["surrogate"]["ckpt"]).model.cuda().eval()
    trainer = LegacyKProtoTrainer(cfg, method, torch.device("cuda"), model)
    if not set(LAYERS.values()) <= set(trainer.shape_layers):
        raise RuntimeError("P3/P4/P5 feature hooks are unavailable")
    frozen_hash = state_hash(model)
    reference = torch.load(Path(args.reference_artifact_dir) / "non_target_subspaces.pt",
                           map_location="cpu", weights_only=True)
    previous = json.loads(Path(args.previous_p3_scan).read_text(encoding="utf-8"))
    data = Path(cfg["data"]["dataset_root"])
    images = sorted(list_images(str(data / cfg["data"]["train_images"])))
    label_dir = data / cfg["data"]["train_labels"]
    nt_classes = [c for c in range(cfg["experiment"]["num_classes"]) if c != trainer.target_class_id]
    sums = {name: {"target": None, "nt": {c: None for c in nt_classes}} for name in LAYERS}
    target_count = 0
    nt_counts = {c: 0 for c in nt_classes}
    manifest = hashlib.sha256()
    started = time.time()
    write_json(output / "status.json", {"state": "running", "purpose": "P3_P4_P5_separability_only",
                                         "images_total": len(images), "started_unix": started})
    for image_index, image_path in enumerate(images):
        image = load_image_rgb_float(image_path)
        label_path = label_path_for_image(image_path, str(label_dir))
        annotations = read_yolo_annotations(label_path)
        manifest.update((Path(image_path).stem + ":" + sha256(image_path) + ":" + sha256(label_path) + "\n").encode())
        target_count += sum(int(a["cls"]) == trainer.target_class_id for a in annotations)
        if annotations:
            tensor = trainer._to_tensor(image).requires_grad_(True)
            trainer.multi_features.clear()
            decoded = trainer._forward_raw(tensor)
            features = [trainer.multi_features[layer] for layer in LAYERS.values()]
            for cls, evidence in gt_evidence(decoded, annotations, image.shape[:2], trainer.imgsz):
                grads = torch.autograd.grad(evidence, features, retain_graph=True)
                for name, gradient in zip(LAYERS, grads):
                    sensitivity = gradient.detach().abs().mean(dim=(0, 2, 3)).double().cpu()
                    if not torch.isfinite(sensitivity).all():
                        raise RuntimeError(f"Nonfinite {name} sensitivity")
                    key = "target" if cls == trainer.target_class_id else "nt"
                    if key == "target":
                        prior = sums[name]["target"]
                        sums[name]["target"] = sensitivity if prior is None else prior + sensitivity
                    elif cls in sums[name]["nt"]:
                        prior = sums[name]["nt"][cls]
                        sums[name]["nt"][cls] = sensitivity if prior is None else prior + sensitivity
                if cls in nt_counts:
                    nt_counts[cls] += 1
            del features, decoded, tensor
        if (image_index + 1) % 100 == 0:
            write_json(output / "status.json", {"state": "running", "images_done": image_index + 1,
                "images_total": len(images), "target_objects": target_count,
                "non_target_objects": sum(nt_counts.values()), "elapsed_seconds": time.time() - started})
            print(f"{image_index + 1}/{len(images)} elapsed={time.time() - started:.1f}s", flush=True)
    if state_hash(model) != frozen_hash:
        raise RuntimeError("Frozen reference changed during P3/P4/P5 scan")
    ref_identity = reference["identity"]
    if (ref_identity["feature_layer"] != trainer.feature_layer
            or ref_identity["checkpoint_sha256"] != sha256(cfg["surrogate"]["ckpt"])
            or ref_identity["frozen_state_sha256"] != frozen_hash
            or ref_identity["clean_train_manifest_sha256"] != manifest.hexdigest()
            or previous["clean_train_manifest_sha256"] != manifest.hexdigest()):
        raise RuntimeError("P3/P4/P5 scan does not match the formal clean-train reference")
    if any(count <= 0 for count in nt_counts.values()) or target_count <= 0:
        raise RuntimeError("Incomplete class-wise sensitivity scan")
    reports = {}
    for name in LAYERS:
        target = sums[name]["target"] / target_count
        nt = {c: sums[name]["nt"][c] / nt_counts[c] for c in nt_classes}
        rows, arrays = feasibility_scan(target, nt, float(method["semantic"]["orth_eps"]),
                                        float(strict["target_channel_fraction"]),
                                        int(strict["sensitivity_worst_k"]))
        st = torch.tensor(arrays["target_normalized"])
        worst = torch.tensor(arrays["worst_nt_normalized"])
        top40 = next(row for row in rows if row["target_top_fraction"] == .40 and row["worst_nt_quantile"] == .60)
        top40_mask = st >= torch.quantile(st, .60)
        stats = {"target_top40_mean": float(st[top40_mask].mean()),
                 "worst_nt_on_target_top40_mean": float(worst[top40_mask].mean()),
                 "target_to_worst_nt_ratio_of_means_on_top40": float(st[top40_mask].mean() / worst[top40_mask].mean()),
                 "pearson_target_worst_nt": float(torch.corrcoef(torch.stack((st, worst)))[0, 1]),
                 "top40_Q60_eligible_count": top40["eligible_count"]}
        reports[name] = {"layer": LAYERS[name], "channels": len(arrays["target_normalized"]),
                         "separability": stats, "gate_rows": rows, "sensitivity_arrays": arrays}
    p3_reference = previous["sensitivity_arrays"]
    p3_max_abs_diff = max(max(abs(a - b) for a, b in zip(reports["P3"]["sensitivity_arrays"][key], p3_reference[key]))
                          for key in ("target_normalized", "worst_nt_normalized"))
    if p3_max_abs_diff > 1e-6:
        raise RuntimeError(f"P3 rescan differs from previous frozen statistics: {p3_max_abs_diff}")
    report = {"state": "complete", "purpose": "P3_P4_P5_separability_only", "layers": reports,
              "clean_train_manifest_sha256": manifest.hexdigest(), "frozen_state_unchanged": True,
              "p3_previous_scan_max_abs_diff": p3_max_abs_diff, "images": len(images),
              "target_objects": target_count, "non_target_objects_by_class": nt_counts,
              "validation_used": False, "noise_updated": False, "victim_trained": False}
    write_json(output / "fpn_separability.json", report)
    write_json(output / "status.json", {"state": "complete", "purpose": "P3_P4_P5_separability_only",
               "duration_seconds": time.time() - started, "images_done": len(images),
               "report": str(output / "fpn_separability.json")})
    print(json.dumps({name: item["separability"] for name, item in reports.items()}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
