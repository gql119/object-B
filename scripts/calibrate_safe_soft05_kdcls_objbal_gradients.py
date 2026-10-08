"""One-shot object-balanced KD gradient calibration on the locked 16 train queries."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from ue_framework.config import load_config
from ue_framework.data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.b2_semantic import B2SemanticTrainer
from ue_framework.methods.kd_utility import KDAdaptiveLearner
from ue_framework.methods.semantic_coordinates import sha256


PARENT_CACHE = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/"
    "ue_project/ablation_safe_bg_20261004/candidate/"
    "noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt"
)
PARENT_CACHE_SHA256 = "12889ba0f80b29c194a59bd9eb2a6d8b137e9aa4977fdb94ad84444628b316a7"
QUERY_INDICES = (0, 4, 8, 12)
EPS = 1e-12


class ObjectGradientTrainer(B2SemanticTrainer):
    def __init__(self, cfg, method, device, frozen, adaptive):
        super().__init__(cfg, method, device, frozen, adaptive)
        self.adaptive = KDAdaptiveLearner(
            adaptive, self.adaptive_feature_layer, self.target_class_id,
            self.imgsz, float(method["adaptive"]["inner_lr"]),
        )
        self.adaptive.collect_kd_object_diagnostics = True

    def _compose_delta_batched(self, *args, **kwargs):
        result = super()._compose_delta_batched(*args, **kwargs)
        self._calibration_delta = result[1]
        return result


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty or nonfinite calibration distribution")
    result = {"min": float(values.min()), "max": float(values.max()),
              "mean": float(values.mean()), "std": float(values.std(ddof=0))}
    for name, q in (("Q10", .1), ("Q25", .25), ("median", .5),
                    ("Q75", .75), ("Q90", .9)):
        result[name] = float(np.quantile(values, q))
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write empty calibration rows")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fixed-set", required=True)
    parser.add_argument("--old-gradient-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    fixed_path = Path(args.fixed_set)
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    if (fixed["seed"] != 0 or fixed["optimizer_step_count"] != 0
            or fixed["batch_size"] != 16
            or fixed["query_indices_per_batch"] != list(QUERY_INDICES)
            or len(fixed["selected_stems"]) != 64 or len(fixed["rows"]) != 16
            or not all(fixed["checks"].values())):
        raise ValueError("The preregistered calibration set changed")
    if sha256(PARENT_CACHE) != PARENT_CACHE_SHA256:
        raise ValueError("Parent prototype cache changed")

    cfg = load_config(args.config)
    method = cfg["methods"]["legacy_kproto_ret"]
    device = torch.device("cuda")
    frozen = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    adaptive = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    trainer = ObjectGradientTrainer(cfg, method, device, frozen, adaptive)
    before = (state_hash(frozen), state_hash(adaptive))
    coeff_before = trainer.fourier_coeff.detach().clone()
    small_before = trainer.suppress_small.detach().clone()
    data = Path(cfg["data"]["dataset_root"])
    images = data / cfg["data"]["train_images"]
    labels = data / cfg["data"]["train_labels"]
    cache = torch.load(PARENT_CACHE, map_location="cpu", weights_only=True)
    if cache["cache_key"] != trainer._cache_key(str(images), str(labels)):
        raise ValueError("Parent cache key differs from data/model protocol")
    prototypes = cache["prototypes"].to(device)
    mixed = []
    for path in trainer._collect_target_images(str(images), str(labels)):
        anns = read_yolo_annotations(label_path_for_image(path, str(labels)))
        if any(int(ann["cls"]) != 14 for ann in anns):
            mixed.append(path)
        if len(mixed) == 64:
            break
    if [Path(p).stem for p in mixed] != fixed["selected_stems"]:
        raise ValueError("The locked 64 training image stems changed")

    trainer._epoch = 0
    trainer._active_idx = trainer._get_active_freq_indices(0, 0)
    trainer._active_coords = [tuple(trainer.freq_candidate_coords[int(i)])
                              for i in trainer._active_idx.cpu().tolist()]
    trainer._semantic_calls = 1
    rows, object_rows, batch_rows = [], [], []
    for batch_id in range(4):
        paths = mixed[batch_id * 16:(batch_id + 1) * 16]
        items = []
        for path in paths:
            image = load_image_rgb_float(path)
            anns = read_yolo_annotations(label_path_for_image(path, str(labels)))
            support, _, _ = trainer._build_support(image.shape, anns, image_path=path)
            if support.sum() <= 10:
                raise ValueError("A locked train image lacks target support")
            items.append((image, anns, support))
        for query_index in QUERY_INDICES:
            expected = fixed["rows"][len(rows)]
            stem = Path(paths[query_index]).stem
            if (expected["batch"] != batch_id or expected["query_index"] != query_index
                    or expected["query_stem"] != stem):
                raise ValueError("Locked query order changed")
            trainer._adaptive_calls = query_index
            result = trainer._batch_loss(items, prototypes)
            if result is None:
                raise RuntimeError("No loss for locked query")
            _, metrics = result
            delta = trainer._calibration_delta
            ret = trainer._semantic_terms["ret"]
            kd_unit = metrics["L_KD_cls_nt_raw"]
            attack = (trainer._semantic_terms["BG_sel"]
                      + trainer._semantic_terms["rel_sel"] + metrics["L_adaptive"])
            gradients = [torch.autograd.grad(term, delta, retain_graph=True)[0]
                         for term in (ret, kd_unit, attack)]
            nr, nk, na = [float(g.norm().detach()) for g in gradients]
            ret_kd_dot = float((gradients[0] * gradients[1]).sum().detach())
            kd_ret = nk / (nr + EPS)
            kd_attack = nk / (na + EPS)
            objects = trainer.adaptive.kd_object_rows
            positive_count = int(metrics["kd_positive_count"].detach())
            valid_count = int(metrics["valid_kd_object_count"].detach())
            nt_count = int(metrics["non_target_gt_count"].detach())
            zero_count = int(metrics["zero_positive_object_count"].detach())
            if (len(objects) != valid_count or sum(obj["positive_count"] for obj in objects) != positive_count
                    or valid_count + zero_count != nt_count):
                raise RuntimeError("Object assignment accounting failed")
            for obj in objects:
                object_rows.append({"batch_id": batch_id, "query_index": query_index,
                                    "query_id": stem, **obj})
            obj_counts = [obj["positive_count"] for obj in objects]
            obj_losses = [obj["object_KD"] for obj in objects]
            count_stats = distribution(obj_counts) if objects else None
            loss_stats = distribution(obj_losses) if objects else None
            row = {
                "batch_id": batch_id, "query_index": query_index, "query_id": stem,
                "non_target_gt_count": nt_count,
                "valid_kd_object_count": valid_count,
                "zero_positive_object_count": zero_count,
                "kd_positive_count": positive_count,
                "L_ret_raw": float(metrics["L_ret"].detach()),
                "L_KD_raw": float(kd_unit.detach()),
                "grad_ret_norm": nr, "grad_KD_unit_norm": nk, "grad_attack_norm": na,
                "ret_KD_unit_dot": ret_kd_dot,
                "KD_unit_ret_ratio": kd_ret, "KD_unit_attack_ratio": kd_attack,
                "positives_per_object_min": count_stats["min"] if count_stats else 0,
                "positives_per_object_Q25": count_stats["Q25"] if count_stats else 0,
                "positives_per_object_median": count_stats["median"] if count_stats else 0,
                "positives_per_object_Q75": count_stats["Q75"] if count_stats else 0,
                "positives_per_object_max": count_stats["max"] if count_stats else 0,
                "object_KD_min": loss_stats["min"] if loss_stats else 0,
                "object_KD_Q25": loss_stats["Q25"] if loss_stats else 0,
                "object_KD_median": loss_stats["median"] if loss_stats else 0,
                "object_KD_Q75": loss_stats["Q75"] if loss_stats else 0,
                "object_KD_Q90": loss_stats["Q90"] if loss_stats else 0,
                "object_KD_max": loss_stats["max"] if loss_stats else 0,
                "kd_person_GT_count": float(metrics["kd_target_gt_count"].detach()),
                "kd_person_logit_count": float(metrics["kd_person_logit_in_distribution"].detach()),
                "kd_shared_index_flag": float(metrics["kd_teacher_student_index_identical"].detach()),
            }
            if not all(math.isfinite(float(v)) for k, v in row.items() if k != "query_id"):
                raise FloatingPointError("Nonfinite object-balanced KD calibration")
            rows.append(row)
            del result, metrics, delta, ret, kd_unit, attack, gradients
            trainer._calibration_delta = None
            trainer._last_poison = None
            trainer._semantic_terms = {}
            trainer._frozen_total = None
            trainer._orth_raw = None
            print(json.dumps({"query_id": stem, "KD_unit_ret": kd_ret,
                              "KD_unit_attack": kd_attack, "objects": valid_count}), flush=True)
        subset = rows[-4:]
        batch_objects = [r for r in object_rows if r["batch_id"] == batch_id]
        positives = distribution([r["positive_count"] for r in batch_objects])
        losses = distribution([r["object_KD"] for r in batch_objects])
        batch_rows.append({
            "batch_id": batch_id,
            "non_target_gt_count": sum(r["non_target_gt_count"] for r in subset),
            "valid_kd_object_count": sum(r["valid_kd_object_count"] for r in subset),
            "zero_positive_object_count": sum(r["zero_positive_object_count"] for r in subset),
            "positives_per_object_min": positives["min"],
            "positives_per_object_Q25": positives["Q25"],
            "positives_per_object_median": positives["median"],
            "positives_per_object_Q75": positives["Q75"],
            "positives_per_object_max": positives["max"],
            "object_KD_min": losses["min"], "object_KD_Q25": losses["Q25"],
            "object_KD_median": losses["median"], "object_KD_Q75": losses["Q75"],
            "object_KD_Q90": losses["Q90"], "object_KD_max": losses["max"],
            "median_KD_unit_ret": distribution([r["KD_unit_ret_ratio"] for r in subset])["median"],
        })

    m_ret = distribution([r["KD_unit_ret_ratio"] for r in rows])["median"]
    q90_attack = distribution([r["KD_unit_attack_ratio"] for r in rows])["Q90"]
    max_attack = max(r["KD_unit_attack_ratio"] for r in rows)
    candidates = {
        "median_KD_ret": .25 / (m_ret + EPS),
        "Q90_attack": .10 / (q90_attack + EPS),
        "max_attack": .50 / (max_attack + EPS),
    }
    binding = min(candidates, key=candidates.get)
    lam = candidates[binding]
    for row in rows:
        row["grad_KD_norm"] = lam * row["grad_KD_unit_norm"]
        row["KD_ret_ratio"] = lam * row["KD_unit_ret_ratio"]
        row["KD_attack_ratio"] = lam * row["KD_unit_attack_ratio"]
        # This is the exact squared norm of g_ret + lambda*g_KD_unit;
        # retain only one scalar dot product per query, not 16 large tensors.
        utility_sq = (row["grad_ret_norm"] ** 2 + row["grad_KD_norm"] ** 2
                      + 2 * lam * row["ret_KD_unit_dot"])
        row["grad_util_norm"] = math.sqrt(max(utility_sq, 0.0))
        row["util_attack_ratio"] = row["grad_util_norm"] / (row["grad_attack_norm"] + EPS)

    distributions = {name: distribution([r[name] for r in rows]) for name in (
        "grad_ret_norm", "grad_KD_unit_norm", "grad_KD_norm", "grad_attack_norm",
        "grad_util_norm", "KD_unit_ret_ratio", "KD_unit_attack_ratio",
        "KD_ret_ratio", "KD_attack_ratio", "util_attack_ratio")}
    old = json.loads(Path(args.old_gradient_summary).read_text(encoding="utf-8"))
    old_ratio = old["distributions"]["KD_ret_ratio"]
    old_tail = old_ratio["Q90"] / old_ratio["median"]
    new_ratio = distributions["KD_ret_ratio"]
    new_tail = new_ratio["Q90"] / new_ratio["median"] if new_ratio["median"] > 0 else math.inf
    object_count = sum(r["valid_kd_object_count"] for r in rows)
    zero_count = sum(r["zero_positive_object_count"] for r in rows)
    after = (state_hash(frozen), state_hash(adaptive))
    checks = {
        "same_preregistered_queries": len(rows) == 16,
        "teacher_frozen": before[0] == after[0],
        "adaptive_base_frozen": before[1] == after[1],
        "noise_unchanged": bool(torch.equal(coeff_before, trainer.fourier_coeff)
                                and torch.equal(small_before, trainer.suppress_small)),
        "person_GT_excluded": all(r["kd_person_GT_count"] == 0 for r in rows),
        "person_logit_excluded": all(r["kd_person_logit_count"] == 0 for r in rows),
        "teacher_indices_shared": all(r["kd_shared_index_flag"] == 1 for r in rows),
        "object_accounting": all(r["non_target_gt_count"] == r["valid_kd_object_count"]
                                 + r["zero_positive_object_count"] for r in rows),
        "finite": all(math.isfinite(float(r[key])) for r in rows for key in (
            "L_ret_raw", "L_KD_raw", "grad_ret_norm", "grad_KD_norm",
            "grad_attack_norm", "grad_util_norm", "KD_ret_ratio",
            "KD_attack_ratio", "util_attack_ratio")),
        "median_KD_ret_at_least_0_10": new_ratio["median"] >= .10,
        "Q90_KD_attack_at_most_0_10": distributions["KD_attack_ratio"]["Q90"] <= .10 + 1e-10,
        "max_KD_attack_at_most_0_50": distributions["KD_attack_ratio"]["max"] <= .50 + 1e-10,
        "Q90_util_attack_below_1": distributions["util_attack_ratio"]["Q90"] < 1,
        "max_util_attack_below_10": distributions["util_attack_ratio"]["max"] < 10,
    }
    if not checks["Q90_KD_attack_at_most_0_10"] or not checks["max_KD_attack_at_most_0_50"]:
        gate = "KD_TOO_STRONG"
    elif not checks["median_KD_ret_at_least_0_10"] and binding in ("Q90_attack", "max_attack"):
        gate = "KD_CALIBRATION_INFEASIBLE"
    elif all(checks.values()):
        gate = "KD_OBJECT_BALANCED_CALIBRATION_PASS"
    else:
        gate = "KD_CALIBRATION_FAILED"
    artifact = {
        "schema": "safe-soft05-kdcls-object-balanced-lambda-v1",
        "seed": 0, "optimizer_step_count": 0, "batch_size": 16,
        "query_indices_per_batch": list(QUERY_INDICES),
        "selected_stems": fixed["selected_stems"], "query_count": len(rows),
        "fixed_set_sha256": sha256(fixed_path),
        "parent_prototype_cache_sha256": PARENT_CACHE_SHA256,
        "config_sha256": sha256(args.config),
        "m_ret": m_ret, "Q90_attack_unit": q90_attack,
        "max_attack_unit": max_attack,
        "lambda_from_ret": candidates["median_KD_ret"],
        "lambda_from_q90": candidates["Q90_attack"],
        "lambda_from_max": candidates["max_attack"],
        "final_lambda_KD": lam, "binding_constraint": binding,
        "valid_kd_object_count": object_count,
        "zero_positive_object_count": zero_count,
        "positive_unit_count": sum(r["kd_positive_count"] for r in rows),
        "positives_per_object": distribution([r["positive_count"] for r in object_rows]),
        "object_KD": distribution([r["object_KD"] for r in object_rows]),
        "distributions": distributions,
        "old_KD_ret": old_ratio, "old_Q90_median": old_tail,
        "new_Q90_median": new_tail,
        "tail_reduction_factor": old_tail / new_tail if math.isfinite(new_tail) and new_tail > 0 else None,
        "tail_reduction_success": bool(new_tail < old_tail),
        "checks": checks, "gate": gate,
        "frozen_state_sha256_before": before[0], "frozen_state_sha256_after": after[0],
        "adaptive_state_sha256_before": before[1], "adaptive_state_sha256_after": after[1],
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    write_csv(output / "query_gradients.csv", rows)
    write_csv(output / "object_diagnostics.csv", object_rows)
    write_csv(output / "batch_summary.csv", batch_rows)
    (output / "kd_preregistered_lambda.json").write_text(
        json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"gate": gate, "final_lambda_KD": lam,
                      "binding_constraint": binding, "checks": checks}), flush=True)
    if gate != "KD_OBJECT_BALANCED_CALIBRATION_PASS":
        raise RuntimeError("KD calibration gate failed: " + gate)


if __name__ == "__main__":
    main()
