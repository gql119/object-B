"""Train-only, no-update KD gradient audit on the locked 16 calibration queries."""

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
from ue_framework.methods.b2_safe_kd import SafeSoft05KDTrainer
from ue_framework.methods.semantic_coordinates import sha256


PARENT_CACHE = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/"
    "ue_project/ablation_safe_bg_20261004/candidate/"
    "noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt"
)
PARENT_CACHE_SHA256 = "12889ba0f80b29c194a59bd9eb2a6d8b137e9aa4977fdb94ad84444628b316a7"
QUERY_INDICES = (0, 4, 8, 12)
DENOM_EPS = 1e-12
METRICS = (
    "grad_ret_norm", "grad_KD_norm", "grad_util_norm", "grad_attack_norm",
    "KD_ret_ratio", "KD_attack_ratio", "util_attack_ratio",
    "cos_KD_ret", "cos_KD_attack", "cos_util_attack",
)


class DeltaAuditTrainer(SafeSoft05KDTrainer):
    """Expose the composed delta tensor without changing production loss code."""

    def _compose_delta_batched(self, *args, **kwargs):
        result = super()._compose_delta_batched(*args, **kwargs)
        self._audit_delta = result[1]
        return result


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _cosine(left, right):
    return float(((left * right).sum() / (left.norm() * right.norm()).clamp_min(DENOM_EPS)).detach())


def _distribution(values):
    data = np.asarray(values, dtype=np.float64)
    if data.size != 16 or not np.isfinite(data).all():
        raise ValueError("A gradient distribution needs 16 finite query measurements")
    result = {"min": float(data.min()), "max": float(data.max()),
              "mean": float(data.mean()), "std": float(data.std(ddof=0))}
    for label, fraction in (("Q10", .10), ("Q25", .25), ("median", .50),
                            ("Q75", .75), ("Q90", .90)):
        result[label] = float(np.quantile(data, fraction))
    return result


def _tail(summary):
    median = summary["median"]
    if median <= DENOM_EPS:
        return summary["max"] > DENOM_EPS
    return summary["max"] / median > 10 or summary["Q90"] / median > 5


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    cfg = load_config(args.config)
    method = cfg["methods"]["legacy_kproto_ret"]
    kd_cfg = method["semantic"]["kd_cls"]
    calibration_path = Path(kd_cfg["calibration_artifact"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (calibration["seed"] != 0 or calibration["optimizer_step_count"] != 0
            or calibration["batch_size"] != 16
            or tuple(calibration["query_indices_per_batch"]) != QUERY_INDICES
            or len(calibration["selected_stems"]) != 64 or len(calibration["rows"]) != 16
            or not all(calibration["checks"].values())):
        raise ValueError("The pre-registered calibration set or checks changed")
    if sha256(PARENT_CACHE) != PARENT_CACHE_SHA256:
        raise ValueError("The parent prototype cache changed")

    device = torch.device("cuda")
    frozen = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    adaptive = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    trainer = DeltaAuditTrainer(cfg, method, device, frozen, adaptive)
    frozen_before, adaptive_before = _state_sha256(frozen), _state_sha256(adaptive)
    coeff_before = trainer.fourier_coeff.detach().clone()
    small_before = trainer.suppress_small.detach().clone()
    data = Path(cfg["data"]["dataset_root"])
    image_dir = data / cfg["data"]["train_images"]
    label_dir = data / cfg["data"]["train_labels"]
    cache = torch.load(PARENT_CACHE, map_location="cpu", weights_only=True)
    if cache["cache_key"] != trainer._cache_key(str(image_dir), str(label_dir)):
        raise ValueError("Parent prototype cache key differs from this train/model protocol")
    prototypes = cache["prototypes"].to(device)

    mixed = []
    for path in trainer._collect_target_images(str(image_dir), str(label_dir)):
        annotations = read_yolo_annotations(label_path_for_image(path, str(label_dir)))
        if any(int(ann["cls"]) != 14 for ann in annotations):
            mixed.append(path)
        if len(mixed) == 64:
            break
    if [Path(path).stem for path in mixed] != calibration["selected_stems"]:
        raise ValueError("The first 64 sorted mixed training images changed")
    trainer._epoch = 0
    trainer._active_idx = trainer._get_active_freq_indices(0, 0)
    trainer._active_coords = [tuple(trainer.freq_candidate_coords[int(i)])
                              for i in trainer._active_idx.cpu().tolist()]
    # Production gradient diagnostics are redundant here; the audit differentiates
    # each weighted term against the *same composed delta* on every query.
    trainer._semantic_calls = 1
    rows = []
    batch_rows = []
    for batch_id in range(4):
        paths = mixed[batch_id * 16:(batch_id + 1) * 16]
        items = []
        for path in paths:
            image = load_image_rgb_float(path)
            annotations = read_yolo_annotations(label_path_for_image(path, str(label_dir)))
            support, _, _ = trainer._build_support(image.shape, annotations, image_path=path)
            if support.sum() <= 10:
                raise ValueError("A pre-registered image has insufficient target support")
            items.append((image, annotations, support))
        batch_query_ids = [Path(paths[index]).stem for index in QUERY_INDICES]
        for query_index in QUERY_INDICES:
            expected = calibration["rows"][len(rows)]
            query_id = Path(paths[query_index]).stem
            if (expected["batch"] != batch_id or expected["query_index"] != query_index
                    or expected["query_stem"] != query_id):
                raise ValueError("Pre-registered query ordering changed")
            trainer._adaptive_calls = query_index
            result = trainer._batch_loss(items, prototypes)
            if result is None:
                raise RuntimeError("A locked calibration query produced no loss")
            total, metrics = result
            delta = trainer._audit_delta
            weighted_ret = trainer._semantic_terms["ret"]
            weighted_kd = metrics["L_KD_weighted"]
            weighted_attack = (trainer._semantic_terms["BG_sel"]
                               + trainer._semantic_terms["rel_sel"]
                               + metrics["L_adaptive"])
            g_ret = torch.autograd.grad(weighted_ret, delta, retain_graph=True)[0]
            g_kd = torch.autograd.grad(weighted_kd, delta, retain_graph=True)[0]
            g_attack = torch.autograd.grad(weighted_attack, delta, retain_graph=True)[0]
            g_util = g_ret + g_kd
            nr, nk, na, nu = (float(g.norm().detach()) for g in
                              (g_ret, g_kd, g_attack, g_util))
            if not all(math.isfinite(x) for x in (nr, nk, na, nu)):
                raise FloatingPointError("Non-finite delta gradient")
            query_anns = items[query_index][1]
            row = {
                "batch_id": batch_id, "query_index": query_index, "query_id": query_id,
                "batch_query_ids": "|".join(batch_query_ids),
                "non_target_gt_count": sum(int(ann["cls"]) != 14 for ann in query_anns),
                "kd_positive_count": float(metrics["kd_positive_count"].detach()),
                "kd_gt_object_count": float(metrics["kd_gt_object_count"].detach()),
                "L_ret_raw": float(metrics["L_ret"].detach()),
                "L_KD_raw": float(metrics["L_KD_cls_nt_raw"].detach()),
                "L_BG": float(metrics["L_BG"].detach()),
                "L_rel": float(metrics["L_rel"].detach()),
                "L_adapt": float(metrics["L_adaptive"].detach()),
                "L_orth": float(metrics["L_orth_worst"].detach()),
                "grad_ret_norm": nr, "grad_KD_norm": nk,
                "grad_util_norm": nu, "grad_attack_norm": na,
                "KD_ret_ratio": nk / (nr + DENOM_EPS),
                "KD_attack_ratio": nk / (na + DENOM_EPS),
                "util_attack_ratio": nu / (na + DENOM_EPS),
                "cos_KD_ret": _cosine(g_kd, g_ret),
                "cos_KD_attack": _cosine(g_kd, g_attack),
                "cos_util_attack": _cosine(g_util, g_attack),
                "KD_bicycle": float(metrics["KD_bicycle"].detach()),
                "KD_dog": float(metrics["KD_dog"].detach()),
                "KD_horse": float(metrics["KD_horse"].detach()),
                "kd_worst_class_1": int(metrics["kd_worst_class_1"].detach()),
                "kd_worst_class_loss_1": float(metrics["kd_worst_class_loss_1"].detach()),
                "kd_person_GT_count": float(metrics["kd_target_gt_count"].detach()),
                "kd_person_logit_count": float(metrics["kd_person_logit_in_distribution"].detach()),
                "kd_shared_index_flag": float(metrics["kd_teacher_student_index_identical"].detach()),
            }
            if (not all(math.isfinite(float(value)) for key, value in row.items()
                        if key not in ("query_id", "batch_query_ids"))
                    or row["kd_person_GT_count"] != 0 or row["kd_person_logit_count"] != 0
                    or (row["kd_positive_count"] > 0 and row["kd_shared_index_flag"] != 1)):
                raise RuntimeError("A KD exclusion/correspondence or finite check failed")
            rows.append(row)
            print(json.dumps({"batch_id": batch_id, "query_id": query_id,
                              "KD_ret_ratio": row["KD_ret_ratio"],
                              "KD_attack_ratio": row["KD_attack_ratio"],
                              "util_attack_ratio": row["util_attack_ratio"]}), flush=True)
            del result, total, metrics, delta, weighted_ret, weighted_kd, weighted_attack
            del g_ret, g_kd, g_attack, g_util
            trainer._audit_delta = None
            trainer._last_poison = None
            trainer._semantic_terms = {}
            trainer._frozen_total = None
            trainer._orth_raw = None
        subset = rows[-4:]
        summary_row = {"batch_id": batch_id, "query_ids": "|".join(batch_query_ids)}
        for key in ("non_target_gt_count", "kd_positive_count", "L_ret_raw", "L_KD_raw",
                    "L_BG", "L_rel", "L_adapt", "L_orth") + METRICS:
            summary_row[key] = float(np.mean([float(row[key]) for row in subset]))
        batch_rows.append(summary_row)

    distributions = {key: _distribution([row[key] for row in rows]) for key in METRICS}
    tails = {key: _tail(distributions[key]) for key in
             ("grad_KD_norm", "KD_ret_ratio", "KD_attack_ratio")}
    frozen_after, adaptive_after = _state_sha256(frozen), _state_sha256(adaptive)
    checks = {
        "same_calibration_queries": len(rows) == 16,
        "frozen_teacher_unchanged": frozen_before == frozen_after,
        "adaptive_base_unchanged": adaptive_before == adaptive_after,
        "noise_unchanged": bool(torch.equal(coeff_before, trainer.fourier_coeff)
                                and torch.equal(small_before, trainer.suppress_small)),
        "person_GT_excluded": all(row["kd_person_GT_count"] == 0 for row in rows),
        "person_logit_excluded": all(row["kd_person_logit_count"] == 0 for row in rows),
        "shared_teacher_indices": all(row["kd_shared_index_flag"] == 1
                                      or row["kd_positive_count"] == 0 for row in rows),
        "all_finite": all(math.isfinite(float(row[key])) for row in rows for key in METRICS),
    }
    result = {
        "schema": "safe-soft05-kdcls-gradient-audit-v1",
        "optimizer_step_count": 0,
        "config_sha256": sha256(args.config),
        "calibration_sha256": sha256(calibration_path),
        "parent_prototype_cache_sha256": PARENT_CACHE_SHA256,
        "seed": 0, "batch_count": 4, "query_count": 16,
        "non_target_positive_units": sum(row["kd_positive_count"] for row in rows),
        "kd_scale": float(trainer.kd_scale.detach()),
        "lambda_ret": trainer.lambda_ret,
        "effective_lambda_KD": trainer.lambda_ret * trainer.kd_beta * float(trainer.kd_scale.detach()),
        "denominator_epsilon": DENOM_EPS,
        "gradient_variable": "composed delta tensor, same tensor for all four gradients",
        "statistics": "16 query observations; linear quantiles; population std (ddof=0)",
        "distributions": distributions, "long_tail": tails,
        "checks": checks,
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "adaptive_state_sha256_before": adaptive_before,
        "adaptive_state_sha256_after": adaptive_after,
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    _write_csv(output / "kd_gradient_calibration_batches.csv", rows)
    _write_csv(output / "kd_gradient_calibration_batch_summary.csv", batch_rows)
    (output / "kd_gradient_calibration_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError("Gradient audit checks failed: " + str(checks))
    print(json.dumps({"checks": checks, "long_tail": tails,
                      "non_target_positive_units": result["non_target_positive_units"]}))


if __name__ == "__main__":
    main()
