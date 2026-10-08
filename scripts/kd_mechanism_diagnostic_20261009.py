"""Read-only KD mechanism diagnostic on the completed formal perturbation."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.tal import make_anchors

from ue_framework.config import load_config
from ue_framework.data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.b2_safe_kd import SafeSoft05KDTrainer
from ue_framework.methods.kd_utility import KDAdaptiveLearner, object_balanced_mean
from ue_framework.methods.semantic_coordinates import CLASSES, sha256


CONFIG_SHA = "8107f4a3f7859b3d775cfa3c37d461a774bc379b2782206ec7b0c39b363fc741"
CACHE_SHA = "12889ba0f80b29c194a59bd9eb2a6d8b137e9aa4977fdb94ad84444628b316a7"
QUERY_INDICES = (0, 4, 8, 12)
TARGET = 14
TAU = 2.0
LAMBDA_KD = 0.7160846249749698


class RecordedAdaptive(KDAdaptiveLearner):
    def _predict_with_raw(self, images, params, buffers, *, grad):
        value = super()._predict_with_raw(images, params, buffers, grad=grad)
        self.audit_predictions.append(value)
        return value


class AuditTrainer(SafeSoft05KDTrainer):
    def __init__(self, cfg, method, device, frozen, adaptive):
        super().__init__(cfg, method, device, frozen, adaptive)
        self.adaptive = RecordedAdaptive(
            adaptive, self.adaptive_feature_layer, self.target_class_id,
            self.imgsz, float(method["adaptive"]["inner_lr"]))
        self.adaptive.audit_predictions = []
        self.audit_delta = None

    def _compose_delta_batched(self, *args, **kwargs):
        result = super()._compose_delta_batched(*args, **kwargs)
        self.audit_delta = result[1]
        return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou_xyxy(boxes, gt):
    inter = (torch.minimum(boxes[:, 2:], gt[:, 2:])
             - torch.maximum(boxes[:, :2], gt[:, :2])).clamp_min(0).prod(-1)
    area_box = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    area_gt = (gt[:, 2:] - gt[:, :2]).clamp_min(0).prod(-1)
    return inter / (area_box + area_gt - inter).clamp_min(1e-12)


def cosine(left, right):
    denom = left.norm() * right.norm()
    if float(denom) < 1e-15:
        return float("nan")
    return float(((left * right).sum() / denom).detach())


def json_vector(tensor):
    return json.dumps([round(float(v), 7) for v in tensor.detach().cpu().tolist()],
                      separators=(",", ":"))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_query(trainer, clean_batch, expected_softmax):
    """Reproduce formal teacher TAL indices and compare raw-head outputs."""
    if len(trainer.adaptive.audit_predictions) != 3:
        raise RuntimeError("Expected teacher clean, student clean and student poison predictions")
    t_decoded, _, t_raw = trainer.adaptive.audit_predictions[0]
    s_decoded, _, s_raw = trainer.adaptive.audit_predictions[1]
    errors = []
    raw_extrema = []
    for decoded, raw in ((t_decoded, t_raw), (s_decoded, s_raw)):
        logits = raw["scores"]
        if logits.shape != decoded[:, 4:, :].shape:
            raise RuntimeError("Raw and decoded score shapes differ")
        error = float((logits.sigmoid() - decoded[:, 4:, :]).abs().max().detach())
        if error > 1e-5:
            raise RuntimeError(f"Raw scores are not logits: sigmoid/decoded error={error}")
        errors.append(error)
        raw_extrema.append((float(logits.min().detach()), float(logits.max().detach())))
    criterion = trainer.adaptive.model.criterion
    t_score = t_raw["scores"].permute(0, 2, 1).contiguous().detach()
    s_score = s_raw["scores"].permute(0, 2, 1).contiguous()
    targets = torch.cat((clean_batch["batch_idx"].view(-1, 1),
                         clean_batch["cls"].view(-1, 1), clean_batch["bboxes"]), 1)
    size = torch.tensor(t_raw["feats"][0].shape[2:], device=t_score.device,
                        dtype=t_score.dtype) * criterion.stride[0]
    targets = criterion.preprocess(targets.to(t_score.device), t_score.shape[0],
                                   scale_tensor=size[[1, 0, 1, 0]])
    gt_labels, gt_boxes = targets.split((1, 4), 2)
    nt_mask = gt_boxes.sum(2, keepdim=True).gt(0) & gt_labels.long().ne(TARGET)
    anchor, stride = make_anchors(t_raw["feats"], criterion.stride, .5)
    t_boxes = criterion.bbox_decode(
        anchor, t_raw["boxes"].permute(0, 2, 1).contiguous().detach())
    s_boxes = criterion.bbox_decode(
        anchor, s_raw["boxes"].permute(0, 2, 1).contiguous().detach())
    with torch.no_grad():
        _, _, _, t_fg, t_match = criterion.assigner(
            t_score.sigmoid(), (t_boxes * stride).type(gt_boxes.dtype),
            anchor * stride, gt_labels, gt_boxes, nt_mask)
        _, _, _, s_fg, s_match = criterion.assigner(
            s_score.detach().sigmoid(), (s_boxes * stride).type(gt_boxes.dtype),
            anchor * stride, gt_labels, gt_boxes, nt_mask)
    pos_b, pos_i = torch.nonzero(t_fg, as_tuple=True)
    if not len(pos_b):
        raise RuntimeError("No teacher positives in locked query")
    gt_idx = t_match[pos_b, pos_i].long()
    gt_class = gt_labels[pos_b, gt_idx, 0].long()
    if bool((gt_class == TARGET).any()):
        raise RuntimeError("Person GT entered non-target diagnostic")
    keep = torch.arange(20, device=t_score.device) != TARGET
    teacher = t_score[pos_b, pos_i][:, keep].double()
    student = s_score[pos_b, pos_i][:, keep].double()
    t_logsoft = F.log_softmax(teacher / TAU, -1)
    s_logsoft = F.log_softmax(student / TAU, -1)
    t_soft = t_logsoft.exp()
    s_soft = s_logsoft.exp()
    soft_kl = ((t_soft * (t_logsoft - s_logsoft)).sum(-1) * TAU**2).clamp_min(0)
    t_logp, t_logq = F.logsigmoid(teacher / TAU), F.logsigmoid(-teacher / TAU)
    s_logp, s_logq = F.logsigmoid(student / TAU), F.logsigmoid(-student / TAU)
    sigmoid_kl = ((t_logp.exp() * (t_logp - s_logp)
                   + t_logq.exp() * (t_logq - s_logq)).sum(-1) * TAU**2).clamp_min(0)
    pair_id = pos_b * gt_labels.shape[1] + gt_idx
    soft_object, _, _, _ = object_balanced_mean(soft_kl, pair_id)
    sigmoid_object, _, _, _ = object_balanced_mean(sigmoid_kl, pair_id)
    if abs(float(soft_object.float().detach()) - float(expected_softmax.detach())) > 1e-6:
        raise RuntimeError("Recomputed Softmax KD differs from the formal objective")
    gt_kept = gt_class - (gt_class > TARGET).long()
    rows = torch.arange(len(pos_b), device=t_score.device)
    t_logit, s_logit = teacher[rows, gt_kept], student[rows, gt_kept]
    t_prob, s_prob = t_logit.sigmoid(), s_logit.sigmoid()
    t_iou = iou_xyxy((t_boxes[pos_b, pos_i] * stride[pos_i]).detach(),
                     gt_boxes[pos_b, gt_idx])
    s_iou = iou_xyxy((s_boxes[pos_b, pos_i] * stride[pos_i]).detach(),
                     gt_boxes[pos_b, gt_idx])
    alpha, beta = float(criterion.assigner.alpha), float(criterion.assigner.beta)
    t_evidence = t_prob.detach().float().pow(alpha) * t_iou.pow(beta)
    s_evidence = s_prob.detach().float().pow(alpha) * s_iou.pow(beta)
    t_top, s_top = teacher.argmax(-1), student.argmax(-1)
    t_top = t_top + (t_top >= TARGET).long()
    s_top = s_top + (s_top >= TARGET).long()
    return locals()


def main():
    parser = argparse.ArgumentParser()
    for option in ("config", "fixed-set", "parent-cache", "global-params", "comparison", "output"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if digest(args.config) != CONFIG_SHA or sha256(args.parent_cache) != CACHE_SHA:
        raise RuntimeError("Formal config or parent prototype cache changed")
    fixed = json.loads(Path(args.fixed_set).read_text(encoding="utf-8"))
    if (fixed["seed"] != 0 or fixed["optimizer_step_count"] != 0
            or fixed["batch_size"] != 16
            or fixed["query_indices_per_batch"] != list(QUERY_INDICES)
            or len(fixed["selected_stems"]) != 64 or len(fixed["rows"]) != 16
            or not all(fixed["checks"].values())):
        raise RuntimeError("Locked fixed query set changed")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    cfg = load_config(args.config)
    method = cfg["methods"]["legacy_kproto_ret"]
    device = torch.device("cuda")
    frozen = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    adaptive = YOLO(cfg["surrogate"]["ckpt"]).model.to(device)
    trainer = AuditTrainer(cfg, method, device, frozen, adaptive)
    if abs(trainer.lambda_kd - LAMBDA_KD) > 1e-12:
        raise RuntimeError("Formal lambda_KD changed")
    pack = torch.load(args.global_params, map_location="cpu", weights_only=True)
    saved_coords = [tuple(c) for c in pack["coords"]]
    mapping = {tuple(c): i for i, c in enumerate(trainer.freq_candidate_coords)}
    if len(saved_coords) != 16 or any(c not in mapping for c in saved_coords):
        raise RuntimeError("Formal Fourier coordinates changed")
    idx = torch.tensor([mapping[c] for c in saved_coords], device=device)
    if len(set(idx.cpu().tolist())) != len(idx):
        raise RuntimeError("Duplicate formal Fourier coordinates")
    with torch.no_grad():
        trainer.fourier_coeff.zero_()
        trainer.fourier_coeff[idx] = pack["fourier_coeff"].to(device)
        trainer.suppress_small.copy_(pack["suppress_small"].to(device))
    trainer.freq_active_idx = idx
    trainer._active_idx = idx
    trainer._active_coords = saved_coords
    trainer._epoch = 23
    trainer.is_universal_training = False  # Final materialization regime; no random dropout.
    trainer._semantic_calls = 1
    data = Path(cfg["data"]["dataset_root"])
    images, labels = data / cfg["data"]["train_images"], data / cfg["data"]["train_labels"]
    cache = torch.load(args.parent_cache, map_location="cpu", weights_only=True)
    if cache["cache_key"] != trainer._cache_key(str(images), str(labels)):
        raise RuntimeError("Prototype cache protocol differs")
    prototypes = cache["prototypes"].to(device)
    mixed = []
    for path in trainer._collect_target_images(str(images), str(labels)):
        anns = read_yolo_annotations(label_path_for_image(path, str(labels)))
        if any(int(ann["cls"]) != TARGET for ann in anns):
            mixed.append(path)
        if len(mixed) == 64:
            break
    if [Path(path).stem for path in mixed] != fixed["selected_stems"]:
        raise RuntimeError("Fixed image selection changed")
    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    ap = {row["class"]: row for row in comparison["per_class"]}
    functional, gradients, objects = [], [], []
    errors, extrema = [], []
    for batch_id in range(4):
        paths = mixed[batch_id * 16:(batch_id + 1) * 16]
        items = []
        for path in paths:
            image = load_image_rgb_float(path)
            anns = read_yolo_annotations(label_path_for_image(path, str(labels)))
            support, _, _ = trainer._build_support(image.shape, anns, image_path=path)
            if support.sum() <= 10:
                raise RuntimeError("Locked image has no target support")
            items.append((image, anns, support))
        for query_index in QUERY_INDICES:
            expected = fixed["rows"][len(gradients)]
            stem = Path(paths[query_index]).stem
            if (expected["batch"] != batch_id or expected["query_index"] != query_index
                    or expected["query_stem"] != stem):
                raise RuntimeError("Fixed query order changed")
            trainer._adaptive_calls = query_index
            trainer.adaptive.audit_predictions = []
            result = trainer._batch_loss(items, prototypes)
            if result is None:
                raise RuntimeError("No loss for locked query")
            _, metrics = result
            delta = trainer.audit_delta
            if delta is None or not delta.requires_grad:
                raise RuntimeError("No differentiable delta captured")
            image = items[query_index][0]
            pad_h, pad_w = trainer._last_poison.shape[-2:]
            clean = F.pad(torch.from_numpy(image).permute(2, 0, 1).float(),
                          (0, pad_w - image.shape[1], 0, pad_h - image.shape[0]), value=.5)
            batch = trainer.adaptive._batch(clean[None].to(device), [items[query_index]])
            detail = read_query(trainer, batch, metrics["L_KD_cls_nt_raw"])
            errors.extend(detail["errors"])
            extrema.extend(detail["raw_extrema"])
            attack = (trainer._semantic_terms["BG_sel"]
                      + trainer._semantic_terms["rel_sel"] + metrics["L_adaptive"])
            terms = {"ret": trainer._semantic_terms["ret"],
                     "soft": metrics["L_KD_cls_nt_raw"],
                     "sig": detail["sigmoid_object"], "attack": attack}
            g = {key: torch.autograd.grad(term, delta, retain_graph=True)[0].detach()
                 for key, term in terms.items()}
            ret, soft_raw, sig_raw, attack_grad = (g[key] for key in
                                                    ("ret", "soft", "sig", "attack"))
            soft, sig = trainer.lambda_kd * soft_raw, trainer.lambda_kd * sig_raw
            ret_norm, attack_norm = float(ret.norm()), float(attack_grad.norm())
            soft_norm, sig_norm = float(soft.norm()), float(sig.norm())
            gradients.append({
                "batch_id": batch_id, "query_index": query_index, "query_id": stem,
                "ret_weighted_norm": ret_norm, "softmax_raw_norm": float(soft_raw.norm()),
                "softmax_weighted_norm": soft_norm, "sigmoid_raw_norm": float(sig_raw.norm()),
                "sigmoid_weighted_same_lambda_norm": sig_norm,
                "attack_weighted_norm": attack_norm,
                "softmax_to_ret": soft_norm / max(ret_norm, 1e-12),
                "sigmoid_to_ret_same_lambda": sig_norm / max(ret_norm, 1e-12),
                "softmax_to_attack": soft_norm / max(attack_norm, 1e-12),
                "sigmoid_to_attack_same_lambda": sig_norm / max(attack_norm, 1e-12),
                "softmax_cos_attack_unit": cosine(soft, attack_grad),
                "sigmoid_cos_attack_unit": cosine(sig, attack_grad),
                "softmax_cos_ret_unit": cosine(soft, ret),
                "sigmoid_cos_ret_unit": cosine(sig, ret),
                "softmax_sigmoid_cos_unit": cosine(soft, sig),
            })
            d = {key: detail[key].detach().cpu().tolist() for key in
                 ("pos_b", "pos_i", "gt_idx", "gt_class", "soft_kl", "sigmoid_kl",
                  "t_logit", "s_logit", "t_prob", "s_prob", "t_iou", "s_iou",
                  "t_evidence", "s_evidence", "t_top", "s_top")}
            for i, (b, loc, obj, cls) in enumerate(zip(
                    d["pos_b"], d["pos_i"], d["gt_idx"], d["gt_class"])):
                if b != 0:
                    raise RuntimeError("Adaptive query batch size changed")
                same = bool(detail["s_fg"][0, loc].item()) and int(
                    detail["s_match"][0, loc].item()) == obj
                functional.append({
                    "query_id": stem, "object_key": f"{stem}:{obj}",
                    "gt_index": obj, "gt_class_id": cls, "gt_class": CLASSES[cls],
                    "teacher_positive_location": loc,
                    "teacher_raw_logits_19": json_vector(detail["teacher"][i]),
                    "student_raw_logits_19": json_vector(detail["student"][i]),
                    "teacher_sigmoid_19": json_vector(detail["teacher"][i].sigmoid()),
                    "student_sigmoid_19": json_vector(detail["student"][i].sigmoid()),
                    "teacher_softmax_tau2_19": json_vector(detail["t_soft"][i]),
                    "student_softmax_tau2_19": json_vector(detail["s_soft"][i]),
                    "softmax_KL_tau2": d["soft_kl"][i],
                    "sigmoid_Bernoulli_KL_tau2_sum19": d["sigmoid_kl"][i],
                    "teacher_GT_logit": d["t_logit"][i],
                    "student_GT_logit": d["s_logit"][i],
                    "teacher_GT_sigmoid": d["t_prob"][i],
                    "student_GT_sigmoid": d["s_prob"][i],
                    "GT_sigmoid_change": d["s_prob"][i] - d["t_prob"][i],
                    "teacher_top1_class": CLASSES[d["t_top"][i]],
                    "student_top1_class": CLASSES[d["s_top"][i]],
                    "top1_agreement": int(d["t_top"][i] == d["s_top"][i]),
                    "teacher_GT_aligned_IoU": d["t_iou"][i],
                    "student_GT_aligned_IoU": d["s_iou"][i],
                    "GT_aligned_IoU_change": d["s_iou"][i] - d["t_iou"][i],
                    "teacher_foreground_evidence": d["t_evidence"][i],
                    "student_foreground_evidence": d["s_evidence"][i],
                    "foreground_evidence_change": d["s_evidence"][i] - d["t_evidence"][i],
                    "same_object_student_assignment": int(same),
                    "blindspot_KL1e-4_drop0p05": int(
                        d["soft_kl"][i] <= 1e-4
                        and d["s_prob"][i] - d["t_prob"][i] <= -0.05),
                })
            for obj in sorted(set(d["gt_idx"])):
                members = [row for row in functional if row["object_key"] == f"{stem}:{obj}"]
                t_mask = detail["t_fg"][0] & (detail["t_match"][0] == obj)
                s_mask = detail["s_fg"][0] & (detail["s_match"][0] == obj)
                intersection = int((t_mask & s_mask).sum())
                union = int((t_mask | s_mask).sum())
                objects.append({
                    "object_key": f"{stem}:{obj}", "class_id": members[0]["gt_class_id"],
                    "positives": len(members), "student_positives": int(s_mask.sum()),
                    "assignment_jaccard": intersection / union if union else 1.0,
                    "same_positive_retention": intersection / len(members),
                    **{key: float(np.mean([row[field] for row in members]))
                       for key, field in (
                           ("softmax_KL", "softmax_KL_tau2"),
                           ("sigmoid_KL", "sigmoid_Bernoulli_KL_tau2_sum19"),
                           ("GT_sigmoid_change", "GT_sigmoid_change"),
                           ("top1_agreement", "top1_agreement"),
                           ("IoU_change", "GT_aligned_IoU_change"),
                           ("evidence_change", "foreground_evidence_change"))},
                })
            print(json.dumps({"query": stem, "objects": len(set(d["gt_idx"])),
                              "positives": len(d["pos_i"])}), flush=True)
            trainer.adaptive.audit_predictions = []
            trainer.audit_delta = None
            trainer._last_poison = None
            trainer._semantic_terms = {}
            del result, metrics, delta, detail, g
    if len(gradients) != 16 or not objects or not functional:
        raise RuntimeError("Incomplete fixed-query diagnostic")
    for row in gradients:
        if not all(np.isfinite(value) for key, value in row.items()
                   if key != "query_id"):
            raise FloatingPointError("Non-finite gradient diagnostic")
    by_class = defaultdict(list)
    for row in objects:
        by_class[row["class_id"]].append(row)
    classwise = []
    for cls, name in enumerate(CLASSES):
        if cls == TARGET:
            continue
        obj = by_class[cls]
        classwise.append({
            "class_id": cls, "class": name, "GT_objects": len(obj),
            "teacher_positive_locations": sum(r["positives"] for r in obj),
            "parent_AP50": ap[name]["Safe-Soft05_AP50"],
            "KD_AP50": ap[name]["KD-ObjBal_AP50"],
            "KD_minus_parent_AP50": ap[name]["Delta_AP50"],
            **{key: float(np.mean([r[field] for r in obj])) if obj else ""
               for key, field in (
                   ("softmax_KL_object_mean", "softmax_KL"),
                   ("sigmoid_KL_object_mean", "sigmoid_KL"),
                   ("GT_sigmoid_change_object_mean", "GT_sigmoid_change"),
                   ("top1_agreement_object_mean", "top1_agreement"),
                   ("assignment_jaccard_object_mean", "assignment_jaccard"),
                   ("same_positive_retention_object_mean", "same_positive_retention"),
                   ("GT_aligned_IoU_change_object_mean", "IoU_change"),
                   ("foreground_evidence_change_object_mean", "evidence_change"))},
        })
    output.mkdir(parents=True)
    write_csv(output / "kd_functional_drift.csv", functional)
    write_csv(output / "kd_gradient_comparison.csv", gradients)
    write_csv(output / "kd_classwise_diagnostics.csv", classwise)
    (output / "audit_metadata.json").write_text(json.dumps({
        "config_sha256": digest(args.config),
        "fixed_set_sha256": digest(args.fixed_set),
        "global_params_sha256": digest(args.global_params),
        "parent_cache_sha256": sha256(args.parent_cache),
        "query_count": len(gradients), "object_count": len(objects),
        "teacher_positive_count": len(functional), "lambda_KD": trainer.lambda_kd,
        "tau": TAU, "no_optimizer_steps": True,
        "raw_to_decoded_sigmoid_max_abs_error": max(errors),
        "raw_logit_min": min(lo for lo, _ in extrema),
        "raw_logit_max": max(hi for _, hi in extrema),
        "comparison_source": args.comparison,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "queries": len(gradients),
                      "objects": len(objects), "positives": len(functional),
                      "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
