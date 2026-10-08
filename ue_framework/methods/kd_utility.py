"""Teacher-aligned non-target classification KD for the Safe-Soft05 ablation.

The existing one-step adaptive update, target evidence, and feature retention
remain unchanged.  Only raw classification logits at frozen-teacher TAL
positives on the existing clean query are distilled.
"""

import torch
from torch.func import functional_call
from ultralytics.utils.tal import make_anchors

from .adaptive_learner import AdaptiveLearner
from .legacy_kproto_ret import non_target_retention_loss


FOCUS_CLASSES = {1: "bicycle", 11: "dog", 12: "horse"}


def object_balanced_mean(per_positive, pair_ids):
    """Mean within each assigned GT object, then mean across valid objects."""
    if per_positive.ndim != 1 or pair_ids.shape != per_positive.shape:
        raise ValueError("KD values and teacher GT pair IDs must align")
    if per_positive.numel() == 0:
        return per_positive.sum() * 0.0, pair_ids, per_positive, pair_ids
    unique_ids, inverse, counts = torch.unique(
        pair_ids, sorted=True, return_inverse=True, return_counts=True)
    object_sums = torch.zeros(unique_ids.numel(), device=per_positive.device,
                              dtype=per_positive.dtype).scatter_add(0, inverse, per_positive)
    object_means = object_sums / counts.to(per_positive.dtype)
    return object_means.mean(), unique_ids, object_means, counts


def teacher_aligned_non_target_kd(criterion, teacher_raw, student_raw, batch,
                                  target_class_id=14, tau=2.0,
                                  collect_object_diagnostics=False):
    """Return object-balanced KL(T||S) at frozen-teacher TAL positives."""
    if tau != 2.0 or target_class_id != 14:
        raise ValueError("Safe-Soft05-KDcls has fixed tau=2 and person class=14")
    if set(teacher_raw) != {"boxes", "scores", "feats"} or set(student_raw) != set(teacher_raw):
        raise ValueError("Expected the installed YOLO one-to-many raw head")
    t_scores = teacher_raw["scores"].permute(0, 2, 1).contiguous().detach()
    s_scores = student_raw["scores"].permute(0, 2, 1).contiguous()
    if t_scores.shape != s_scores.shape or t_scores.shape[-1] != 20:
        raise ValueError("Teacher/student VOC20 raw classification locations differ")
    if len(teacher_raw["feats"]) != len(student_raw["feats"]):
        raise ValueError("Teacher/student feature pyramid levels differ")
    if any(a.shape[-2:] != b.shape[-2:] for a, b in zip(teacher_raw["feats"], student_raw["feats"])):
        raise ValueError("Teacher/student feature-map location geometry differs")

    batch_size = t_scores.shape[0]
    zero = s_scores.sum() * 0.0
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    size = torch.tensor(teacher_raw["feats"][0].shape[2:], device=t_scores.device,
                        dtype=t_scores.dtype) * criterion.stride[0]
    targets = criterion.preprocess(targets.to(t_scores.device), batch_size,
                                   scale_tensor=size[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_nt = gt_bboxes.sum(2, keepdim=True).gt(0) & gt_labels.long().ne(target_class_id)
    non_target_count = mask_nt.sum()
    if not bool(mask_nt.any()):
        return zero, _empty_diagnostics(zero, non_target_count), []

    anchor_points, stride_tensor = make_anchors(teacher_raw["feats"], criterion.stride, 0.5)
    t_dist = teacher_raw["boxes"].permute(0, 2, 1).contiguous().detach()
    t_boxes = criterion.bbox_decode(anchor_points, t_dist).detach()
    # Mask person GT *before* TAL.  No student prediction participates in assignment.
    _, _, _, fg_mask, matched_gt = criterion.assigner(
        t_scores.sigmoid(), (t_boxes * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_nt,
    )
    positive_b, positive_i = torch.nonzero(fg_mask, as_tuple=True)
    if positive_b.numel() == 0:
        return zero, _empty_diagnostics(zero, non_target_count), []
    gt_index = matched_gt[positive_b, positive_i].long()
    matched_class = gt_labels[positive_b, gt_index, 0].long()
    if bool((matched_class == target_class_id).any()) or bool((matched_class < 0).any()):
        raise RuntimeError("Target or invalid GT reached non-target KD")
    # The same (batch, location) index extracts both logits; person logit is absent.
    keep = torch.arange(t_scores.shape[-1], device=t_scores.device) != target_class_id
    # The virtual update can make T and S nearly identical.  FP32 KL may then
    # become slightly negative through cancellation; only these selected logits
    # use FP64, leaving assignment and the mathematical objective unchanged.
    teacher = t_scores[positive_b, positive_i][:, keep].double()
    student = s_scores[positive_b, positive_i][:, keep].double()
    log_teacher = (teacher / tau).log_softmax(-1)
    log_student = (student / tau).log_softmax(-1)
    p_teacher = log_teacher.exp()
    per_positive = ((p_teacher * (log_teacher - log_student)).sum(-1) * tau**2).clamp_min(0.0)
    gt_pair_id = positive_b * gt_labels.shape[1] + gt_index
    balanced, object_ids, object_means, object_counts = object_balanced_mean(
        per_positive, gt_pair_id)
    kd = balanced.float()
    if not bool(torch.isfinite(kd)):
        raise FloatingPointError("Non-finite non-target classification KD")
    p_student = log_student.exp()
    diagnostics = {
        "kd_positive_count": kd.new_tensor(float(positive_b.numel())),
        "kd_gt_object_count": object_ids.numel() * kd.new_ones(()),
        "non_target_gt_count": non_target_count.to(kd.dtype),
        "valid_kd_object_count": object_ids.numel() * kd.new_ones(()),
        "zero_positive_object_count": (non_target_count - object_ids.numel()).to(kd.dtype),
        "kd_teacher_entropy": -(p_teacher * log_teacher).sum(-1).mean().float(),
        "kd_student_entropy": -(p_student * log_student).sum(-1).mean().float(),
        "kd_kl_mean": kd,
        "teacher_student_top1_agreement": (teacher.argmax(-1) == student.argmax(-1)).float().mean(),
        "kd_target_gt_count": kd.new_zeros(()),
        "kd_person_logit_in_distribution": kd.new_zeros(()),
        "kd_teacher_student_index_identical": kd.new_ones(()),
    }
    class_means = []
    object_classes = gt_labels[object_ids // gt_labels.shape[1],
                               object_ids % gt_labels.shape[1], 0].long()
    for class_id, name in FOCUS_CLASSES.items():
        selected = object_classes == class_id
        diagnostics["KD_" + name] = object_means[selected].mean().float() if bool(selected.any()) else kd.new_zeros(())
        diagnostics["KD_count_" + name] = selected.sum().to(kd.dtype)
    for class_id in object_classes.unique().tolist():
        selected = object_classes == class_id
        class_means.append((int(class_id), float(object_means[selected].detach().mean())))
    class_means.sort(key=lambda item: (-item[1], item[0]))
    for rank in range(3):
        class_id, value = class_means[rank] if rank < len(class_means) else (-1, 0.0)
        diagnostics[f"kd_worst_class_{rank + 1}"] = kd.new_tensor(float(class_id))
        diagnostics[f"kd_worst_class_loss_{rank + 1}"] = kd.new_tensor(value)
    object_rows = []
    if collect_object_diagnostics:
        for object_id, class_id, count, value in zip(
                object_ids.detach().cpu().tolist(), object_classes.detach().cpu().tolist(),
                object_counts.detach().cpu().tolist(), object_means.detach().cpu().tolist()):
            object_rows.append({
                "class_label": int(class_id),
                "batch_index": int(object_id // gt_labels.shape[1]),
                "object_index": int(object_id % gt_labels.shape[1]),
                "positive_count": int(count), "object_KD": float(value),
            })
    return kd, diagnostics, object_rows


def _empty_diagnostics(zero, non_target_count):
    keys = ("kd_positive_count", "kd_gt_object_count", "kd_teacher_entropy",
            "kd_student_entropy", "kd_kl_mean", "teacher_student_top1_agreement",
            "kd_target_gt_count", "kd_person_logit_in_distribution",
            "kd_teacher_student_index_identical", "valid_kd_object_count")
    result = {key: zero.detach().clone() for key in keys}
    result["non_target_gt_count"] = non_target_count.to(zero.dtype)
    result["zero_positive_object_count"] = non_target_count.to(zero.dtype)
    for name in FOCUS_CLASSES.values():
        result["KD_" + name] = zero.detach().clone()
        result["KD_count_" + name] = zero.detach().clone()
    for rank in range(1, 4):
        result[f"kd_worst_class_{rank}"] = zero.new_tensor(-1.0)
        result[f"kd_worst_class_loss_{rank}"] = zero.detach().clone()
    return result


class KDAdaptiveLearner(AdaptiveLearner):
    """The existing virtual update with an additional clean-query raw-logit read."""

    def _predict_with_raw(self, images, params, buffers, *, grad):
        capture = {}

        def hook(_module, _inputs, output):
            capture["feature"] = output

        handle = self.feature_module.register_forward_hook(hook)
        try:
            with self._mode(training=False):
                if grad:
                    result = functional_call(self.model, (params, buffers), (images,), strict=True)
                else:
                    with torch.no_grad():
                        result = functional_call(self.model, (params, buffers), (images,), strict=True)
        finally:
            handle.remove()
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("Expected YOLO eval (decoded, raw_head) output")
        decoded, raw = result
        if decoded.ndim != 3 or decoded.shape[1] < 4 + self.target_class_id + 1:
            raise ValueError("Expected decoded YOLO predictions [B, 4+C, N]")
        return decoded, capture["feature"], raw

    def losses(self, clean, poison, items, non_target_masks, pool):
        poison_batch = self._batch(poison, items)
        clean_batch = self._batch(clean, items)
        params, buffers = self._initial_state()
        with self._mode(training=True):
            inner_loss, _ = functional_call(self.model, (params, buffers), (poison_batch,), strict=True)
        inner_loss = inner_loss.sum()
        if not torch.isfinite(inner_loss):
            raise FloatingPointError("Non-finite adaptive detection loss")
        gradients = torch.autograd.grad(
            inner_loss, [params[name] for name in self.trainable],
            create_graph=True, allow_unused=False,
        )
        updated = {**params, **{
            name: params[name] - self.inner_lr * gradient
            for name, gradient in zip(self.trainable, gradients)
        }}
        update_norm = torch.stack([gradient.detach().norm() for gradient in gradients]).norm() * self.inner_lr
        before_pred, before_feat, teacher_raw = self._predict_with_raw(clean_batch["img"], params, buffers, grad=False)
        after_pred, after_feat, student_raw = self._predict_with_raw(clean_batch["img"], updated, buffers, grad=True)
        with torch.no_grad():
            poison_pred, _, _ = self._predict_with_raw(poison_batch["img"], updated, buffers, grad=False)
        before_score = self._target_evidence(before_pred, clean_batch)
        after_score = self._target_evidence(after_pred, clean_batch)
        poison_score = self._target_evidence(poison_pred, clean_batch)
        before_nt, after_nt = [], []
        for index, masks in enumerate(non_target_masks):
            before_nt.extend(pool(before_feat[index:index + 1], masks))
            after_nt.extend(pool(after_feat[index:index + 1], masks))
        if before_nt:
            nt_loss = non_target_retention_loss(torch.stack(after_nt), torch.stack(before_nt))
        else:
            nt_loss = after_feat.sum() * 0.0
        kd, kd_metrics, self.kd_object_rows = teacher_aligned_non_target_kd(
            self.model.criterion, teacher_raw, student_raw, clean_batch,
            target_class_id=self.target_class_id, tau=2.0,
            collect_object_diagnostics=getattr(self, "collect_kd_object_diagnostics", False),
        )
        metrics = {
            "L_adapt_target": after_score,
            "L_adapt_nt": nt_loss,
            "target_score_clean_before": before_score,
            "target_score_clean_after": after_score,
            "target_score_poison_after": poison_score,
            "cue_gap": poison_score - after_score,
            "cue_ratio": poison_score / (after_score + 1e-6),
            "inner_detection_loss": inner_loss,
            "inner_update_norm": update_norm,
            "L_KD_cls_nt_raw": kd,
            **kd_metrics,
        }
        return after_score, nt_loss, metrics
