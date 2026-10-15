"""Pinned YOLOv8 CNLP adapter; train-only, GT-positive object/class reduction.

Shared detached TAL is computed from the clean-updated query detector once.
Missing valid NT GT supervision is a gate failure, never silently zero loss.
This matching/reduction is a DRAFT to audit on real train data before locking.
"""
import copy

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, make_anchors

from .counterfactual_utility import CounterfactualSGD


class YOLOCounterfactual:
    def __init__(self, adaptive, config):
        # Never change Parent's Adaptive model/criterion or reference checkpoint.
        self.model = copy.deepcopy(adaptive.model)
        self.config = config
        self.criterion = v8DetectionLoss(self.model)
        self.engine = CounterfactualSGD(self.model, config, adaptive.trainable)
        if self.criterion.nc != config.num_classes or self.criterion.bbox_loss.dfl_loss is None:
            raise ValueError('Requires matching VOC20 YOLOv8 head with DFL')

    @staticmethod
    def support_loss(forward, batch):
        # Native all-GT training criterion, exactly the Parent's virtual learner.
        loss, _ = forward(batch)
        return loss.sum()

    def _raw(self, output):
        raw = output[1] if isinstance(output, tuple) else output
        if not isinstance(raw, dict) or not {'scores', 'boxes', 'feats'} <= set(raw):
            raise ValueError('Expected pinned Ultralytics 8.4.150 raw head dict')
        scores = raw['scores'].permute(0, 2, 1).contiguous()
        distribution = raw['boxes'].permute(0, 2, 1).contiguous()
        if scores.shape[-1] != self.config.num_classes:
            raise ValueError('Class count changed')
        if not torch.isfinite(scores).all() or not torch.isfinite(distribution).all():
            raise FloatingPointError('Nonfinite RAW logits or DFL logits')
        anchors, stride = make_anchors(raw['feats'], self.criterion.stride, .5)
        decoded = self.criterion.bbox_decode(anchors, distribution)
        return scores, distribution, decoded, anchors, stride

    def query_loss(self, clean_output, poison_output, batch):
        clean = self._raw(clean_output)
        poison = self._raw(poison_output)
        cs, _, cb, anchors, stride = clean
        ps, pd, pb, pa, pst = poison
        if cs.shape != ps.shape or not torch.equal(anchors, pa) or not torch.equal(stride, pst):
            raise ValueError('Query anchor geometry differs across virtual paths')
        size = batch['img'].shape[-2:]
        targets = torch.cat((batch['batch_idx'].reshape(-1, 1).to(cs),
                             batch['cls'].reshape(-1, 1).to(cs), batch['bboxes'].to(cs)), 1)
        if targets.numel() and (bool((targets[:, 1] < 0).any())
                               or bool((targets[:, 1] >= self.config.num_classes).any())
                               or bool((targets[:, 4:] <= 0).any())):
            raise ValueError('Invalid query GT')
        targets = self.criterion.preprocess(targets, cs.shape[0],
                    scale_tensor=cs.new_tensor([size[1], size[0], size[1], size[0]]))
        labels, boxes = targets.split((1, 4), 2)
        valid = boxes.sum(-1, keepdim=True) > 0
        with torch.no_grad():
            # Sigmoid exactly once on RAW logits, all original GT compete.
            _, truth, scores, foreground, gt_idx = self.criterion.assigner(
                cs.sigmoid(), cb * stride, anchors * stride, labels, boxes, valid)
        keep = [c for c in range(self.config.num_classes) if c != self.config.target_class_id]
        gain = cs.new_tensor([self.criterion.hyp.box, self.criterion.hyp.cls, self.criterion.hyp.dfl])
        regmax = self.criterion.bbox_loss.dfl_loss.reg_max
        truth_grid = truth / stride
        distance = bbox2dist(anchors, truth_grid, regmax - 1)
        object_losses = ({}, {})
        records = []
        for image_idx in range(cs.shape[0]):
            for gt in range(labels.shape[1]):
                if not bool(valid[image_idx, gt]) or int(labels[image_idx, gt, 0]) == self.config.target_class_id:
                    continue
                cls = int(labels[image_idx, gt, 0])
                hit = foreground[image_idx] & (gt_idx[image_idx] == gt)
                native_count = int(hit.sum())
                object_scores = scores[image_idx]
                object_truth = truth_grid[image_idx]
                object_distance = distance[image_idx]
                mode = 'native_all_GT_competition'
                if not bool(hit.any()) and self.config.matching_policy == 'native_shared_gt_coverage_v2':
                    # Explicit GT-conditioned query design adjustment: only an
                    # uncovered valid NT GT competes alone in the SAME native TAL.
                    # Clean-updated logits define ONE detached assignment shared
                    # by clean/poison/initial paths; covered objects are unchanged.
                    with torch.no_grad():
                        _, recovered_truth, recovered_scores, recovered_fg, _ = self.criterion.assigner(
                            cs[image_idx:image_idx+1].sigmoid(), cb[image_idx:image_idx+1] * stride,
                            anchors * stride, labels[image_idx:image_idx+1,gt:gt+1],
                            boxes[image_idx:image_idx+1,gt:gt+1], valid[image_idx:image_idx+1,gt:gt+1])
                    hit = recovered_fg[0]
                    object_scores = recovered_scores[0]
                    object_truth = recovered_truth[0] / stride
                    object_distance = bbox2dist(anchors, object_truth.unsqueeze(0), regmax - 1)[0]
                    mode = 'independent_same_TAL_for_uncovered_valid_GT'
                if not bool(hit.any()):
                    raise ValueError(f'CNLP_NO_VALID_POSITIVE: image={batch["image_ids"][image_idx]},GT={gt},class={cls}')
                weights = object_scores[hit].sum(-1).detach()
                if not torch.isfinite(weights).all() or float(weights.sum()) <= 0:
                    raise ValueError('CNLP_NO_FOREGROUND_EVIDENCE')
                weights = weights / weights.sum()
                components = []
                for path_idx, (logits, dist, decoded, _, _) in enumerate((clean, poison)):
                    bce = F.binary_cross_entropy_with_logits(logits[image_idx, hit][:, keep],
                                    object_scores[hit][:, keep], reduction='none').sum(-1)
                    ciou = 1 - bbox_iou(decoded[image_idx, hit], object_truth[hit],
                                       xywh=False, CIoU=True).flatten()
                    dfl = self.criterion.bbox_loss.dfl_loss(dist[image_idx, hit].reshape(-1, regmax),
                                                          object_distance[hit]).flatten()
                    terms = torch.stack([(x * weights).sum() for x in (ciou, bce, dfl)])
                    value = (terms * gain).sum()
                    if not torch.isfinite(value):
                        raise FloatingPointError('Nonfinite GT-conditioned query loss')
                    object_losses[path_idx].setdefault(cls, []).append(value)
                    components.append(terms.detach())
                records.append(dict(image_id=batch['image_ids'][image_idx], gt_index=gt,
                                    cls=cls, positive_count=int(hit.sum()),native_positive_count=native_count,coverage_mode=mode,
                                    clean_components=components[0], poison_components=components[1]))
        c_loss, p_loss = [{c: torch.stack(rows).mean() for c, rows in path.items()}
                          for path in object_losses]
        return c_loss, p_loss, dict(objects=records, gains=gain.detach(),
                                  matching=self.config.matching_policy,
                                  reduction='softscore_positive_mean_then_object_mean_per_class',
                                  legal_no_nt_gt=not bool(c_loss))

    def evaluate(self, clean, poison, query):
        return self.engine.evaluate(clean, poison, query, self.support_loss, self.query_loss)
