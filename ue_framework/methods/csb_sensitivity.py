"""Classwise NT-GT positive detection sensitivity, used ONLY for frozen budgets.

All original GT compete in detached native TAL. Per-GT positive means precede
class means. Native box/cls/DFL gains are retained; background anchors and the
Person logit are excluded. This positive-object reduction is disclosed and is
not the original all-anchor YOLO training loss or an added S1 optimization loss.
"""
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors, bbox2dist


def gradients(model, image, annotations, imgsz=640, *, coverage_policy='strict_native_TAL', coverage_only=False):
    if coverage_policy not in ['strict_native_TAL', 'independent_same_TAL_for_uncovered_valid_GT']:
        raise ValueError('Unknown predeclared CSB supervision coverage policy')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('CSB teacher must be eval/frozen, with input gradient enabled')
    nt = [a for a in annotations if int(a['cls']) != 14]
    if not nt:
        return {}, []
    device = next(model.parameters()).device
    x = torch.from_numpy(image).permute(2, 0, 1)[None].to(device).float().requires_grad_(not coverage_only)
    h, w = image.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    nh, nw = round(h * ratio), round(w * ratio)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    lb = F.pad(F.interpolate(x, (nh, nw), mode='bilinear', align_corners=False),
               (left, imgsz - nw - left, top, imgsz - nh - top), value=.447)
    output = model(lb)
    raw = output[1] if isinstance(output, tuple) else output
    if not isinstance(raw, dict) or not {'scores', 'boxes', 'feats'} <= set(raw):
        raise ValueError('Expected pinned Ultralytics8.4.150 raw head outputs')
    if isinstance(model.args, dict):
        defaults = vars(get_cfg())
        model.args = get_cfg(overrides={k: v for k, v in model.args.items() if k in defaults})
    criterion = v8DetectionLoss(model)
    pd = raw['boxes'].permute(0, 2, 1).contiguous()
    ps = raw['scores'].permute(0, 2, 1).contiguous()
    if ps.shape[-1] != 20 or not torch.isfinite(ps).all() or not torch.isfinite(pd).all():
        raise ValueError('Invalid raw logits/distribution; do not sigmoid twice')
    anchors, stride = make_anchors(raw['feats'], criterion.stride, .5)
    decoded = criterion.bbox_decode(anchors, pd)
    classes = torch.tensor([a['cls'] for a in annotations], device=device, dtype=torch.float32)
    boxes = torch.tensor([a['bbox'] for a in annotations], device=device, dtype=torch.float32)
    boxes = boxes * boxes.new_tensor([w * ratio, h * ratio, w * ratio, h * ratio])
    boxes[:, 0] += left; boxes[:, 1] += top
    targets = torch.cat((torch.zeros_like(classes[:, None]), classes[:, None], boxes / imgsz), 1)
    targets = criterion.preprocess(targets, 1, scale_tensor=ps.new_tensor([imgsz] * 4))
    labels, gtboxes = targets.split((1, 4), 2)
    with torch.no_grad():
        _, assigned_boxes, scores, fg, gi = criterion.assigner(ps.detach().sigmoid(),
            decoded.detach() * stride, anchors * stride, labels, gtboxes, gtboxes.sum(-1, keepdim=True) > 0)
    losses = {}
    records = []
    keep = [c for c in range(20) if c != 14]
    for j, ann in enumerate(annotations):
        cls = int(ann['cls'])
        if cls == 14:
            continue
        hit = fg[0] & (gi[0] == j)
        native_count = int(hit.sum())
        object_boxes, object_scores = assigned_boxes, scores
        mode = 'native_all_GT_competition'
        if not bool(hit.any()) and coverage_policy == 'independent_same_TAL_for_uncovered_valid_GT':
            # Budget-estimation supervision only: retain native matching for every
            # covered GT. A valid uncovered GT competes alone in the SAME TAL,
            # without stealing another object's matches or using zero sensitivity.
            with torch.no_grad():
                _, object_boxes, object_scores, object_fg, _ = criterion.assigner(
                    ps.detach().sigmoid(), decoded.detach() * stride, anchors * stride,
                    labels[:, j:j+1], gtboxes[:, j:j+1], gtboxes[:, j:j+1].sum(-1, keepdim=True) > 0)
            hit = object_fg[0]
            mode = 'independent_same_TAL_rematch'
        if not bool(hit.any()):
            if coverage_only:
                records.append({'gt_index': j, 'cls': cls, 'native_positive_count': native_count,
                    'positive_count': 0, 'coverage_mode': mode, 'coverage_status': 'NO_VALID_POSITIVE'})
                continue
            raise ValueError(f'NO_VALID_POSITIVE: GT index={j}, class={cls}')
        weight = object_scores[0, hit].sum(-1).detach()
        if not bool(torch.isfinite(weight).all()) or float(weight.sum()) <= 0:
            if coverage_only:
                records.append({'gt_index': j, 'cls': cls, 'native_positive_count': native_count,
                    'positive_count': int(hit.sum()), 'coverage_mode': mode, 'coverage_status': 'NO_FOREGROUND_EVIDENCE'})
                continue
            raise ValueError(f'NO_FOREGROUND_EVIDENCE: GT={j}, class={cls}')
        if coverage_only:
            records.append({'gt_index': j, 'cls': cls, 'native_positive_count': native_count,
                'positive_count': int(hit.sum()), 'coverage_mode': mode, 'coverage_status': 'COVERED'})
            continue
        weight = weight / weight.sum()
        bce = F.binary_cross_entropy_with_logits(ps[0, hit][:, keep], object_scores[0, hit][:, keep], reduction='none').sum(-1)
        truth = object_boxes / stride
        ciou = 1 - bbox_iou(decoded[0, hit], truth[0, hit], xywh=False, CIoU=True).flatten()
        if criterion.bbox_loss.dfl_loss is None:
            raise ValueError('Parent YOLOv8 reference requires DFL')
        reg_max = criterion.bbox_loss.dfl_loss.reg_max
        ltrb = bbox2dist(anchors, truth, reg_max - 1)
        dfl = criterion.bbox_loss.dfl_loss(pd[0, hit].reshape(-1, reg_max), ltrb[0, hit]).flatten()
        terms = torch.stack(((ciou * weight).sum(), (bce * weight).sum(), (dfl * weight).sum()))
        gains = terms.new_tensor([criterion.hyp.box, criterion.hyp.cls, criterion.hyp.dfl])
        loss = (terms * gains).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite NT detection loss')
        losses.setdefault(cls, []).append(loss)
        records.append({'gt_index': j, 'cls': cls, 'positive_count': int(hit.sum()),
                        'native_positive_count': native_count, 'coverage_mode': mode, 'coverage_status': 'COVERED',
                        'box': float(terms[0].detach()), 'cls_bce19': float(terms[1].detach()),
                        'dfl': float(terms[2].detach()), 'gains': gains.tolist(),
                        'weighted_loss': float(loss.detach())})
    maps = {}
    for cls, rows in sorted(losses.items()):
        loss = torch.stack(rows).mean()
        g = torch.autograd.grad(loss, x, retain_graph=True, allow_unused=False)[0]
        if not torch.isfinite(g).all() or float(g.norm()) <= 0:
            raise FloatingPointError(f'NT input gradient failure: class={cls}')
        maps[cls] = g[0].norm(dim=0).detach().cpu().numpy().astype(np.float32)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Teacher accumulated parameter gradients')
    return maps, records
