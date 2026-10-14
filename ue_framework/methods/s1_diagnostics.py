"""Frozen-teacher NT confidence/localization proxies; never part of optimization."""
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.metrics import bbox_iou
from ultralytics.cfg import get_cfg


@torch.no_grad()
def nt_drift(model, clean, poison, annotations, imgsz=640):
    nt = [i for i, a in enumerate(annotations) if int(a['cls']) != 14]
    if not nt: return []
    device = next(model.parameters()).device
    h, w = clean.shape[:2]; ratio = min(imgsz / h, imgsz / w)
    nh, nw = round(h * ratio), round(w * ratio)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    def predict(image):
        x = torch.from_numpy(image).permute(2, 0, 1)[None].to(device).float()
        x = F.pad(F.interpolate(x, (nh, nw), mode='bilinear', align_corners=False),
                  (left, imgsz - nw - left, top, imgsz - nh - top), value=.447)
        out = model(x)
        raw = out[1] if isinstance(out, tuple) else out
        if not torch.isfinite(raw['scores']).all() or not torch.isfinite(raw['boxes']).all():
            raise FloatingPointError('NT diagnostic output nonfinite')
        return raw
    raw = predict(clean); pr = predict(poison)
    if isinstance(model.args, dict):
        defaults = vars(get_cfg())
        model.args = get_cfg(overrides={k: v for k, v in model.args.items() if k in defaults})
    c = v8DetectionLoss(model)
    a, stride = make_anchors(raw['feats'], c.stride, .5)
    z, pz = raw['scores'].permute(0, 2, 1), pr['scores'].permute(0, 2, 1)
    boxes = c.bbox_decode(a, raw['boxes'].permute(0, 2, 1)) * stride
    pboxes = c.bbox_decode(a, pr['boxes'].permute(0, 2, 1)) * stride
    classes = z.new_tensor([q['cls'] for q in annotations])
    gt = z.new_tensor([q['bbox'] for q in annotations]) * z.new_tensor([w * ratio, h * ratio, w * ratio, h * ratio])
    gt[:, 0] += left; gt[:, 1] += top
    targets = c.preprocess(torch.cat((torch.zeros_like(classes[:, None]), classes[:, None], gt / imgsz), 1),
                           1, z.new_tensor([imgsz] * 4))
    labels, truth = targets.split((1, 4), 2)
    _, _, _, fg, gi = c.assigner(z.sigmoid(), boxes, a * stride, labels, truth, truth.sum(-1, keepdim=True) > 0)
    _, _, _, pfg, pgi = c.assigner(pz.sigmoid(), pboxes, a * stride, labels, truth, truth.sum(-1, keepdim=True) > 0)
    rows = []
    for j in nt:
        cls = int(classes[j]); hit = fg[0] & (gi[0] == j)
        if not hit.any(): raise ValueError('NT diagnostic clean coverage missing')
        phit = pfg[0] & (pgi[0] == j)
        t = truth[0, j].expand(int(hit.sum()), 4)
        iou = bbox_iou(boxes[0, hit], t, xywh=False).flatten()
        piou = bbox_iou(pboxes[0, hit], t, xywh=False).flatten()
        prob, pprob = z[0, hit, cls].sigmoid(), pz[0, hit, cls].sigmoid()
        rows.append(dict(cls=cls, gt_index=j, clean_positive_count=int(hit.sum()), poison_positive_count=int(phit.sum()),
            sigmoid_confidence_change=float((pprob - prob).mean()), clean_gt_sigmoid=float(prob.mean()),
            poison_gt_sigmoid=float(pprob.mean()), mean_iou_change=float((piou-iou).mean()),
            clean_mean_iou=float(iou.mean()), poison_mean_iou=float(piou.mean()),
            class_top1_agreement=float((z[0, hit].argmax(-1) == pz[0, hit].argmax(-1)).float().mean()),
            assignment_jaccard=float((hit & phit).sum() / (hit | phit).sum().clamp_min(1)),
            foreground_evidence_change=float((pprob.sqrt() * piou.clamp_min(0).pow(6) - prob.sqrt() * iou.clamp_min(0).pow(6)).mean())))
    return rows
