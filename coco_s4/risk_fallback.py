"""COCO80 NT risk-estimator boundary extension, isolated validation candidate.

Original successful calls are delegated unchanged. Independent candidates never
claim native TAL-positive status. See the versioned mathematical specification.
"""
POLICY='coco80-ntgt-native-tal-then-gt-risk-v1'
REFERENCE_SHA256='e25118c6c8cbd5b2d7b852b0ed9481b8bcbfbd1e3995e268454df5b21cd2d58e'
from ue_framework.methods.tfss_nt_sensitivity import gradients as _original_gradients
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors, bbox2dist


def _with_fallback_gradients(model, image, annotations, imgsz=640, *, coverage_policy='strict_native_TAL', coverage_only=False,
              target_class_id=14, num_classes=20):
    if not 0 <= target_class_id < num_classes:
        raise ValueError('Invalid target/class schema')
    if any(int(a['cls']) != a['cls'] or not 0 <= int(a['cls']) < num_classes for a in annotations):
        raise ValueError('GT class outside configured schema')
    if coverage_policy not in ['strict_native_TAL', 'independent_same_TAL_for_uncovered_valid_GT']:
        raise ValueError('Unknown predeclared CSB supervision coverage policy')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('CSB teacher must be eval/frozen, with input gradient enabled')
    nt = [a for a in annotations if int(a['cls']) != target_class_id]
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
    if ps.shape[-1] != num_classes or not torch.isfinite(ps).all() or not torch.isfinite(pd).all():
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
    keep = [c for c in range(num_classes) if c != target_class_id]
    for j, ann in enumerate(annotations):
        cls = int(ann['cls'])
        if cls == target_class_id:
            continue
        hit = fg[0] & (gi[0] == j)
        native_count = int(hit.sum())
        object_boxes, object_scores = assigned_boxes, scores
        mode = 'native_all_GT_competition'
        if (not bool(hit.any()) or float(object_scores[0,hit].sum())<=0) and coverage_policy == 'independent_same_TAL_for_uncovered_valid_GT':
            # Budget-estimation supervision only: retain native matching for every
            # covered GT. A valid uncovered GT competes alone in the SAME TAL,
            # without stealing another object's matches or using zero sensitivity.
            with torch.no_grad():
                _, object_boxes, object_scores, object_fg, _ = criterion.assigner(
                    ps.detach().sigmoid(), decoded.detach() * stride, anchors * stride,
                    labels[:, j:j+1], gtboxes[:, j:j+1], gtboxes[:, j:j+1].sum(-1, keepdim=True) > 0)
            hit = object_fg[0]
            mode = 'independent_same_TAL_rematch'
        native_evidence_positive_count=int(hit.sum())
        native_evidence_weight_sum=float(object_scores[0,hit].sum().detach())
        independent_risk=False
        selection=None
        if not bool(hit.any()) or native_evidence_weight_sum<=0:
            # These are explicitly risk-estimation candidates, NOT TAL positives.
            hit,selection=risk_candidate_mask(criterion.assigner,anchors*stride,gtboxes[:,j:j+1])
            object_boxes=gtboxes[:,j:j+1].expand(-1,assigned_boxes.shape[1],-1)
            object_scores=torch.zeros_like(scores)
            object_scores[0,hit,cls]=1
            independent_risk=True
            mode='independent_GT_conditioned_risk_candidates'
        weight = object_scores[0, hit].sum(-1).detach()
        if not bool(torch.isfinite(weight).all()) or float(weight.sum())<=0:
            raise ValueError('Independent risk supervision is nonfinite or empty; no zero-risk substitution')
        weight=weight/weight.sum()
        if coverage_only:
            row={'gt_index':j,'cls':cls,'native_positive_count':native_count,
                 'positive_count':0 if independent_risk else int(hit.sum()),'coverage_mode':mode,
                 'coverage_status':'RISK_CANDIDATES_AVAILABLE' if independent_risk else 'COVERED'}
            if independent_risk:row.update(risk_candidate_count=int(hit.sum()),native_evidence_positive_count=native_evidence_positive_count,
                native_evidence_weight_sum=native_evidence_weight_sum,risk_estimator_policy=POLICY,selection=selection,
                risk_gradient_validated=False)
            records.append(row)
            continue
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
        row={'gt_index': j, 'cls': cls, 'positive_count': int(hit.sum()),
             'native_positive_count': native_count, 'coverage_mode': mode, 'coverage_status': 'COVERED',
             'box': float(terms[0].detach()), f'cls_bce{num_classes-1}': float(terms[1].detach()),
             'dfl': float(terms[2].detach()), 'gains': gains.tolist(), 'weighted_loss': float(loss.detach())}
        if independent_risk:
            row.update(positive_count=0,coverage_status='INDEPENDENT_RISK_ESTIMATED',
                risk_candidate_count=int(hit.sum()),native_evidence_positive_count=native_evidence_positive_count,
                native_evidence_weight_sum=native_evidence_weight_sum,risk_estimator_policy=POLICY,selection=selection,
                risk_class_target='hard true NT class; all other NT classes negative',risk_candidate_weights='uniform')
        records.append(row)
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

def risk_candidate_mask(assigner, anchor_pixels, box):
    import hashlib
    if box.shape!=(1,1,4) or not torch.isfinite(box).all():raise ValueError('Invalid actual GT box')
    size=box[0,0,2:]-box[0,0,:2]
    if not bool((size>0).all()):raise ValueError('Degenerate transformed GT; do not invent epsilon supervision')
    valid=box.sum(-1,keepdim=True)>0
    mask=assigner.select_candidates_in_gts(anchor_pixels,box,valid)[0,0].bool()
    rule='pinned native GT geometric candidate rule; not assignment'
    if not bool(mask.any()):
        # Promote the actual endpoints before deriving center/size. Deriving in
        # float32 first can round an off-center GT into a false distance tie.
        box64=box[0,0].double()
        size64=box64[2:]-box64[:2]
        center64=(box64[:2]+box64[2:])/2
        distance=((anchor_pixels.double()-center64)/size64).square().sum(-1)
        if not torch.isfinite(distance).all():raise FloatingPointError('Nonfinite GT-normalized candidate distance')
        mask=torch.zeros(anchor_pixels.shape[0],dtype=torch.bool,device=anchor_pixels.device)
        mask[int(distance.argmin())]=True
        rule='single nearest GT-size-normalized center; tie native anchor index'
    indices=mask.nonzero().flatten().detach().cpu().numpy().astype('<i8')
    details=dict(rule=rule,count=len(indices),anchor_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        indices=indices.tolist() if len(indices)<=32 else None,GT_xyxy=box[0,0].detach().cpu().tolist())
    return mask,details


def gradients(model,image,annotations,imgsz=640,*,coverage_policy='strict_native_TAL',coverage_only=False,
              target_class_id=14,num_classes=20):
    kwargs=dict(imgsz=imgsz,coverage_policy=coverage_policy,coverage_only=coverage_only,
                target_class_id=target_class_id,num_classes=num_classes)
    if num_classes!=80 or target_class_id!=0 or coverage_policy!='independent_same_TAL_for_uncovered_valid_GT':
        return _original_gradients(model,image,annotations,**kwargs)
    for ann in annotations:
        box=np.asarray(ann['bbox'],dtype=np.float64)
        if (box.shape!=(4,) or not np.isfinite(box).all() or not np.all(box[2:]>0)
                or np.any(box<0) or np.any(box>1)):
            raise ValueError('Invalid actual GT bbox: no independent risk fallback for bad data')
    if not coverage_only:
        try:
            # Exact original output/records/gradient and one original forward on covered images.
            return _original_gradients(model,image,annotations,**kwargs)
        except ValueError as exc:
            if not str(exc).startswith(('NO_VALID_POSITIVE:','NO_FOREGROUND_EVIDENCE:')):raise
    return _with_fallback_gradients(model,image,annotations,**kwargs)


def bind(trainer,manifest,root):
    """Bind only an explicit isolated COCO80 candidate and include it in cache identity."""
    from pathlib import Path
    from .runtime import sha,load
    policy=manifest.get('risk_estimator_policy')
    if policy is None:return {}
    if policy!=POLICY or getattr(trainer,'num_classes',None)!=80 or getattr(trainer,'target_class_id',None)!=0:
        raise ValueError('Independent NT-risk policy is restricted to explicit COCO80/Person0')
    if manifest.get('validation_only') is not True:
        from .runtime import assert_formal_launch_authorized
        assert_formal_launch_authorized(manifest)
        if manifest.get('risk_estimator_policy')!=POLICY:
            raise ValueError('Formal risk extension requires its independent protocol')
    root=Path(root);name='coco_s4/risk_fallback.py';spec='coco_s4/configs/RISK_FALLBACK_PROTOCOL.json';adapter='coco_s4/risk_fallback_cache.py'
    if sha(root/name)!=manifest['files'][name] or sha(root/spec)!=manifest['files'][spec]:raise ValueError('Risk policy source/spec drift')
    if sha(root/'ue_framework/methods/tfss_nt_sensitivity.py')!=REFERENCE_SHA256:raise ValueError('Original covered estimator changed')
    if load(root/spec)['protocol_version']!=POLICY:raise ValueError('Independent risk protocol version mismatch')
    trainer.nt_risk_estimator=gradients;trainer.risk_estimator_policy=POLICY
    return dict(risk_estimator_policy=POLICY,risk_estimator_code_sha256=sha(root/name),
                risk_policy_spec_sha256=sha(root/spec),risk_cache_adapter_sha256=sha(root/adapter),normal_native_estimator_sha256=REFERENCE_SHA256)
