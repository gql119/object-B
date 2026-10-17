"""Clean TRAIN maps on original image coordinates; no noise/model updates."""
import torch
import torch.nn.functional as F
import time

from .legacy_kproto_ret import _rectangle
from .tfss_spatial import normalize_map, target_union
from .tfss_nt_sensitivity import gradients


def build_maps(trainer, items):
    """One per-object input gradient, max-merge Persons, equal-class NT risk."""
    device = trainer.device
    pad_h = (max(im.shape[0] for im, _, _ in items) + 31) // 32 * 32
    pad_w = (max(im.shape[1] for im, _, _ in items) + 31) // 32 * 32
    maps, records = [], []
    for image, annotations, support in items:
        started = time.perf_counter()
        for ann in annotations:
            box = torch.as_tensor(ann['bbox'])
            if (int(ann['cls']) != ann['cls'] or not 0 <= int(ann['cls']) < 20
                or box.shape != (4,) or not torch.isfinite(box).all()
                or bool((box < 0).any()) or bool((box > 1).any())
                or not bool((box[2:] > 0).all())):
                raise ValueError('Invalid real GT; no skip/zero sensitivity fallback')
        x = torch.from_numpy(image).permute(2, 0, 1)[None].to(device).float()
        x.requires_grad_(True)
        h, w = image.shape[:2]
        padded = F.pad(x, (0, pad_w-w, 0, pad_h-h), value=.5)
        feature = trainer._feature(padded, grad=True)
        target_masks, _, _ = trainer._masks(annotations, support, (h, w), feature, (pad_h, pad_w))
        anns = [a for a in annotations if int(a['cls']) == 14]
        if len(target_masks) != len(anns) or not anns:
            raise ValueError('Person mask/GT mismatch')
        object_maps, object_records = [], []
        target = torch.as_tensor(support > 0, device=device)
        invalid_q_domain = torch.zeros_like(target)
        for ann, feature_mask in zip(anns, target_masks):
            pixel_mask = _rectangle((h, w), ann['bbox'])[0, 0].to(device).bool()
            if not bool(pixel_mask.any()):
                raise ValueError('Empty real Person pixel domain; no zero fallback')
            feature_mass = float(feature_mask.sum())
            if not bool(torch.isfinite(feature_mask).all()) or feature_mass < 0:
                raise ValueError('Invalid canonical Person feature mask')
            if feature_mass < 1:
                # A valid pixel GT can be too small for canonical S0 pooling
                # (min_pixels=1). It has undefined Qt, not an invalid label.
                # Apply the ALREADY LOCKED invalid-Qt strength-preservation
                # policy; retain this GT and explicitly report no attribution.
                q = pixel_mask.to(dtype=torch.float32)
                invalid_q_domain |= pixel_mask
                object_maps.append((q, pixel_mask))
                object_records.append({'valid': False,
                    'reason': 'canonical_P5_pool_undercoverage',
                    'q10': None, 'q90': None, 'raw_norm': None,
                    'feature_mask_sum': feature_mass, 'bbox': ann['bbox']})
                continue
            selected = feature[:, trainer.channels]
            score = (selected.square() * feature_mask).sum() / (feature_mask.sum() * len(trainer.channels))
            grad = torch.autograd.grad(score, x, retain_graph=True)[0][0]
            if not torch.isfinite(grad).all():
                raise FloatingPointError('Nonfinite TSFL input gradient')
            q, evidence = normalize_map(grad.norm(dim=0), pixel_mask, constant_policy='preserve')
            if not evidence['valid']:
                invalid_q_domain |= pixel_mask
            object_maps.append((q, pixel_mask))
            object_records.append(dict(evidence, bbox=ann['bbox'], raw_norm=float(grad.norm())))
        q = target_union(object_maps, target)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        target_seconds = time.perf_counter()-started
        risk_started = time.perf_counter()
        # Original eval/frozen teacher, real all-GT native TAL; only uncovered
        # valid NTGT uses the SAME TAL alone. Invalid/no evidence still fails.
        class_gradients, matching = gradients(trainer.surrogate, image, annotations,
            imgsz=trainer.imgsz, coverage_policy='independent_same_TAL_for_uncovered_valid_GT')
        risks, risk_records = [], []
        for cls, raw in sorted(class_gradients.items()):
            r, evidence = normalize_map(torch.as_tensor(raw, device=device), target,
                                         constant_policy='preserve')
            risks.append(r)
            risk_records.append(dict(evidence, cls=cls))
        risk = torch.stack(risks).mean(0) if risks else torch.zeros_like(q)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        maps.append((q.detach(), risk.detach(), (target & ~invalid_q_domain).detach()))
        records.append({'target_objects': object_records, 'risk_classes': risk_records,
                        'nt_matching': matching, 'no_nt_gt': not risks,
                        'q_target_seconds': target_seconds,
                        'extra_nt_risk_seconds': time.perf_counter()-risk_started})
    return maps, records
