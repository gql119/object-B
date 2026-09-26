"""Fixed clean-training semantic coordinates for B2-T/TO; no adaptive changes."""
import hashlib
import math
from pathlib import Path

import torch

from .adaptive_learner import AdaptiveLearner

CLASSES = ('aeroplane bicycle bird boat bottle bus car cat chair cow diningtable '
           'dog horse motorbike person pottedplant sheep sofa train tvmonitor').split()


def sha256(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            d.update(b)
    return d.hexdigest()


def select_channels(target, non_target, fraction=.2, specificity=1., eps=1e-12):
    if not 0 < fraction <= 1 or target.shape != non_target.shape:
        raise ValueError('Invalid TSFL coordinates')
    if not torch.isfinite(target).all() or not torch.isfinite(non_target).all():
        raise ValueError('Nonfinite sensitivity')
    if target.mean() <= 0 or non_target.mean() <= 0:
        raise ValueError('Zero sensitivity: evidence is not connected')
    st, sn = target / (target.mean() + eps), non_target / (non_target.mean() + eps)
    score = st - specificity * sn
    # ceil prevents underselecting the requested fraction; stable ties use channel ID.
    indices = torch.argsort(score, descending=True, stable=True)[:max(1, math.ceil(fraction * len(score)))]
    return indices.sort().values, st, sn, score


def gt_evidence(decoded, annotations, image_shape, imgsz):
    """B2 task-alignment formula per real GT; original -> centered letterbox geometry."""
    h, w = image_shape
    r = min(imgsz / h, imgsz / w)
    nh, nw = max(1, round(h*r)), max(1, round(w*r))
    top, left = (imgsz-nh)//2, (imgsz-nw)//2
    result = []
    for ann in annotations:
        cx, cy, bw, bh = ann['bbox']
        gt = decoded.new_tensor([[cx*w*r+left, cy*h*r+top, bw*w*r, bh*h*r]])
        c = int(ann['cls'])
        probs = decoded[0, 4+c].clamp(1e-6, 1.)
        align = probs[:, None].sqrt() * AdaptiveLearner._iou(decoded[0, :4].T, gt).clamp_min(1e-6).pow(6)
        result.append((c, align.topk(min(100, align.shape[0]), dim=0).values.mean()))
    return result


def class_subspace(features, variance=.90, min_rank=4, max_rank_fraction=.25):
    z = torch.stack(features).double()
    n, channels = z.shape
    if n < 2 or not 0 < max_rank_fraction <= 1 or not 0 < variance <= 1:
        raise ValueError('Insufficient class features or invalid PCA configuration')
    mean = z.mean(0)
    _, s, vh = torch.linalg.svd(z-mean, full_matrices=False)
    energy = s.square()
    if energy.sum() <= 0:
        raise ValueError('Degenerate class features')
    cumulative = energy.cumsum(0)/energy.sum()
    needed = int(torch.searchsorted(cumulative, torch.tensor(variance, dtype=cumulative.dtype)))+1
    cap = min(n-1, channels, max(1, math.floor(channels*max_rank_fraction)))
    rank = min(cap, max(min_rank, needed))
    u = vh[:rank].T
    # SVD sign convention for reproducible serialized coordinates.
    pivots = u.abs().argmax(0)
    signs = u[pivots, torch.arange(rank)].sign()
    u = u * signs
    return {'basis': u.float(), 'mean': mean.float(), 'count': n, 'rank': rank,
            'rank_for_variance': needed, 'max_rank': cap,
            'explained_variance': float(cumulative[rank-1]), 'variance_capped': rank < needed}


def normalized_orth(displacement, bases, overlap_weights, worst_k=3, eps=1e-12):
    """Full-C displacement. Class conflict -> top-k classes -> weighted instance mean."""
    classes = sorted(bases)
    energy = displacement.square().sum(-1).clamp_min(eps)
    ratios = torch.stack([(displacement @ bases[c]).square().sum(-1)/
                          (displacement.square().sum(-1)+eps) for c in classes], -1)
    # Equal instance averaging chooses worst classes; overlap then weights their instances.
    class_conflicts = ratios.mean(0)
    order = torch.argsort(class_conflicts.detach(), descending=True, stable=True)[:min(worst_k, len(classes))]
    value = (ratios[:, order].mean(-1)*overlap_weights).mean()
    return value, class_conflicts, [classes[int(i)] for i in order]


def semantic_identity(cfg, root):
    s = cfg['methods']['legacy_kproto_ret'].get('semantic', {})
    if not s.get('enabled', False):
        return None
    root = Path(root).resolve()
    artifacts = {}
    for name in ('target_specific_channels.pt', 'non_target_subspaces.pt', 'coordinates_identity.json'):
        p = Path(s['artifact_dir']) / name
        if not p.resolve().is_relative_to(root):
            raise ValueError('Semantic artifacts must stay in isolated project')
        artifacts[name] = sha256(p)
    methods = root / 'ue_framework' / 'methods'
    return {'artifacts': artifacts, 'semantic_code_sha256': sha256(methods/'b2_semantic.py'),
            'coordinate_code_sha256': sha256(methods/'semantic_coordinates.py'),
            'lambda_orth': float(s['lambda_orth'])}
