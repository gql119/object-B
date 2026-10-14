"""Frozen clean-teacher spatial feasibility budgets; no optimization loss is added."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .legacy_kproto_ret import _rectangle


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def config_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def item_key(image, annotations):
    h = hashlib.sha256(np.ascontiguousarray(image, dtype=np.float32).tobytes())
    h.update(json.dumps(annotations, sort_keys=True, separators=(',', ':')).encode())
    return h.hexdigest()


def domains(shape, annotations):
    target = np.zeros(shape[:2], dtype=bool)
    other = np.zeros_like(target)
    for ann in annotations:
        cls = int(ann['cls'])
        box = np.asarray(ann['bbox'], dtype=float)
        if not 0 <= cls < 20 or not np.isfinite(box).all() or box.shape != (4,):
            raise ValueError('Invalid GT class/box')
        if np.any(box[:2] < 0) or np.any(box[:2] > 1) or np.any(box[2:] <= 0) or np.any(box[2:] > 1):
            raise ValueError('Invalid normalized GT coordinates')
        mask = _rectangle(shape[:2], box)[0, 0].numpy().astype(bool)
        if not mask.any():
            raise ValueError('GT rectangle has no original-resolution pixels')
        if cls == 14:
            target |= mask
        else:
            other |= mask
    return target, other


def normalize_map(g, target, settings):
    """Normalize each class separately; deterministic midpoint for constant support."""
    if not np.isfinite(g).all() or np.any(g < 0):
        raise FloatingPointError('Nonfinite/negative sensitivity')
    if not target.any():
        raise ValueError('Normalization requires Person support')
    if float(g.max()) <= 0:
        raise ValueError('NT supervision produced no input gradient; not a fallback')
    qlo, qhi = np.quantile(g[target], [settings['q_low'], settings['q_high']])
    floor = max(float(g.max()) * 1e-7, 1e-20)
    degenerate = float(qhi - qlo) <= floor
    if degenerate:
        result = np.where(target, .5, 0).astype(np.float32)
    else:
        result = np.where(target, np.clip((g - qlo) / (qhi - qlo), 0, 1), 0).astype(np.float32)
    return result, {'q_low_value': float(qlo), 'q_high_value': float(qhi),
                    'degenerate': bool(degenerate), 'degenerate_policy': 'constant_midpoint_0.5'}


def construct_budget(class_maps, target, settings, *, no_nt_gt=False):
    if not 0 < settings['eta'] <= 1 or settings['gamma'] <= 0 or settings['epsilon_max'] > 8 / 255 + 1e-9:
        raise ValueError('CSB mapping outside declared range')
    if not target.any():
        return np.zeros(target.shape, np.float32), np.zeros(target.shape, np.float32), []
    if no_nt_gt:
        if class_maps:
            raise ValueError('Contradictory no-NT fallback')
        return target.astype(np.float32) * settings['epsilon_max'], np.zeros(target.shape, np.float32), []
    if not class_maps:
        raise ValueError('NT GT exists but sensitivity is missing')
    normalized, stats = [], []
    for cls in sorted(class_maps):
        z, row = normalize_map(class_maps[cls], target, settings)
        normalized.append(z); stats.append(dict(row, cls=int(cls)))
    z = torch.from_numpy(np.stack(normalized).mean(0))[None, None]
    m = torch.from_numpy(target.astype(np.float32))[None, None]
    k = settings['smooth_kernel']
    if k % 2 != 1 or k < 1:
        raise ValueError('Masked smoothing requires a positive odd kernel')
    numerator = F.avg_pool2d(z * m, k, stride=1, padding=k // 2)
    denominator = F.avg_pool2d(m, k, stride=1, padding=k // 2)
    normalized = ((numerator / denominator.clamp_min(1e-12)).clamp(0, 1) * m)[0, 0].numpy()
    budget = (settings['epsilon_max'] * target *
              (settings['eta'] + (1 - settings['eta']) * (1 - normalized) ** settings['gamma'])).astype(np.float32)
    if not np.isfinite(budget).all() or np.any(budget[~target] != 0):
        raise FloatingPointError('Invalid budget outside support')
    return budget, normalized, stats


def png_project(clean_uint8, poisoned_float, budget):
    """Enforce continuous budgets after quantization; never rely on a +1 byte tolerance."""
    cap = np.floor(budget.astype(np.float64) * 255 + 1e-6).astype(np.int16)[..., None]
    wanted = np.rint(np.clip(poisoned_float, 0, 1) * 255).astype(np.int16)
    diff = np.clip(wanted - clean_uint8.astype(np.int16), -cap, cap)
    return np.clip(clean_uint8.astype(np.int16) + diff, 0, 255).astype(np.uint8)


def spearman(a, b):
    def ranks(values):
        _, indices, counts = np.unique(values, return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)
        return (ends - (counts - 1) / 2)[indices]
    x, y = ranks(a), ranks(b)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


class BudgetStore:
    def __init__(self, root, expected_identity):
        self.root = Path(root)
        manifest = json.loads((self.root / 'manifest.json').read_text())
        if manifest['identity'] != expected_identity or manifest['state'] != 'complete':
            raise ValueError('Incomplete or unbound frozen CSB cache')
        self.rows = {r['item_key']: r for r in manifest['images']}
        if len(self.rows) != len(manifest['images']):
            raise ValueError('Duplicate CSB image identity')

    def get(self, image, annotations):
        row = self.rows[item_key(image, annotations)]
        path = self.root / row['map_file']
        if digest(path) != row['map_sha256']:
            raise ValueError('Frozen spatial budget checksum changed')
        with np.load(path, allow_pickle=False) as maps:
            b = maps['budget'].copy()
        if b.shape != image.shape[:2] or not np.isfinite(b).all():
            raise ValueError('Budget geometry/nonfinite mismatch')
        return b
