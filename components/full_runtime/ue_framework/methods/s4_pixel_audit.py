"""Original matched PNG budget audit helpers; no CSB method path."""
import numpy as np
from .legacy_kproto_ret import _rectangle

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

def png_project(clean_uint8, poisoned_float, budget):
    """Enforce continuous budgets after quantization; never rely on a +1 byte tolerance."""
    cap = np.floor(budget.astype(np.float64) * 255 + 1e-6).astype(np.int16)[..., None]
    wanted = np.rint(np.clip(poisoned_float, 0, 1) * 255).astype(np.int16)
    diff = np.clip(wanted - clean_uint8.astype(np.int16), -cap, cap)
    return np.clip(clean_uint8.astype(np.int16) + diff, 0, 255).astype(np.uint8)
