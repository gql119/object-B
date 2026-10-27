"""Full VOC20 clean-validation metrics and signed matched-budget scores."""
import math
from .data import VOC_NAMES


def extract(box):
    import numpy as np
    ids = np.asarray(box.ap_class_index).astype(int).tolist()
    ap = np.asarray(box.all_ap, dtype=float)
    if ids != list(range(20)) or ap.shape != (20, 10) or not np.isfinite(ap).all() or np.any((ap < 0) | (ap > 1)):
        raise ValueError('Full ordered VOC20 AP across all ten IoU thresholds required')
    for values in (box.r, box.p):
        a = np.asarray(values, dtype=float)
        if a.shape != (20,) or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
            raise ValueError('Full finite VOC20 precision/recall required')
    return dict(class_ids=ids, names=list(VOC_NAMES), AP50=ap[:, 0].tolist(),
                AP75=ap[:, 5].tolist(), AP50_95=ap.mean(1).tolist(),
                Recall=np.asarray(box.r, dtype=float).tolist(),
                Precision=np.asarray(box.p, dtype=float).tolist(), target_id=14, validation_count=4952)


def pair(clean, poison):
    for row in (clean, poison):
        if row['class_ids'] != list(range(20)) or row['validation_count'] != 4952:
            raise ValueError('Unmatched or incomplete VOC evaluation')
        for key in ('AP50', 'AP75', 'AP50_95', 'Recall', 'Precision'):
            if len(row[key]) != 20 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in row[key]):
                raise ValueError('Invalid fractional per-class metrics')
    c, p = clean['AP50'], poison['AP50']
    nt = [i for i in range(20) if i != 14]
    P = 1 - p[14] / c[14] if c[14] else None
    U = sum(p[i] for i in nt) / sum(c[i] for i in nt) if sum(c[i] for i in nt) else None
    S = 2 * P * U / (P + U) if P is not None and U is not None and P + U != 0 else None
    damage = {VOC_NAMES[i]: 1 - p[i] / c[i] if c[i] else None for i in nt}
    ranked = sorted(((k, v) for k, v in damage.items() if v is not None), key=lambda x: x[1], reverse=True)
    complete = len(ranked) == 19
    return dict(P_AT=P, U_AT=U, S_AT=S,
                undefined_reasons=dict(P_AT=None if P is not None else 'Clean-AT Person AP50=0',
                                       U_AT=None if U is not None else 'Clean-AT NT AP50=0',
                                       S_AT=None if S is not None else 'undefined component or P+U=0'),
                Person_change_pp=100 * (p[14] - c[14]),
                NT_change_pp=100 * (sum(p[i] - c[i] for i in nt) / 19),
                WCD=ranked[0][1] if complete else None,
                TCD3=sum(v for _, v in ranked[:3]) / 3 if complete else None,
                WCD_class=ranked[0][0] if complete else None, relative_NT_harm=damage,
                collateral_reason=None if complete else 'a Clean-AT NT class AP50=0',
                per_class_change_pp={VOC_NAMES[i]: 100 * (p[i] - c[i]) for i in range(20)},
                scores_clipped=False, denominator='same epsilon Clean-AT last@200')
