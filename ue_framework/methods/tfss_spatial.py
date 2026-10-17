"""Detached clean-map soft scaling after the exact S0 composition.

No optimizer, loss, TAL assignment, frequency selection or model update lives here.
"""
import torch
import torch.nn.functional as F


def normalize_map(raw, mask, *, constant_policy):
    """Masked 3x3 smoothing followed by within-object Q10/Q90 normalization."""
    if raw.shape != mask.shape or raw.ndim != 2 or not bool(mask.any()):
        raise ValueError('Map/domain mismatch or empty valid GT domain')
    if not torch.isfinite(raw).all() or bool((raw < 0).any()):
        raise ValueError('Invalid sensitivity')
    domain = mask.to(raw.dtype)[None, None]
    num = F.avg_pool2d((raw * mask)[None, None], 3, 1, 1)
    den = F.avg_pool2d(domain, 3, 1, 1)
    smooth = (num / den.clamp_min(torch.finfo(raw.dtype).tiny))[0, 0]
    lo, hi = torch.quantile(smooth[mask], raw.new_tensor([.1, .9]))
    valid = bool(hi - lo > max(float(smooth[mask].max()) * 1e-7, 1e-20))
    if not valid:
        if constant_policy != 'preserve':
            raise ValueError('Unrecognized degenerate-map policy')
        # Preserve candidate strength when target attribution is uninformative;
        # for a nonzero flat risk map every supported position is equally risky.
        result = mask.to(raw.dtype)
    else:
        result = ((smooth - lo) / (hi - lo)).clamp(0, 1) * mask
    return result.detach(), {'valid': valid, 'q10': float(lo), 'q90': float(hi)}


def target_union(object_maps, target_mask):
    if not object_maps:
        raise ValueError('No Person supervision')
    merged = torch.zeros_like(target_mask, dtype=torch.float32)
    for q, mask in object_maps:
        if q.shape != merged.shape or mask.shape != merged.shape:
            raise ValueError('Object map is not in original image coordinates')
        merged = torch.maximum(merged, q.to(merged) * mask)
    return (merged * target_mask).detach()


def spatial_weights(q, risk, support, *, mode, eta=.85):
    if not 0 < eta <= 1 or q.shape != support.shape or risk.shape != support.shape:
        raise ValueError('Invalid TFSS shape/floor')
    if not all(bool(torch.isfinite(v).all()) for v in (q, risk, support)):
        raise ValueError('Nonfinite map')
    if any(bool(((v < 0) | (v > 1)).any()) for v in (q, risk)):
        raise ValueError('Normalized attribution/risk outside [0,1]')
    if not bool(((support == 0) | (support == .5) | (support == 1)).all()):
        raise ValueError('Not the original S0 1/.5/0 support')
    if mode == 'S0':
        w = torch.ones_like(q)
    elif mode == 'TFSS-C':
        w = eta + (1 - eta) * q
    elif mode == 'TFSS-U':
        w = 1 - (1 - eta) * risk * (1 - q)
    else:
        raise ValueError('Unknown candidate')
    # W is neutral outside support; actual delta and gradient remain zero there.
    return torch.where(support > 0, w, torch.ones_like(w)).detach()


def realized_rms(image, delta, support):
    domain = (support > 0).expand_as(delta)
    if not bool(domain.any()):
        raise ValueError('Empty Person domain')
    actual = (image + delta).clamp(0, 1) - image
    return actual[domain].double().square().mean().sqrt()


def match_realized_rms(image, s0_delta, support, target, *, tolerance=1e-7,
                       relative_tolerance=1e-4):
    """Per-image uniform alpha, matched AFTER image clamp; never amplify S0."""
    if not torch.isfinite(torch.as_tensor(target)) or target <= 0:
        raise ValueError('Invalid/nonpositive target RMS')
    if tolerance <= 0 or relative_tolerance <= 0:
        raise ValueError('Invalid numerical matching tolerance')
    with torch.no_grad():
        def measured(alpha):
            # Evaluate the SAME typed alpha used in the actual batched forward.
            a = s0_delta.new_tensor(alpha)
            return float(a), float(realized_rms(image, s0_delta*a, support))

        maximum = float(realized_rms(image, s0_delta, support))
        if target > maximum and (target-maximum > tolerance or
                                 (target-maximum)/target > relative_tolerance):
            raise ValueError('Uniform target exceeds own S0 maximum: no silent amplification')
        lo, hi = 0., 1.
        for _ in range(48):
            alpha = (lo + hi) / 2
            if measured(alpha)[1] < target:
                lo = alpha
            else:
                hi = alpha
        # The float32 image-clamped RMS is a staircase. Its midpoint need not be
        # closer than either attainable bracket endpoint. Include boundary alpha
        # 1 for exact no-attenuation/no-NT cases and never exceed [0, 1].
        candidates = [measured(a) for a in (lo, hi, (lo+hi)/2, 0., 1.)]
        alpha, value = min(candidates, key=lambda p: (abs(p[1]-target), p[0]))
        error = abs(value-target)
        if error > tolerance or error/target > relative_tolerance:
            raise ValueError(f'RMS matching failed: absolute={error}, relative={error/target}, '
                             f'target={target}, maximum={maximum}')
    return alpha, error


class SoftSpatialProjection:
    """Mixin before B2SemanticTrainer; exact no-op mode returns original tuple."""
    tfss_mode = 'S0'
    tfss_weight = None
    tfss_uniform_targets = None

    def _compose_delta_batched(self, img_t, inner_t, ring_t, *args, **kwargs):
        original = super()._compose_delta_batched(img_t, inner_t, ring_t, *args, **kwargs)
        raw, delta, poison, binary, jnd = original
        self.tfss_s0_delta = delta
        self.tfss_input = img_t
        self.tfss_support = inner_t
        self.tfss_matching_errors = []
        self.tfss_matching_relative_errors = []
        if self.tfss_mode == 'S0':
            self.tfss_delta = delta
            self.tfss_applied_weight = torch.ones_like(inner_t)
            return original
        if self.tfss_mode.startswith('Uniform-'):
            if self.tfss_uniform_targets is None or len(self.tfss_uniform_targets) != len(delta):
                raise ValueError('Missing paired step/EOT/image candidate RMS targets')
            alphas = []
            for image, d, mask, target in zip(img_t, delta, inner_t, self.tfss_uniform_targets):
                alpha, error = match_realized_rms(image, d, mask, target,
                    tolerance=self.tfss_uniform_absolute_tolerance,
                    relative_tolerance=self.tfss_uniform_relative_tolerance)
                alphas.append(alpha)
                self.tfss_matching_errors.append(error)
                self.tfss_matching_relative_errors.append(error/target)
            w = delta.new_tensor(alphas)[:, None, None, None]
        else:
            w = self.tfss_weight
            if w is None or w.shape != inner_t.shape or w.requires_grad:
                raise ValueError('Missing detached fixed clean weight in exact padded coordinates')
            if not torch.isfinite(w).all() or bool(((w < .85) | (w > 1)).any()):
                raise ValueError('TFSS weight outside preregistered range')
        # Keep raw/JND/TV/budget paths unchanged. Scale only final a0-composed delta.
        delta = delta * w
        poison = (img_t + delta).clamp(0, 1)
        self._last_poison = poison  # Adaptive must consume the ACTUAL scaled image.
        self.tfss_delta = delta
        self.tfss_applied_weight = w.expand_as(inner_t)
        return raw, delta, poison, binary, jnd
