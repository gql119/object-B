"""GT-supervised PGD-10 on the actual augmented, normalized detector input."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib

import torch
from torch.nn.modules.batchnorm import _BatchNorm


@dataclass(frozen=True)
class AttackConfig:
    epsilon_pixels: int
    step_pixels: int
    steps: int = 10
    random_start: bool = True

    def __post_init__(self):
        if (self.epsilon_pixels, self.step_pixels) not in ((4, 1), (8, 2)):
            raise ValueError('Only project-defined AT4/AT8 PGD steps are supported')
        if self.steps != 10 or self.random_start is not True:
            raise ValueError('PGD-10 with uniform random start is frozen')


def tensor_sha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str((t.dtype, tuple(t.shape))).encode() + t.numpy().tobytes()).hexdigest()


def gt_identity(batch):
    return {k: tensor_sha(batch[k]) for k in ('batch_idx', 'cls', 'bboxes')}


def validate_batch(batch, nc):
    x, cls, boxes, idx = (batch[k] for k in ('img', 'cls', 'bboxes', 'batch_idx'))
    if x.ndim != 4 or x.shape[1] != 3 or x.dtype != torch.float32:
        raise ValueError('PGD must follow native normalization on NCHW FP32 images')
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('Invalid normalized detector input')
    if cls.numel() != idx.numel() or boxes.shape != (cls.numel(), 4):
        raise ValueError('GT shape mismatch')
    if not all(torch.isfinite(t).all() for t in (cls, boxes, idx)):
        raise ValueError('Nonfinite GT')
    if cls.numel() and (not torch.equal(cls, cls.round()) or cls.min() < 0 or cls.max() >= nc):
        raise ValueError('Invalid full-class GT mapping')
    if idx.numel() and (not torch.equal(idx, idx.round()) or idx.min() < 0 or idx.max() >= len(x)):
        raise ValueError('Invalid GT image index')
    if boxes.numel() and (boxes.min() < 0 or boxes.max() > 1 or (boxes[:, 2:] <= 0).any()):
        raise ValueError('Invalid normalized xywh GT')


@contextmanager
def inner_model_state(model):
    """Freeze only BN running-stat updates and parameter differentiation; restore exactly."""
    params = list(model.parameters())
    requires = [p.requires_grad for p in params]
    prior_grads = [p.grad for p in params]
    bn = [(m, m.training, {k: v.detach().clone() for k, v in m.named_buffers(recurse=False)})
          for m in model.modules() if isinstance(m, _BatchNorm)]
    try:
        for p in params:
            p.requires_grad_(False)
        for m, _, _ in bn:
            m.training = False
        yield
    finally:
        bad_grad = any(p.grad is not prior for p, prior in zip(params, prior_grads))
        for p, required, prior in zip(params, requires, prior_grads):
            p.requires_grad_(required)
        bad_bn = False
        for m, training, buffers in bn:
            m.training = training
            if any(not torch.equal(v, dict(m.named_buffers(recurse=False))[k]) for k, v in buffers.items()):
                bad_bn = True
        if bad_grad or bad_bn:
            raise RuntimeError('PGD changed parameter gradients or BN buffers; all training flags restored')


def safe_interval(x, epsilon):
    exact = x.double()
    exact_lo, exact_hi = (exact - epsilon).clamp(0, 1), (exact + epsilon).clamp(0, 1)
    lo, hi = exact_lo.to(x.dtype), exact_hi.to(x.dtype)
    # Form endpoints before casting; float32 subtraction near epsilon can
    # cancel to zero, where one nextafter would not repair the budget.
    lo = torch.where(lo.double() < exact_lo, torch.nextafter(lo, x), lo)
    hi = torch.where(hi.double() > exact_hi, torch.nextafter(hi, x), hi)
    return lo, hi


def flatten_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from flatten_tensors(item)


class PGDAttack:
    def __init__(self, config, device, seed=0, nc=20):
        self.config = config if isinstance(config, AttackConfig) else AttackConfig(**config)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.nc = nc
        self.calls = 0

    def __call__(self, model, batch):
        validate_batch(batch, self.nc)
        if not model.training or model.model[-1].nc != self.nc:
            raise ValueError('PGD requires the native training detector and full class head')
        before_gt = gt_identity(batch)
        x = batch['img'].detach()
        epsilon, alpha = self.config.epsilon_pixels / 255., self.config.step_pixels / 255.
        lo, hi = safe_interval(x, epsilon)
        adv = torch.maximum(lo, torch.minimum(hi, x + torch.empty_like(x).uniform_(
            -epsilon, epsilon, generator=self.generator))).detach()
        if getattr(model, 'criterion', None) is None:
            model.criterion = model.init_criterion()
        criterion = model.criterion
        if not hasattr(criterion, 'assigner'):
            raise ValueError('Expected unmodified native YOLOv8 GT assigner')
        assignments, raw_devices, steps = [], set(), []

        def assigned(module, inputs, output):
            fg, local_gt = output[3], output[4]
            valid = inputs[5].squeeze(-1).bool()
            submitted = int(valid.sum())
            if submitted != batch['cls'].numel():
                raise ValueError('Native loss did not receive all transformed GT')
            matched = sum(len(torch.unique(local_gt[i][fg[i]])) for i in range(len(fg)))
            assignments.append(dict(submitted_GT=submitted, matched_GT=int(matched),
                                    foreground_anchors=int(fg.sum())))
            if submitted and not bool(fg.any()):
                raise ValueError('Nonempty batch has no native GT positive supervision')

        def raw_hook(module, inputs, output):
            tensors = list(flatten_tensors(output))
            if not tensors or any(not torch.isfinite(t).all() for t in tensors):
                raise FloatingPointError('Nonfinite native pre-NMS training predictions')
            raw_devices.update((str(t.device), t.device.index) for t in tensors)
            if any(t.device != x.device for t in tensors):
                raise ValueError('Native prediction device differs from actual PGD input')

        handles = [criterion.assigner.register_forward_hook(assigned),
                   model.model[-1].register_forward_hook(raw_hook)]
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        try:
            with torch.random.fork_rng(devices=devices), inner_model_state(model), torch.enable_grad(), \
                    torch.autocast(device_type=self.device.type, enabled=False):
                for step in range(self.config.steps):
                    adv = adv.detach().requires_grad_(True)
                    loss, parts = model(dict(batch, img=adv))
                    if len(parts) != 3 or not all(torch.isfinite(v).all() for v in parts.values()):
                        raise FloatingPointError('Native box/cls/DFL loss missing or nonfinite')
                    loss = loss.sum()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite full detection loss in PGD')
                    grad, = torch.autograd.grad(loss, adv, only_inputs=True)
                    if not torch.isfinite(grad).all() or not bool(grad.abs().max() > 0):
                        raise FloatingPointError('PGD input gradient is nonfinite or identically zero')
                    steps.append(dict(step=step, loss=float(loss.detach()),
                                      loss_parts={k: float(v) for k, v in parts.items()},
                                      input_grad_norm=float(grad.detach().norm()),
                                      **assignments[-1]))
                    adv = torch.maximum(lo, torch.minimum(hi, adv + alpha * grad.sign())).detach()
        finally:
            for handle in handles:
                handle.remove()
        if before_gt != gt_identity(batch):
            raise ValueError('PGD changed transformed GT')
        delta = (adv.double() - x.double()).abs()
        linf = float(delta.max())
        if linf > epsilon or adv.min() < 0 or adv.max() > 1 or linf == 0:
            raise ValueError(f'PGD budget/range/nonzero failure: linf={linf!r}, epsilon={epsilon!r}, min={float(adv.min())!r}, max={float(adv.max())!r}')
        self.calls += 1
        return adv, dict(pgd_steps=steps, actual_inner_forwards=len(steps),
                         epsilon=epsilon, alpha=alpha, linf=linf,
                         actual_raw_devices=sorted(raw_devices, key=str),
                         GT_sha256=before_gt, GT_count=int(batch['cls'].numel()),
                         class_histogram=torch.bincount(batch['cls'].long().flatten(), minlength=self.nc).tolist(),
                         BN_running_state_unchanged=True, parameter_grad_accumulation_unchanged=True,
                         attack_scope='whole actual augmented normalized training input',
                         TAL_definition='unmodified native assignment; unmatched GT are reported',
                         prediction_head_classes=self.nc, call=self.calls)
