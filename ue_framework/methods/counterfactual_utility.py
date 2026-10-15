"""Experimental CNLP/CUP primitives. No defaults authorize an experiment.

The poison path retains exact higher-order SGD derivatives. The clean path is
detached. BN running statistics are frozen; functional states isolate the base
model. Callers supply train-only batches with already shared preprocessing.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import random

import numpy as np
import torch
from torch.func import functional_call


@dataclass(frozen=True)
class CNLPConfig:
    inner_steps: int
    inner_lr: float
    lambda_cnlp: float
    target_class_id: int = 14
    num_classes: int = 20
    matching_policy: str = 'strict_native_all_GT'

    def __post_init__(self):
        if type(self.inner_steps) is not int or self.inner_steps not in (1, 3, 5):
            raise ValueError('CNLP J must be explicitly selected among 1/3/5')
        if not math.isfinite(self.inner_lr) or self.inner_lr <= 0:
            raise ValueError('Invalid virtual SGD learning rate')
        if not math.isfinite(self.lambda_cnlp) or self.lambda_cnlp < 0:
            raise ValueError('lambda_CNLP requires an explicit nonnegative value (zero only for equivalence probes)')
        if not 0 <= self.target_class_id < self.num_classes:
            raise ValueError('Invalid target class')
        if self.matching_policy not in ('strict_native_all_GT','native_shared_gt_coverage_v2'):
            raise ValueError('Unknown explicit CNLP matching policy')


@dataclass(frozen=True)
class CUPConfig:
    beta: float
    temperature: float
    min_weight_factor: float
    max_weight_factor: float
    damage_clip: float

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite CUP settings')
        if not 0 <= self.beta < 1 or self.temperature <= 0 or self.damage_clip <= 0:
            raise ValueError('Invalid CUP EMA/temperature/clipping')
        if not 0 < self.min_weight_factor <= 1 <= self.max_weight_factor:
            raise ValueError('CUP bounds must contain uniform weights')


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def same_rng(a, b):
    na, nb = a['numpy'], b['numpy']
    return (a['python'] == b['python'] and na[0] == nb[0]
            and np.array_equal(na[1], nb[1]) and na[2:] == nb[2:]
            and torch.equal(a['torch'], b['torch']) and len(a['cuda']) == len(b['cuda'])
            and all(torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda'])))


@contextmanager
def isolated_mode(model, training):
    states = [(m, m.training) for m in model.modules()]
    model.train(training)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    try:
        yield
    finally:
        for m, mode in states:
            m.training = mode


def validate_batch(batch, *, clean):
    if batch.get('split') != 'train':
        raise ValueError('CNLP support/query must originate from train, never validation')
    ids = tuple(batch['image_ids'])
    image = batch['img']
    if not ids or len(ids) != image.shape[0] or len(set(ids)) != len(ids):
        raise ValueError('Invalid or duplicate batch image identities')
    if image.ndim != 4 or not torch.isfinite(image).all():
        raise ValueError('Invalid image tensor')
    if bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError('Images must be RGB floats in [0,1]')
    if clean and image.requires_grad:
        raise ValueError('Clean reference input must be detached')
    return ids


def validate_pair(clean, poison, query):
    ids = validate_batch(clean, clean=True)
    if ids != validate_batch(poison, clean=False):
        raise ValueError('Clean/poison support identities differ')
    qids = validate_batch(query, clean=True)
    if set(ids) & set(qids):
        raise ValueError('Clean query overlaps support IDs')
    if clean['img'].shape != poison['img'].shape:
        raise ValueError('Support geometry differs')
    if clean.get('preprocessing_id') != poison.get('preprocessing_id') or not clean.get('preprocessing_id'):
        raise ValueError('Shared preprocessing identity is required')
    for key in ('cls', 'bboxes', 'batch_idx'):
        if not torch.equal(clean[key], poison[key]):
            raise ValueError('Clean/poison support GT differs: ' + key)
    if not poison['img'].requires_grad:
        raise ValueError('Poison support must retain its perturbation graph')


class CounterfactualSGD:
    """Two reset SGD(momentum=0, weight_decay=0) paths from the same theta0.

    This is a short virtual learner, not an independent victim. `inner_loss`
    receives a functional-forward callback and a batch. No first-order shortcut.
    """
    def __init__(self, model, config, trainable_names):
        self.model, self.config = model, config
        self.trainable = tuple(trainable_names)
        names = dict(model.named_parameters())
        if not self.trainable or set(self.trainable) - names.keys():
            raise ValueError('Invalid virtual trainable parameter set')

    def _initial(self):
        return ({k: v.detach().clone().requires_grad_(k in self.trainable)
                 for k, v in self.model.named_parameters()},
                {k: v.detach().clone() for k, v in self.model.named_buffers()})

    def _path(self, params, buffers, support, inner_loss, poison):
        losses = []
        for _ in range(self.config.inner_steps):
            def forward(arg):
                return functional_call(self.model, (params, buffers), (arg,), strict=True)
            with isolated_mode(self.model, True):
                loss = inner_loss(forward, support)
            if loss.numel() != 1 or not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite/nonscalar virtual support loss')
            grads = torch.autograd.grad(loss, [params[k] for k in self.trainable],
                                        create_graph=poison, allow_unused=False)
            if not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite virtual update gradient')
            updates = {k: params[k] - self.config.inner_lr * g for k, g in zip(self.trainable, grads)}
            params = {**params, **updates}
            if not poison:
                params = {k: v.detach().requires_grad_(k in self.trainable) for k, v in params.items()}
            losses.append(loss.detach())
        if not poison:
            params = {k: v.detach() for k, v in params.items()}
        return params, buffers, torch.stack(losses)

    def evaluate(self, clean, poison, query, inner_loss, query_loss):
        validate_pair(clean, poison, query)
        initial_rng = capture_rng()
        p0, b0 = self._initial()
        p1 = {k: v.detach().clone().requires_grad_(k in self.trainable) for k, v in p0.items()}
        b1 = {k: v.clone() for k, v in b0.items()}
        try:
            c, cb, clean_support = self._path(p0, b0, clean, inner_loss, False)
            clean_rng = capture_rng()
            restore_rng(initial_rng)
            p, pb, poison_support = self._path(p1, b1, poison, inner_loss, True)
            if not same_rng(clean_rng, capture_rng()):
                raise ValueError('Clean/poison virtual paths consumed different random draws')
            with isolated_mode(self.model, False):
                with torch.no_grad():
                    c_pred = functional_call(self.model, (c, cb), (query['img'],), strict=True)
                    initial_pred = functional_call(self.model, (p0, b0), (query['img'],), strict=True)
                p_pred = functional_call(self.model, (p, pb), (query['img'],), strict=True)
                c_loss, p_loss, evidence = query_loss(c_pred, p_pred, query)
                # Reuse the SAME clean-updated reference matching to measure
                # learning gain; this diagnostic is not an additional loss.
                _, initial_loss, _ = query_loss(c_pred, initial_pred, query)
        finally:
            # The extra virtual paths do not change Parent/EOT/query sampler RNG.
            restore_rng(initial_rng)
        if set(c_loss) != set(p_loss) or set(c_loss) != set(initial_loss):
            raise ValueError('Classwise query loss keys differ')
        damage = {}
        for cls in sorted(c_loss):
            if cls == self.config.target_class_id or not 0 <= cls < self.config.num_classes:
                raise ValueError('Invalid non-target class in query comparison')
            if not torch.isfinite(c_loss[cls]) or not torch.isfinite(p_loss[cls]):
                raise FloatingPointError('Nonfinite classwise query loss')
            damage[cls] = p_loss[cls] - c_loss[cls].detach()
        zero = poison['img'].sum() * 0
        loss = torch.stack([d.relu() for d in damage.values()]).mean() if damage else zero
        return dict(loss=loss, damage=damage, clean_loss=c_loss, poison_loss=p_loss,
                    initial_loss=initial_loss,
                    clean_support_loss=clean_support, poison_support_loss=poison_support,
                    evidence=evidence)


class CUPHistory:
    """Detached, checkpointed class history; updated ONCE after an outer step.

    All EOT draws use the same pre-step history. Absent classes retain their
    history without decay; first observation initializes without EMA bias.
    Weight projection has exact sum=1 and relative-to-uniform lower/upper bounds.
    """
    def __init__(self, config, num_classes=20, target_class_id=14):
        self.config = config
        self.num_classes, self.target = num_classes, target_class_id
        self.mean = torch.zeros(num_classes, dtype=torch.float64)
        self.observations = torch.zeros(num_classes, dtype=torch.long)
        self.steps = 0

    def _check_classes(self, classes):
        if len(set(classes)) != len(classes) or any(c == self.target or not 0 <= c < self.num_classes for c in classes):
            raise ValueError('Invalid CUP non-target classes')

    def weights(self, classes, reference):
        classes = list(classes)
        self._check_classes(classes)
        if not classes:
            return reference.new_empty(0)
        cfg, n = self.config, len(classes)
        scores = self.mean[classes] / cfg.temperature
        probability = torch.softmax(scores, 0)
        lo, hi = cfg.min_weight_factor / n, cfg.max_weight_factor / n
        # Euclidean projection on a bounded simplex; plain clamp+renormalize
        # would violate the lower/upper bounds.
        left, right = float(probability.min()) - hi, float(probability.max()) - lo
        for _ in range(80):
            mid = (left + right) / 2
            if float((probability - mid).clamp(lo, hi).sum()) > 1:
                left = mid
            else:
                right = mid
        return (probability - (left + right) / 2).clamp(lo, hi).to(reference).detach()

    def aggregate(self, damage, zero):
        classes = sorted(damage)
        weights = self.weights(classes, zero)
        loss = (weights * torch.stack([damage[c].relu() for c in classes])).sum() if classes else zero
        return loss, weights

    def commit(self, eot_damage):
        grouped = {}
        for sample in eot_damage:
            self._check_classes(list(sample))
            for cls, value in sample.items():
                value = float(value.detach()) if torch.is_tensor(value) else float(value)
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite CUP damage')
                grouped.setdefault(cls, []).append(value)
        # Validate everything before changing the checkpointable state.
        for cls, values in grouped.items():
            value = sum(values) / len(values)
            value = max(-self.config.damage_clip, min(self.config.damage_clip, value))
            self.mean[cls] = (self.config.beta * self.mean[cls] + (1-self.config.beta)*value
                              if self.observations[cls] else value)
            self.observations[cls] += len(values)
        self.steps += 1

    def state_dict(self):
        return dict(config=asdict(self.config), num_classes=self.num_classes, target=self.target,
                    mean=self.mean.clone(), observations=self.observations.clone(), steps=self.steps)

    def load_state_dict(self, state):
        if (state['config'] != asdict(self.config) or state['num_classes'] != self.num_classes
                or state['target'] != self.target):
            raise ValueError('CUP resume configuration mismatch')
        mean, count = state['mean'], state['observations']
        if (mean.shape != self.mean.shape or count.shape != self.observations.shape
                or not torch.isfinite(mean).all() or bool((count < 0).any())
                or bool((mean.abs() > self.config.damage_clip).any()) or state['steps'] < 0
                or count[self.target] != 0 or mean[self.target] != 0):
            raise ValueError('Invalid CUP checkpoint')
        self.mean.copy_(mean); self.observations.copy_(count); self.steps = state['steps']
