"""Parent losses/update are inherited; only post-synthesis feasibility changes."""
import numpy as np
import torch
import torch.nn.functional as F

from .b2_semantic import B2SemanticTrainer
from .legacy_kproto_ret import LegacyKProtoGenerator


class SpatialProjection:
    def _compose_delta_batched(self, img_t, inner_t, ring_t, *args, **kwargs):
        raw, old_delta, old_poison, binary, jnd = super()._compose_delta_batched(
            img_t, inner_t, ring_t, *args, **kwargs)
        if self.budget_mode == 'parent_fixed':
            return raw, old_delta, old_poison, binary, jnd
        budget = self._batch_budget.to(device=img_t.device, dtype=img_t.dtype)
        if budget.shape != inner_t.shape or not torch.isfinite(budget).all():
            raise ValueError('Invalid padded CSB map')
        if bool((budget < 0).any()) or bool((budget > self.eps + 1e-9).any()):
            raise ValueError('CSB epsilon bound')
        if bool((budget[inner_t == 0] != 0).any()):
            raise ValueError('CSB budget outside Person support')
        # Undo only the parent's final overlap amplitude multiplier. The latent
        # tanh/JND/dropout/clamp/ST-floor is unchanged, including its RNG usage.
        base = old_delta / torch.where(inner_t > 0, inner_t, torch.ones_like(inner_t))
        delta = torch.maximum(torch.minimum(base, budget), -budget) * (inner_t > 0)
        # Keep raw TV/budget regularization and all loss masks exactly parent-defined.
        return raw, delta, (img_t + delta).clamp(0, 1), binary, jnd


class S1CSBTrainer(SpatialProjection, B2SemanticTrainer):
    def __init__(self, *args, budget_store=None, budget_mode='csb', **kwargs):
        self.budget_mode = budget_mode
        self.budget_store = budget_store
        if budget_mode not in ('csb', 'parent_fixed', 'uniform_matched'):
            raise ValueError('Unknown budget mode')
        super().__init__(*args, **kwargs)

    def _batch_loss(self, items, prototypes):
        if self.budget_mode != 'parent_fixed':
            h = (max(im.shape[0] for im, _, _ in items) + 31) // 32 * 32
            w = (max(im.shape[1] for im, _, _ in items) + 31) // 32 * 32
            budgets = []
            for im, anns, support in items:
                b = self.budget_store.get(im, anns)
                if self.budget_mode == 'uniform_matched':
                    target = support > 0
                    b = np.where(target, b[target].mean(), 0).astype(np.float32)
                budgets.append(F.pad(torch.from_numpy(b)[None], (0, w - im.shape[1], 0, h - im.shape[0])))
            self._batch_budget = torch.stack(budgets)
        return super()._batch_loss(items, prototypes)


class S1CSBGenerator(SpatialProjection, LegacyKProtoGenerator):
    def __init__(self, *args, budget_store, budget_mode='csb', **kwargs):
        self.budget_store = budget_store
        self.budget_mode = budget_mode
        super().__init__(*args, **kwargs)

    def generate(self, image, annotations, *args, **kwargs):
        self._batch_budget = torch.from_numpy(self.budget_store.get(image, annotations))[None, None]
        return super().generate(image, annotations, *args, **kwargs)
