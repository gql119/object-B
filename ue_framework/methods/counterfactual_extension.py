"""Transactional outer-loss extension; original Parent files stay unchanged.

The production loop must begin once per optimizer step, add each EOT loss,
then commit only after a successful optimizer step, and checkpoint this state.
No standalone formal training entry point is enabled by this module.
"""
import copy
from dataclasses import asdict

import torch

from .counterfactual_utility import CUPHistory


class CounterfactualExtension:
    def __init__(self, adapter, cup_config=None):
        self.adapter = adapter
        config = adapter.config
        self.cup = (CUPHistory(cup_config, config.num_classes, config.target_class_id)
                    if cup_config is not None else None)
        self.completed_steps = 0
        self.pending = None

    def begin_step(self, step):
        if self.pending is not None or step != self.completed_steps:
            raise ValueError('CNLP optimizer-step transaction mismatch')
        self.pending = []

    def add(self, parent_loss, clean, poison, query):
        if self.pending is None:
            raise ValueError('Begin CNLP outer optimizer step first')
        result = self.adapter.evaluate(clean, poison, query)
        raw = result['loss']
        if self.cup:
            raw, weights = self.cup.aggregate(result['damage'], raw * 0)
        else:
            n = len(result['damage'])
            weights = raw.new_full((n,), 1/n) if n else raw.new_empty(0)
        total = parent_loss + self.adapter.config.lambda_cnlp * raw
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite Parent+CNLP loss')
        self.pending.append({k: v.detach().clone() for k, v in result['damage'].items()})
        result.update(loss=raw, weights=weights, weighted_loss=self.adapter.config.lambda_cnlp * raw)
        return total, result

    def commit_step(self):
        if not self.pending:
            raise ValueError('Cannot commit an empty optimizer transaction')
        if self.cup:
            self.cup.commit(self.pending)
        self.completed_steps += 1
        self.pending = None

    def abort_step(self):
        # Caller must also restore its own Adam/RNG if it had already updated.
        self.pending = None

    def state_dict(self):
        if self.pending is not None:
            raise ValueError('CNLP checkpoint requires a completed optimizer boundary')
        return dict(cnlp_config=asdict(self.adapter.config), completed_steps=self.completed_steps,
                    cup=self.cup.state_dict() if self.cup else None)

    def load_state_dict(self, state):
        if self.pending is not None or state['cnlp_config'] != asdict(self.adapter.config):
            raise ValueError('CNLP resume config/boundary mismatch')
        if state['completed_steps'] < 0 or (state['cup'] is None) != (self.cup is None):
            raise ValueError('CNLP/CUP resume mode mismatch')
        if self.cup:
            restored = copy.deepcopy(self.cup)
            restored.load_state_dict(state['cup'])
            if restored.steps != state['completed_steps']:
                raise ValueError('CUP and outer completed-step counters differ')
            self.cup = restored
        self.completed_steps = state['completed_steps']
