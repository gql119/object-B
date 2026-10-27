"""Identity-bound FP32 epoch checkpoints, without claiming worker-prefetch equivalence."""
import random
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def capture(model, optimizer, scheduler, scaler, attack, identity, completed_epochs, ema=None, loader=None):
    pending = any(p.grad is not None and bool(p.grad.detach().abs().max() > 0) for p in model.parameters())
    return dict(schema='s4.at.fullstate.v1', identity=deepcopy(identity),
                completed_epochs=completed_epochs, model=deepcopy(model.state_dict()),
                optimizer=deepcopy(optimizer.state_dict()), scheduler=deepcopy(scheduler.state_dict()),
                scaler=deepcopy(scaler.state_dict()), pgd_rng=attack.generator.get_state(),
                pgd_calls=attack.calls, rng=rng_state(next(model.parameters()).device),
                ema=deepcopy(ema.ema.state_dict()) if ema else None,
                ema_updates=ema.updates if ema else None,
                loader_rng=loader.generator.get_state() if loader and loader.generator else None,
                resume_safe_optimizer_boundary=not pending,
                data_loader_worker_prefetch_rng_verified=False,
                resume_semantics='same run/protocol at completed epoch and cleared optimizer boundary; '
                                 'native workers16 prefetch is not a bitwise trajectory guarantee')


def restore(payload, model, optimizer, scheduler, scaler, attack, identity, ema=None, loader=None):
    if payload['schema'] != 's4.at.fullstate.v1' or payload['identity'] != identity:
        raise ValueError('Checkpoint is not from this exact AT run/source/config/data identity')
    if not payload['resume_safe_optimizer_boundary']:
        raise ValueError('Pending accumulated gradients: epoch checkpoint is not an allowed resume boundary')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['scaler'])
    attack.generator.set_state(payload['pgd_rng'])
    attack.calls = payload['pgd_calls']
    if ema is not None:
        ema.ema.load_state_dict(payload['ema'], strict=True)
        ema.updates = payload['ema_updates']
    if loader is not None and loader.generator is not None and payload['loader_rng'] is not None:
        loader.generator.set_state(payload['loader_rng'])
    restore_rng(payload['rng'], next(model.parameters()).device)
    return payload['completed_epochs']


def save_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        torch.save(payload, f)
