"""Identity-bound FP32 epoch checkpoints, without claiming worker-prefetch equivalence."""
import random
import hashlib
import json
import os
import shutil
import time
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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path):
    if os.name == 'nt':
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_header(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != 's4.at.fullstate.v1':
        raise ValueError(f'Unexpected retained checkpoint schema: {path}')
    if not isinstance(payload.get('completed_epochs'), int) or payload['completed_epochs'] < 1:
        raise ValueError(f'Invalid retained checkpoint epoch: {path}')
    return payload


def _validate_fullstate(path, expected_epoch, expected_identity):
    loaded = _load_header(path)
    required = ('model', 'optimizer', 'scheduler', 'scaler', 'pgd_rng', 'pgd_calls',
                'rng', 'ema', 'ema_updates', 'resume_safe_optimizer_boundary')
    if loaded['completed_epochs'] != expected_epoch or loaded.get('identity') != expected_identity:
        raise ValueError('Checkpoint epoch/identity mismatch')
    if any(key not in loaded for key in required):
        raise ValueError('Checkpoint is missing full training state')
    if loaded['resume_safe_optimizer_boundary'] is not True:
        raise ValueError('Checkpoint is not at the validated optimizer accumulation boundary')
    if not isinstance(loaded['model'], dict) or not loaded['model']:
        raise ValueError('Checkpoint model state is empty')
    if not isinstance(loaded['optimizer'], dict) or 'param_groups' not in loaded['optimizer']:
        raise ValueError('Checkpoint optimizer state is malformed')
    if not isinstance(loaded['scheduler'], dict) or not isinstance(loaded['scaler'], dict):
        raise ValueError('Checkpoint scheduler/scaler state is malformed')
    if not isinstance(loaded['rng'], dict) or loaded['pgd_rng'] is None or loaded['pgd_calls'] is None:
        raise ValueError('Checkpoint RNG/PGD state is malformed')
    if loaded['ema'] is None or loaded['ema_updates'] is None:
        raise ValueError('Checkpoint EMA state is missing')


def save_retained(root, payload, *, milestone_every=25, final_epoch=200):
    """Atomically keep two rolling states and immutable 25-epoch/final milestones.

    The storage profile uses two fixed slots. Updating one slot with os.replace
    preserves the other recent state and avoids a cleanup sweep over old files.
    Milestones share an inode with the corresponding recent slot where supported.
    """
    root = Path(root)
    epoch = payload.get('completed_epochs')
    if not isinstance(epoch, int) or not 1 <= epoch <= final_epoch:
        raise ValueError('Cloud full-state checkpoint epoch is outside 1..200')
    if milestone_every != 25 or final_epoch != 200:
        raise ValueError('Cloud retention policy is fixed at every25/final200')
    identity = payload.get('identity')
    if not isinstance(identity, dict):
        raise ValueError('Checkpoint identity is missing')
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / 'retention.json'
    previous = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else None
    if previous:
        if previous.get('schema') != 's4.at.checkpoint-retention.v1':
            raise ValueError('Checkpoint retention index schema mismatch')
        if previous.get('identity') != identity:
            raise ValueError('Checkpoint retention index belongs to a different run')
        if epoch != previous.get('completed_epochs', 0) + 1:
            raise ValueError('Checkpoint epochs must be saved contiguously')
    elif epoch != 1:
        raise ValueError('First cloud checkpoint must be completed epoch 1')

    slot = root / f'recent_{epoch % 2}.pt'
    temporary = root / f'.{slot.name}.{os.getpid()}.{time.time_ns()}.tmp'
    milestone = root / f'milestone_{epoch:03d}.pt' if epoch % milestone_every == 0 or epoch == final_epoch else None
    milestone_temporary = (root / f'.{milestone.name}.{os.getpid()}.{time.time_ns()}.tmp'
                           if milestone is not None else None)
    try:
        with temporary.open('xb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256(temporary)
        _validate_fullstate(temporary, epoch, identity)
        if milestone is not None:
            if milestone.exists():
                if _sha256(milestone) != digest:
                    raise FileExistsError(f'Milestone collision with different bytes: {milestone}')
            else:
                try:
                    os.link(temporary, milestone)
                except OSError:
                    # Some restricted filesystems disallow hard links. Preserve
                    # the milestone with its own verified atomic copy instead.
                    with milestone_temporary.open('xb') as stream:
                        with temporary.open('rb') as source:
                            shutil.copyfileobj(source, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if _sha256(milestone_temporary) != digest:
                        raise IOError('Copied checkpoint milestone hash mismatch')
                    _validate_fullstate(milestone_temporary, epoch, identity)
                    os.replace(milestone_temporary, milestone)
        # Only this one named rolling slot is atomically replaced. Its mate
        # remains available until the following epoch is fully written.
        os.replace(temporary, slot)
        _fsync_directory(root)

        prior_recent = previous.get('recent', []) if previous else []
        recent = [entry for entry in prior_recent if entry.get('path') != slot.name]
        recent.append(dict(path=slot.name, completed_epochs=epoch, sha256=digest))
        recent.sort(key=lambda entry: entry['completed_epochs'])
        recent = recent[-2:]
        milestones = list(previous.get('milestones', [])) if previous else []
        if milestone is not None:
            milestones.append(dict(path=milestone.name, completed_epochs=epoch, sha256=digest))
        from .common import atomic
        index = dict(schema='s4.at.checkpoint-retention.v1', identity=identity,
                     policy='two_atomic_rolling_slots_plus_each_25_epochs_and_final_200',
                     completed_epochs=epoch, recent=recent, milestones=milestones,
                     latest_path=slot.name, latest_sha256=digest,
                     final_epoch_retained=(epoch == final_epoch or bool(
                         previous and previous.get('final_epoch_retained'))))
        atomic(index_path, index)
        _fsync_directory(root)
        return index
    finally:
        if temporary.exists():
            temporary.unlink()
        if milestone_temporary is not None and milestone_temporary.exists():
            milestone_temporary.unlink()


def retained_checkpoint_epochs(root):
    """Read the small fixed set of cloud checkpoints, including unindexed crash tails."""
    root = Path(root)
    paths = list(root.glob('recent_[01].pt')) + list(root.glob('milestone_*.pt'))
    epochs = set()
    for path in paths:
        epochs.add(_load_header(path)['completed_epochs'])
    return epochs


def find_retained_checkpoint(root, epoch):
    root = Path(root)
    paths = list(root.glob('recent_[01].pt')) + list(root.glob('milestone_*.pt'))
    matches = [path for path in paths if _load_header(path)['completed_epochs'] == epoch]
    if not matches:
        raise FileNotFoundError(f'No retained checkpoint exists for epoch {epoch}')
    if len(matches) == 1:
        return matches[0]
    digests = {_sha256(path) for path in matches}
    if len(digests) != 1:
        raise ValueError(f'Conflicting retained checkpoints exist for epoch {epoch}')
    milestones = [path for path in matches if path.name.startswith('milestone_')]
    return milestones[0] if milestones else matches[0]
