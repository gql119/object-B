"""CPU-only tests for the cloud checkpoint retention contract."""
import json
import tempfile
from pathlib import Path
import torch

from .checkpoint import find_retained_checkpoint, retained_checkpoint_epochs, save_retained


class RetainedTempDirectory:
    """Create an isolated fixture tree without recursively deleting it on exit."""
    def __enter__(self):
        self.path = Path(tempfile.mkdtemp(prefix='s4-at-checkpoint-policy-'))
        return self.path

    def __exit__(self, exc_type, exc, traceback):
        return False


def payload(epoch, identity):
    weight = torch.tensor([float(epoch), 2.0])
    return dict(schema='s4.at.fullstate.v1', identity=identity, completed_epochs=epoch,
                model={'weight': weight}, optimizer={'state': {}, 'param_groups': [{'lr': .01}]},
                scheduler={'last_epoch': epoch}, scaler={'scale': 1024.0},
                pgd_rng=torch.tensor([epoch], dtype=torch.uint8), pgd_calls=epoch * 10,
                rng={'python': (3, (1, 2, 3), None), 'numpy': ('MT19937',), 'cpu': torch.tensor([epoch])},
                ema={'weight': weight.clone()}, ema_updates=epoch * 36,
                resume_safe_optimizer_boundary=True)


def run():
    identity = {'run_id': 'storage-policy-cpu-test', 'source_commit': 'fixture'}
    with RetainedTempDirectory() as temp:
        root = Path(temp) / 'training' / 'full_state'
        for epoch in range(1, 201):
            save_retained(root, payload(epoch, identity))
        epochs = retained_checkpoint_epochs(root)
        expected_milestones = set(range(25, 201, 25))
        if not expected_milestones.issubset(epochs):
            raise AssertionError(f'missing milestone epochs: {expected_milestones - epochs}')
        if not {199, 200}.issubset(epochs):
            raise AssertionError('latest two full-state epochs are missing')
        if len(list(root.glob('recent_[01].pt'))) != 2:
            raise AssertionError('rolling window must use exactly two named slots')
        if len(list(root.glob('milestone_*.pt'))) != 8:
            raise AssertionError('expected milestones at 25..200, every 25 epochs')
        index = json.loads((root / 'retention.json').read_text(encoding='utf-8'))
        if index['completed_epochs'] != 200 or not index['final_epoch_retained']:
            raise AssertionError('final full-state receipt is missing')
        latest = find_retained_checkpoint(root, 200)
        restored = torch.load(latest, map_location='cpu', weights_only=False)
        if restored['completed_epochs'] != 200 or restored['optimizer']['param_groups'][0]['lr'] != .01:
            raise AssertionError('full-state round-trip did not preserve optimizer/epoch state')
        before = sorted((p.name, p.stat().st_size) for p in root.glob('*.pt'))
        failure_root = Path(temp) / 'failure' / 'full_state'
        save_retained(failure_root, payload(1, identity))
        before_failure = sorted((p.name, p.read_bytes()) for p in failure_root.iterdir())
        broken = payload(2, identity)
        broken['scaler'] = None
        try:
            save_retained(failure_root, broken)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed checkpoint unexpectedly passed validation')
        after_failure = sorted((p.name, p.read_bytes()) for p in failure_root.iterdir())
        if before_failure != after_failure:
            raise AssertionError('failed round-trip changed previously retained checkpoint/index bytes')
        if before != sorted((p.name, p.stat().st_size) for p in root.glob('*.pt')):
            raise AssertionError('out-of-range epoch changed retained checkpoint files')
        return dict(state='PASS_CPU_ONLY', epochs=200, retained_epoch_entries=len(epochs),
                    recent_slots=2, milestones=sorted(expected_milestones),
                    final200=True, failed_roundtrip_preserved_prior_files=True,
                    no_training_or_cuda=True)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
