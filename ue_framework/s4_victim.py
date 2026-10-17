"""Finite-state safety and retained checkpoints around the unchanged native victim.

No model/loss/update/preprocessing/augmentation implementation is replaced.
Native data-loader worker prefetch RNG is not asserted recoverable; an interrupted
victim cannot be automatically resumed without a separate exact-state validation.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from .methods.s4_tfss import rng_state


class S4VictimTrainer(DetectionTrainer):
    def _finite(self):
        for value in (getattr(self,'loss',None),getattr(self,'tloss',None)):
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite victim loss; native NaN recovery prohibited')
        for model in (self.model,self.ema.ema if self.ema else None):
            if model is not None and not all(torch.isfinite(v).all() for v in model.state_dict().values() if torch.is_tensor(v)):
                raise FloatingPointError('Nonfinite victim/EMA model; no repair or checkpoint sanitization')

    def optimizer_step(self):
        self._finite()
        if not all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite victim gradient; stop before update')
        return super().optimizer_step()

    def _handle_nan_recovery(self,epoch):
        self._finite()
        if self.metrics and any(not __import__('math').isfinite(float(v)) for v in self.metrics.values()):
            raise FloatingPointError('Nonfinite validator output; automatic recovery prohibited')
        return False

    def save_model(self):
        self._finite()
        saved=super().save_model()
        if saved:
            root=Path(self.save_dir)/'retained_epoch_checkpoints';root.mkdir(exist_ok=True)
            dest=root/f'epoch_{self.epoch+1:03d}.pt'
            if dest.exists():raise FileExistsError('No overwrite of completed victim epoch snapshot')
            shutil.copyfile(self.last,dest)
        return saved


def attach_safety_callbacks(model):
    def batch_start(t):
        if t.args.batch!=36 or t.batch_size!=36:
            raise ValueError('Matched victim batch changed (including native OOM retry); stop')
    def snapshot(t):
        # Save original FP32 state and scheduler/RNG metadata, without changing
        # native last/best serialization or optimizer arithmetic.
        root=Path(t.save_dir)/'state_snapshots';root.mkdir(exist_ok=True)
        epoch=t.epoch+1
        if epoch%10==0 or epoch==200:
            target=root/f'epoch_{epoch:03d}_full_state.pt'
            if target.exists():raise FileExistsError('Victim full state collision')
            torch.save(dict(epoch=epoch,model={k:v.detach().cpu() for k,v in t.model.state_dict().items()},
                optimizer=deepcopy(t.optimizer.state_dict()),scheduler=t.scheduler.state_dict(),
                scaler=t.scaler.state_dict(),rng=rng_state(),train_args=vars(t.args),
                data_loader_worker_prefetch_rng_verified=False,automatic_resume_allowed=False),target)
        (root/'latest.json').write_text(json.dumps(dict(completed_epoch=epoch,
            snapshot_every_epochs=10,native_epoch_snapshots_retained=True,
            automatic_resume_allowed=False,
            reason='worker/prefetch RNG exact continuation needs verification before any interrupted victim recovery')))
    model.add_callback('on_train_batch_start',batch_start)
    model.add_callback('on_fit_epoch_end',snapshot)
