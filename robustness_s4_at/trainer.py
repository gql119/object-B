"""Native detection training with PGD only at the normalized input sink."""
import json
from pathlib import Path
import torch
from ultralytics.engine import trainer as native
from ue_framework.s4_victim import S4VictimTrainer
from ue_framework.s4_amp_observer import attach_amp_observer
from .pgd import PGDAttack
from .checkpoint import capture, restore, save_once
from .common import atomic, sha, write_once, now
from .native_loop import phase_preserving_loop


def native_amp_check(model):
    """Accepted ST AMP readiness calculation, without any downloaded weights."""
    flags = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        sample = torch.zeros((1, 3, 256, 256), device=next(model.parameters()).device)
        with torch.inference_mode():
            full = model(sample)
            with torch.autocast(device_type='cuda', enabled=True):
                mixed = model(sample)
        full = full[0] if isinstance(full, (tuple, list)) else full
        mixed = mixed[0] if isinstance(mixed, (tuple, list)) else mixed
        return bool(full.shape == mixed.shape and torch.isfinite(full).all()
                    and torch.isfinite(mixed).all() and torch.allclose(full.float(), mixed.float(), atol=.5))
    finally:
        for module, flag in flags:
            module.training = flag


class ATTrainer(S4VictimTrainer):
    _do_train = phase_preserving_loop()

    def __init__(self, *, at_config, identity, audit_root, validation_only=False,
                 resume_fullstate=None, **kwargs):
        self.at_identity, self.at_root = identity, Path(audit_root)
        self.validation_only = validation_only
        self.resume_fullstate = Path(resume_fullstate) if resume_fullstate else None
        self.at_batches = self.prior_updates = self.prior_skips = 0
        self.at_last_opt_step = -1
        super().__init__(**kwargs)
        self.attack = PGDAttack(at_config, self.device, seed=0)
        self.at_holder = attach_amp_observer(self, self.at_root / 'amp', enforce=True,
                                             debug_hashes=validation_only)
        self.add_callback('on_fit_epoch_end', self._at_epoch_checkpoint)
        self.add_callback('on_train_batch_end', self._at_progress)

    def _setup_train(self):
        old = native.check_amp
        try:
            native.check_amp = native_amp_check
            super()._setup_train()
        finally:
            native.check_amp = old
        if not self.validation_only and (self.args.batch != 36 or self.batch_size != 36 or not self.amp):
            raise ValueError('Formal batch36/AMP changed')

    def _build_train_pipeline(self):
        if hasattr(self, '_at_expected_batch') and self.batch_size != self._at_expected_batch:
            raise RuntimeError('Native OOM retry tried to change the declared AT batch; stop')
        self._at_expected_batch = self.batch_size
        result = super()._build_train_pipeline()
        self._at_pipeline_built = True
        return result

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if not self.validation_only and (self.args.batch != 36 or self.batch_size != 36):
            raise ValueError('AT batch silently changed')
        adv, audit = self.attack(self.model, batch)
        self.at_batches += 1
        audit.update(batch=self.at_batches, epoch=self.epoch + 1, UTC=now(),
                     run_id=self.at_identity['run_id'], normalized_after_native_augmentation=True,
                     actual_input_shape=list(adv.shape), actual_input_device=str(adv.device))
        self.at_root.mkdir(parents=True, exist_ok=True)
        with (self.at_root / 'pgd_batches.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(audit, allow_nan=False) + '\n')
        self.at_latest = audit
        return dict(batch, img=adv)

    def _at_progress(self, trainer):
        observer = self.at_holder['observer']
        atomic(self.at_root / 'progress.json', dict(run_id=self.at_identity['run_id'],
               epoch=self.epoch + 1, completed_epochs=self.epoch, at_batches=self.at_batches,
               actual_updates=self.prior_updates + observer.updates,
               legal_amp_skips=self.prior_skips + observer.skips,
               loss=float(self.loss.detach()), latest_PGD=self.at_latest, UTC=now()))

    def save_model(self):
        if any(v.is_floating_point() and bool((v.detach().abs() > 65504).any())
               for v in self.ema.ema.state_dict().values()):
            raise FloatingPointError('Native EMA half serialization would overflow; no sanitization')
        return super().save_model()

    def _at_epoch_checkpoint(self, trainer):
        epoch = self.epoch + 1
        if epoch > self.epochs:
            raise ValueError('Unexpected extra fit epoch callback')
        payload = capture(self.model, self.optimizer, self.scheduler, self.scaler,
                          self.attack, self.at_identity, epoch, self.ema, self.train_loader)
        observer = self.at_holder['observer']
        payload.update(best_fitness=self.best_fitness, at_batches=self.at_batches,
                       actual_updates=self.prior_updates + observer.updates,
                       legal_amp_skips=self.prior_skips + observer.skips,
                       native_epoch_checkpoint_sha256=sha(self.last),
                       native_loop_phase=dict(last_opt_step=self.at_last_opt_step,
                           accumulate=int(self.accumulate), batches_per_epoch=len(self.train_loader)))
        # A cleared buffer alone is insufficient: retain only a phase whose
        # latest attempt was the last completed batch as a safe resume boundary.
        payload['resume_safe_optimizer_boundary'] &= self.at_last_opt_step == epoch * len(self.train_loader) - 1
        save_once(Path(self.save_dir) / 'full_state' / f'epoch_{epoch:03d}.pt', payload)

    def resume_training(self, ckpt):
        if self.resume_fullstate is None:
            if ckpt is not None or self.resume:
                raise ValueError('Only explicit same-run fullstate resume is allowed')
            return
        # Keep RNG ByteTensors on CPU; state_dict loaders move model/optimizer
        # tensors to the existing devices without corrupting RNG restoration.
        payload = torch.load(self.resume_fullstate, map_location='cpu', weights_only=False)
        phase = payload.get('native_loop_phase', {})
        nb = len(self.train_loader)
        if phase.get('batches_per_epoch') != nb or phase.get('last_opt_step') != payload['completed_epochs'] * nb - 1:
            raise ValueError('Missing or incompatible native optimizer-attempt phase')
        epoch = restore(payload, self.model, self.optimizer, self.scheduler, self.scaler,
                        self.attack, self.at_identity, self.ema, self.train_loader)
        if not 0 < epoch < self.epochs:
            raise ValueError('AT checkpoint is complete or invalid for resume')
        self.start_epoch = epoch
        self.at_last_opt_step = phase['last_opt_step']
        self.accumulate = phase['accumulate']
        self.at_batches, self.best_fitness = payload['at_batches'], payload['best_fitness']
        self.prior_updates, self.prior_skips = payload['actual_updates'], payload['legal_amp_skips']
        if epoch >= self.epochs - self.args.close_mosaic:
            self._close_dataloader_mosaic()
            self.train_loader.reset()

    def final_eval(self):
        write_once(self.at_root / 'final_eval_deferred.json', dict(
            primary='last.pt@200', secondary='best.pt', native_checkpoint_strip_disabled=True,
            clean_evaluation_by_separate_plain_validator=True))
