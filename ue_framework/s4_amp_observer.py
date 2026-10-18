"""Passive AMP accounting using public optimizer hooks and trainer callbacks.

Never changes gradients, loss scale, parameters, scheduling or update ordering.
Native GradScaler remains the sole authority deciding whether SGD.step is called.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch


def tree_hash(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            v = x.detach().contiguous().cpu()
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(repr(key).encode()); visit(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x: visit(item)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def tensors_finite(values):
    flags = [torch.isfinite(v.detach()).all() for v in values if torch.is_tensor(v)]
    if not flags: return True
    # Reduce per device, avoiding one host synchronization per parameter.
    groups = {}
    for flag in flags: groups.setdefault(str(flag.device), []).append(flag)
    return all(bool(torch.stack(group).all().item()) for group in groups.values())


def optimizer_finite(optimizer):
    return tensors_finite(v for state in optimizer.state.values() for v in state.values())


class AMPAuditObserver:
    def __init__(self, directory, *, max_consecutive_skips=16, enforce=True, debug_hashes=False):
        self.directory = Path(directory)
        self.max_consecutive_skips = int(max_consecutive_skips)
        if self.max_consecutive_skips != 16:
            raise ValueError('Recovery AMP skip limit is frozen at16 before CUDA smoke')
        self.enforce = enforce
        self.debug_hashes = debug_hashes
        self.updates = self.skips = self.attempts = self.consecutive_skips = 0
        self.rows = [] if debug_hashes else None
        self.started = time.monotonic()
        self.input_batch = None
        self.input_hash = None
        self.input_ids = []
        self.parameter_hash_before = None
        self.batch_index = -1
        self.total_batches = 0
        self.handles = []

    def on_train_start(self, trainer):
        self.trainer = trainer
        trainer.amp_audit = self
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory/'AMP_SCALE_AND_SKIP_LOG.csv').open('x', newline='')
        self.writer = None
        self.handles.append(trainer.optimizer.register_step_pre_hook(self._before_actual_step))
        self.handles.append(trainer.optimizer.register_step_post_hook(self._after_actual_step))
        self.handles.append(trainer.model.register_forward_pre_hook(self._capture_input))
        self.initial = self.state_fingerprint(trainer) if self.debug_hashes else None

    def on_train_epoch_start(self, trainer):
        self.batch_index = -1

    def on_train_batch_start(self, trainer):
        self.batch_index += 1
        self.total_batches += 1
        self.scale_before = float(trainer.scaler.get_scale())
        self.parameter_hash_before = tree_hash(dict(trainer.model.named_parameters())) if self.debug_hashes else None
        self.updates_before = self.updates
        self.post_clip_norm = None
        self.input_batch = None
        self.input_hash = None
        self.input_ids = []

    def _capture_input(self, module, args):
        if not module.training or not args or not isinstance(args[0], dict): return
        batch = args[0]
        self.input_batch = batch
        self.input_ids = [Path(p).stem for p in batch.get('im_file', [])]
        if self.debug_hashes:
            self.input_hash = tree_hash({k:batch[k] for k in ('img','cls','bboxes','batch_idx') if k in batch})

    def _before_actual_step(self, optimizer, args, kwargs):
        # Called by SGD only when native scaler permits the actual update.
        # Native unscale and clipping have already run. Never touches p.grad.
        trainer = self.trainer
        gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        if self.enforce and (not tensors_finite(gradients) or not tensors_finite([trainer.loss])):
            self.fail(trainer, 'Nonfinite actual-update gradient/loss after native AMP handling')
        if gradients:
            self.post_clip_norm = float(torch.stack([g.detach().float().norm(2) for g in gradients]).norm(2).item())

    def _after_actual_step(self, optimizer, args, kwargs):
        self.updates += 1

    def state_fingerprint(self, trainer):
        ema = getattr(trainer, 'ema', None)
        return {'model':tree_hash(trainer.model.state_dict()),
                'optimizer':tree_hash(trainer.optimizer.state_dict()),
                'scheduler':tree_hash(trainer.scheduler.state_dict()),
                'scaler':tree_hash(trainer.scaler.state_dict()),
                'EMA':tree_hash(ema.ema.state_dict()) if ema else None,
                'EMA_updates':getattr(ema,'updates',None)}

    def on_train_batch_end(self, trainer):
        after = float(trainer.scaler.get_scale())
        executed = self.updates-self.updates_before
        backoff = after < self.scale_before
        skipped = bool(backoff and executed == 0)
        attempted = bool(executed or backoff)
        if executed: self.consecutive_skips = 0
        elif skipped: self.consecutive_skips += 1
        self.skips += int(skipped)
        self.attempts += int(attempted)
        loss_finite = tensors_finite([getattr(trainer,'loss',None),getattr(trainer,'tloss',None)])
        model_finite = tensors_finite(trainer.model.state_dict().values())
        ema = getattr(trainer,'ema',None)
        ema_finite = tensors_finite(ema.ema.state_dict().values()) if ema else True
        opt_finite = optimizer_finite(trainer.optimizer)
        cuda = getattr(trainer,'device',torch.device('cpu')).type == 'cuda'
        raw_loss = float(trainer.loss.detach().sum().item())
        row = {'epoch':int(trainer.epoch)+1,'batch_index':self.batch_index,'total_batches':self.total_batches,
               'raw_loss':raw_loss if math.isfinite(raw_loss) else str(raw_loss),
               'raw_loss_items':json.dumps(trainer.loss_items.detach().cpu().tolist()) if torch.is_tensor(getattr(trainer,'loss_items',None)) else None,
               'amp_scale_before':self.scale_before,'amp_scale_after':after,
               'optimizer_attempt_observed':attempted,'optimizer_update_executed':bool(executed),
               'native_scaler_nonfinite_skip':skipped,'backoff':backoff,
               'consecutive_skips':self.consecutive_skips,'actual_updates_total':self.updates,'skips_total':self.skips,
               'model_finite':model_finite,'EMA_finite':ema_finite,'optimizer_finite':opt_finite,'raw_loss_finite':loss_finite,
               'post_native_clip_unscaled_gradient_norm':self.post_clip_norm,
               'cuda_memory_allocated':torch.cuda.memory_allocated(trainer.device) if cuda else 0,
               'cuda_peak_memory_allocated':torch.cuda.max_memory_allocated(trainer.device) if cuda else 0,
               'elapsed_seconds':time.monotonic()-self.started,'input_ids':json.dumps(self.input_ids),
               'input_tensor_sha256':self.input_hash}
        if self.writer is None:
            self.writer=csv.DictWriter(self.stream,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row);self.stream.flush()
        if self.debug_hashes:
            parameter_hash_after=tree_hash(dict(trainer.model.named_parameters()))
            self.rows.append(dict(row,state=self.state_fingerprint(trainer),parameter_hash_before=self.parameter_hash_before,
                                  parameter_hash_after=parameter_hash_after,parameters_unchanged_on_skip=(self.parameter_hash_before==parameter_hash_after) if skipped else None))
        summary={k:row[k] for k in ('epoch','batch_index','total_batches','actual_updates_total','skips_total','consecutive_skips','amp_scale_after','model_finite','EMA_finite','optimizer_finite','elapsed_seconds')}
        tmp=self.directory/'latest.json.writing';tmp.write_text(json.dumps(summary,allow_nan=False));tmp.replace(self.directory/'latest.json')
        if self.enforce:
            if executed not in (0,1) or (backoff and executed): self.fail(trainer,'Unexpected duplicate update or scaler accounting')
            if not (loss_finite and model_finite and ema_finite and opt_finite): self.fail(trainer,'Nonfinite raw loss/model/EMA/optimizer')
            if not math.isfinite(after) or after<=0: self.fail(trainer,'Invalid native AMP scale')
            if self.consecutive_skips>=self.max_consecutive_skips: self.fail(trainer,'Persistent native AMP nonfinite skips reached predeclared16')

    def fail(self, trainer, reason):
        from .methods.s4_tfss import rng_state
        p=self.directory/f'failure_epoch_{int(trainer.epoch)+1:03d}_batch_{self.batch_index:04d}.pt'
        if p.exists(): raise FileExistsError('Failure snapshot collision')
        torch.save({'reason':reason,'epoch':int(trainer.epoch)+1,'batch_index':self.batch_index,
                    'model':trainer.model.state_dict(),'optimizer':trainer.optimizer.state_dict(),
                    'scheduler':trainer.scheduler.state_dict(),'scaler':trainer.scaler.state_dict(),
                    'rng':rng_state(),'input_batch':self.input_batch,'actual_updates':self.updates,'skips':self.skips},p)
        (self.directory/'failure.json').write_text(json.dumps({'reason':reason,'snapshot':str(p),'image_ids':self.input_ids}))
        raise FloatingPointError(reason)


def attach_amp_observer(model, directory=None, *, enforce=True, debug_hashes=False):
    holder={}
    def start(t):
        observer=AMPAuditObserver(directory or Path(t.save_dir)/'amp_audit',enforce=enforce,debug_hashes=debug_hashes)
        holder['observer']=observer;observer.on_train_start(t)
    model.add_callback('on_train_start',start)
    model.add_callback('on_train_epoch_start',lambda t:holder['observer'].on_train_epoch_start(t))
    model.add_callback('on_train_batch_start',lambda t:holder['observer'].on_train_batch_start(t))
    model.add_callback('on_train_batch_end',lambda t:holder['observer'].on_train_batch_end(t))
    return holder
