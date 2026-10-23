"""S4 resumable outer loop: same S0 shuffle/EOT/Adam/frequency/scheduler math.

Derived from the numerically replayed full S2 orchestration; all CNLP/CUP calls
and dependencies removed. Core Parent files remain byte-identical.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from ..data_utils import load_image_rgb_float, read_yolo_annotations, label_path_for_image
from .legacy_kproto_ret import _snapshot_metrics
from .semantic_coordinates import CLASSES
from .s4_tfss import rng_state, restore_rng, model_hash


def optimize_s4(trainer, images, label_dir, prototypes, output, identity, *, stop_after=0, resume=None, observe=None, formal_lock=None):
    """stop_after is an admission interruption/step bound, never a formal schedule override."""
    from coco_s4.protocol import expected_noise_schedule
    schedule = expected_noise_schedule(getattr(trainer, 'num_classes', 20), len(images))
    class_names = getattr(trainer, 'class_names', CLASSES)
    if not stop_after:
        if not formal_lock or formal_lock.get('noise_schedule') != schedule:
            raise ValueError('Missing frozen S4 schedule')
    output = Path(output).absolute()
    if os.name == 'nt' and not str(output).startswith('\\\\?\\'):
        value = str(output)
        value = '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
        output = Path(value)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'checkpoints').mkdir()
    opt = torch.optim.Adam([{'params': [trainer.fourier_coeff], 'lr': trainer.universal_lr_fourier}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=trainer.universal_epochs, eta_min=1e-5)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
    state = dict(epoch=0, cursor=0, order=None, step=0, diagnostics=[])
    expected_images = hashlib.sha256('\n'.join(str(Path(p).resolve()) for p in images).encode()).hexdigest()
    (output/'ordered_support_manifest.json').write_text(json.dumps([
        dict(index=i,image_id=Path(p).stem,path=str(Path(p).resolve())) for i,p in enumerate(images)],indent=2))
    identity = dict(identity, ordered_images_sha256=expected_images)
    frozen_before = model_hash(trainer.surrogate)
    adaptive_before = model_hash(trainer.adaptive.model)
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['identity'] != identity or saved['boundary'] != 'complete_optimizer_step':
            raise ValueError('Resume config/cache/source/image identity or checkpoint boundary mismatch')
        if saved['frozen_model_sha256'] != frozen_before or saved['adaptive_model_sha256'] != adaptive_before:
            raise ValueError('Resume model initialization changed')
        for name in ['fourier_coeff', 'suppress_small', 'freq_score', 'freq_usage', 'freq_active_idx']:
            with torch.no_grad(): getattr(trainer, name).copy_(saved['trainer'][name].to(trainer.device))
        trainer._adaptive_calls = saved['trainer']['adaptive_calls']
        trainer._semantic_calls = saved['trainer']['semantic_calls']
        opt.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        state = saved['cursor_state']
        if 'diagnostics_journal' in saved:
            reference=saved['diagnostics_journal']
            with Path(reference['path']).open('rb') as stream:prefix=stream.read(reference['bytes'])
            if hashlib.sha256(prefix).hexdigest()!=reference['sha256']:
                raise ValueError('Resume diagnostic journal prefix changed')
            state['diagnostics']=[json.loads(line) for line in prefix.decode().splitlines()]
            if len(state['diagnostics'])!=state['step']:
                raise ValueError('Journal/Adam boundary mismatch')
        if state['step'] > 0:
            # PyTorch's scheduler ordering marker is not part of Adam.state_dict.
            # A restored completed-update boundary has already performed steps.
            opt._opt_called = True
        restore_rng(saved['rng'])
        if rng_state()['python'] != saved['rng']['python'] or not torch.equal(torch.get_rng_state(), saved['rng']['torch']):
            raise ValueError('RNG restore failed')
        for p in opt.param_groups[0]['params']:
            if opt.state[p] and not torch.isfinite(opt.state[p]['exp_avg']).all():
                raise ValueError('Restored Adam state nonfinite')
    start = time.perf_counter()
    journal_path=output/'diagnostics_journal.jsonl';journal_hash=hashlib.sha256();journal_bytes=0
    with journal_path.open('xb') as journal:
        for row in state['diagnostics']:
            line=(json.dumps(row,allow_nan=False)+'\n').encode();journal.write(line);journal_hash.update(line);journal_bytes+=len(line)
    checkpoint = Path(resume).resolve() if resume else None
    status = dict(state='running', epoch=min(state['epoch'] + 1, trainer.universal_epochs),
        steps=state['step'], epoch_image_cursor=state['cursor'], elapsed_seconds=0.,
        identity=identity, latest=state['diagnostics'][-1] if state['diagnostics'] else {},
        resume_from=str(resume) if resume else None)

    def save_checkpoint(*,live=False):
        nonlocal checkpoint
        pack = dict(identity=identity, boundary='complete_optimizer_step', cursor_state=state.copy(),
            trainer={k: getattr(trainer, k).detach().cpu() for k in
                ['fourier_coeff', 'suppress_small', 'freq_score', 'freq_usage', 'freq_active_idx']},
            optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
            frozen_model_sha256=frozen_before, adaptive_model_sha256=adaptive_before)
        pack['trainer'].update(adaptive_calls=trainer._adaptive_calls, semantic_calls=trainer._semantic_calls)
        target = (output/'latest_continuation.pt') if live else output / 'checkpoints' / f"step_{state['step']:06d}_epoch_{state['epoch']:02d}_cursor_{state['cursor']:05d}.pt"
        if live:
            pack['cursor_state']=dict(state,diagnostics=[])
            pack['diagnostics_journal']=dict(path=str(journal_path),bytes=journal_bytes,sha256=journal_hash.hexdigest())
        if target.exists() and not live:
            return
        temp = target.with_suffix('.writing')
        # Python file handles avoid the native writer's Windows long-path limit.
        with temp.open('xb') as stream:
            torch.save(pack, stream)
        temp.replace(target)
        if live:
            (output/'live_checkpoint.json').write_text(json.dumps(dict(path=str(target),step=state['step'],epoch=state['epoch'],cursor=state['cursor'])))
            return
        checkpoint=target
        (output / 'latest_checkpoint.json').write_text(json.dumps({'path': str(checkpoint),
            'step': state['step'], 'epoch': state['epoch'], 'cursor': state['cursor']}))

    with (output / 'diagnostics.csv').open('w', newline='') as f:
        writer = None
        if state['diagnostics']:
            writer = csv.DictWriter(f, fieldnames=list(state['diagnostics'][0])); writer.writeheader()
            writer.writerows(state['diagnostics']); f.flush()
        while state['epoch'] < trainer.universal_epochs:
            epoch = state['epoch']; trainer._epoch = epoch
            if state['order'] is None:
                state['order'] = np.random.permutation(len(images)); state['cursor'] = 0
            order = state['order']
            while state['cursor'] < len(order):
                begin = state['cursor']
                end = min(len(order), begin + trainer.universal_batch_size)
                opt.zero_grad(set_to_none=True)
                trainer._active_idx = trainer._get_active_freq_indices(epoch, state['step'])
                trainer._active_coords = [tuple(trainer.freq_candidate_coords[int(i)]) for i in trainer._active_idx.cpu().tolist()]
                items, item_ids = [], []
                for index in order[begin:end]:
                    path = images[int(index)]; im = load_image_rgb_float(path)
                    ann = read_yolo_annotations(label_path_for_image(path, str(label_dir)))
                    support, _, _ = trainer._build_support(im.shape, ann, image_path=path)
                    if support.sum() > 10:
                        items.append((im, ann, support))
                        item_ids.append(str(Path(path).resolve()))
                metrics = []
                if items:
                    trainer.s4_item_paths = item_ids
                    trainer.s4_measure_protection = begin == 0
                    for eot in range(max(1, trainer.eot_samples)):
                        result = trainer._batch_loss(items, prototypes)
                        if result is None: raise ValueError('Missing canonical supervision')
                        loss, values = result
                        if not torch.isfinite(loss): raise FloatingPointError('Parent+CNLP loss nonfinite')
                        if observe: observe(trainer, items, loss, values, state['step'], eot)
                        if trainer.tfss_mode != 'S0':
                            with (output/'spatial_per_image.jsonl').open('a') as spatial:
                                for row in trainer.s4_spatial_rows:
                                    spatial.write(json.dumps(dict(row,epoch=epoch+1,step=state['step']+1,eot=eot+1),allow_nan=False)+'\n')
                        loss.backward(); metrics.append(_snapshot_metrics(values))
                state['cursor'] = end
                if not metrics:
                    continue
                trainer.fourier_coeff.grad.div_(len(metrics))
                grad_norm = float(trainer.fourier_coeff.grad.norm())
                if not np.isfinite(grad_norm): raise FloatingPointError('Fourier gradient nonfinite')
                if trainer.enable_adaptive_freq_basis and epoch < trainer.adaptive_freq_warmup_epochs:
                    with torch.no_grad():
                        trainer.freq_score[trainer._active_idx] += trainer.fourier_coeff.grad[trainer._active_idx].abs().mean(dim=1)
                        trainer.freq_usage[trainer._active_idx] += 1
                opt.step()
                latest = {k: (';'.join(row[k] for row in metrics) if k == 'selected_proto_ids'
                    else (sum(row[k] for row in metrics if row[k] is not None) / sum(row[k] is not None for row in metrics)
                    if any(row[k] is not None for row in metrics) else None)) for k in metrics[0]}
                latest.update(epoch=epoch + 1, step=state['step'], grad_norm_fourier=grad_norm)
                worst = sorted(trainer.bases, key=lambda c: (-latest['L_orth_' + class_names[c]], c))[:trainer.worst_k]
                for j, cls in enumerate(worst):
                    latest[f'worst_class_{j+1}'] = cls
                    latest[f'worst_Lorth_{j+1}'] = latest['L_orth_' + class_names[cls]]
                if not all(np.isfinite(v) for v in latest.values() if isinstance(v, (float, int))):
                    raise FloatingPointError('CNLP diagnostic nonfinite')
                state['diagnostics'].append(latest)
                line=(json.dumps(latest,allow_nan=False)+'\n').encode()
                with journal_path.open('ab') as journal:journal.write(line)
                journal_hash.update(line);journal_bytes+=len(line)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(latest)); writer.writeheader()
                writer.writerow(latest); f.flush()
                state['step'] += 1
                save_checkpoint(live=True)
                status = dict(state='running', epoch=epoch + 1, steps=state['step'],
                    epoch_image_cursor=end, elapsed_seconds=time.perf_counter() - start,
                    identity=identity, latest=latest, resume_from=str(resume) if resume else None)
                (output / 'status.json').write_text(json.dumps(status, indent=2))
                if state['step'] % 50 == 0:
                    save_checkpoint()
                if stop_after and state['step'] >= stop_after:
                    if model_hash(trainer.surrogate)!=frozen_before or model_hash(trainer.adaptive.model)!=adaptive_before:
                        raise ValueError('Bounded production mutated Frozen/Adaptive base state')
                    save_checkpoint()
                    status.update(state='bounded_interruption', checkpoint=str(checkpoint), schedule_complete=False,
                                  frozen_model_unchanged=True,adaptive_model_unchanged=True)
                    (output / 'status.json').write_text(json.dumps(status, indent=2))
                    return status
            if trainer.enable_adaptive_freq_basis and epoch + 1 == trainer.adaptive_freq_warmup_epochs:
                score = trainer.freq_score / trainer.freq_usage.clamp_min(1)
                trainer.freq_active_idx = torch.topk(score, trainer.freq_active_num_bases).indices.sort().values
            scheduler.step()
            epoch_rows=[r for r in state['diagnostics'] if r['epoch']==epoch+1]
            epoch_summary=dict(epoch=epoch+1,Adam_updates=len(epoch_rows),
                aggregation='outer-step means; per-image distribution summaries; EOT observations not unique images',
                protection_gradient_scope='first canonical batch of each epoch, both EOT draws')
            for k in epoch_rows[0]:
                numbers=[r[k] for r in epoch_rows if isinstance(r[k],(int,float))]
                if numbers:
                    epoch_summary[k]=max(numbers) if k in ('s4_actual_linf','s4_overlap_linf') else sum(numbers)/len(numbers)
            epoch_summary['epoch']=epoch+1
            with (output/'epoch_summary.jsonl').open('a') as stream:
                stream.write(json.dumps(epoch_summary,allow_nan=False)+'\n')
            state.update(epoch=epoch + 1, cursor=0, order=None)
            save_checkpoint(live=True)
            save_checkpoint()
    if model_hash(trainer.surrogate) != frozen_before or model_hash(trainer.adaptive.model) != adaptive_before:
        raise ValueError('Frozen/Adaptive base model changed')
    if not stop_after and (state['step'] != schedule['optimizer_steps'] or state['epoch'] != schedule['epochs']):
        raise ValueError('S4 must complete the frozen dataset-specific 24-epoch schedule')
    if state['step'] == 0:
        raise ValueError('No valid optimization steps')
    saved_idx = trainer.freq_active_idx
    pack = dict(coords=[list(trainer.freq_candidate_coords[int(i)]) for i in saved_idx.cpu().tolist()],
        fourier_coeff=trainer.fourier_coeff.detach()[saved_idx].cpu(), suppress_small=trainer.suppress_small.detach().cpu(),
        method='legacy_kproto_ret', support_mode=trainer.support_mode, target_class_id=trainer.target_class_id, eps=trainer.eps,
        feature_layer=trainer.feature_layer, s4_identity=identity, num_prototypes=trainer.num_prototypes,
        margin=trainer.margin)
    with (output / 'global_params.pt').open('xb') as stream:
        torch.save(pack, stream)
    status.update(state='complete', steps=state['step'], epochs_completed=state['epoch'], schedule_complete=True,
                  epoch=state['epoch'], checkpoint=str(checkpoint), frozen_model_unchanged=True,
                  adaptive_model_unchanged=True, elapsed_seconds=time.perf_counter() - start)
    (output / 'status.json').write_text(json.dumps(status, indent=2))
    return status
