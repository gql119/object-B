"""Bound production Teacher phase and bounded CUDA validation of that same route.

Teacher has no validation split. Post-training official evaluation is a distinct
serial phase and can never select a checkpoint or update method statistics.
"""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import sys

from .runtime import ROOT, DATA_ROOT, atomic, commit, load, now, require_review, sha, verify_release, write_once


def local_amp_check(model):
    """Historical accepted selfcheck, without external reference weight downloads.

    Native AMP/scaler/SGD remains unchanged. This is the same preflight used by
    the accepted original defense runs, now measured on the COCO80 random model.
    """
    import torch
    was_training = model.training
    try:
        model.eval()
        sample = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=next(model.parameters()).device)
        with torch.inference_mode():
            full = model(sample)
            with torch.autocast(device_type='cuda', enabled=True):
                mixed = model(sample)
        full = full[0] if isinstance(full, (tuple, list)) else full
        mixed = mixed[0] if isinstance(mixed, (tuple, list)) else mixed
        return bool(full.shape == mixed.shape and torch.isfinite(full).all() and torch.isfinite(mixed).all()
                    and torch.allclose(full.float(), mixed.float(), atol=.5))
    finally:
        model.train(was_training)


def baseline_probe(t, output):
    """Pre-update candidate native loss vs independently instantiated native model."""
    import numpy as np
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ue_framework.methods.s4_tfss import rng_state, restore_rng
    # This extra read/forward exists only in the bounded, non-result smoke.
    batch = t.preprocess_batch(next(iter(t.train_loader)))
    state = {k: v.detach().clone() for k, v in t.model.state_dict().items()}
    weights = output / 'baseline_preupdate_state.pt'
    torch.save(state, weights)
    ref = DetectionModel(str(t.args.model), ch=3, nc=80, verbose=False).to(t.device)
    ref.args = copy.deepcopy(t.model.args)
    ref.names = copy.deepcopy(t.model.names)
    ref.stride = t.model.stride
    ref.load_state_dict(state)
    ref.train()
    rng = rng_state()
    try:
        t.model.train()
        with torch.no_grad():
            a = ref.loss(batch)[0]
            a = sum(a.values()) if isinstance(a, dict) else a
            left = float(a.sum().cpu())
        restore_rng(rng)
        with torch.no_grad():
            b = t.model.loss(batch)[0]
            b = sum(b.values()) if isinstance(b, dict) else b
            right = float(b.sum().cpu())
        if not math.isfinite(left) or left != right:
            raise ValueError('Canonical pre-update loss equivalence failed')
    finally:
        t.model.load_state_dict(state)
        restore_rng(rng)
    proof = dict(reference_id='pinned ultralytics.nn.tasks.DetectionModel.loss',
                 candidate_id='COCOTrainOnlyTeacher production model.loss',
                 reference_weights_path=str(weights), candidate_weights_path=str(weights),
                 weights_sha256=sha(weights), metric_name='pre-update native detection loss sum',
                 reference_metric=left, candidate_metric=right, absolute_difference=abs(left-right), tolerance=0,
                 evaluation_setting=dict(dataset=getattr(t, 'baseline_probe_dataset', 'first native augmented clean25K batch'), seed=0,
                                         sample_count=int(batch['img'].shape[0]),
                                         parameters=dict(imgsz=int(t.args.imgsz), nc=80, Person=0),
                                         post_processing='native raw detection loss, identical tensors, no NMS'),
                 reference_computation_module=type(ref).__module__+'.'+type(ref).__name__,
                 native_loss_inherited=t.model.__class__.loss is DetectionModel.loss,
                 official_metrics_disabled=True)
    write_once(output / 'baseline_equivalence.json', proof)


def train(release_path, output, *, smoke_batches=0, review=None, smoke_evidence=None):
    release = verify_release(release_path)
    output = Path(output).resolve()
    if smoke_batches and not 1 <= smoke_batches <= 100:
        raise ValueError('Smoke must have 1..100 native training batches')
    if not smoke_batches:
        require_review(review, release_path, smoke_evidence)
    output.mkdir(parents=True, exist_ok=False)
    # All local caches/settings belong to this new phase, never old experiments.
    os.environ['YOLO_CONFIG_DIR'] = str(output/'yolo_settings')
    os.environ['MPLCONFIGDIR'] = str(output/'mpl')
    os.environ['WANDB_MODE'] = 'disabled'
    (output/'yolo_settings').mkdir()
    (output/'mpl').mkdir()
    import numpy as np
    import torch
    import yaml
    from ultralytics.engine import trainer as native
    from ue_framework.s4_victim import attach_safety_callbacks
    from ue_framework.methods.s4_tfss import rng_state
    from .teacher import COCOTrainOnlyTeacher, teacher_overrides
    torch.set_num_threads(1)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    config = yaml.safe_load((ROOT/release['config_path']).read_text(encoding='utf-8'))
    config['victim']['init'] = str(ROOT/config['victim']['init'])
    mapping = load(ROOT/'docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json')['mapping']
    data_yaml = output/'teacher_train_only.yaml'
    data_yaml.write_text(yaml.safe_dump(dict(path=str(DATA_ROOT), train='images/train', val=None, nc=80,
                                          names={r['yolo_class_id']: r['name'] for r in mapping}), sort_keys=False), encoding='utf-8')
    overrides = teacher_overrides(config, data_yaml, output/'train')
    write_once(output/'phase_spec.json', dict(source_commit=commit(), release_sha256=sha(release_path),
               config_sha256=sha(ROOT/release['config_path']), train_args=overrides, smoke_batches=smoke_batches,
               native_AMP_preflight='historical accepted random-model FP32/autocast selfcheck, no downloaded weights',
               strict_train_only=True, primary='last@200', started_UTC=now()))
    status = dict(state='RUNNING', source_commit=commit(), official_metrics_disabled=True,
                  artifact_ingest_disabled=bool(smoke_batches), smoke_batches=smoke_batches)
    atomic(output/'status.json', status)
    trainer = COCOTrainOnlyTeacher(overrides=overrides)
    holder = attach_safety_callbacks(trainer, amp_log_dir=output/'amp_audit', debug_hashes=False)
    if smoke_batches:
        trainer.add_callback('on_pretrain_routine_end', lambda t: baseline_probe(t, output))
    def started(t):
        if not t.amp or t.device.type != 'cuda' or t.model.nc != 80 or t.model.names[0] != 'person':
            raise ValueError('Wrong AMP/device/COCO80/Person production state')
    trainer.add_callback('on_train_start', started)
    def boundary(t):
        if smoke_batches and holder['observer'].total_batches >= smoke_batches:
            t.stop = True  # supported native between-batch boundary; still saves native last/full train-only audit
    trainer.add_callback('on_train_batch_end', boundary)
    original_check = native.check_amp
    try:
        native.check_amp = local_amp_check
        trainer.train()
        obs = holder['observer']
        trainer._finite()
        audit = load(output/'train/TEACHER_TRAIN_ONLY_AUDIT.json')
        if audit['validation_image_count'] or audit['validation_forwards'] or audit['EMA_forwards'] or trainer.validator.calls:
            raise ValueError('Teacher validation isolation failed')
        last = output/'train/weights/last.pt'
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck['optimizer'] is None or ck['scaler'] is None or ck['ema']._forward_pre_hooks:
            raise ValueError('Native Teacher checkpoint completeness failed')
        rows = list(csv.DictReader((output/'train/results.csv').open()))
        if smoke_batches:
            if obs.total_batches != smoke_batches or obs.updates < 1:
                raise ValueError('Bounded smoke did not execute a real native AMP SGD update')
            expected_epoch = 0
        else:
            expected_epoch = 199
            if len(rows) != 200 or int(rows[-1]['epoch']) != 200:
                raise ValueError('Teacher did not finish 200 CSV epochs')
            snapshot = output/'train/state_snapshots/epoch_200_full_state.pt'
            if not snapshot.is_file():
                raise ValueError('Missing complete native optimizer/scheduler/scaler/RNG state')
        if ck['epoch'] != expected_epoch:
            raise ValueError('Wrong embedded checkpoint epoch')
        status.update(state='passed' if smoke_batches else 'ACCEPTED', completed_UTC=now(), finite_loss=True,
                      production_entrypoint_reached=True, computation_kind='training',
                      total_actual_batches=obs.total_batches, actual_native_SGD_updates=obs.updates,
                      legal_native_AMP_skips=obs.skips, zero_validation_forwards=True,
                      release_sha256=sha(release_path), last_checkpoint_sha256=sha(last),
                      checkpoint_paths_remaining=[str(p) for p in output.rglob('*.pt')],
                      cleanup_policy='retain_all', checkpoint_cleanup_completed=False, deleted_paths=[])
        if smoke_batches:
            status['baseline_equivalence_probe'] = load(output/'baseline_equivalence.json')
            status['baseline_equivalence_required'] = True
            write_once(output/'smoke_summary.json', status)
        else:
            status['completed_epochs'] = 200
            status['full_state_sha256'] = sha(snapshot)
            write_once(output/'accepted.json', status)
    except BaseException as exc:
        status.update(state='FAILED', error=repr(exc), ended_UTC=now(), deleted_paths=[])
        raise
    finally:
        native.check_amp = original_check
        atomic(output/'status.json', status)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--smoke-batches', default=0, type=int)
    p.add_argument('--review', type=Path)
    p.add_argument('--smoke-evidence', type=Path)
    a = p.parse_args()
    train(a.release, a.output, smoke_batches=a.smoke_batches, review=a.review, smoke_evidence=a.smoke_evidence)


if __name__ == '__main__':
    main()
