"""One isolated AT arm, either bounded validation or a gated 200-epoch run."""
import argparse
import csv
from pathlib import Path
import random
import traceback
import numpy as np
import torch
from .common import ROOT, PROTOCOL, load, protocol, run_config, sha, commit, atomic, write_once, native_identity, scientific_source, now
from .data import data_view
from .trainer import ATTrainer
from .checkpoint import capture, save_once


class BoundedComplete(Exception):
    pass


def gate(receipt, source):
    r = load(receipt)
    if r.get('state') != 'PREFLIGHT_ACCEPTED' or r.get('source_commit') != source \
            or r.get('protocol_sha256') != sha(PROTOCOL):
        raise ValueError('Missing current-source AT preflight acceptance')
    for name in ('CPU_native', 'CUDA_native', 'GT_and_budget', 'resume', 'scientific_main_closure'):
        item = r.get(name, {})
        path = ROOT / item.get('path', '__missing__')
        if item.get('passed') is not True or not path.is_file() or sha(path) != item.get('sha256'):
            raise ValueError('Incomplete AT gate: ' + name)
    if r.get('CUDA_actual_device') != 'cuda:0' or r.get('original_S4_source_unchanged') is not True:
        raise ValueError('AT CUDA/source gate is not actual evidence')
    return r


def evaluate(checkpoint, data, out, device):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from .metrics import extract
    holder = {}

    class CountValidator(DetectionValidator):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.at_ids = []
            holder['validator'] = self

        def update_metrics(self, preds, batch):
            self.at_ids.extend(Path(p).stem for p in batch['im_file'])
            return super().update_metrics(preds, batch)

    before = sha(checkpoint)
    metrics = YOLO(str(checkpoint)).val(validator=CountValidator, data=str(data), split='val',
        imgsz=640, batch=36, workers=16, device=device, half=False, augment=False,
        conf=.001, iou=.7, max_det=300, plots=False, verbose=False,
        project=str(out), name='clean_val', exist_ok=False)
    ids = holder['validator'].at_ids
    import yaml
    cfg = yaml.safe_load(Path(data).read_text())
    expected = {p.stem for p in (Path(cfg['path']) / 'images/val').iterdir() if p.is_file()}
    if len(ids) != 4952 or len(set(ids)) != 4952 or set(ids) != expected or sha(checkpoint) != before:
        raise ValueError('Clean full-validation identity/count/checkpoint changed')
    result = extract(metrics.box)
    result.update(checkpoint_sha256=before, actual_image_ids=ids, validation_attacked=False)
    write_once(out / 'metrics.json', result)
    return result


def run(args):
    p, cfg = protocol(), run_config(args.run_id)
    native = native_identity()
    source = commit()
    closure = scientific_source()
    if args.formal and closure['uncommitted_files']:
        raise ValueError('Formal AT source bytes are not the actual committed closure')
    for item in (p['authorization'], p['reused_acceptance']):
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Human authorization or retained acceptance identity changed')
    for rel, digest in p['original_sources'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError('Original S4 source changed: ' + rel)
    if args.formal:
        if source != args.source_commit or args.fixture_root is not None or args.device != '0':
            raise ValueError('Formal source/data/GPU binding invalid')
        gate(args.preflight, source)
    if args.device != 'cpu':
        from .queue import resource_status
        r = resource_status()
        if not r['admitted']:
            raise ValueError('Another job/controller or resource gate blocks this new AT GPU work: ' + str(r))
    if args.resume_fullstate:
        if not args.formal or not args.out.is_dir() or not args.resume_fullstate.resolve().is_relative_to(args.out.resolve()):
            raise ValueError('Resume must use an existing checkpoint from this same AT output')
        identity = load(args.out / 'identity.json')
        if identity['run_id'] != args.run_id or identity['source_commit'] != source or identity['protocol_sha256'] != sha(PROTOCOL):
            raise ValueError('Resume identity mismatch')
        payload = torch.load(args.resume_fullstate, map_location='cpu', weights_only=False)
        epoch = payload['completed_epochs']
        later = [v for v in (args.out / 'training/full_state').glob('epoch_*.pt')
                 if int(v.stem.split('_')[1]) > epoch]
        results_csv = args.out / 'training/results.csv'
        if (args.out / 'accepted.json').exists() or later or (results_csv.exists() and
                len(list(csv.DictReader(results_csv.open()))) > epoch):
            raise ValueError('Later completed epochs/results exist; resume would overwrite retained evidence')
        data = args.out / 'data.yaml'
        segment = args.out / 'resume_segments' / now().replace(':', '').replace('+', '_')
        segment.mkdir(parents=True, exist_ok=False)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        data = data_view(args.out, cfg['arm'], fixture_root=args.fixture_root)
        identity = dict(run_id=args.run_id, source_commit=source, protocol_sha256=sha(PROTOCOL),
                        data_audit_sha256=sha(args.out / 'data_audit.json'), arm=cfg['arm'],
                        attack=cfg['attack'], model_yaml_sha256=p['model_yaml_sha256'], source_closure=closure)
        write_once(args.out / 'identity.json', identity)
        segment = args.out / 'audit'
    train_args = dict(p['train_args'], data=str(data), model=str(ROOT / 'configs/voc_yolov8n_20cls.yaml'),
                      project=str(args.out), name='training', device=args.device)
    if not args.formal:
        train_args.update(epochs=200, val=False, save=False, plots=False, cache=False,
                          batch=36 if args.device != 'cpu' else 2,
                          imgsz=640 if args.device != 'cpu' else 160,
                          workers=16 if args.device != 'cpu' else 0,
                          amp=args.device != 'cpu')
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if args.device != 'cpu':
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(0)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    trainer = ATTrainer(at_config=cfg['attack'], identity=identity, audit_root=segment,
                        validation_only=not args.formal, resume_fullstate=args.resume_fullstate,
                        overrides=train_args)
    def initial_state(t):
        from ue_framework.s4_amp_observer import tree_hash
        current = tree_hash(dict(t.model.named_parameters()))
        t.at_initial_model_sha256 = load(args.out / 'audit/initialization.json')['parameter_sha256'] \
            if args.resume_fullstate else current
        write_once(segment / 'initialization.json', dict(
            parameter_sha256=t.at_initial_model_sha256, current_parameter_sha256=current, seed=0,
            kind='same-run resume' if args.resume_fullstate else 'fresh random YAML',
            model_yaml_sha256=p['model_yaml_sha256']))
    trainer.add_callback('on_train_start', initial_state)
    if not args.formal:
        def forbid_validation(t):
            raise AssertionError('Bounded AT validation must not execute a validator')

        def finish(t):
            observer = t.at_holder['observer']
            if observer.updates >= 1:
                if args.device != 'cpu' and (t.at_latest['actual_input_shape'][0] != 36 or
                        t.at_latest['actual_raw_devices'] != [('cuda:0', 0)]):
                    raise ValueError('CUDA AT smoke did not reach actual native batch36/cuda0')
                payload = capture(t.model, t.optimizer, t.scheduler, t.scaler, t.attack, identity,
                                  0, t.ema, t.train_loader)
                save_once(args.out / 'retained_bounded_state.pt', payload)
                write_once(args.out / 'smoke_summary.json', dict(state='PASSED_BOUNDED_NATIVE_TRAINING',
                    source_commit=source, protocol_sha256=sha(PROTOCOL), identity=identity,
                    native_sources=native, source_closure=closure,
                    initial_model_sha256=t.at_initial_model_sha256, actual_updates=observer.updates,
                    legal_AMP_skips=observer.skips, outer_batches=observer.total_batches,
                    actual_PGD_forwards=t.attack.calls * 10, actual_outer_forwards=observer.total_batches,
                    latest_PGD=t.at_latest, finite_loss=bool(torch.isfinite(t.loss)),
                    device=str(t.device), actual_input_batch=t.at_latest['actual_input_shape'][0],
                    CUDA_peak_allocated_MiB=torch.cuda.max_memory_allocated(0) / 2**20 if args.device != 'cpu' else None,
                    validation_forwards=0, official_metrics_disabled=True, artifact_ingest_disabled=True,
                    cleanup_policy='retain_all', deleted_paths=[]))
                raise BoundedComplete()
            if observer.total_batches >= args.max_batches:
                raise ValueError('No actual native optimizer update within bounded PGD validation')
        trainer.add_callback('on_val_start', forbid_validation)
        trainer.add_callback('on_train_batch_end', finish)
    try:
        trainer.train()
    except BoundedComplete:
        return
    finally:
        for loader in (getattr(trainer, 'train_loader', None), getattr(trainer, 'test_loader', None)):
            if loader is not None and hasattr(loader, 'close'):
                loader.close()
    if not args.formal:
        raise ValueError('Bounded validation exited without an actual optimizer update')
    rows = list(csv.DictReader((Path(trainer.save_dir) / 'results.csv').open()))
    ckpt = torch.load(trainer.last, map_location='cpu', weights_only=False)
    if len(rows) != 200 or ckpt['epoch'] != 199:
        raise ValueError('AT training is not complete last@200')
    primary = evaluate(trainer.last, data, args.out / 'last_primary', args.device)
    secondary = evaluate(trainer.best, data, args.out / 'best_secondary', args.device) if trainer.best.exists() else None
    write_once(args.out / 'accepted.json', dict(state='ACCEPTED', **identity, epochs=200,
        initial_model_sha256=trainer.at_initial_model_sha256,
        last_sha256=sha(trainer.last), best_sha256=sha(trainer.best) if secondary else None,
        primary_metrics_sha256=sha(args.out / 'last_primary/metrics.json'),
        secondary_metrics_sha256=sha(args.out / 'best_secondary/metrics.json') if secondary else None,
        completed_UTC=now(), primary='last.pt@200', full_VOC20=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--source-commit')
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--fixture-root', type=Path)
    parser.add_argument('--device', default='cpu', choices=['cpu', '0'])
    parser.add_argument('--max-batches', type=int, default=12)
    parser.add_argument('--resume-fullstate', type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_batches <= 100:
        parser.error('Bounded validation permits 1..100 outer training batches')
    if not args.formal and args.fixture_root is None:
        parser.error('Bounded runs require explicit training fixtures')
    if args.formal and args.preflight is None:
        parser.error('Formal training requires current published preflight acceptance')
    if args.out.exists() and not args.resume_fullstate:
        raise FileExistsError('Output already exists; no writes or implicit restart allowed')
    try:
        run(args)
    except BaseException as exc:
        if args.out.exists():
            failure = args.out / 'failures' / (now().replace(':', '').replace('+', '_') + '.json')
            write_once(failure, dict(error=repr(exc), traceback=traceback.format_exc(),
                                    UTC=now(), no_automatic_retry=True, cleanup_policy='retain_all'))
        raise


if __name__ == '__main__':
    main()
