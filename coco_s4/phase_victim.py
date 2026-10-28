"""Train one matched, train-only COCO victim for the Original S4 comparison.

The clean and poisoned victims share the unchanged native YOLO/AMP/SGD protocol.
Training validation is blocked for both; final COCOeval is a later read-only stage.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random

import yaml

from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT, phase_release
from .runtime import DATA_ROOT, atomic, load, now, sha, write_once
from .dimension_protocol import validate_receipt_dimension

FROZEN_TRAINING = dict(epochs=200, imgsz=640, seed=0, batch=36, workers=16,
                       optimizer='SGD', cos_lr=True, close_mosaic=100, cache=True,
                       amp=True, save_period=10, lr0=.01, lrf=.01, momentum=.937,
                       weight_decay=.0005, patience=0, device='0')
PHASES = {'clean200', 's4_200'}
MAPPING_SHA256 = '72d8e90134dade3cc3dc6f7a2b51df2c5758a60e315165a255f15258d26a8814'


def validate_phase_config(cfg, phase=None):
    if not isinstance(cfg, dict) or cfg.get('schema') != 's4.coco.victim-phase.v1':
        raise ValueError('Unknown COCO matched-victim config schema')
    phase = cfg.get('phase') if phase is None else phase
    if phase not in PHASES or cfg.get('phase') != phase or cfg.get('workflow_run_id') != WORKFLOW:
        raise ValueError('Wrong matched-victim phase or workflow identity')
    role, predecessor, route = {
        'clean200': ('clean_control', 'export25k', 'accepted clean COCO minitrain2017 train25K'),
        's4_200': ('original_s4_poisoned_victim', 'clean200',
                   'accepted Original S4 poisoned COCO minitrain2017 train25K'),
    }[phase]
    if (cfg.get('role') != role or cfg.get('predecessor_phase') != predecessor
            or cfg.get('dataset_route') != route or cfg.get('image_count') != 25000
            or cfg.get('class_count') != 80 or cfg.get('target_class_id') != 0):
        raise ValueError('Matched-victim dataset/role contract changed')
    if cfg.get('initialization') != 'random YOLOv8n 80-class YAML; pretrained=false; resume=false':
        raise ValueError('Victim initialization must remain random and independent')
    if cfg.get('training') != FROZEN_TRAINING:
        raise ValueError('Clean and poisoned victim training settings must exactly match the frozen protocol')
    validation = cfg.get('validation')
    if validation != dict(training_val=False, final_epoch_validation=False, final_eval=False,
                          selection_from_val=False, official_metrics=False):
        raise ValueError('Final clean val2017 cannot enter training or checkpoint selection')
    if (cfg.get('primary_checkpoint') != 'last.pt@200'
            or cfg.get('automatic_resume') is not False
            or cfg.get('cleanup_policy') != 'retain_all'):
        raise ValueError('Victim checkpoint/resume/retention policy changed')
    return True


def expected_data_root(phase, workflow_output=WORKFLOW_OUTPUT):
    if phase == 'clean200':
        return DATA_ROOT
    if phase == 's4_200':
        return Path(workflow_output).resolve()/'export25k'/'poisoned'
    raise ValueError('Unknown COCO victim phase')


def validate_phase_binding(manifest, cfg, phase, workflow_output=WORKFLOW_OUTPUT):
    validate_phase_config(cfg, phase)
    deps = manifest.get('phase_dependencies')
    if not isinstance(deps, dict):
        raise ValueError('Matched-victim release omits accepted phase dependency bindings')
    expected_names = (('teacher200', 'teacher_eval', 'calibration', 'fourier24', 'export25k')
                      if phase == 'clean200' else
                      ('teacher200', 'teacher_eval', 'calibration', 'fourier24', 'export25k', 'clean200'))
    if set(deps) != set(expected_names):
        raise ValueError('Matched-victim dependency set is not the exact serial predecessor chain')
    output = Path(workflow_output).resolve()
    for name in expected_names:
        dep = deps[name]
        accepted_path = output/name/'accepted.json'
        if dep.get('accepted_path') != str(accepted_path) or not accepted_path.is_file():
            raise ValueError('Missing or rerouted accepted predecessor: '+name)
        actual = sha(accepted_path)
        stage = load(output/'pipeline_state.json').get('stages', {}).get(name, {})
        receipt = load(accepted_path)
        if (dep.get('accepted_sha256') != actual or stage.get('state') != 'ACCEPTED'
                or stage.get('receipt_sha256') != actual or receipt.get('state') != 'ACCEPTED'):
            raise ValueError('Predecessor acceptance changed: '+name)
    teacher = load(output/'teacher200/accepted.json')
    last = output/'teacher200/train/weights/last.pt'
    if (teacher.get('completed_epochs') != 200 or teacher.get('zero_validation_forwards') is not True
            or sha(last) != teacher.get('last_checkpoint_sha256')
            or deps['teacher200'].get('checkpoint_sha256') != sha(last)):
        raise ValueError('Fixed train-only Teacher last@200 dependency changed')
    evaluation = load(output/'teacher_eval/accepted.json')
    if (evaluation.get('image_count') != 5000 or evaluation.get('classes') != 80
            or evaluation.get('parameter_updates') is not False
            or evaluation.get('teacher_checkpoint_sha256') != sha(last)):
        raise ValueError('Separate fixed-last Teacher assessment is not accepted')
    calibration = load(output/'calibration/accepted.json')
    validate_receipt_dimension(calibration)
    if (calibration.get('images') != 25000
            or calibration.get('validation_used') is not False
            or calibration.get('parameter_updates') is not False):
        raise ValueError('COCO train-only selected-space calibration is not accepted')
    fourier = load(output/'fourier24/accepted.json')
    if (fourier.get('epochs_completed') != 24 or fourier.get('validation_used') is not False
            or fourier.get('calibration_receipt_sha256') != deps['calibration'].get('accepted_sha256')):
        raise ValueError('Original S4 Fourier24 dependency changed')
    export = load(output/'export25k/accepted.json')
    if (export.get('image_count') != 25000 or export.get('labels_byte_identical') is not True
            or export.get('decoded_budget_passed') is not True
            or export.get('validation_used') is not False
            or export.get('manifest_sha256') != deps['export25k'].get('manifest_sha256')):
        raise ValueError('Accepted complete Original S4 poisoned export changed')
    export_manifest = output/'export25k'/'poisoned_manifest.jsonl'
    if not export_manifest.is_file() or sha(export_manifest) != export.get('manifest_sha256'):
        raise ValueError('Poisoned train manifest differs from accepted export receipt')
    if phase == 's4_200':
        clean = load(output/'clean200/accepted.json')
        if (clean.get('completed_epochs') != 200 or clean.get('pretrained') is not False
                or clean.get('primary_checkpoint_sha256') != deps['clean200'].get('checkpoint_sha256')
                or not clean.get('initial_state_sha256')
                or clean.get('initial_state_sha256') != deps['clean200'].get('initial_state_sha256')):
            raise ValueError('Matched clean-victim last@200 dependency changed')
    return dict(expected_data_root=str(expected_data_root(phase, output)),
                train_manifest_sha256=export.get('original_train_manifest_sha256'),
                poisoned_manifest_sha256=export.get('manifest_sha256'),
                matched_clean_initial_state_sha256=(load(output/'clean200/accepted.json').get('initial_state_sha256')
                                                    if phase == 's4_200' else None),
                previous_receipt_sha256=deps['export25k']['accepted_sha256'] if phase == 'clean200'
                else deps['clean200']['accepted_sha256'])


def _phase_formal_gate(registration, release_path, manifest, config_sha, phase):
    from .runtime import assert_formal_launch_authorized
    assert_formal_launch_authorized()
    registration = Path(registration).resolve()
    if registration != (WORKFLOW_OUTPUT/'receipts'/(phase+'_READY.json')).resolve():
        raise ValueError('Wrong serial controller registration path')
    ready = load(registration)
    if (ready.get('phase') != phase or ready.get('workflow_run_id') != WORKFLOW
            or ready.get('module') != 'coco_s4.phase_victim' or ready.get('pre_run_result') != 'pass'
            or ready.get('source_root') != str(Path(__file__).resolve().parents[1])
            or ready.get('release_sha256') != sha(release_path)):
        raise ValueError('Victim READY record is not bound to this immutable release')
    pred = manifest['phase_dependencies'][cfg_predecessor(manifest, phase)]
    if ready.get('previous_stage_receipt_sha256') != pred.get('accepted_sha256'):
        raise ValueError('Victim READY is not bound to its exact accepted predecessor')
    closure_path = Path(ready.get('scientific_closure_path', '')).resolve()
    if (closure_path.parent != (WORKFLOW_OUTPUT/'receipts').resolve()
            or sha(closure_path) != ready.get('scientific_closure_sha256')):
        raise ValueError('Victim scientific closure path/hash invalid')
    closure = load(closure_path)
    expected = dict(phase=phase, source_commit=manifest['source_commit'],
                    release_sha256=sha(release_path), phase_config_sha256=config_sha,
                    reviewer_count=1, pre_run_result='pass',
                    teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
                    reviewer_independence=True, scientific_verdict='allow_run', blockers=[])
    if any(closure.get(k) != v for k, v in expected.items()):
        raise ValueError('One independent candidate-specific victim review is required')
    review_sha = closure.get('review_sha256')
    if (not isinstance(closure.get('review_path'), str) or not closure['review_path']
            or not isinstance(review_sha, str) or len(review_sha) != 64
            or any(c not in '0123456789abcdef' for c in review_sha)):
        raise ValueError('Victim review evidence identity is incomplete')
    smoke_path = Path(closure.get('smoke_path', '')).resolve()
    if (smoke_path != Path(ready.get('smoke_path', '')).resolve()
            or not smoke_path.is_relative_to((OWNED/'validation').resolve())
            or sha(smoke_path) != closure.get('smoke_sha256')
            or ready.get('smoke_sha256') != closure.get('smoke_sha256')):
        raise ValueError('Victim closure is not bound to its retained isolated smoke')
    smoke = load(smoke_path)
    required = dict(state='passed', phase=phase, source_commit=manifest['source_commit'],
                    release_sha256=sha(release_path), phase_config_sha256=config_sha,
                    teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
                    official_metrics_disabled=True, artifact_ingest_disabled=True,
                    validation_used=False, production_entrypoint_reached=True, finite_outputs=True)
    if (any(smoke.get(k) != v for k, v in required.items())
            or not 1 <= smoke.get('actual_native_batches', 0) <= 100
            or smoke.get('actual_native_updates', 0) < 1
            or smoke.get('validation_forwards') != 0):
        raise ValueError('Current candidate bounded native-train smoke required')
    return closure, smoke


def cfg_predecessor(manifest, phase):
    expected = 'export25k' if phase == 'clean200' else 'clean200'
    if manifest.get('phase_config', {}).get('predecessor_phase') != expected:
        raise ValueError('Matched-victim phase predecessor mismatch')
    return expected


def _workflow_slot(phase, *, smoke, manifest):
    from .workflow_admission import validate as validate_controller_admission
    if manifest.get('phase') != phase:
        raise ValueError('Victim phase differs from its immutable release')
    admission = validate_controller_admission(manifest)
    pipeline_path = WORKFLOW_OUTPUT/'pipeline_state.json'
    pipeline = load(pipeline_path)
    controller = pipeline.get('controller', {})
    pid = controller.get('pid')
    proc = Path('/proc')/str(pid)
    if not proc.is_dir():
        raise ValueError('Existing serial controller identity is no longer live')
    fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    if (proc.stat().st_uid != 1014 or int(fields[19]) != controller.get('start_ticks')
            or fields[0] == 'Z'):
        raise ValueError('Existing serial controller PID/start/owner identity changed')
    if (pipeline.get('run_id') != WORKFLOW
            or pipeline.get('source_commit') != admission['source_commit']):
        raise ValueError('Wrong active serial workflow binding')
    if smoke:
        if (pipeline.get('state') != 'WAITING_PHASE_RELEASE' or pipeline.get('current_stage') != phase
                or pipeline.get('active_job') is not None or pipeline.get('active_child') is not None):
            raise ValueError('Victim smoke requires its existing serial slot to wait with no active job')
    else:
        child = pipeline.get('active_child')
        if (pipeline.get('state') != 'RUNNING' or pipeline.get('current_stage') != phase
                or pipeline.get('active_job') != phase or not isinstance(child, dict)):
            raise ValueError('Formal victim does not match the existing serial controller active slot')
        child_proc = Path('/proc')/str(child.get('pid'))
        if not child_proc.is_dir():
            raise ValueError('Formal victim child identity is no longer live')
        child_fields = (child_proc/'stat').read_text().rsplit(')', 1)[1].split()
        current = Path('/proc')/str(os.getpid())
        if (child_proc.stat().st_uid != child.get('uid') or child.get('uid') != 1014
                or int(child_fields[19]) != child.get('start_ticks') or child_fields[0] == 'Z'
                or child.get('pid') != os.getpid()
                or current.stat().st_uid != child.get('uid')):
            raise ValueError('Formal victim process differs from the controller-bound child identity')
    return pipeline


def canonical_model_state_sha256(state):
    """Hash ordered tensor identity/content without PyTorch archive metadata."""
    import torch

    digest = hashlib.sha256()
    digest.update(len(state).to_bytes(8, 'big'))
    for name, tensor in sorted(state.items()):
        if not torch.is_tensor(tensor):
            raise TypeError('Model state contains a non-tensor entry: ' + str(name))
        value = tensor.detach().cpu().contiguous()
        for metadata in (name.encode('utf-8'), str(value.dtype).encode('ascii'),
                         json.dumps(list(value.shape), separators=(',', ':')).encode('ascii')):
            digest.update(len(metadata).to_bytes(8, 'big'))
            digest.update(metadata)
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(payload).to_bytes(8, 'big'))
        digest.update(payload)
    return digest.hexdigest()


class COCOMatchedTrainOnlyVictim:
    """Deferred subclass factory to keep import-time CPU contracts lightweight."""
    @staticmethod
    def build(expected_root, phase):
        from .teacher import COCOTrainOnlyTeacher

        class _Victim(COCOTrainOnlyTeacher):
            def __init__(self, *, overrides):
                self._expected_victim_root = Path(expected_root).resolve()
                self._victim_phase = phase
                super().__init__(overrides=overrides)

            def get_dataset(self):
                data = super().get_dataset()
                if Path(data['path']).resolve() != self._expected_victim_root:
                    raise ValueError('Victim train data root differs from its accepted phase binding')
                return data

        _Victim.__name__ = 'COCOMatchedTrainOnlyVictim'
        return _Victim


def _baseline_probe(trainer, output, phase):
    import copy
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ue_framework.methods.s4_tfss import restore_rng, rng_state

    batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
    state = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    weights = output/'baseline_preupdate_state.pt'
    torch.save(state, weights)
    initial_state_sha256 = canonical_model_state_sha256(state)
    ref = DetectionModel(str(trainer.args.model), ch=3, nc=80, verbose=False).to(trainer.device)
    candidate_conv = next(p for p in trainer.model.parameters() if p.ndim == 4)
    if candidate_conv.is_contiguous(memory_format=torch.channels_last):
        ref.to(memory_format=torch.channels_last)
    ref.args = copy.deepcopy(trainer.model.args)
    ref.names = copy.deepcopy(trainer.model.names)
    ref.stride = trainer.model.stride
    ref.load_state_dict(state)
    rng = rng_state()
    try:
        trainer.model.train(); ref.train()
        restore_rng(rng)
        with torch.no_grad():
            a = ref.loss(batch)[0]
            a = sum(a.values()) if isinstance(a, dict) else a
            left = float(a.sum().cpu())
        restore_rng(rng)
        with torch.no_grad():
            b = trainer.model.loss(batch)[0]
            b = sum(b.values()) if isinstance(b, dict) else b
            right = float(b.sum().cpu())
    finally:
        trainer.model.load_state_dict(state)
        restore_rng(rng)
    if not math.isfinite(left) or left != right:
        raise ValueError('Matched-victim initial native loss differs from the independent reference')
    proof = dict(phase=phase, reference_id='pinned ultralytics.nn.tasks.DetectionModel.loss',
                 candidate_id='COCOMatchedTrainOnlyVictim original native model.loss',
                 reference_metric=left, candidate_metric=right,
                 absolute_difference=abs(left-right), tolerance=0,
                 weights_sha256=sha(weights), initial_state_sha256=initial_state_sha256,
                 official_metrics_disabled=True)
    write_once(output/'baseline_equivalence.json', proof)
    return proof


def run(release_path, output, *, registration=None, smoke_batches=0):
    manifest = None
    phase_hint = None
    release_path = Path(release_path).resolve()
    # The config is included in the immutable release manifest; its phase is used
    # only to select the exact serial slot and is rechecked by verify_release.
    release_doc = load(release_path)
    phase_hint = release_doc.get('phase')
    if phase_hint not in PHASES:
        raise ValueError('Release is not a matched-victim phase')
    manifest, cfg, checkpoint, teacher = phase_release(release_path, phase_hint)
    validate_phase_config(cfg, phase_hint)
    config_sha = sha(Path(__file__).resolve().parents[1]/manifest['phase_config_path'])
    binding = validate_phase_binding(manifest, cfg, phase_hint)
    output = Path(output).resolve()
    if smoke_batches and not 1 <= smoke_batches <= 100:
        raise ValueError('Victim isolated smoke is bounded to 1..100 actual native batches')
    if smoke_batches:
        if registration is not None or not output.is_relative_to((OWNED/'validation').resolve()):
            raise ValueError('Victim smoke must use a fresh isolated owned validation root')
        _workflow_slot(phase_hint, smoke=True, manifest=manifest)
    else:
        if registration is None or output != (WORKFLOW_OUTPUT/phase_hint).resolve():
            raise ValueError('Formal victim must use its registered existing serial phase output')
        _workflow_slot(phase_hint, smoke=False, manifest=manifest)
        _phase_formal_gate(registration, release_path, manifest, config_sha, phase_hint)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_CONFIG_DIR'] = str(output/'yolo_settings')
    os.environ['MPLCONFIGDIR'] = str(output/'mpl')
    os.environ['WANDB_MODE'] = 'disabled'
    (output/'yolo_settings').mkdir(); (output/'mpl').mkdir()
    import numpy as np
    import torch
    from .teacher import attach_safety_callbacks, teacher_overrides
    from .phase_teacher import local_amp_check

    if not torch.cuda.is_available():
        raise ValueError('Matched-victim production requires the admitted CUDA device')
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1); torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    data_root = Path(binding['expected_data_root']).resolve()
    if data_root != expected_data_root(phase_hint, WORKFLOW_OUTPUT).resolve():
        raise ValueError('Victim dataset route differs from the frozen phase selection')
    data_yaml = output/'victim_train_only.yaml'
    mapping_path = Path(__file__).resolve().parents[1]/'docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json'
    mapping_bytes = mapping_path.read_bytes()
    if hashlib.sha256(mapping_bytes).hexdigest() != MAPPING_SHA256:
        raise ValueError('Official COCO category map changed')
    mapping = load(mapping_path)['mapping']
    names = {r['yolo_class_id']: r['name'] for r in mapping}
    data_yaml.write_text(yaml.safe_dump(dict(path=str(data_root), train='images/train', val=None,
                                             nc=80, names=names), sort_keys=False), encoding='utf-8')
    main_cfg_path = Path(__file__).resolve().parents[1]/manifest['config_path']
    main_cfg = yaml.safe_load(main_cfg_path.read_text(encoding='utf-8'))
    config = main_cfg['victim']
    if any(config.get(k) != v for k, v in FROZEN_TRAINING.items()):
        raise ValueError('Canonical clean/poisoned victim settings drifted')
    overrides = teacher_overrides(main_cfg, data_yaml, output/'train')
    if (overrides.get('pretrained') is not False or overrides.get('resume') is not False
            or overrides.get('val') is not False or Path(overrides['model']).suffix != '.yaml'):
        raise ValueError('Victim must use independent random YAML and no train/eval resume')
    victim_cls = COCOMatchedTrainOnlyVictim.build(data_root, phase_hint)
    trainer = victim_cls(overrides=overrides)
    holder = attach_safety_callbacks(trainer, amp_log_dir=output/'amp_audit', debug_hashes=False)
    def baseline(t):
        proof = _baseline_probe(t, output, phase_hint)
        expected_initial = binding.get('matched_clean_initial_state_sha256')
        if expected_initial is not None and proof['initial_state_sha256'] != expected_initial:
            raise ValueError('Clean and poisoned victims do not share the same random initialization')
    trainer.add_callback('on_pretrain_routine_end', baseline)
    def started(t):
        if (not t.amp or t.device.type != 'cuda' or t.model.nc != 80
                or t.model.names[0] != 'person' or t.args.batch != 36):
            raise ValueError('Victim actual native trainer protocol differs from COCO80/Person0 lock')
    trainer.add_callback('on_train_start', started)
    def boundary(t):
        if smoke_batches and holder['observer'].total_batches >= smoke_batches:
            t.stop = True
    trainer.add_callback('on_train_batch_end', boundary)
    status = dict(phase=phase_hint, source_commit=manifest['source_commit'],
                  release_sha256=sha(release_path), phase_config_sha256=config_sha,
                  previous_stage_receipt_sha256=binding['previous_receipt_sha256'],
                  dataset_root=str(data_root), dataset_manifest_sha256=binding['train_manifest_sha256'],
                  teacher_checkpoint_sha256=sha(checkpoint), computation_kind='matched-victim-training',
                  validation_used=False, official_metrics_disabled=True,
                  artifact_ingest_disabled=True, automatic_resume_allowed=False,
                  cleanup_policy='retain_all', deleted_paths=[])
    atomic(output/'status.json', dict(status, state='RUNNING', started_UTC=now()))
    write_once(output/'phase_spec.json', dict(status, config=cfg, native_train_args=overrides))
    native = __import__('ultralytics.engine.trainer', fromlist=['trainer'])
    old_amp = native.check_amp
    try:
        native.check_amp = local_amp_check
        trainer.train()
        trainer._finite()
        audit = load(output/'train/TEACHER_TRAIN_ONLY_AUDIT.json')
        if (audit.get('validation_image_count') != 0 or audit.get('validation_forwards') != 0
                or audit.get('EMA_forwards') != 0 or trainer.validator.calls != 0):
            raise ValueError('Matched victim reached a forbidden validation forward')
        rows = list(csv.DictReader((output/'train/results.csv').open(newline='', encoding='utf-8')))
        observer = holder['observer']
        if smoke_batches:
            if len(rows) < 1 or observer.total_batches != smoke_batches or observer.updates < 1:
                raise ValueError('Victim smoke did not execute its bounded native optimizer update')
            result = dict(status, state='passed', production_entrypoint_reached=True,
                          finite_outputs=True, actual_native_batches=observer.total_batches,
                          actual_native_updates=observer.updates, legal_amp_skips=observer.skips,
                          validation_forwards=0, baseline_equivalence=load(output/'baseline_equivalence.json'),
                          checkpoint_paths_remaining=[str(p) for p in output.rglob('*.pt')],
                          completed_UTC=now())
            write_once(output/'smoke_summary.json', result)
            atomic(output/'status.json', result)
            return result
        expected_epochs = list(range(1, 201))
        epoch_col = rows[0].keys() if rows else ()
        epoch_key = next((k for k in epoch_col if k.strip().lower() == 'epoch'), None)
        if (len(rows) != 200 or epoch_key is None
                or [int(float(row[epoch_key])) for row in rows] != expected_epochs):
            raise ValueError('Victim formal training must complete contiguous epochs 1..200')
        if any(not math.isfinite(float(v)) for row in rows for k, v in row.items() if 'loss' in k.lower()):
            raise FloatingPointError('Nonfinite recorded victim loss')
        last = output/'train/weights/last.pt'; best = output/'train/weights/best.pt'
        full = output/'train/state_snapshots/epoch_200_full_state.pt'
        if not all(p.is_file() for p in (last, best, full)):
            raise ValueError('Missing retained native victim checkpoints/full state')
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck.get('epoch') != 199 or ck.get('optimizer') is None or ck.get('scaler') is None:
            raise ValueError('Native last@200 is incomplete or has the wrong embedded epoch')
        snapshot = torch.load(full, map_location='cpu', weights_only=False)
        if (snapshot.get('epoch') != 200 or snapshot.get('automatic_resume_allowed') is not False
                or snapshot.get('data_loader_worker_prefetch_rng_verified') is not False):
            raise ValueError('Full native final state is incomplete or overstates resume fidelity')
        accepted = dict(status, state='ACCEPTED', completed_epochs=200, csv_rows=200,
                        primary_checkpoint='last.pt@200', primary_checkpoint_sha256=sha(last),
                        best_secondary_checkpoint_sha256=sha(best),
                        full_state_sha256=sha(full),
                        initial_state_sha256=load(output/'baseline_equivalence.json')['initial_state_sha256'],
                        initial_state_file_sha256=sha(output/'baseline_preupdate_state.pt'),
                        matched_clean_initial_state=(phase_hint == 'clean200' or
                            load(output/'baseline_equivalence.json')['initial_state_sha256'] == binding.get('matched_clean_initial_state_sha256')),
                        pretrained=False, actual_native_batches=observer.total_batches,
                        actual_native_updates=observer.updates, legal_amp_skips=observer.skips,
                        zero_validation_forwards=True, validation_forwards=0,
                        checkpoint_selection_from_validation=False,
                        clean_control_checkpoint_sha256=sha(last) if phase_hint == 'clean200' else None,
                        completed_UTC=now())
        write_once(output/'accepted.json', accepted)
        atomic(output/'status.json', accepted)
        return accepted
    except BaseException as exc:
        atomic(output/'status.json', dict(status, state='FAILED', error=repr(exc), ended_UTC=now()))
        raise
    finally:
        native.check_amp = old_amp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--registration', type=Path)
    p.add_argument('--smoke-batches', type=int, default=0)
    a = p.parse_args()
    run(a.release, a.output, registration=a.registration, smoke_batches=a.smoke_batches)


if __name__ == '__main__':
    main()
