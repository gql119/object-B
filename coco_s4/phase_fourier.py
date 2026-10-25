"""Original S4 COCO Fourier stage, bound to accepted clean-train calibration.

The native Fourier/B2 loss and optimizer implementations remain unchanged.
This wrapper enforces the frozen COCO eligible-image schedule and imports the
accepted calibration artifacts byte-for-byte into this phase's retained output.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

from .runtime import DATA_ROOT, ROOT, atomic, load, now, safe_relative, sha, write_once
from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT, phase_release
from .phase_calibration import validate_original_config, train_manifest
from .dimension_protocol import PROTOCOL, validate_receipt_dimension

PHASE = 'fourier24'
MAIN_CONFIG = 'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml'
CALIBRATION_ARTIFACT_SUFFIXES = (
    'prototype_bank.pt',
    'fixed_p5_identity.json',
    'collected/eligible_support_manifest.json',
    'method/coordinates/target_specific_channels.pt',
    'method/coordinates/non_target_subspaces.pt',
    'method/coordinates/coordinates_identity.json',
    'method/coordinates/tsfl_summary.json',
    'method/coordinates/subspace_summary.json',
    'method/coordinates/strict_tsfl_channels.pt',
    'method/coordinates/safe_bg_selected_subspaces.pt',
    'method/coordinates/strict_tsfl_summary.json',
    'method/coordinates/tsfl_sensitivity_stats.json',
    'method/coordinates/orth_target_specific_channels.pt',
    'method/coordinates/orth_non_target_subspaces.pt',
    'method/coordinates/strict_tsfl_selected_layer.pt',
    'method/coordinates/strict_tsfl_selected_layer_summary.json',
    'method/reference/target_specific_channels.pt',
    'method/reference/non_target_subspaces.pt',
    'method/reference/coordinates_identity.json',
)
SEMANTIC_SOURCE_FILES = (
    'ue_framework/methods/b2_semantic.py',
    'ue_framework/methods/semantic_coordinates.py',
    'ue_framework/methods/orth_reference_design.py',
    'ue_framework/methods/safebg.py',
)


def validate_phase_config(cfg):
    expected = dict(schema='s4.coco.fourier24-config.v1', phase=PHASE,
                    workflow_run_id=WORKFLOW, predecessor_phase='calibration',
                    predecessor_receipt_kind='accepted.json',
                    teacher_policy='fixed native EMA last@200',
                    optimizer='native Adam over Fourier coefficients', epochs=24,
                    batch_size=16, eot_samples=2, seed=0,
                    support_mode='bbox_half_overlap',
                    support_eligibility='accepted calibration manifest; original support.sum()>10',
                    optimizer_steps='24*ceil(eligible_person_image_count/16)',
                    target_class_id=0, num_classes=80, selected_space_dimension='actual StrictTSFL selection',
                    dimension_protocol=PROTOCOL,
                    eps=0.03137254901960784, validation_used=False,
                    parameter_updates='Fourier coefficients only; original S4 inner adaptive step retained',
                    official_metrics_enabled=False, artifact_ingest_enabled=False,
                    checkpoint_selection_allowed=False, optimizer_resume_claim=False,
                    cleanup_policy='retain_all')
    if cfg != expected:
        raise ValueError('Frozen Original S4 COCO Fourier phase contract changed')


def validate_fourier_source_config(cfg):
    validate_original_config(cfg)
    method = cfg['methods']['legacy_kproto_ret']
    if method['semantic']['safe_bg'].get('dimension_protocol')!=PROTOCOL:
        raise ValueError('Current COCO Fourier source requires the authorized selected-dimension protocol')
    exact = [
        (method['support_mode'], 'bbox_half_overlap'),
        (method['shortcut_num_bases'], 16),
        (method['suppress_small_size'], 32),
        (method['enable_adaptive_freq_basis'], True),
        (method['freq_candidate_num_bases'], 64),
        (method['freq_active_num_bases'], 16),
        (method['adaptive_freq_warmup_epochs'], 5),
        (method['freq_basis_seed'], 0),
        (method['universal_epochs'], 24),
        (method['universal_batch_size'], 16),
        (method['eot_samples'], 2),
        (method['jnd_floor'], .4),
        (method['jnd_ceiling'], 1.0),
        (method['universal_lr_fourier'], .001),
        (method['lambda_tv'], .0001),
        (method['kproto']['num_prototypes'], 8),
        (method['kproto']['feature_layer'], 'model.21'),
        (method['kproto']['margin'], .2),
        (method['kproto']['seed'], 0),
        (method['kproto']['context_inner'], 12),
        (method['kproto']['context_outer'], 28),
        (method['kproto']['max_images'], 0),
        (method['loss']['lambda_bg'], 1.0),
        (method['loss']['lambda_rel'], 1.0),
        (method['loss']['lambda_ret'], .25),
        (method['adaptive']['enabled'], True),
        (method['adaptive']['inner_steps'], 1),
        (method['adaptive']['inner_lr'], .0001),
        (method['adaptive']['batch_size'], 1),
        (method['adaptive']['lambda_adapt'], 1.0),
        (method['adaptive']['lambda_nt'], .25),
        (method['adaptive']['feature_layer'], 'model.15'),
    ]
    if any(actual != expected for actual, expected in exact):
        raise ValueError('Original S4 COCO Fourier loss, basis, EOT, or schedule changed')
    if int(method['kproto']['max_steps']) != 0:
        raise ValueError('Formal Fourier training cannot retain a smoke step limit')


def eligible_paths_from_manifest(document, train_image_root):
    """Validate the accepted calibration schedule without opening any image."""
    root = Path(train_image_root).resolve()
    if (document.get('predicate') != 'original bbox_half_overlap support.sum()>10, Person0'
            or document.get('epochs') != 24 or document.get('batch') != 16):
        raise ValueError('Accepted calibration eligibility predicate/protocol changed')
    paths = document.get('paths')
    count = document.get('count')
    if not isinstance(paths, list) or not isinstance(count, int) or count != len(paths) or count < 1:
        raise ValueError('Accepted calibration eligible-image list is empty or malformed')
    resolved = []
    for raw in paths:
        path = Path(raw).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in ('.jpg', '.jpeg'):
            raise ValueError('Eligible image escaped the accepted COCO train split')
        resolved.append(path)
    if len(set(resolved)) != count or resolved != sorted(resolved):
        raise ValueError('Eligible image list is duplicated or not in canonical sorted order')
    steps = 24 * ((count + 15) // 16)
    if document.get('optimizer_steps') != steps:
        raise ValueError('Accepted Fourier step count does not match eligible COCO support images')
    return resolved, steps


def accepted_calibration_dependency(manifest, *, validation=False):
    candidate=manifest.get('dimension_reconstruction_dependency') if validation else None
    path = Path(candidate['receipt_path']) if candidate else WORKFLOW_OUTPUT/'calibration'/'accepted.json'
    expected = manifest.get('calibration_dependency', {})
    if not path.is_file() or sha(path) != expected.get('accepted_sha256'):
        raise ValueError('Accepted Original S4 calibration receipt is absent or changed')
    receipt = load(path)
    required = dict(state='ACCEPTED', phase='calibration', images=25000,
                    person_ground_truth=55096,
                    validation_used=False, parameter_updates=False,
                    checkpoint_selection_allowed=False)
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError('Calibration receipt is not a valid train-only selected-dimension COCO result')
    validate_receipt_dimension(receipt)
    if candidate:
        if (not path.resolve().is_relative_to((OWNED/'validation').resolve())
                or sha(path)!=candidate['receipt_sha256']
                or manifest.get('dimension_reconstruction_validation_allowed') is not True
                or receipt.get('acceptance_scope')!='CPU_RECONSTRUCTION_COMPONENT'
                or receipt.get('formal_use_requires_current_review') is not True):
            raise ValueError('Unreviewed reconstruction is permitted only in a bound isolated validation release')
        teacher=manifest['teacher_dependency']
        if (receipt.get('teacher_checkpoint_sha256')!=teacher['checkpoint_sha256']
                or receipt.get('teacher_receipt_sha256')!=teacher['accepted_sha256']):
            raise ValueError('Retained reconstruction used a different accepted Teacher')
        return path,receipt,None
    pipeline = load(WORKFLOW_OUTPUT/'pipeline_state.json')
    stage = pipeline.get('stages', {}).get('calibration', {})
    if stage.get('state') != 'ACCEPTED' or stage.get('receipt_sha256') != sha(path):
        raise ValueError('Existing serial controller has not accepted calibration')
    teacher = manifest['teacher_dependency']
    if (receipt.get('teacher_checkpoint_sha256') != teacher['checkpoint_sha256']
            or receipt.get('teacher_receipt_sha256') != teacher['accepted_sha256']):
        raise ValueError('Calibration was produced from a different Teacher dependency')
    return path, receipt, pipeline


def validate_original_support_manifest(document, train_image_root):
    paths, steps = eligible_paths_from_manifest(document, train_image_root)
    return paths, steps


def select_validation_images(eligible, *, smoke_images=0, cpu_probe_images=0):
    """Keep both isolated probe types to one accepted eligible train image."""
    if smoke_images not in (0, 1) or cpu_probe_images not in (0, 1):
        raise ValueError('Fourier smoke and CPU-probe image counts must be zero or one')
    if smoke_images + cpu_probe_images > 1:
        raise ValueError('Fourier smoke and CPU probe are separate bounded runs')
    if smoke_images or cpu_probe_images:
        return list(eligible[:1])
    return list(eligible)


def copy_calibration_consumer_artifacts(output, dependency):
    """Copy only accepted B2/S4 consumer artifacts; never import raw training data."""
    from .runtime import sha as file_sha

    output = Path(output).resolve()
    spec = dependency['phase_spec']
    source_root = Path(spec['source_root']).resolve()
    owned_releases = (OWNED/'releases').resolve()
    if not source_root.is_relative_to(owned_releases):
        raise ValueError('Accepted calibration source is outside this owner’s release root')
    source_release = source_root/'source_release.json'
    if file_sha(source_release) != dependency['receipt']['release_sha256']:
        raise ValueError('Calibration source release no longer matches its accepted receipt')
    artifact_root = Path(spec['artifact_root']).resolve()
    if not artifact_root.is_relative_to(source_root):
        raise ValueError('Accepted calibration artifacts escaped their immutable source release')
    accepted_files = {row['path']: row['sha256'] for row in dependency['receipt']['artifact_files']}
    import_root = output/'imported_calibration'
    imported = {}
    for suffix in CALIBRATION_ARTIFACT_SUFFIXES:
        source = artifact_root/suffix
        relative = source.relative_to(source_root).as_posix()
        expected = accepted_files.get(relative)
        if not expected or not source.is_file() or file_sha(source) != expected:
            raise ValueError('Required accepted calibration artifact is absent or changed: '+suffix)
        destination = import_root/suffix
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_sha(destination) != expected:
            raise ValueError('Imported calibration artifact hash mismatch: '+suffix)
        imported[suffix] = dict(source_path=str(source), path=str(destination), sha256=expected,
                                bytes=destination.stat().st_size)
    semantic_code_root = output/'ue_framework'/'methods'
    for relative in SEMANTIC_SOURCE_FILES:
        expected = dependency['manifest']['files'].get(relative)
        source = ROOT/safe_relative(relative)
        if not expected or file_sha(source) != expected:
            raise ValueError('Current immutable release lacks an exact semantic source: '+relative)
        destination = semantic_code_root/Path(relative).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_sha(destination) != expected:
            raise ValueError('Copied semantic source hash mismatch: '+relative)
    return import_root, imported


def runtime_config(phase_cfg, manifest, imported_root, *, smoke):
    import yaml
    if manifest.get('config_path') != MAIN_CONFIG or MAIN_CONFIG not in manifest.get('files', {}):
        raise ValueError('Fourier phase must bind the fixed Original S4 COCO configuration')
    cfg = yaml.safe_load((ROOT/manifest['config_path']).read_text(encoding='utf-8'))
    validate_fourier_source_config(cfg)
    checkpoint = Path(manifest['teacher_dependency']['checkpoint_path']).resolve()
    cfg['surrogate']['ckpt'] = str(checkpoint)
    cfg['data'] = dict(dataset_root=DATA_ROOT.as_posix(),
                       train_images=(DATA_ROOT/'images/train').as_posix(),
                       train_labels=(DATA_ROOT/'labels/train').as_posix(),
                       instance_mask_dir='', allow_pseudo_mask_fallback=False)
    method = cfg['methods']['legacy_kproto_ret']
    method['kproto']['prototype_cache'] = str(imported_root/'prototype_bank.pt')
    semantic = method['semantic']
    semantic['artifact_dir'] = str(imported_root/'method'/'coordinates')
    semantic['safe_bg']['reference_artifact_dir'] = str(imported_root/'method'/'reference')
    semantic['strict_tsfl']['layer_selection_path'] = str(imported_root/'fixed_p5_identity.json')
    if smoke:
        method['kproto']['max_steps'] = 1
        method['universal_batch_size'] = 1
    return cfg, checkpoint


def formal_gate(registration, release_path, manifest, config_sha):
    registration = Path(registration).resolve()
    expected_registration = (WORKFLOW_OUTPUT/'receipts'/'fourier24_READY.json').resolve()
    if registration != expected_registration:
        raise ValueError('Wrong existing-controller Fourier phase registration route')
    ready = load(registration)
    release_path = Path(release_path).resolve()
    dependency = manifest['calibration_dependency']
    expected_ready = dict(phase=PHASE, workflow_run_id=WORKFLOW, pre_run_result='pass',
                          module='coco_s4.phase_fourier', source_root=str(ROOT.resolve()),
                          release_sha256=sha(release_path),
                          previous_stage_receipt_sha256=dependency['accepted_sha256'])
    if (any(ready.get(key) != value for key, value in expected_ready.items())
            or (ROOT/safe_relative(ready.get('release_relative', ''))).resolve() != release_path):
        raise ValueError('Fourier READY registration is not bound to this accepted calibration/release')
    closure = Path(ready['scientific_closure_path']).resolve()
    if closure.parent != (WORKFLOW_OUTPUT/'receipts').resolve() or sha(closure) != ready['scientific_closure_sha256']:
        raise ValueError('Fourier scientific closure path/hash mismatch')
    review = load(closure)
    expected_review = dict(phase=PHASE, source_commit=manifest['source_commit'],
                           release_sha256=sha(release_path), phase_config_sha256=config_sha,
                           calibration_receipt_sha256=dependency['accepted_sha256'],
                           teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
                           pre_run_result='pass', reviewer_count=1,
                           scientifically_correct=True, allow_run=True, blockers=[])
    if any(review.get(key) != value for key, value in expected_review.items()):
        raise ValueError('Fourier formal phase requires one current-release scientific review and main closure')
    smoke_path = Path(review['smoke_path']).resolve()
    if not smoke_path.is_relative_to((OWNED/'validation').resolve()) or sha(smoke_path) != review['smoke_sha256']:
        raise ValueError('Fourier smoke must be retained under the owned validation root')
    smoke = load(smoke_path)
    expected_smoke = dict(state='passed', phase=PHASE, source_commit=manifest['source_commit'],
                          release_sha256=sha(release_path), phase_config_sha256=config_sha,
                          calibration_receipt_sha256=dependency['accepted_sha256'],
                          teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
                          computation_kind='fourier-gradient', production_entrypoint_reached=True,
                          finite_outputs=True, frozen_teacher_unchanged=True, validation_used=False,
                          parameter_updates=True, actual_fourier_optimizer_updates=1,
                          official_metrics_disabled=True, artifact_ingest_disabled=True,
                          actual_device='cuda:0')
    if any(smoke.get(key) != value for key, value in expected_smoke.items()):
        raise ValueError('Current Fourier release lacks the required bounded native CUDA evidence')
    if smoke.get('actual_training_images') != 1 or not 1 <= smoke.get('production_forward_calls', 0) <= 100:
        raise ValueError('Fourier smoke must execute one eligible training image within 100 forwards')
    return review, smoke


def _rng_state(torch, np, *, include_cuda):
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state())
    if include_cuda:
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _hash_eligible(paths, expected_hashes):
    result = {}
    for path in paths:
        label_path = DATA_ROOT/'labels'/'train'/(path.stem+'.txt')
        image_sha, label_sha = sha(path), sha(label_path)
        expected = expected_hashes.get(path.stem)
        if expected != (image_sha, label_sha):
            raise ValueError('Eligible COCO image/label differs from accepted per-file manifest: '+path.stem)
        result[path.stem] = (image_sha, label_sha)
    return result


def _finite_number(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return x == x and abs(x) != float('inf')


def run(release_path, output, *, registration=None, smoke_images=0, cpu_probe_images=0):
    import numpy as np
    import torch
    from ue_framework.methods.b2_semantic import B2SemanticTrainer
    from ue_framework.methods.s4_tfss import model_hash
    from .phase_calibration import load_teacher, train_manifest

    output = Path(output).resolve()
    manifest, phase_cfg, checkpoint, teacher_receipt = phase_release(release_path, PHASE)
    validate_phase_config(phase_cfg)
    validate_fourier_source_config(__import__('yaml').safe_load(
        (ROOT/manifest['config_path']).read_text(encoding='utf-8')))
    config_sha = sha(ROOT/manifest['phase_config_path'])
    if smoke_images or cpu_probe_images:
        if (smoke_images not in (0, 1) or cpu_probe_images not in (0, 1)
                or smoke_images + cpu_probe_images != 1
                or not output.is_relative_to((OWNED/'validation').resolve())):
            raise ValueError('Fourier validation probe is one real eligible Person train image in owned validation')
    else:
        if output != (WORKFLOW_OUTPUT/'fourier24').resolve() or registration is None:
            raise ValueError('Formal Fourier must use the existing serial stage output and READY gate')
    calibration_path, calibration, pipeline = accepted_calibration_dependency(manifest,validation=bool(smoke_images or cpu_probe_images))
    if smoke_images == 0 and cpu_probe_images == 0:
        review, smoke = formal_gate(registration, release_path, manifest, config_sha)
    else:
        review = smoke = None
    if not cpu_probe_images and not torch.cuda.is_available():
        raise ValueError('Original S4 Fourier production path requires admitted CUDA0')
    if output.exists():
        raise FileExistsError('Fourier output already exists; preserve it and use a unique recovery identity')
    output.mkdir(parents=True, exist_ok=False)
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['YOLO_CONFIG_DIR'] = str(output/'yolo_settings')
    os.environ['MPLCONFIGDIR'] = str(output/'mpl')
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device('cpu') if cpu_probe_images else torch.device('cuda:0')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    common = dict(phase=PHASE, source_commit=manifest['source_commit'],
                  release_sha256=sha(release_path), phase_config_sha256=config_sha,
                  teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
                  teacher_receipt_sha256=manifest['teacher_dependency']['accepted_sha256'],
                  calibration_receipt_sha256=sha(calibration_path),
                  accepted_calibration_manifest_sha256=calibration['accepted_clean_train_manifest_sha256'],
                  validation_used=False, official_metrics_disabled=True, artifact_ingest_disabled=True,
                  cleanup_policy='retain_all', deleted_paths=[])
    status = dict(common, state='RUNNING', started_UTC=now(), smoke=bool(smoke_images),
                  cpu_probe=bool(cpu_probe_images))
    atomic(output/'status.json', status)
    try:
        imported_root, imported = copy_calibration_consumer_artifacts(
            output, dict(manifest=manifest, receipt=calibration,
                         phase_spec=load(Path(manifest['dimension_reconstruction_dependency']['phase_spec_path']))
                         if manifest.get('dimension_reconstruction_dependency') else
                         load(WORKFLOW_OUTPUT/'calibration'/'phase_spec.json')))
        cfg, checkpoint = runtime_config(phase_cfg, manifest, imported_root,
                                         smoke=bool(smoke_images or cpu_probe_images))
        images, expected_hashes, expected_digest = train_manifest()
        if expected_digest != calibration['accepted_clean_train_manifest_sha256']:
            raise ValueError('Accepted fixed25K train image/label manifest changed')
        eligible_document = load(imported_root/'collected'/'eligible_support_manifest.json')
        eligible, expected_steps = validate_original_support_manifest(eligible_document, DATA_ROOT/'images/train')
        if (len(eligible) != calibration['eligible_person_image_count']
                or expected_steps != calibration['original_fourier_optimizer_steps']
                or calibration.get('selected_channel_count') != validate_receipt_dimension(calibration)):
            raise ValueError('Fourier schedule or selected space differs from accepted calibration')
        if not set(map(str, eligible)).issubset(set(map(str, images))):
            raise ValueError('Accepted eligible image list is not a subset of the fixed train split')
        # Revalidate only the support-eligible images actually consumed by Fourier.
        probe_paths = eligible[:1] if smoke_images or cpu_probe_images else eligible
        eligible_before = _hash_eligible(probe_paths, expected_hashes)
        teacher, frozen_sha = load_teacher(checkpoint,
                                           manifest['teacher_dependency']['checkpoint_sha256'],
                                           device)
        original_teacher = copy.deepcopy(teacher).eval()
        adaptive_teacher = copy.deepcopy(teacher).eval()
        method_cfg = cfg['methods']['legacy_kproto_ret']
        from ue_framework.methods.semantic_coordinates import semantic_identity
        semantic_identity_result = semantic_identity(cfg, output)
        expected_identity = copy.deepcopy(calibration['semantic_identity'])
        # Phase artifact roots change by design. Preserve every accepted byte
        # and source identity while normalizing only the bound consumer paths.
        expected_identity['strict_tsfl_settings']['layer_selection_path'] = method_cfg['semantic']['strict_tsfl']['layer_selection_path']
        expected_identity['safe_bg_settings']['reference_artifact_dir'] = method_cfg['semantic']['safe_bg']['reference_artifact_dir']
        if semantic_identity_result != expected_identity:
            raise ValueError('Imported calibration consumer identity differs from accepted Original S4 artifacts')
        class EligibleCOCOB2Trainer(B2SemanticTrainer):
            def _get_active_freq_indices(self, epoch, step):
                self.audit_epoch, self.audit_step = epoch, step
                return super()._get_active_freq_indices(epoch, step)

            def _batch_loss(self, *args, **kwargs):
                result = super()._batch_loss(*args, **kwargs)
                if result is None:
                    return None
                total, metrics = result
                # Preserve categorical IDs before the inherited numeric EOT averaging.
                ids = [int(metrics['active_nt_class_'+str(i)].detach().item()) for i in (1, 2, 3)]
                with (output/'categorical_diagnostics.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(epoch=self.audit_epoch, step=self.audit_step,
                        active_nt_class_ids=ids, scope='per-EOT-draw; native CSV numeric means are not class IDs'))+'\n')
                if smoke_images or cpu_probe_images:
                    from .fourier_gradient_audit import measure
                    self.coefficient_gradient_draws.append(measure(self._semantic_terms, self.fourier_coeff))
                return total, metrics

            def _collect_target_images(self, train_img_dir, train_label_dir):
                if (Path(train_img_dir).resolve() != (DATA_ROOT/'images/train').resolve()
                        or Path(train_label_dir).resolve() != (DATA_ROOT/'labels/train').resolve()):
                    raise ValueError('Fourier attempted to read a different COCO split')
                selected = select_validation_images(eligible, smoke_images=smoke_images,
                                                    cpu_probe_images=cpu_probe_images)
                return [str(path) for path in selected]

        trainer = EligibleCOCOB2Trainer(cfg, method_cfg, device,
                                        original_teacher, adaptive_teacher)
        trainer.coefficient_gradient_draws = []
        if (trainer.channels.numel() != validate_receipt_dimension(calibration) or len(trainer.bases) != 79
                or len(trainer.safe_bases) != 79 or trainer.safe_bg_rank != 8
                or trainer.safe_bg_removal_alpha != .5 or trainer.safe_bg_active_k != 3
                or trainer.orth_feature_layer != 'model.15'):
            raise ValueError('Actual Original S4 B2 consumer rejected accepted COCO calibration')
        cache_path = Path(method_cfg['kproto']['prototype_cache'])
        bank = torch.load(cache_path, map_location='cpu', weights_only=True)
        expected_cache_key = trainer._cache_key(DATA_ROOT/'images/train', DATA_ROOT/'labels/train')
        prototypes = bank.get('prototypes')
        if (bank.get('cache_key') != expected_cache_key or bank.get('count', 0) < 8
                or tuple(prototypes.shape) != (8, trainer.multi_features.get('model.21', prototypes).shape[-1])
                or not torch.isfinite(prototypes).all()):
            raise ValueError('Accepted clean-train K=8 P5 prototype cache is not the exact consumer bank')
        if len(eligible) != calibration['eligible_person_image_count']:
            raise ValueError('Eligible support schedule changed before the original trainer call')
        forward_calls = []
        baseline_equivalence=None
        if smoke_images or cpu_probe_images:
            from .dimension_equivalence import probe
            baseline_equivalence=probe(manifest['canonical_safebg_reference'],device,checkpoint)
        forward_devices=set();forward_cuda_indices=set()
        def audit_forward(module,args,result):
            # Every bounded-probe forward is checked. Formal training retains
            # first-forward device proof and the native per-step loss checks.
            if forward_calls and not (smoke_images or cpu_probe_images):
                forward_calls.append(1);return
            def tensors(value):
                if torch.is_tensor(value):yield value
                elif isinstance(value,(tuple,list)):
                    for item in value:yield from tensors(item)
                elif isinstance(value,dict):
                    for item in value.values():yield from tensors(item)
            outputs=list(tensors(result))
            if not outputs:raise ValueError('Actual native forward returned no auditable tensors')
            for tensor in outputs:
                if tensor.device!=device or not torch.isfinite(tensor).all():
                    raise FloatingPointError('Native pre-loss forward tensor has wrong device or nonfinite outputs')
                forward_devices.add(str(tensor.device))
                if tensor.device.type=='cuda':forward_cuda_indices.add(tensor.device.index)
            forward_calls.append(1)
        hooks = [model.register_forward_hook(audit_forward)
                 for model in (trainer.surrogate, trainer.adaptive.model)]
        before_coeff = trainer.fourier_coeff.detach().clone()
        rng_before = _rng_state(torch, np, include_cuda=(device.type == 'cuda'))
        write_once(output/'phase_spec.json', dict(common,
                   computation_kind=('fourier-gradient-cpu-validation' if cpu_probe_images else 'fourier-gradient'),
                   cpu_probe=bool(cpu_probe_images), actual_device=str(device),
                   config=phase_cfg, original_config_sha256=sha(ROOT/manifest['config_path']),
                   eligible_person_images=len(eligible), expected_optimizer_steps=expected_steps,
                   accepted_eligible_manifest_sha256=sha(imported_root/'collected'/'eligible_support_manifest.json'),
                   imported_calibration_artifacts=imported, semantic_identity=semantic_identity_result,
                   calibration_acceptance=dict(receipt_sha256=sha(calibration_path),
                       channel_count=calibration['selected_channel_count'],
                       checkpoint_sha256=calibration['teacher_checkpoint_sha256']),
                   teacher_eval_metrics_read=False, validation_images_opened=False,
                   parameter_update_scope='original S4 Fourier coefficients plus existing one-step adaptive inner model',
                   optimizer_resume_claim=False, resume_policy='complete accepted stages only; retain failed outputs'))
        try:
            params_path = output/'global_params.pt'
            csv_path = output/'diagnostics.csv'
            json_path = output/'diagnostics.json'
            trainer.train_universal(str(DATA_ROOT/'images/train'), str(DATA_ROOT/'labels/train'),
                                    str(params_path), str(csv_path), str(json_path), seed=0)
        finally:
            for hook in hooks:
                hook.remove()
        rng_after = _rng_state(torch, np, include_cuda=(device.type == 'cuda'))
        torch.save(dict(before=rng_before, after=rng_after), output/'native_rng_state.pt')
        if model_hash(trainer.surrogate) != frozen_sha or sha(checkpoint) != manifest['teacher_dependency']['checkpoint_sha256']:
            raise ValueError('Frozen accepted Teacher changed during Fourier optimization')
        if not torch.isfinite(trainer.fourier_coeff).all():
            raise FloatingPointError('Nonfinite learned Fourier coefficients')
        learned_change = float((trainer.fourier_coeff.detach()-before_coeff).abs().max().item())
        if learned_change <= 0:
            raise ValueError('Original S4 Fourier coefficient optimizer did not make a measurable update')
        if not params_path.is_file() or not csv_path.is_file() or not json_path.is_file():
            raise ValueError('Original Fourier trainer did not retain all required outputs')
        with csv_path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if smoke_images or cpu_probe_images:
            if not rows or int(float(rows[-1]['step'])) != 0 or int(float(rows[-1]['epoch'])) != 1:
                raise ValueError('Bounded Fourier smoke did not execute exactly one native optimizer step')
            if not 1 <= len(forward_calls) <= 100:
                raise ValueError('Fourier smoke exceeded the 1..100 actual model-forward bound')
            scalar_keys = ('L_BG', 'L_rel', 'L_ret', 'L_total', 'max_abs_delta',
                           'outside_support_max', 'grad_norm_fourier')
            if any(not _finite_number(rows[-1].get(key)) for key in scalar_keys):
                raise FloatingPointError('Nonfinite Original S4 Fourier full loss/delta/gradient')
            if float(rows[-1]['grad_norm_fourier']) <= 0 or float(rows[-1]['outside_support_max']) > 1e-7:
                raise ValueError('Fourier smoke has zero gradient or perturbation outside the original support')
            from .fourier_gradient_audit import MEASUREMENT, validate_summary
            draws = trainer.coefficient_gradient_draws
            if len(draws) != 2:
                raise ValueError('Expected two actual EOT coefficient-gradient measurements')
            loss_gradients = {name:float(torch.stack([d[name] for d in draws]).mean(0).norm().item())
                              for name in ('L_BG', 'L_rel')}
            coefficient_proof = dict(measurement=MEASUREMENT, target='trainer.fourier_coeff',
                parameter_shape=list(trainer.fourier_coeff.shape), parameter_device=str(trainer.fourier_coeff.device),
                parameter_is_leaf=trainer.fourier_coeff.is_leaf, EOT_draws=len(draws),
                optimizer_gradient_accumulation_unchanged=True,
                per_draw_norms=[{name:float(g.norm().item()) for name,g in d.items()} for d in draws],
                mean_EOT_gradient_norms=loss_gradients,
                gradients=[{name:g.cpu().tolist() for name,g in d.items()} for d in draws])
            write_once(output/'per_loss_fourier_gradients.json', coefficient_proof)
            pack = torch.load(params_path, map_location='cpu', weights_only=True)
            if (len(pack.get('coords', [])) != 16 or tuple(pack['fourier_coeff'].shape) != (16, 3)
                    or abs(float(pack['eps'])-8/255) > 1e-9
                    or not torch.isfinite(pack['fourier_coeff']).all()):
                raise ValueError('Fourier smoke output violates the original 8/4/0 and 16-basis contract')
            is_cpu_probe = bool(cpu_probe_images)
            status.update(state='passed', computation_kind=('fourier-gradient-cpu-validation' if is_cpu_probe
                                                             else 'fourier-gradient'),
                          production_entrypoint_reached=True, finite_outputs=True,
                          frozen_teacher_unchanged=True, actual_device=str(device), actual_training_images=1,
                          actual_forward_tensor_devices=sorted(forward_devices),
                          actual_cuda_device_indices=sorted(forward_cuda_indices),
                          selected_channel_count=trainer.channels.numel(),non_target_class_count=len(trainer.safe_bases),
                          independent_P3_selected_count=trainer.orth_channels.numel(),
                          SafeBG_loss_Fourier_gradient_norms=loss_gradients,cache_memory=trainer.safe_bg_cache.memory_report(),
                          per_loss_fourier_gradient_measurement={k:v for k,v in coefficient_proof.items() if k!='gradients'},
                          per_loss_fourier_gradient_proof_sha256=sha(output/'per_loss_fourier_gradients.json'),
                          poison_image_gradient_norms={name:float(rows[-1][key]) for name,key in
                              (('L_BG','grad_BG_safe_norm'),('L_rel','grad_rel_safe_norm'))},
                          baseline_equivalence_probe=baseline_equivalence,
                          production_forward_calls=len(forward_calls), actual_fourier_optimizer_updates=1,
                          actual_eot_samples=2, validation_used=False, parameter_updates=True,
                          official_metrics_disabled=True, artifact_ingest_disabled=True,
                          Fourier_gradient_norm=float(rows[-1]['grad_norm_fourier']),
                          max_abs_delta=float(rows[-1]['max_abs_delta']),
                          outside_support_max=float(rows[-1]['outside_support_max']),
                          learned_fourier_max_abs_change=learned_change,
                          native_rng_sha256=sha(output/'native_rng_state.pt'),
                          retained=['phase_spec.json','global_params.pt','diagnostics.csv','diagnostics.json',
                                    'native_rng_state.pt','imported_calibration/'], completed_UTC=now())
            probe_name = 'cpu_probe_summary.json' if is_cpu_probe else 'smoke_summary.json'
            validate_summary(status)
            write_once(output/probe_name, status)
            atomic(output/'status.json', status)
            return status
        counts = [0]*24
        for index, row in enumerate(rows):
            step = int(float(row['step']))
            epoch = int(float(row['epoch']))
            if step != index or not 1 <= epoch <= 24:
                raise ValueError('Fourier optimizer step sequence is incomplete or reordered')
            counts[epoch-1] += 1
            if any(not _finite_number(value) for key, value in row.items() if key not in ('selected_proto_ids',)):
                raise FloatingPointError('Nonfinite full Fourier diagnostic row')
        per_epoch = (len(eligible)+15)//16
        if len(rows) != expected_steps or counts != [per_epoch]*24:
            raise ValueError('Formal Fourier did not perform exactly 24*ceil(eligible COCO Person/16) updates')
        pack = torch.load(params_path, map_location='cpu', weights_only=True)
        if (len(pack.get('coords', [])) != 16 or tuple(pack['fourier_coeff'].shape) != (16, 3)
                or abs(float(pack['eps'])-8/255) > 1e-9 or pack.get('support_mode') != 'bbox_half_overlap'
                or pack.get('target_class_id') != 0 or pack.get('feature_layer') != 'model.21'
                or not torch.isfinite(pack['fourier_coeff']).all()):
            raise ValueError('Formal Fourier artifact violates the frozen Original S4 protocol')
        eligible_after = _hash_eligible(eligible, expected_hashes)
        if eligible_after != eligible_before:
            raise ValueError('Support-eligible COCO image/label bytes changed during Fourier')
        accepted = dict(common, state='ACCEPTED', computation_kind='original-S4-Fourier24',
                        production_entrypoint_reached=True, finite_outputs=True,
                        frozen_teacher_unchanged=True, validation_used=False,
                        parameter_updates='native Fourier optimizer and original one-step adaptive inner learner',
                        official_metrics_disabled=True, artifact_ingest_disabled=True,
                        train_manifest_sha256=expected_digest, eligible_person_image_count=len(eligible),
                        optimizer_steps=expected_steps, optimizer_steps_per_epoch=per_epoch,
                        epochs_completed=24, steps_per_epoch=counts,
                        selected_channels=calibration['selected_channels'],
                        calibration_receipt_sha256=sha(calibration_path),
                        semantic_identity=semantic_identity_result,
                        Fourier_params_sha256=sha(params_path), diagnostics_csv_sha256=sha(csv_path),
                        diagnostics_json_sha256=sha(json_path), native_rng_sha256=sha(output/'native_rng_state.pt'),
                        learned_fourier_max_abs_change=learned_change,
                        clean_train_eligible_files_unchanged=True,
                        cleanup_policy='retain_all', deleted_paths=[], completed_UTC=now())
        write_once(output/'accepted.json', accepted)
        status.update(accepted)
        atomic(output/'status.json', status)
        return accepted
    except BaseException as exc:
        status.update(state='FAILED', error=repr(exc), ended_UTC=now(),
                      cleanup_policy='retain_all', deleted_paths=[])
        atomic(output/'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--smoke-images', type=int, default=0)
    parser.add_argument('--cpu-probe-images', type=int, default=0)
    args = parser.parse_args()
    run(args.release, args.output, registration=args.registration, smoke_images=args.smoke_images,
        cpu_probe_images=args.cpu_probe_images)
