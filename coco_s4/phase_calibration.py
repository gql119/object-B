"""Original S4 COCO clean-train calibration phase and its bounded GPU smoke."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re

from .runtime import DATA_ROOT, ROOT, atomic, load, now, safe_relative, sha, verify_release, write_once
from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT, phase_release
from .calibration import ProtocolSpaceBlocked, build_coordinates, collect

PHASE = 'calibration'
MAIN_CONFIG = 'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml'
TRAIN_MANIFEST = DATA_ROOT/'metadata/TRAIN25K_KAGGLE_IMAGE_MANIFEST_20261024.jsonl'
FIXED_COUNTS = dict(images=25000, ground_truth=181475, person_ground_truth=55096)


def validate_phase_config(cfg):
    """Reject protocol drift before loading a model or touching image data."""
    expected = dict(schema='s4.coco.calibration-config.v1', phase=PHASE,
                    workflow_run_id=WORKFLOW, teacher_policy='fixed native EMA last@200',
                    predecessor_phase='teacher_eval', predecessor_receipt_kind='accepted.json',
                    validation_used=False, parameter_updates=False, checkpoint_selection_allowed=False,
                    method_statistics_from_validation=False, selected_space_dimension=24,
                    support_mode='bbox_half_overlap', target_class_id=0, num_classes=80,
                    prototype_count=8, prototype_seed=0, prototype_max_images=0,
                    strict_tsfl=dict(target_candidate_fraction=.4, nt_safety_quantile=.6,
                                     sensitivity_worst_k=3, target_channel_fraction=.15,
                                     min_selected_channels=4, selected_layer='P5', feature_layer='model.21'),
                    safe_bg=dict(enabled=True, mode='low_rank', rank=8, removal_alpha=.5, active_k=3,
                                 reference_layer='model.15'))
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen Original S4 COCO calibration protocol changed')
    relative = str(safe_relative(cfg['artifact_relative_root']))
    if not re.fullmatch(r'execution/generated_calibration/coco_calibration_[a-z0-9_]{1,40}', relative):
        raise ValueError('Calibration artifacts must use a unique owned release-local execution path')


def validate_original_config(cfg):
    exp, surrogate = cfg['experiment'], cfg['surrogate']
    method = cfg['methods']['legacy_kproto_ret']
    semantic = method['semantic']
    strict, safe = semantic['strict_tsfl'], semantic['safe_bg']
    exact = [
        (exp['dataset'], 'coco_minitrain25k'), (exp['target_class_id'], 0),
        (exp['num_classes'], 80), (exp['steps'], [24]), (exp['seeds'], [0]),
        (surrogate['model'], 'yolov8n'), (surrogate['num_classes'], 80), (surrogate['imgsz'], 640),
        (method['support_mode'], 'bbox_half_overlap'),
        (method['kproto']['feature_layer'], 'model.21'), (method['kproto']['num_prototypes'], 8),
        (method['kproto']['seed'], 0), (method['kproto']['max_images'], 0),
        (method['universal_epochs'], 24), (method['universal_batch_size'], 16),
        (method['eot_samples'], 2), (method['adaptive']['inner_steps'], 1),
        (method['adaptive']['feature_layer'], 'model.15'),
        (semantic['target_channel_fraction'], .75), (semantic['lambda_specificity'], 1.0),
        (semantic['subspace_variance'], .9), (semantic['subspace_min_rank'], 4),
        (semantic['subspace_max_rank_fraction'], .5), (semantic['lambda_orth'], .0005),
        (strict['enabled'], True), (strict['target_candidate_fraction'], .4),
        (strict['nt_safety_quantile'], .6), (strict['strict_tsfl_feature_level'], 'P5'),
        (strict['sensitivity_worst_k'], 3), (strict['target_channel_fraction'], .15),
        (strict['min_selected_channels'], 4), (safe['enabled'], True),
        (safe['mode'], 'low_rank'), (safe['rank'], 8), (safe['removal_alpha'], .5),
        (safe['active_k'], 3),
    ]
    if any(actual != wanted for actual, wanted in exact):
        raise ValueError('Original S4 COCO experiment configuration changed')
    if abs(float(exp['eps']) - 8 / 255) > 1e-9:
        raise ValueError('Original S4 8/255 perturbation budget changed')
    if len(exp['class_names']) != 80 or exp['class_names'][0] != 'person':
        raise ValueError('Original COCO80 Person0 mapping changed')


def accepted_eval_dependency(manifest):
    """Read the previous phase receipt without reading or using its AP values."""
    path = WORKFLOW_OUTPUT/'teacher_eval'/'accepted.json'
    expected = manifest.get('teacher_eval_dependency', {})
    if not path.is_file() or sha(path) != expected.get('accepted_sha256'):
        raise ValueError('Accepted fixed Teacher evaluation receipt changed or is absent')
    receipt = load(path)
    if any(receipt.get(k) != v for k, v in dict(state='ACCEPTED', phase='teacher_eval',
                                                    image_count=5000, classes=80,
                                                    parameter_updates=False,
                                                    checkpoint_selection_allowed=False,
                                                    method_statistics_allowed=False).items()):
        raise ValueError('COCO Teacher evaluation dependency is not the fixed read-only 5K result')
    if receipt.get('teacher_checkpoint_sha256') != manifest['teacher_dependency']['checkpoint_sha256']:
        raise ValueError('Teacher evaluation used a different checkpoint from calibration')
    pipeline = load(WORKFLOW_OUTPUT/'pipeline_state.json')
    stage = pipeline.get('stages', {}).get('teacher_eval', {})
    if stage.get('state') != 'ACCEPTED' or stage.get('receipt_sha256') != sha(path):
        raise ValueError('Serial controller has not accepted the preceding Teacher evaluation')
    if expected.get('accepted_sha256') != stage['receipt_sha256']:
        raise ValueError('Calibration is not bound to the exact preceding phase receipt')
    return path, receipt


def train_manifest():
    """Load the already accepted image manifest and return exact expected hashes."""
    records = [json.loads(line) for line in TRAIN_MANIFEST.read_text(encoding='utf-8').splitlines()]
    if len(records) != FIXED_COUNTS['images']:
        raise ValueError('Accepted COCO train manifest must contain exactly25K records')
    hashes, images = {}, []
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda r: r['image']):
        image_id = int(row['image_id'])
        image_rel = f'images/train/{image_id:012d}.jpg'
        label_rel = f'labels/train/{image_id:012d}.txt'
        if row.get('image') != image_rel or row.get('label') != label_rel:
            raise ValueError('Accepted train manifest has an unexpected path or split')
        stem = Path(image_rel).stem
        if stem in hashes:
            raise ValueError('Duplicate COCO train image ID in accepted manifest')
        image_sha, label_sha = row.get('image_sha256'), row.get('label_sha256')
        if len(image_sha or '') != 64 or len(label_sha or '') != 64:
            raise ValueError('Accepted per-file image/label SHA256 missing')
        hashes[stem] = (image_sha, label_sha)
        images.append(DATA_ROOT/image_rel)
        digest.update(f'{stem}:{image_sha}:{label_sha}\n'.encode())
    if len({int(r['image_id']) for r in records}) != FIXED_COUNTS['images']:
        raise ValueError('Duplicate COCO train image IDs')
    return images, hashes, digest.hexdigest()


def runtime_config(phase_cfg, manifest):
    import yaml
    if manifest.get('config_path') != MAIN_CONFIG or MAIN_CONFIG not in manifest.get('files', {}):
        raise ValueError('Calibration must bind the fixed Original S4 COCO experiment configuration')
    cfg = yaml.safe_load((ROOT/manifest['config_path']).read_text(encoding='utf-8'))
    validate_original_config(cfg)
    checkpoint = Path(manifest['teacher_dependency']['checkpoint_path']).resolve()
    cfg['surrogate']['ckpt'] = str(checkpoint)
    cfg['data'] = dict(dataset_root=DATA_ROOT.as_posix(),
                       train_images=(DATA_ROOT/'images/train').as_posix(),
                       train_labels=(DATA_ROOT/'labels/train').as_posix(),
                       instance_mask_dir='', allow_pseudo_mask_fallback=False)
    artifact_root = (ROOT/safe_relative(phase_cfg['artifact_relative_root'])).resolve()
    if not artifact_root.is_relative_to(ROOT.resolve()):
        raise ValueError('Calibration artifact root escaped its isolated source release')
    method = cfg['methods']['legacy_kproto_ret']
    method['kproto']['prototype_cache'] = str(artifact_root/'prototype_bank.pt')
    semantic = method['semantic']
    semantic['artifact_dir'] = str(artifact_root/'method'/'coordinates')
    semantic['safe_bg']['reference_artifact_dir'] = str(artifact_root/'method'/'reference')
    semantic['strict_tsfl']['layer_selection_path'] = str(artifact_root/'fixed_p5_identity.json')
    return cfg, checkpoint, artifact_root


def load_teacher(path, checkpoint_sha, device):
    import torch
    from ue_framework.methods.s4_tfss import model_hash
    if sha(path) != checkpoint_sha:
        raise ValueError('Fixed last@200 Teacher checkpoint hash drift')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('epoch') != 199 or checkpoint.get('ema') is None:
        raise ValueError('Calibration requires the native EMA last@200 checkpoint')
    # Native last.pt stores the DetectionModel itself in ema; .model would
    # discard its forward/head semantics and return only a Sequential.
    model = checkpoint['ema']
    if model.model[-1].nc != 80 or model.names[0] != 'person':
        raise ValueError('Teacher checkpoint is not COCO80 Person0')
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite fixed Teacher parameters')
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise ValueError('Teacher checkpoint contains unexpected forward audit hooks')
    # Native serialization stores EMA in FP16. Canonical YOLO checkpoint
    # loading converts it to FP32 before the original calibration path.
    model.float().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model_hash(model)


def formal_gate(registration, release_path, manifest, config_sha):
    registration = Path(registration).resolve()
    ready_path = (WORKFLOW_OUTPUT/'receipts'/'calibration_READY.json').resolve()
    if registration != ready_path:
        raise ValueError('Wrong existing-controller calibration registration route')
    ready = load(registration)
    release_path = Path(release_path).resolve()
    if (ready.get('phase') != PHASE or ready.get('workflow_run_id') != WORKFLOW
            or ready.get('pre_run_result') != 'pass'
            or ready.get('module') != 'coco_s4.phase_calibration'
            or Path(ready.get('source_root', '')).resolve() != ROOT.resolve()
            or ready.get('release_sha256') != sha(release_path)
            or (ROOT/safe_relative(ready.get('release_relative', ''))).resolve() != release_path
            or ready.get('previous_stage_receipt_sha256') != manifest['teacher_eval_dependency']['accepted_sha256']):
        raise ValueError('Calibration READY registration is not bound to this accepted predecessor/release')
    closure = Path(ready['scientific_closure_path']).resolve()
    if closure.parent != (WORKFLOW_OUTPUT/'receipts').resolve() or sha(closure) != ready['scientific_closure_sha256']:
        raise ValueError('Calibration scientific closure path/hash mismatch')
    review = load(closure)
    expected = dict(phase=PHASE, source_commit=manifest['source_commit'], release_sha256=sha(release_path),
                    phase_config_sha256=config_sha, teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
                    previous_stage_receipt_sha256=manifest['teacher_eval_dependency']['accepted_sha256'],
                    pre_run_result='pass', reviewer_count=1)
    if (any(review.get(k) != v for k, v in expected.items())
            or review.get('scientifically_correct') is not True
            or review.get('allow_run') is not True or review.get('blockers') != []):
        raise ValueError('Calibration requires the one current-candidate independent scientific review')
    smoke_path = Path(review['smoke_path']).resolve()
    if not smoke_path.is_relative_to((OWNED/'validation').resolve()) or sha(smoke_path) != review['smoke_sha256']:
        raise ValueError('Calibration smoke must be the retained bounded output for this release')
    smoke = load(smoke_path)
    required = dict(state='passed', phase=PHASE, source_commit=manifest['source_commit'],
                    release_sha256=sha(release_path), phase_config_sha256=config_sha,
                    teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
                    previous_stage_receipt_sha256=manifest['teacher_eval_dependency']['accepted_sha256'],
                    computation_kind='gradient-calibration', production_entrypoint_reached=True,
                    finite_outputs=True, model_unchanged=True, validation_used=False,
                    parameter_updates=False, official_metrics_disabled=True, artifact_ingest_disabled=True,
                    actual_device='cuda:0')
    if any(smoke.get(k) != v for k, v in required.items()) or smoke.get('actual_training_images') != 1:
        raise ValueError('Current calibration release needs one real bounded CUDA production-path image')
    baseline = smoke.get('baseline_equivalence_probe', {})
    if (not 1 <= smoke.get('production_forward_calls', 0) <= 100
            or not 1 <= smoke.get('total_production_forward_calls', 0) <= 100
            or smoke.get('actual_forward_tensor_devices') != ['cuda:0']
            or smoke.get('actual_cuda_device_indices') != [0]
            or baseline.get('all_sensitivity_sums_exact') is not True
            or baseline.get('absolute_difference') != 0 or baseline.get('tolerance') != 0
            or baseline.get('reference_actual_forward_tensor_devices') != ['cuda:0']
            or baseline.get('reference_actual_cuda_device_indices') != [0]):
        raise ValueError('Calibration smoke exceeded its 1..100 forward bound')
    return review, smoke


def _finite(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return not isinstance(value, float) or (value == value and abs(value) != float('inf'))


def canonical_gradient_reference(trainer, image_path, label_dir, raw):
    """Independent canonical per-layer loops, same frozen Teacher and image.

    The producer collects both layers from one forward. This reference follows
    the original builder's separate forward for each layer and compares every
    target/NT sensitivity sum rather than only checking that arrays exist.
    """
    import torch
    from ue_framework.data_utils import load_image_rgb_float, label_path_for_image, read_yolo_annotations
    from ue_framework.methods.semantic_coordinates import gt_evidence
    from ue_framework.methods.s4_tfss import model_hash
    from .coco_eval import raw_tensor_device_details, raw_signature
    model=trainer.surrogate;before=model_hash(model);devices=set();indices=set();calls=[]
    def hook(module,inputs,result):
        ds,ix=raw_tensor_device_details(result);devices.update(ds);indices.update(ix)
        raw_signature(result);calls.append(int(inputs[0].shape[0]))
    handle=model.register_forward_hook(hook)
    image=load_image_rgb_float(str(image_path))
    anns=read_yolo_annotations(label_path_for_image(str(image_path),str(label_dir)))
    compared=0
    try:
        for layer in ('model.15','model.21'):
            target=None;nt=None;by={c:None for c in range(1,80)}
            x=trainer._to_tensor(image).requires_grad_(True)
            trainer.multi_features.clear();decoded=trainer._forward_raw(x);feature=trainer.multi_features[layer]
            for cls,evidence in gt_evidence(decoded,anns,image.shape[:2],trainer.imgsz):
                g=torch.autograd.grad(evidence,feature,retain_graph=True)[0].detach().abs().mean((0,2,3)).double().cpu()
                if cls==0:target=g.clone() if target is None else target+g
                else:
                    nt=g.clone() if nt is None else nt+g
                    by[cls]=g.clone() if by[cls] is None else by[cls]+g
            item=raw['layers'][layer]
            for candidate,reference in [(item['target_sum'],target),(item['nt_sum'],nt),
                                         *((item['nt_sums'][c],by[c]) for c in by)]:
                if candidate is None or reference is None:
                    if candidate is not None or reference is not None:raise ValueError('Canonical missing-class statistics differ')
                elif not torch.equal(candidate,reference):
                    raise ValueError('Actual P3/P5 statistics differ from independent canonical gradient loops')
                else:compared+=candidate.numel()
            del x,decoded,feature
    finally:handle.remove()
    if model_hash(model)!=before:raise ValueError('Canonical gradient reference changed frozen Teacher')
    return dict(reference_id='independent original per-layer GT-evidence gradient loops',
        reference_metric=0,candidate_metric=0,metric_name='P3/P5 target, pooledNT and per-class sensitivity sums exact identity',
        absolute_difference=0,tolerance=0,all_sensitivity_sums_exact=True,compared_values=compared,
        reference_forward_calls=len(calls),reference_forward_input_batch_sizes=calls,
        reference_actual_forward_tensor_devices=sorted(devices),reference_actual_cuda_device_indices=sorted(indices),
        frozen_model_unchanged=True)


def run(release_path, output, *, registration=None, smoke_images=0):
    import numpy as np
    import torch
    import yaml
    from ue_framework.data_utils import label_path_for_image, load_image_rgb_float, read_yolo_annotations
    from ue_framework.methods.legacy_kproto_ret import LegacyKProtoTrainer
    from ue_framework.methods.b2_semantic import B2SemanticTrainer
    from ue_framework.methods.semantic_coordinates import semantic_identity
    from ue_framework.methods.s4_tfss import model_hash

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Calibration output already exists; inspect it and use a unique recovery identity')
    manifest, phase_cfg, checkpoint, teacher_receipt = phase_release(release_path, PHASE)
    validate_phase_config(phase_cfg)
    eval_path, _ = accepted_eval_dependency(manifest)
    if smoke_images:
        if smoke_images != 1 or not output.is_relative_to((OWNED/'validation').resolve()):
            raise ValueError('Calibration GPU smoke is exactly one real train-only image in owned validation root')
    else:
        if output != (WORKFLOW_OUTPUT/'calibration').resolve() or registration is None:
            raise ValueError('Formal calibration must use the existing serial stage output and READY gate')
    review = smoke = None
    if not smoke_images:
        review, smoke = formal_gate(registration, release_path, manifest, sha(ROOT/manifest['phase_config_path']))
    if not torch.cuda.is_available():
        raise ValueError('Calibration phase requires the admitted CUDA0 device')
    device = torch.device('cuda:0')
    torch.set_num_threads(1)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    cfg, checkpoint, artifact_root = runtime_config(phase_cfg, manifest)
    images, expected_hashes, expected_digest = train_manifest()
    if len(images) != FIXED_COUNTS['images']:
        raise ValueError('COCO clean-train image list count changed')
    teacher, frozen_sha = load_teacher(checkpoint, manifest['teacher_dependency']['checkpoint_sha256'], device)
    consumer_seed = copy.deepcopy(teacher)
    method_cfg = cfg['methods']['legacy_kproto_ret']
    trainer = LegacyKProtoTrainer(cfg, method_cfg, device, teacher)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['WANDB_MODE'] = 'disabled'
    common = dict(phase=PHASE, source_commit=manifest['source_commit'], release_sha256=sha(release_path),
                  phase_config_sha256=sha(ROOT/manifest['phase_config_path']),
                  teacher_checkpoint_sha256=sha(checkpoint),
                  teacher_receipt_sha256=manifest['teacher_dependency']['accepted_sha256'],
                  teacher_eval_receipt_sha256=manifest['teacher_eval_dependency']['accepted_sha256'],
                  previous_stage_receipt_sha256=manifest['teacher_eval_dependency']['accepted_sha256'],
                  accepted_teacher_eval_hash=sha(eval_path), validation_used=False, parameter_updates=False,
                  checkpoint_selection_allowed=False, official_metrics_disabled=True, artifact_ingest_disabled=True,
                  cleanup_policy='retain_all', deleted_paths=[])
    atomic(output/'status.json', dict(common, state='RUNNING', started_UTC=now(), smoke=bool(smoke_images)))
    write_once(output/'phase_spec.json', dict(common, computation_kind='gradient-calibration',
               image_count=(1 if smoke_images else FIXED_COUNTS['images']), class_count=80,
               target_class_id=0, config_path=manifest['config_path'], artifact_root=str(artifact_root),
               source_root=str(ROOT.resolve()),
               teacher_eval_metrics_read=False, validation_images_opened=False,
               expected_statistics_manifest_sha256=expected_digest if not smoke_images else None,
               expected_teacher_receipt=teacher_receipt,
               expected_teacher_eval_receipt=dict(sha256=sha(eval_path), image_count=5000, classes=80,
                                                  teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'])))
    stats_root = output/'smoke_statistics' if smoke_images else artifact_root/'collected'
    if smoke_images:
        selected = None
        for path in images:
            label = label_path_for_image(str(path), str(DATA_ROOT/'labels/train'))
            annotations = read_yolo_annotations(label)
            if not any(int(a['cls']) == 0 for a in annotations):
                continue
            image = load_image_rgb_float(path)
            support, _, _ = trainer._build_support(image.shape, annotations, image_path=str(path))
            if support.sum() > 10:
                selected = path
                break
        if selected is None:
            raise ValueError('No eligible clean Person image found for bounded calibration smoke')
        chosen = [selected]
    else:
        chosen = images
    before = model_hash(teacher)
    forward_calls = []; forward_devices=set(); forward_cuda_indices=set()
    def audit_forward(module, inputs, result):
        from .coco_eval import raw_tensor_device_details, raw_signature
        devices,indices=raw_tensor_device_details(result)
        forward_devices.update(devices);forward_cuda_indices.update(indices)
        raw_signature(result)  # finite raw tensors, before GT evidence reduction
        forward_calls.append(int(inputs[0].shape[0]))
    hook = teacher.register_forward_hook(audit_forward) if smoke_images else None
    try:
        try:
            raw = collect(trainer, teacher, chosen, DATA_ROOT/'labels/train', stats_root,
                          expected_hashes=expected_hashes)
        finally:
            if hook is not None:
                hook.remove()
        if not _finite(raw) or raw['validation_used'] or model_hash(teacher) != before or before != frozen_sha:
            raise ValueError('Calibration produced nonfinite statistics or changed the frozen Teacher')
        if smoke_images:
            if (raw['images'] != 1 or raw['target_count'] <= 0 or len(raw['eligible_person_images']) != 1
                    or not 1 <= len(forward_calls) <= 100
                    or forward_devices!={'cuda:0'} or forward_cuda_indices!={0}):
                raise ValueError('One-image calibration smoke did not exercise eligible Person statistics')
            reference_trainer=LegacyKProtoTrainer(cfg,method_cfg,device,consumer_seed)
            baseline=canonical_gradient_reference(reference_trainer,selected,DATA_ROOT/'labels/train',raw)
            total_calls=len(forward_calls)+baseline['reference_forward_calls']
            if (not 1<=total_calls<=100 or baseline['reference_actual_forward_tensor_devices']!=['cuda:0']
                    or baseline['reference_actual_cuda_device_indices']!=[0]):
                raise ValueError('Canonical calibration reference did not prove bounded CUDA0 execution')
            baseline.update(reference_weights_path=str(checkpoint),candidate_weights_path=str(checkpoint),
                weights_sha256=sha(checkpoint),evaluation_setting=dict(dataset=str(selected),sample_count=1,seed=0,
                parameters=dict(imgsz=640,device='cuda:0',target_class_id=0,num_classes=80),
                post_processing='original per-GT evidence and channel sensitivity reduction'))
            status = dict(common, state='passed', computation_kind='gradient-calibration',
                          production_entrypoint_reached=True, finite_outputs=True, model_unchanged=True,
                          actual_device=next(iter(forward_devices)),actual_cuda_device_index=0,
                          actual_forward_tensor_devices=sorted(forward_devices),actual_cuda_device_indices=sorted(forward_cuda_indices),
                          forward_input_batch_sizes=forward_calls,
                          baseline_equivalence_probe=baseline,total_production_forward_calls=total_calls,
                          actual_training_images=1, production_forward_calls=len(forward_calls),
                          target_objects_observed=raw['target_count'], validation_used=False,
                          parameter_updates=False, completed_UTC=now(), retained=['phase_spec.json','smoke_statistics/raw_statistics.pt'],
                          result_manifest_sha256=raw['clean_train_manifest_sha256'])
            write_once(output/'smoke_summary.json', status)
            atomic(output/'status.json', status)
            return status
        non_target_count = sum(raw['layers']['model.21']['nt_counts'].values())
        if (raw['images'] != FIXED_COUNTS['images']
                or raw['target_count'] != FIXED_COUNTS['person_ground_truth']
                or raw['target_count'] + non_target_count != FIXED_COUNTS['ground_truth']):
            raise ValueError('Actual clean COCO statistics do not match the independently accepted 25K/Person counts')
        if raw['clean_train_manifest_sha256'] != expected_digest:
            raise ValueError('Per-image verified clean-train manifest differs from the accepted manifest')
        if not raw['eligible_person_images']:
            raise ValueError('Original S4 support gate selected no eligible Person train images')
        proto_cache = Path(method_cfg['kproto']['prototype_cache'])
        cache_key = trainer._cache_key(DATA_ROOT/'images/train', DATA_ROOT/'labels/train')
        prototypes = trainer._build_bank(images, str(DATA_ROOT/'labels/train'), str(proto_cache), cache_key)
        saved = torch.load(proto_cache, map_location='cpu', weights_only=True)
        if (saved.get('cache_key') != cache_key or saved.get('count', 0) < 8
                or tuple(saved['prototypes'].shape) != (8, trainer.multi_features.get('model.21', prototypes).shape[1])
                or not torch.isfinite(prototypes).all()):
            raise ValueError('Native clean-train K=8 prototype bank is invalid')
        try:
            coordinates = build_coordinates(raw, cfg, artifact_root/'method',
                                            builder_path=ROOT/'ue_framework/methods/safebg.py', source_root=ROOT)
        except ProtocolSpaceBlocked:
            raise
        except ValueError as exc:
            write_once(artifact_root/'BLOCKED_PROTOCOL_COMPATIBILITY.json',
                       dict(state='BLOCKED_PROTOCOL_COMPATIBILITY', reason=str(exc),
                            thresholds_unchanged=True, raw_statistics_retained=True, UTC=now()))
            raise ProtocolSpaceBlocked('Original S4 frozen coordinate/rank gate failed: '+str(exc)) from exc
        if len(coordinates['selected_channels']) != 24:
            raise ProtocolSpaceBlocked('Actual COCO StrictP5 space does not satisfy original24D protocol')
        semantic = semantic_identity(cfg, ROOT)
        frozen_consumer = copy.deepcopy(consumer_seed).eval()
        adaptive_consumer = copy.deepcopy(consumer_seed).eval()
        consumer = B2SemanticTrainer(cfg, method_cfg, device, frozen_consumer, adaptive_consumer)
        if (consumer.channels.numel() != 24 or len(consumer.bases) != 79 or len(consumer.safe_bases) != 79
                or consumer.orth_feature_layer != 'model.15' or consumer.adaptive_feature_layer != 'model.15'
                or consumer.safe_bg_rank != 8 or consumer.safe_bg_removal_alpha != .5
                or consumer.safe_bg_active_k != 3):
            raise ValueError('Actual B2 Safe-BG/orthogonal consumer rejected calibrated COCO artifacts')
        del consumer, frozen_consumer, adaptive_consumer
        if model_hash(teacher) != before or sha(checkpoint) != manifest['teacher_dependency']['checkpoint_sha256']:
            raise ValueError('Read-only COCO calibration modified the accepted Teacher checkpoint')
        artifact_files = []
        for path in sorted(artifact_root.rglob('*')):
            if path.is_file():
                artifact_files.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), bytes=path.stat().st_size))
        if not artifact_files:
            raise ValueError('Calibration generated no retained source-root artifacts')
        report = dict(common, state='ACCEPTED', computation_kind='clean-train-calibration',
                      production_entrypoint_reached=True, finite_outputs=True, model_unchanged=True,
                      validation_used=False, parameter_updates=False, checkpoint_selection_allowed=False,
                      images=raw['images'], person_ground_truth=raw['target_count'],
                      non_target_objects_by_class={str(c): raw['layers']['model.21']['nt_counts'][c] for c in range(1,80)},
                      accepted_clean_train_manifest_sha256=raw['clean_train_manifest_sha256'],
                      eligible_person_image_count=len(raw['eligible_person_images']),
                      original_fourier_optimizer_steps=24*((len(raw['eligible_person_images'])+15)//16),
                      prototypes=dict(path=str(proto_cache), sha256=sha(proto_cache), cache_key=cache_key,
                                      count=saved['count'], shape=list(saved['prototypes'].shape),
                                      source='separate original clean no-grad P5 context pass'),
                      coordinates=coordinates, semantic_identity=semantic,
                      selected_channels=coordinates['selected_channels'], selected_channel_count=24,
                      artifact_files=artifact_files, cleanup_policy='retain_all', deleted_paths=[], completed_UTC=now())
        if len(report['non_target_objects_by_class']) != 79 or any(v <= 0 for v in report['non_target_objects_by_class'].values()):
            raise ValueError('COCO calibration lacks one or more of the 79 non-target classes')
        write_once(output/'accepted.json', report)
        atomic(output/'status.json', report)
        return report
    except ProtocolSpaceBlocked as exc:
        atomic(output/'status.json', dict(common, state='BLOCKED_PROTOCOL_COMPATIBILITY',
                                           error=str(exc), completed_UTC=now(), cleanup_policy='retain_all'))
        raise
    except BaseException as exc:
        atomic(output/'status.json', dict(common, state='FAILED', error=repr(exc), ended_UTC=now(),
                                          cleanup_policy='retain_all'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--smoke-images', type=int, default=0)
    args = parser.parse_args()
    run(args.release, args.output, registration=args.registration, smoke_images=args.smoke_images)
