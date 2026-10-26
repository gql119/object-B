"""Full, retained Original S4 COCO poisoned-train export and decoded-pixel audit."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from .dimension_protocol import validate_receipt_dimension
from .phase_contract import WORKFLOW

PHASE = 'export25k'
MAIN_CONFIG = 'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml'
TFSS_LOCK = 'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json'
TFSS_LOCK_SHA256 = '33d3d7af6c76c2c881db0dfc7d6fc66dafa9b5655026c07332701143ed52d6ec'
TFSS_MAP_VERSION = 'masked3x3-Q10Q90-clean-input-perGT-max-person-equalNTclass-image-canonical-canvas-v2'


def validate_phase_config(cfg):
    expected = dict(schema='s4.coco.export25k-config.v1', phase=PHASE,
                    workflow_run_id=WORKFLOW,
                    predecessor_phase='fourier24', predecessor_receipt_kind='accepted.json',
                    teacher_policy='fixed native EMA last@200',
                    source_split='official fixed COCO minitrain2017 train25K', image_count=25000,
                    class_count=80, target_class_id=0, support_mode='bbox_half_overlap',
                    pixel_budget=dict(person_exclusive=8, person_non_target_overlap=4, outside=0),
                    encoding='lossless PNG, RGB, original dimensions',
                    labels='byte-identical official JSON-derived YOLO80 labels',
                    tfss=dict(mode='TFSS-U', eta=.85, map_version=TFSS_MAP_VERSION,
                              canonical_lock_path=TFSS_LOCK,
                              canonical_lock_sha256=TFSS_LOCK_SHA256),
                    validation_used=False, official_metrics_enabled=False,
                    artifact_ingest_enabled=False, cleanup_policy='retain_all')
    if cfg != expected:
        raise ValueError('Frozen Original S4 COCO export contract changed')


def validate_label_classes(annotations, num_classes=80):
    if num_classes != 80:
        raise ValueError('COCO label class space must contain exactly 80 classes')
    for ann in annotations:
        cls = ann.get('cls')
        box = ann.get('bbox')
        if (not isinstance(cls, int) or isinstance(cls, bool) or not 0 <= cls < num_classes
                or not isinstance(box, (tuple, list)) or len(box) != 4):
            raise ValueError('Export encountered an invalid COCO80 YOLO label row')


def validate_decoded_pair(clean_u8, poisoned_u8, support, audit_pixels):
    import numpy as np
    if not np.isin(support, (0., .5, 1.)).all():
        raise ValueError('Export support is not the original S4 8/4/0 domain')
    audit = audit_pixels(clean_u8, poisoned_u8, support)
    if audit['outside_nonzero'] != 0 or audit['max_bytes'] > 8 or audit['overlap_max_bytes'] > 4:
        raise ValueError('Decoded PNG violates original S4 8/4/0 pixel budgets')
    return audit


def compose_tfss_utility_png(clean_rgb, base_perturbation, q, risk, support, *, eps=8/255, eta=.85):
    """Apply the canonical detached TFSS-U map before Original S4 PNG projection."""
    import numpy as np
    import torch
    from ue_framework.methods.s4_pixel_audit import png_project
    from ue_framework.methods.tfss_spatial import spatial_weights

    clean = np.asarray(clean_rgb, dtype=np.float32)
    delta = np.asarray(base_perturbation, dtype=np.float32)
    support_np = np.asarray(support, dtype=np.float32)
    if clean.ndim != 3 or clean.shape[-1] != 3 or delta.shape != clean.shape:
        raise ValueError('TFSS export requires matching HWC RGB clean image and base delta')
    if support_np.shape != clean.shape[:2] or not np.isfinite(clean).all() or not np.isfinite(delta).all():
        raise ValueError('Invalid TFSS export image, delta, or support geometry')
    if not np.isfinite(eps) or eps != 8/255 or eta != .85:
        raise ValueError('Original S4 TFSS-U COCO export requires frozen eps=8/255 and eta=.85')
    if not np.isin(support_np, (0., .5, 1.)).all():
        raise ValueError('Original S4 support must retain exact 1/.5/0 semantics')
    budget = support_np * eps
    if (np.any(np.abs(delta[support_np == 0]) > 0)
            or np.any(np.abs(delta) > budget[..., None] + 1e-7)):
        raise ValueError('Canonical Fourier delta violates Original S4 support/budget before TFSS')
    q_t = torch.as_tensor(q).detach()
    risk_t = torch.as_tensor(risk).detach()
    support_t = torch.as_tensor(support_np, device=q_t.device, dtype=torch.float32)
    weight_t = spatial_weights(q_t, risk_t, support_t, mode='TFSS-U', eta=eta)
    weight = weight_t.to(device='cpu', dtype=torch.float32).numpy()
    candidate = np.clip(clean + delta * weight[..., None], 0., 1.)
    clean_u8 = np.rint(np.clip(clean, 0., 1.) * 255.).astype(np.uint8)
    poisoned_u8 = png_project(clean_u8, candidate, support_np * eps)
    return candidate, poisoned_u8, weight


def accepted_fourier_dependency(manifest):
    from .phase_contract import WORKFLOW, WORKFLOW_OUTPUT
    from .runtime import load, sha
    path = WORKFLOW_OUTPUT/'fourier24'/'accepted.json'
    expected = manifest.get('fourier_dependency', {})
    if not path.is_file() or sha(path) != expected.get('accepted_sha256'):
        raise ValueError('Accepted Original S4 Fourier24 receipt is absent or changed')
    receipt = load(path)
    from .canonical_s4_training import validate_proof
    validate_proof(receipt.get('tfss_training_proof', {}), draws=receipt.get('optimizer_steps', 0)*2)
    required = dict(state='ACCEPTED', computation_kind='original-S4-Fourier24',
                    validation_used=False, official_metrics_disabled=True,
                    artifact_ingest_disabled=True, epochs_completed=24,
                    clean_train_eligible_files_unchanged=True)
    if any(receipt.get(k) != v for k, v in required.items()):
        raise ValueError('Fourier receipt is not a completed frozen Original S4 phase')
    params = WORKFLOW_OUTPUT/'fourier24'/'global_params.pt'
    if not params.is_file() or sha(params) != receipt.get('Fourier_params_sha256'):
        raise ValueError('Accepted Fourier parameter artifact hash differs')
    pipeline = load(WORKFLOW_OUTPUT/'pipeline_state.json')
    stage = pipeline.get('stages', {}).get('fourier24', {})
    if stage.get('state') != 'ACCEPTED' or stage.get('receipt_sha256') != sha(path):
        raise ValueError('Existing serial controller has not accepted Fourier24')
    if expected.get('accepted_sha256') != stage.get('receipt_sha256'):
        raise ValueError('Export is not bound to the exact accepted Fourier receipt')
    if receipt.get('calibration_receipt_sha256') != manifest['calibration_dependency']['accepted_sha256']:
        raise ValueError('Fourier was created from another calibration')
    return path, receipt, params


def validate_accepted_calibration(calibration):
    dimension=validate_receipt_dimension(calibration)
    channels = calibration.get('selected_channels')
    nt = calibration.get('non_target_objects_by_class')
    if (calibration.get('images') != 25000 or calibration.get('person_ground_truth') != 55096
            or not isinstance(channels, list) or len(channels) != dimension or len(set(channels)) != dimension
            or not isinstance(nt, dict) or set(nt) != {str(i) for i in range(1, 80)}
            or any(not isinstance(v, int) or v <= 0 for v in nt.values())):
        raise ValueError('Export requires the accepted COCO selected-space/79-class calibration')


def validate_accepted_fourier(fourier, calibration):
    if (fourier.get('epochs_completed') != 24
            or fourier.get('optimizer_steps') != calibration.get('original_fourier_optimizer_steps')
            or fourier.get('train_manifest_sha256') != calibration.get('accepted_clean_train_manifest_sha256')
            or fourier.get('selected_channels') != calibration.get('selected_channels')
            or fourier.get('calibration_receipt_sha256') is None
            or fourier.get('clean_train_eligible_files_unchanged') is not True):
        raise ValueError('Export requires the accepted complete Fourier24 result from this calibration')


def _save_png(path, rgb_u8, cv2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Poisoned export path already exists: '+str(path))
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError('OpenCV failed to write lossless poisoned PNG: '+str(path))


def _decode_png(path, cv2):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError('Could not decode exported PNG: '+str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _eligible_probe_path(document, root):
    from .phase_fourier import eligible_paths_from_manifest
    paths, _ = eligible_paths_from_manifest(document, root)
    if not paths:
        raise ValueError('Accepted Fourier eligible support manifest has no image')
    return paths[0]


def formal_gate(registration, release_path, manifest, config_sha):
    from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT
    from .runtime import load, safe_relative, sha
    registration = Path(registration).resolve()
    expected_registration = (WORKFLOW_OUTPUT/'receipts'/'export25k_READY.json').resolve()
    if registration != expected_registration:
        raise ValueError('Wrong existing-controller export registration route')
    ready = load(registration)
    release_path = Path(release_path).resolve()
    dependency = manifest['fourier_dependency']
    expected = dict(phase=PHASE, workflow_run_id=WORKFLOW, pre_run_result='pass',
                    module='coco_s4.phase_export', source_root=str(Path(__file__).resolve().parents[1]),
                    release_sha256=sha(release_path),
                    previous_stage_receipt_sha256=dependency['accepted_sha256'])
    if (any(ready.get(k) != v for k, v in expected.items())
            or (Path(ready['source_root'])/safe_relative(ready.get('release_relative', ''))).resolve()!=release_path):
        raise ValueError('Export READY registration differs from this exact Fourier-bound release')
    closure = Path(ready['scientific_closure_path']).resolve()
    if closure.parent != (WORKFLOW_OUTPUT/'receipts').resolve() or sha(closure) != ready['scientific_closure_sha256']:
        raise ValueError('Export scientific closure path/hash mismatch')
    review = load(closure)
    review_expected = dict(phase=PHASE, source_commit=manifest['source_commit'],
                           release_sha256=sha(release_path), phase_config_sha256=config_sha,
                           Fourier_receipt_sha256=dependency['accepted_sha256'],
                           teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
                           pre_run_result='pass', reviewer_count=1,
                           scientifically_correct=True, allow_run=True, blockers=[])
    if any(review.get(k) != v for k, v in review_expected.items()):
        raise ValueError('Export requires one current-release scientific review and main closure')
    smoke_path = Path(review['smoke_path']).resolve()
    if not smoke_path.is_relative_to((OWNED/'validation').resolve()) or sha(smoke_path) != review['smoke_sha256']:
        raise ValueError('Export smoke evidence escaped the owned validation root')
    smoke = load(smoke_path)
    smoke_expected = dict(state='passed', phase=PHASE, source_commit=manifest['source_commit'],
                          release_sha256=sha(release_path), phase_config_sha256=config_sha,
                          Fourier_receipt_sha256=dependency['accepted_sha256'],
                          computation_kind='poisoned-export', finite_outputs=True,
                          production_entrypoint_reached=True, decoded_budget_passed=True,
                          labels_byte_identical=True, validation_used=False,
                          parameter_updates=False, official_metrics_disabled=True,
                          artifact_ingest_disabled=True, actual_device='cuda:0')
    if any(smoke.get(k) != v for k, v in smoke_expected.items()):
        raise ValueError('Current export release lacks the required bounded CUDA export evidence')
    if smoke.get('actual_training_images') != 1 or not 1 <= smoke.get('production_forward_calls', 0) <= 100:
        raise ValueError('Export smoke must audit one eligible training image within 100 forwards')
    return review, smoke


def run(release_path, output, *, registration=None, smoke_images=0):
    import copy
    import cv2
    import numpy as np
    import torch
    import yaml
    from .data import audit_pixels
    from .phase_calibration import FIXED_COUNTS, load_teacher, train_manifest
    from .phase_contract import DATA_ROOT, OWNED, ROOT, WORKFLOW_OUTPUT, phase_release
    from .phase_fourier import (CALIBRATION_ARTIFACT_SUFFIXES, copy_calibration_consumer_artifacts,
                                runtime_config, validate_fourier_source_config)
    from .runtime import atomic, load, now, sha, write_once
    from ue_framework.data_utils import load_image_rgb_float, read_yolo_annotations
    from ue_framework.methods.legacy_kproto_ret import LegacyKProtoGenerator
    from ue_framework.methods.s4_tfss import CleanMapCache, S4Trainer, model_hash

    output = Path(output).resolve()
    manifest, phase_cfg, checkpoint, _ = phase_release(release_path, PHASE)
    validate_phase_config(phase_cfg)
    if manifest.get('config_path') != MAIN_CONFIG or MAIN_CONFIG not in manifest.get('files', {}):
        raise ValueError('Export must bind the exact Original S4 COCO source config')
    lock_path = ROOT/TFSS_LOCK
    lock_document = load(lock_path)
    if (manifest.get('files', {}).get(TFSS_LOCK) != TFSS_LOCK_SHA256
            or sha(lock_path) != TFSS_LOCK_SHA256
            or lock_document.get('tfss', {}).get('map_version') != TFSS_MAP_VERSION
            or lock_document.get('tfss', {}).get('eta_U') != .85):
        raise ValueError('Export release does not bind the canonical Original S4 TFSS-U lock')
    for relative in ('ue_framework/methods/tfss_clean_maps.py',
                     'ue_framework/methods/tfss_nt_sensitivity.py',
                     'ue_framework/methods/tfss_spatial.py',
                     'ue_framework/methods/s4_pixel_audit.py',
                     'ue_framework/methods/s4_tfss.py'):
        if manifest.get('files', {}).get(relative) != sha(ROOT/relative):
            raise ValueError('Export TFSS/native pixel source is outside the immutable source release: '+relative)
    validate_fourier_source_config(yaml.safe_load((Path(__file__).resolve().parents[1]/MAIN_CONFIG).read_text(encoding='utf-8')))
    config_sha = sha(Path(__file__).resolve().parents[1]/manifest['phase_config_path'])
    if smoke_images:
        if smoke_images != 1 or not output.is_relative_to((OWNED/'validation').resolve()):
            raise ValueError('Export smoke is exactly one real image under the owned validation root')
    elif output != (WORKFLOW_OUTPUT/'export25k').resolve() or registration is None:
        raise ValueError('Formal poisoned export must use the existing serial stage output and READY gate')
    calibration_path, calibration, _ = __import__('coco_s4.phase_fourier', fromlist=['accepted_calibration_dependency']).accepted_calibration_dependency(manifest)
    fourier_path, fourier, params_path = accepted_fourier_dependency(manifest)
    validate_accepted_calibration(calibration)
    validate_accepted_fourier(fourier, calibration)
    if smoke_images:
        review = smoke = None
    else:
        review, smoke = formal_gate(registration, release_path, manifest, config_sha)
    if not torch.cuda.is_available():
        raise ValueError('Original S4 export production path requires the admitted CUDA0 device')
    if output.exists():
        raise FileExistsError('Export output exists; preserve it and use a new diagnosed recovery identity')
    output.mkdir(parents=True, exist_ok=False)
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['YOLO_CONFIG_DIR'] = str(output/'yolo_settings')
    os.environ['MPLCONFIGDIR'] = str(output/'mpl')
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    common = dict(phase=PHASE, source_commit=manifest['source_commit'],
                  release_sha256=sha(release_path), phase_config_sha256=config_sha,
                  teacher_checkpoint_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
                  teacher_receipt_sha256=manifest['teacher_dependency']['accepted_sha256'],
                  calibration_receipt_sha256=sha(calibration_path),
                  Fourier_receipt_sha256=sha(fourier_path), Fourier_params_sha256=sha(params_path),
                  train_manifest_sha256=calibration['accepted_clean_train_manifest_sha256'],
                  TFSS_mode='TFSS-U', TFSS_map_version=TFSS_MAP_VERSION, TFSS_eta=.85,
                  validation_used=False, official_metrics_disabled=True, artifact_ingest_disabled=True,
                  cleanup_policy='retain_all', deleted_paths=[])
    status = dict(common, state='RUNNING', started_UTC=now(), smoke=bool(smoke_images))
    atomic(output/'status.json', status)
    write_once(output/'phase_spec.json', dict(common, phase_config=phase_cfg,
        source_root=str(ROOT.resolve()), output_root=str(output),
        original_config_path=MAIN_CONFIG, canonical_tfss_lock_path=TFSS_LOCK,
        train_image_manifest_sha256=manifest['data_files']['metadata/TRAIN25K_KAGGLE_IMAGE_MANIFEST_20261024.jsonl'],
        train_json_sha256=manifest['data_files']['metadata/instances_minitrain2017.json'],
        tfss_clean_map_cache=str(output/'clean_maps'),
        output_semantics='Person0 rows use Original S4 clean-map TFSS-U weighted pre-clamp Fourier delta; non-Person bytes copied exactly'))
    try:
        imported_root, imported = copy_calibration_consumer_artifacts(
            output, dict(manifest=manifest, receipt=calibration,
                         phase_spec=load(WORKFLOW_OUTPUT/'calibration'/'phase_spec.json')))
        cfg, _ = runtime_config(phase_cfg, manifest, imported_root, smoke=bool(smoke_images))
        method_cfg = cfg['methods']['legacy_kproto_ret']
        images, expected_hashes, expected_digest = train_manifest()
        if expected_digest != calibration['accepted_clean_train_manifest_sha256']:
            raise ValueError('Clean COCO train manifest differs from accepted calibration')
        if fourier['train_manifest_sha256'] != expected_digest:
            raise ValueError('Accepted Fourier phase used a different clean train manifest')
        eligible_document = load(imported_root/'collected'/'eligible_support_manifest.json')
        if smoke_images:
            selected_path = _eligible_probe_path(eligible_document, DATA_ROOT/'images/train')
            selected = [selected_path]
        else:
            selected = images
        expected_hash_map = expected_hashes
        teacher, frozen_sha = load_teacher(checkpoint,
                                           manifest['teacher_dependency']['checkpoint_sha256'], device)
        adaptive = copy.deepcopy(teacher).to(device)
        trainer = S4Trainer(cfg, method_cfg, device, teacher, adaptive)
        trainer.tfss_mode = 'TFSS-U'
        trainer.s4_cache = CleanMapCache(output/'clean_maps', dict(
            reference_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
            maps_code_sha256=manifest['files']['ue_framework/methods/tfss_clean_maps.py'],
            risk_code_sha256=manifest['files']['ue_framework/methods/tfss_nt_sensitivity.py'],
            map_version=phase_cfg['tfss']['map_version'],
            canonical_lock_sha256=phase_cfg['tfss']['canonical_lock_sha256']))
        if (trainer.target_class_id != 0 or trainer.num_classes != 80
                or trainer.feature_layer != 'model.21' or trainer.adaptive_feature_layer != 'model.15'):
            raise ValueError('Export did not construct the fixed Original S4 COCO80 P5/P3 trainer')
        params = torch.load(params_path, map_location='cpu', weights_only=True)
        if (params.get('method') != 'legacy_kproto_ret' or params.get('support_mode') != 'bbox_half_overlap'
                or int(params.get('target_class_id', -1)) != 0 or int(params.get('num_classes', -1)) != 80
                or len(params.get('coords', [])) != 16 or tuple(params.get('fourier_coeff', torch.empty(0)).shape) != (16, 3)
                or abs(float(params.get('eps', 0))-8/255) > 1e-9):
            raise ValueError('Accepted learned Fourier artifact does not match frozen Original S4 COCO protocol')
        generator = LegacyKProtoGenerator(cfg, method_cfg, device, teacher, str(params_path))
        params_hash_before = sha(params_path)
        model_before = model_hash(teacher)
        adaptive_before = model_hash(trainer.adaptive.model)
        forward_calls = []
        hook = teacher.register_forward_hook(lambda module, args, result: forward_calls.append(1))
        images_out = output/'poisoned'/'images'/'train'
        labels_out = output/'poisoned'/'labels'/'train'
        images_out.mkdir(parents=True, exist_ok=False); labels_out.mkdir(parents=True, exist_ok=False)
        manifest_path = output/'poisoned_manifest.jsonl'
        row_count = person_count = nonperson_count = empty_label_count = 0
        changed_pixels = person_exclusive_pixels = overlap_pixels = outside_pixels = 0
        max_bytes = overlap_max = 0
        try:
            with manifest_path.open('x', encoding='utf-8', newline='') as stream:
                for index, image_path in enumerate(selected):
                    stem = image_path.stem
                    expected = expected_hash_map.get(stem)
                    if expected is None or sha(image_path) != expected[0]:
                        raise ValueError('Clean source image differs from accepted manifest: '+stem)
                    label_path = DATA_ROOT/'labels'/'train'/(stem+'.txt')
                    label_bytes = label_path.read_bytes()
                    label_sha = hashlib.sha256(label_bytes).hexdigest()
                    if label_sha != expected[1]:
                        raise ValueError('Clean label differs from accepted manifest: '+stem)
                    anns = read_yolo_annotations(str(label_path))
                    validate_label_classes(anns)
                    clean = load_image_rgb_float(str(image_path))
                    clean_u8 = np.rint(np.clip(clean, 0., 1.)*255.).astype(np.uint8)
                    is_person = any(a['cls'] == 0 for a in anns)
                    support, _, _ = generator._build_support(
                        clean.shape, anns, support_type='mask', image_path=str(image_path))
                    support = np.asarray(support, dtype=np.float32)
                    if is_person:
                        if not np.any(support):
                            raise ValueError('Person GT has no original S4 support: '+stem)
                        maps, map_records = trainer.s4_cache.get(
                            trainer, [(clean, anns, support)], [str(image_path)])
                        q, risk, _valid = maps[0]
                        result = generator.generate(image=clean, annotations=anns, seed=index,
                                                    steps=24, eps=8/255, support_type='mask',
                                                    image_path=str(image_path))
                        result_support = np.asarray(result.support_mask, dtype=np.float32)
                        if not np.array_equal(result_support, support):
                            raise ValueError('Generator support differs from clean TFSS support: '+stem)
                        candidate, poisoned_u8, weight = compose_tfss_utility_png(
                            clean, result.perturbation, q, risk, support, eps=8/255, eta=.85)
                        if not np.isfinite(candidate).all() or poisoned_u8.shape != clean_u8.shape:
                            raise ValueError('TFSS poisoned image is malformed or nonfinite: '+stem)
                        poisoned_image_path = images_out/(stem+'.png')
                        _save_png(poisoned_image_path, poisoned_u8, cv2)
                        decoded = _decode_png(poisoned_image_path, cv2)
                        decoded_audit = validate_decoded_pair(clean_u8, decoded, support, audit_pixels)
                        if not np.array_equal(decoded, poisoned_u8):
                            raise ValueError('Lossless PNG round-trip changed exported pixels: '+stem)
                        weight_summary = dict(min=float(weight[support > 0].min()),
                                              mean=float(weight[support > 0].mean()),
                                              max=float(weight[support > 0].max()))
                        tfss_evidence = dict(mode='TFSS-U', eta=.85,
                                             map_version=TFSS_MAP_VERSION,
                                             map_evidence_sha256=hashlib.sha256(json.dumps(
                                                 map_records[0], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                                             weight=weight_summary,
                                             map_evidence=map_records[0])
                    else:
                        if np.any(support):
                            raise ValueError('Non-Person image unexpectedly received target support: '+stem)
                        poisoned_image_path = images_out/image_path.name
                        shutil.copyfile(image_path, poisoned_image_path)
                        if sha(poisoned_image_path) != expected[0]:
                            raise ValueError('Non-Person source image bytes changed: '+stem)
                        decoded = clean_u8
                        decoded_audit = validate_decoded_pair(clean_u8, decoded, support, audit_pixels)
                        tfss_evidence = None
                    poisoned_label_path = labels_out/(stem+'.txt')
                    with poisoned_label_path.open('xb') as target:
                        target.write(label_bytes)
                    if sha(poisoned_label_path) != label_sha:
                        raise ValueError('Exported label bytes changed: '+stem)
                    if is_person:
                        person_count += 1
                    else:
                        nonperson_count += 1
                    if not anns:
                        empty_label_count += 1
                    target_pixels = int(np.count_nonzero(support == 1.))
                    overlap = int(np.count_nonzero(support == .5))
                    changed = int(np.count_nonzero(np.any(decoded != clean_u8, axis=2)))
                    changed_pixels += changed; person_exclusive_pixels += target_pixels
                    overlap_pixels += overlap; outside_pixels += decoded_audit['outside_nonzero']
                    max_bytes = max(max_bytes, decoded_audit['max_bytes'])
                    overlap_max = max(overlap_max, decoded_audit['overlap_max_bytes'])
                    mse = float(np.mean((decoded.astype(np.float32)-clean_u8.astype(np.float32))**2))
                    psnr = float(10*np.log10((255.*255.)/mse)) if mse > 0 else None
                    row = dict(stem=stem, source_image=f'images/train/{stem}.jpg',
                               poisoned_image=f'poisoned/images/train/{poisoned_image_path.name}',
                               source_image_sha256=expected[0], poisoned_image_sha256=sha(poisoned_image_path),
                               source_label=f'labels/train/{stem}.txt', poisoned_label=f'poisoned/labels/train/{stem}.txt',
                               source_label_sha256=label_sha, poisoned_label_sha256=sha(poisoned_label_path),
                               annotation_count=len(anns), person_present=is_person,
                               support_person_only_pixels=target_pixels, support_overlap_pixels=overlap,
                               changed_pixels=changed, decoded_max_byte_difference=decoded_audit['max_bytes'],
                               decoded_overlap_max_byte_difference=decoded_audit['overlap_max_bytes'],
                               decoded_outside_nonzero_pixels=decoded_audit['outside_nonzero'],
                               PSNR_db=psnr, tfss=tfss_evidence)
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                    row_count += 1
                    if smoke_images and (len(anns) == 0 or target_pixels+overlap == 0):
                        raise ValueError('Bounded exporter smoke image must contain a real eligible Person support')
        finally:
            hook.remove()
        manifest_sha = sha(manifest_path)
        if smoke_images:
            if row_count != 1 or not 1 <= len(forward_calls) <= 100:
                raise ValueError('Export smoke must process one real eligible image within 100 forwards')
        else:
            if row_count != FIXED_COUNTS['images'] or person_count != 13630 or nonperson_count != 11370:
                raise ValueError('Poisoned COCO export image or Person-image counts differ from accepted data')
            if expected_digest != fourier.get('train_manifest_sha256'):
                raise ValueError('Export release train manifest identity differs from accepted Fourier24')
        if (model_hash(teacher) != frozen_sha or model_hash(trainer.adaptive.model) != adaptive_before
                or sha(params_path) != params_hash_before):
            raise ValueError('Export modified a fixed Teacher copy or learned Fourier parameters')
        if smoke_images:
            result = dict(common, state='passed', computation_kind='poisoned-export',
                          production_entrypoint_reached=True, finite_outputs=True,
                          decoded_budget_passed=True, labels_byte_identical=True,
                          actual_device='cuda:0', actual_training_images=1,
                          production_forward_calls=len(forward_calls), parameter_updates=False,
                          validation_used=False, official_metrics_disabled=True,
                          artifact_ingest_disabled=True, row_count=row_count,
                          manifest_sha256=manifest_sha, max_decoded_byte_difference=max_bytes,
                          max_overlap_byte_difference=overlap_max, outside_nonzero_pixels=outside_pixels,
                          cleanup_policy='retain_all', completed_UTC=__import__('coco_s4.runtime', fromlist=['now']).now())
            write_once(output/'smoke_summary.json', result)
            atomic(output/'status.json', result)
            return result
        accepted = dict(common, state='ACCEPTED', computation_kind='original-S4-poisoned-COCO25K',
                        production_entrypoint_reached=True, finite_outputs=True,
                        decoded_budget_passed=True, labels_byte_identical=True,
                        actual_device='cuda:0', image_count=row_count,
                        person_images=person_count, nonperson_images=nonperson_count,
                        empty_label_images=empty_label_count,
                        manifest_path=str(manifest_path), manifest_sha256=manifest_sha,
                        poisoned_image_root=str(images_out), poisoned_label_root=str(labels_out),
                        person_only_support_pixels=person_exclusive_pixels,
                        person_non_target_overlap_support_pixels=overlap_pixels,
                        changed_pixels=changed_pixels, max_decoded_byte_difference=max_bytes,
                        max_overlap_byte_difference=overlap_max, outside_nonzero_pixels=outside_pixels,
                        original_train_manifest_sha256=expected_digest,
                        Fourier_receipt_sha256=sha(fourier_path), calibration_receipt_sha256=sha(calibration_path),
                        teacher_checkpoint_sha256=sha(checkpoint), validation_used=False,
                        official_metrics_disabled=True, artifact_ingest_disabled=True,
                        cleanup_policy='retain_all', deleted_paths=[], completed_UTC=__import__('coco_s4.runtime', fromlist=['now']).now())
        write_once(output/'accepted.json', accepted)
        atomic(output/'status.json', accepted)
        return accepted
    except BaseException as exc:
        status.update(state='FAILED', error=repr(exc), ended_UTC=__import__('coco_s4.runtime', fromlist=['now']).now(),
                      cleanup_policy='retain_all', deleted_paths=[])
        atomic(output/'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--smoke-images', type=int, default=0)
    args = parser.parse_args()
    run(args.release, args.output, registration=args.registration, smoke_images=args.smoke_images)
