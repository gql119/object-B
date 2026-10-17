"""Bounded real S0 Fourier/Adaptive probe. No dataset export/victim entrypoint.

NOT_RUN until independent resources/S3 acceptance, source-equivalence and the
single new TFSS scientific review gates pass. Run via configured rrctl only.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def resource_gate(gpu, s3_control, s3_output, acceptance):
    """Require an empty physical GPU; never allow a shared-S3 exception."""
    binding = json.loads((Path(s3_control)/'binding.json').read_text())
    stage = json.loads((Path(s3_output)/'pipeline_status.json').read_text())
    all_gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                        '--format=csv,noheader,nounits'], text=True)
    devices = {row.split(',')[0].strip(): row.split(',') for row in all_gpus.splitlines()}
    if str(gpu) not in devices:
        raise ValueError('Unknown physical GPU')
    selected = devices[str(gpu)][1].strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                          '--format=csv,noheader,nounits'], text=True)
    rows = [r.split(',') for r in processes.splitlines() if r.strip()]
    if any(r[0].strip() == selected for r in rows):
        raise ValueError('WAIT_FOR_S3_OR_OTHER_USER: selected GPU occupied')
    s3_stat = Path('/proc')/str(binding['workload_pid'])/'stat'
    s3_alive = s3_stat.exists() and int(s3_stat.read_text().split()[21]) == binding['workload_start_ticks']
    if s3_alive:
        s3_rows = [r for r in rows if int(r[1]) == binding['workload_pid']]
        if not s3_rows or any(r[0].strip() == selected for r in s3_rows):
            raise ValueError('Unknown/non-isolated S3 device; do not assume free')
    else:
        if stage.get('state') != 'complete' or not acceptance:
            raise ValueError('S3 absent is not full completion/acceptance')
        a = json.loads(Path(acceptance).read_text())
        if a.get('run_id') != binding['run_id'] or a.get('accepted') is not True:
            raise ValueError('Missing exact-run full S3 archive acceptance')
    if float(devices[str(gpu)][2]) < 24000:
        raise ValueError('Insufficient dedicated free GPU memory')
    if hasattr(os, 'getloadavg') and os.getloadavg()[0] > os.cpu_count()*.75:
        raise ValueError('Shared CPU pressure; preserve S3 resource priority')
    return {'physical_gpu': gpu, 'uuid': selected, 's3_stage': stage, 's3_alive': s3_alive}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', default=str(ROOT/'configs/tfss_short_probe.json'))
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', required=True, type=int)
    p.add_argument('--s3-control', required=True)
    p.add_argument('--s3-output', required=True)
    p.add_argument('--s3-acceptance')
    p.add_argument('--phase', choices=['smoke', 'short'], required=True)
    p.add_argument('--review-receipt')
    a = p.parse_args()
    conf = json.loads(Path(a.protocol).read_text())
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Candidate must be committed and clean')
    if a.phase == 'short':
        if not a.review_receipt:
            raise ValueError('Single scientific review/closure receipt required after current-source GPU smoke')
        receipt = json.loads(Path(a.review_receipt).read_text())
        if receipt.get('source_commit') != source or receipt.get('pre_run_result') != 'pass':
            raise ValueError('Review receipt does not admit current scientific source')
    resource = resource_gate(a.gpu, a.s3_control, a.s3_output, a.s3_acceptance)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    # Paired higher-order CUDA probes must replay the same numerical kernels as
    # well as RNG draws. Apply identically to canonical reference and candidate;
    # this does not change their losses, inputs, updates, or numerical tolerance.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    out = Path(a.output).resolve()
    allowed = Path(conf['remote_output_boundary']).resolve()
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError('Output outside isolated TFSS boundary')
    out.mkdir(parents=True, exist_ok=False)
    for key in ['YOLO_CONFIG_DIR', 'TORCH_HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR']:
        os.environ[key] = str(out/'runtime'/key)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import ultralytics
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.methods.b2_semantic import B2SemanticTrainer
    from ue_framework.methods.legacy_kproto_ret import _snapshot_metrics
    from ue_framework.data_utils import load_image_rgb_float, read_yolo_annotations
    from ue_framework.methods.tfss_spatial import SoftSpatialProjection, spatial_weights, realized_rms
    from ue_framework.methods.tfss_clean_maps import build_maps
    from ue_framework.methods.tfss_probe_metrics import collect
    from ue_framework.methods.tfss_nt_sensitivity import gradients as nt_gradients
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if ultralytics.__version__ != '8.4.150':
        raise ValueError('Pinned actual Parent raw-head version required; no environment changes')
    meta = json.loads((ROOT/'docs/tfss/SOURCE_METADATA.json').read_text())
    identity = json.loads((ROOT/'docs/tfss/S0_CANONICAL_SOURCE_SHA256.json').read_text())
    for rel, expected in identity.items():
        if sha(ROOT/rel) != expected or sha(Path(meta['parent_root'])/rel) != expected:
            raise ValueError('Restored or independent historical S0 bytes changed: '+rel)
    manifest_path = ROOT/'docs/tfss/FIXED_TRAIN_SAMPLE_MANIFEST.json'
    if sha(manifest_path) != conf['manifest_sha256']:
        raise ValueError('Fixed train selection changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest['split'] != 'train' or manifest['count'] != 32:
        raise ValueError('Not the preregistered training-only32')
    cfg_path = ROOT/'ue_framework/configs/exp_voc_person_b2_to_safe_soft05_eps8_seed0_20261004.yaml'
    if sha(cfg_path) != meta['config_sha256']:
        raise ValueError('S0 YAML changed')
    cfg = load_config(str(cfg_path));mc = cfg['methods']['legacy_kproto_ret']
    checkpoint = Path(cfg['surrogate']['ckpt'])
    if not checkpoint.is_file() or sha(checkpoint) != meta['channel_identity']['checkpoint_sha256']:
        raise ValueError('Missing or changed original reference; never download replacement')
    proto_path = Path(meta['parent_root'])/'noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt'
    if sha(proto_path) != conf['prototype_sha256']:
        raise ValueError('Original prototype identity changed')
    device = torch.device('cuda:0')
    prototypes = torch.load(proto_path, map_location=device, weights_only=True)['prototypes']

    def seed():
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)

    def rng():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())

    def restore(r):
        random.setstate(r[0]);np.random.set_state(r[1]);torch.set_rng_state(r[2]);torch.cuda.set_rng_state_all(r[3])

    def state_hash(model):
        h = hashlib.sha256()
        for k, v in model.state_dict().items():
            h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    class Candidate(SoftSpatialProjection, B2SemanticTrainer):
        pass

    class ObservedParent(B2SemanticTrainer):
        def _compose_delta_batched(self, image, support, ring, *args, **kwargs):
            result = super()._compose_delta_batched(image, support, ring, *args, **kwargs)
            self.tfss_input = image
            self.tfss_support = support
            self.tfss_delta = result[1]
            self.tfss_s0_delta = result[1]
            self.tfss_applied_weight = torch.ones_like(support)
            self.tfss_matching_errors = []
            return result  # Instrumentation only: actual canonical tuple unchanged.

    def create(klass):
        seed()
        frozen = YOLO(str(checkpoint)).model.to(device)
        adaptive = YOLO(str(checkpoint)).model.to(device)
        trainer = klass(copy.deepcopy(cfg), copy.deepcopy(mc), device, frozen, adaptive)
        if trainer.channels.tolist() != meta['selected_channels'] or trainer.feature_layer != 'model.21':
            raise ValueError('Actual ordered24 TSFL channels changed')
        if (trainer.universal_batch_size, trainer.eot_samples, trainer.safe_bg_rank,
            trainer.safe_bg_removal_alpha, trainer.safe_bg_active_k) != (16, 2, 8, .5, 3):
            raise ValueError('Parent batch/EOT/SafeBG configuration changed')
        return trainer, frozen, adaptive

    steps = 1 if a.phase == 'smoke' else 5
    batches = [manifest['records'][:16]] if a.phase == 'smoke' else [manifest['records'][:16], manifest['records'][16:]]
    evidence = {'state': 'running', 'source_commit': source, 'phase': a.phase, 'resource': resource,
                'official_metrics_disabled': True, 'artifact_ingest_disabled': True,
                'cleanup_policy': 'retain_all', 'deleted_paths': [], 'trials': [],
                'started_at': time.time(), 'protocol_sha256': sha(a.protocol),
                'environment': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                    'cudnn_deterministic': torch.backends.cudnn.deterministic,
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']}}
    write(out/'status.json', evidence)
    for batch_index, records in enumerate(batches):
        setup, model, adaptive = create(B2SemanticTrainer)
        items = []
        for record in records:
            if sha(record['image_path']) != record['image_sha256'] or sha(record['label_path']) != record['label_sha256']:
                raise ValueError('Fixed train pixels/labels changed')
            image = load_image_rgb_float(record['image_path'])
            annotations = read_yolo_annotations(record['label_path'])
            support, _, _ = setup._build_support(image.shape, annotations, image_path=record['image_path'])
            items.append((image, annotations, support))
        initial_rng = rng()
        fixed_maps, matching = build_maps(setup, items)
        restore(initial_rng)
        write(out/f'maps_batch{batch_index}.json', matching)
        np.savez_compressed(out/f'maps_batch{batch_index}.npz',
            **{f'{name}_{i}': value.detach().cpu().numpy()
               for i, values in enumerate(fixed_maps)
               for name, value in zip(['q', 'nt_risk', 'q_valid'], values)})
        del setup, model, adaptive
        traces = {}
        for method in ['S0-reference', 'S0-noop', 'TFSS-C', 'TFSS-U', 'Uniform-C', 'Uniform-U']:
            trainer, frozen, adaptive = create(ObservedParent if method == 'S0-reference' else Candidate)
            original_hashes = [state_hash(frozen), state_hash(adaptive)]
            trainer.tfss_mode = method if method in ['TFSS-C', 'TFSS-U', 'Uniform-C', 'Uniform-U'] else 'S0'
            # This is the ORIGINAL optimizer; suppress_small is not added to it.
            optimizer = torch.optim.Adam([{'params': [trainer.fourier_coeff], 'lr': trainer.universal_lr_fourier}])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trainer.universal_epochs, eta_min=1e-5)
            trial = out/f'batch{batch_index}_{method}';trial.mkdir()
            rows, targets, gradients, coeffs = [], [], [], []
            restore(initial_rng)
            for step in range(steps):
                trainer._epoch = 0
                optimizer.zero_grad(set_to_none=True)
                trainer._active_idx = trainer._get_active_freq_indices(0, step)
                trainer._active_coords = [tuple(trainer.freq_candidate_coords[int(i)]) for i in trainer._active_idx.cpu().tolist()]
                for eot in range(2):
                    if method in ['TFSS-C', 'TFSS-U']:
                        ph = (max(im.shape[0] for im, _, _ in items)+31)//32*32
                        pw = (max(im.shape[1] for im, _, _ in items)+31)//32*32
                        weights = []
                        for (image, _, support), (q, risk, valid_q_domain) in zip(items, fixed_maps):
                            q = q.to(device);risk = risk.to(device)
                            w = spatial_weights(q, risk, torch.as_tensor(support, device=device), mode=method, eta=.85)
                            weights.append(F.pad(w, (0, pw-w.shape[1], 0, ph-w.shape[0]), value=1))
                        trainer.tfss_weight = torch.stack(weights)[:, None]
                    if method.startswith('Uniform-'):
                        trainer.tfss_uniform_targets = traces['TFSS-'+method[-1]]['targets'][step*2+eot]
                    result = trainer._batch_loss(items, prototypes)
                    if result is None:
                        raise ValueError('No real Parent supervision; do not skip')
                    loss, metrics = result
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite loss')
                    row = _snapshot_metrics(metrics)
                    row.update(step=step+1, eot=eot+1, batch=batch_index, method=method)
                    row['fixed_image_ids'] = [r['image_id'] for r in records]
                    row['adaptive_selected_image_id'] = records[(trainer._adaptive_calls-1) % len(records)]['image_id']
                    ref_path = out/f'reference_batch{batch_index}_step{step+1}_eot{eot+1}.pt'
                    row['per_image'] = collect(trainer, metrics, items, fixed_maps,
                        mode='S0' if method == 'S0-reference' else method,
                        reference_path=ref_path, output_prefix=trial/f'step{step+1}_eot{eot+1}')
                    if method != 'S0-reference':
                        actual = (trainer.tfss_input+trainer.tfss_delta).clamp(0, 1)-trainer.tfss_input
                        support = trainer.tfss_support
                        if float(actual.masked_select((support == 0).expand_as(actual)).abs().max()) != 0:
                            raise ValueError('Outside Person perturbation')
                        if bool((actual.abs() > support*trainer.eps+1e-7).any()):
                            raise ValueError('Original8/4/0 decoded-float budget violation')
                        target_rms = [float(realized_rms(x, d, m)) for x, d, m in zip(trainer.tfss_input, trainer.tfss_delta, support)]
                        targets.append(target_rms)
                        row['actual_person_rms'] = target_rms
                        row['uniform_match_errors'] = trainer.tfss_matching_errors
                        row['weight_quantiles'] = torch.quantile(trainer.tfss_applied_weight[support > 0], torch.tensor([0., .1, .25, .5, .75, .9, 1.], device=device)).tolist()
                    if eot == 1 and step == steps-1:
                        # Derived noise/weight maps only; no original RGB export.
                        np.savez_compressed(trial/'final_recorded_derived_maps.npz',
                            delta=trainer.tfss_delta.detach().cpu().numpy(),
                            actual_delta=((trainer.tfss_input+trainer.tfss_delta).clamp(0, 1)-trainer.tfss_input).detach().cpu().numpy(),
                            weight=trainer.tfss_applied_weight.detach().cpu().numpy(),
                            support=trainer.tfss_support.detach().cpu().numpy())
                    loss.backward()
                    rows.append(row)
                trainer.fourier_coeff.grad.div_(2)
                if not torch.isfinite(trainer.fourier_coeff.grad).all() or trainer.fourier_coeff.grad.norm() <= 0:
                    raise FloatingPointError('Missing/nonfinite real Fourier meta-gradient')
                gradients.append(trainer.fourier_coeff.grad.detach().cpu().clone())
                if trainer.enable_adaptive_freq_basis:
                    with torch.no_grad():
                        trainer.freq_score[trainer._active_idx] += trainer.fourier_coeff.grad[trainer._active_idx].abs().mean(1)
                        trainer.freq_usage[trainer._active_idx] += 1
                optimizer.step()
                coeffs.append(trainer.fourier_coeff.detach().cpu().clone())
                torch.save({'fourier_coeff': trainer.fourier_coeff, 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'rng': rng(), 'step': step+1,
                            'freq_score': trainer.freq_score, 'freq_usage': trainer.freq_usage,
                            'adaptive_calls': trainer._adaptive_calls, 'semantic_calls': trainer._semantic_calls}, trial/f'step{step+1}.pt')
                write(trial/'trace.json', rows)
            if original_hashes != [state_hash(frozen), state_hash(adaptive)]:
                raise ValueError('Frozen or Adaptive base state mutated')
            # Fixed GT-conditioned detection diagnostics of the final RECORDED
            # pre-update forward, not a new loss or post-update/victim result.
            nt_rows = []
            last_poison = trainer._last_poison.detach()
            for index, (image, annotations, support) in enumerate(items):
                h, w = image.shape[:2]
                poison_image = last_poison[index, :, :h, :w].permute(1, 2, 0).cpu().numpy()
                _, clean_records = nt_gradients(frozen, image, annotations, imgsz=trainer.imgsz,
                    coverage_policy='independent_same_TAL_for_uncovered_valid_GT')
                _, poison_records = nt_gradients(frozen, poison_image, annotations, imgsz=trainer.imgsz,
                    coverage_policy='independent_same_TAL_for_uncovered_valid_GT')
                nt_rows.append({'image_id': records[index]['image_id'], 'clean': clean_records,
                                'poison': poison_records, 'forward': 'last recorded pre-update',
                                'assignment': 'native allGT, uncovered sameTAL alone; assignments may change with image'})
            write(trial/'nt_gt_diagnostics.json', nt_rows)
            if original_hashes != [state_hash(frozen), state_hash(adaptive)]:
                raise ValueError('Diagnostic changed model base state')
            traces[method] = {'rows': rows, 'targets': targets, 'gradients': gradients, 'coeffs': coeffs}
            if method == 'S0-noop':
                ref = traces['S0-reference']
                diff = max(float((x-y).abs().max()) for key in ['gradients', 'coeffs'] for x, y in zip(ref[key], traces[method][key]))
                for key in ['L_BG', 'L_rel', 'L_ret', 'L_orth_objective', 'L_total']:
                    diff = max(diff, max(abs(x[key]-y[key]) for x, y in zip(ref['rows'], rows)))
                write(trial/'equivalence.json', {'max_difference': diff, 'tolerance': 1e-7,
                      'reference': 'unchanged canonical B2SemanticTrainer, byte-bound historical S0',
                      'candidate': 'SoftSpatialProjection(S0 no-op)+same canonical trainer'})
                if diff > 1e-7:
                    raise ValueError('S0 restored/no-op production equivalence failed')
            evidence['trials'].append({'batch': batch_index, 'method': method, 'steps': steps, 'state': 'complete',
                                      'frozen_adaptive_before_after': original_hashes})
            write(out/'status.json', evidence)
            del trainer, frozen, adaptive, optimizer, scheduler
    # Analysis/report stage consumes paired sample/gradient/NT-GT evidence.
    # A successful smoke never becomes an efficacy or candidate recommendation.
    evidence.update(state='complete', finite_loss=True, finite_fourier_gradients=True,
                    production_entrypoint_reached=True, candidate_decision='INSUFFICIENT_EVIDENCE',
                    diagnostics_status='RAW_DIAGNOSTICS_READY_FOR_ANALYSIS', ended_at=time.time())
    write(out/'status.json', evidence)


if __name__ == '__main__':
    main()
