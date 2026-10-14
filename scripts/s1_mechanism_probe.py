"""Fixed train 5-step parent/CSB/uniform controls plus actual checkpoint replay.

Calls the original training loop as reference and the resumable S1 production
loop as candidate. Not a victim, and never reads clean validation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def worker(a):
    source = a.parent_root if a.mode == 'native' else str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, source)
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.data_utils import load_image_rgb_float, read_yolo_annotations, label_path_for_image
    from ue_framework.methods.b2_semantic import B2SemanticTrainer
    from ue_framework.methods.legacy_kproto_ret import LegacyKProtoGenerator
    cfg = load_config(a.config); mc = cfg['methods']['legacy_kproto_ret']
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    model = YOLO(cfg['surrogate']['ckpt']).model.to(device)
    learner = YOLO(cfg['surrogate']['ckpt']).model.to(device)
    prototypes = torch.load(a.prototypes, map_location=device, weights_only=True)['prototypes']
    root = Path(cfg['data']['dataset_root'])
    selection = json.loads((Path(a.cache) / 'selection.json').read_text())
    images = [str(root / 'images/train' / x) for x in selection['filenames']]
    label_dir = root / 'labels/train'
    output = Path(a.output)
    if a.mode == 'native':
        output.mkdir(parents=True, exist_ok=False)
        trainer = B2SemanticTrainer(cfg, mc, device, model, learner)
        trainer.max_steps = 5
        trainer._collect_target_images = lambda *_: images
        trainer._build_bank = lambda *_: prototypes
        trainer.train_universal(str(root / 'images/train'), str(label_dir), str(output / 'global_params.pt'),
                                str(output / 'diagnostics.csv'), str(output / 'diagnostics.json'), 0)
        torch.save({'coeff': trainer.fourier_coeff.detach().cpu(), 'suppress': trainer.suppress_small.detach().cpu()},
                   output / 'final_tensors.pt')
        (output / 'receipt.json').write_text(json.dumps({'state': 'complete', 'source_root': source,
            'steps': 5, 'production_entrypoint': 'B2SemanticTrainer.train_universal (original parent loop)',
            'fixed_prototypes_read_only': a.prototypes, 'ids': selection['ids']}))
        return
    from ue_framework.methods.csb_budget import BudgetStore, digest, item_key, png_project, domains
    from ue_framework.methods.s1_csb import S1CSBTrainer, S1CSBGenerator
    from ue_framework.methods.s1_optimization import optimize
    manifest = json.loads((Path(a.cache) / 'manifest.json').read_text())
    store = BudgetStore(a.cache, manifest['identity'])
    trainer = S1CSBTrainer(cfg, mc, device, model, learner, budget_store=store, budget_mode=a.mode)
    trainer.max_steps = 5  # diagnostics only; optimize stop_after controls the bounded step count
    records = []
    def observe(t, items, loss, values, step, eot):
        g = torch.autograd.grad(loss, t._last_poison, retain_graph=True)[0]
        projected = torch.autograd.grad(loss, t._projection_base, retain_graph=True)[0]
        coeff = torch.autograd.grad(loss, t.fourier_coeff, retain_graph=True)[0]
        if not torch.isfinite(g).all() or not torch.isfinite(projected).all():
            raise FloatingPointError('Projection-gradient diagnostic nonfinite')
        records.append({'step': step, 'eot': eot, 'L_BG': float(values['L_BG'].detach()),
            'L_rel': float(values['L_rel'].detach()), 'L_ret': float(values['L_ret'].detach()),
            'L_orth': float(values['L_orth_objective'].detach()), 'L_total': float(loss.detach()),
            'raw_poison_gradient_norm': float(g.norm()), 'projected_base_gradient_norm': float(projected.norm()),
            'Fourier_gradient_norm': float(coeff.norm()), 'person_clean_after_proxy': float(values['target_score_clean_after'].detach()),
            'non_target_feature_drift': float(values['non_target_feature_drift'].detach()),
            'ret_gradient_norm': float(values['grad_ret_norm'].detach()), 'attack_gradient_norm': float(values['grad_attack_norm'].detach()),
            'mean_abs_delta': float(values['mean_abs_delta'].detach()), 'max_abs_delta': float(values['max_abs_delta'].detach())})
    identity = dict(parent_config_sha256=digest(a.config), cache_manifest_sha256=digest(Path(a.cache) / 'manifest.json'),
                    budget_mode=a.mode, optimization_source_sha256=digest(Path(__file__).resolve().parents[1] / 'ue_framework/methods/s1_optimization.py'))
    started = time.perf_counter()
    status = optimize(trainer, images, label_dir, prototypes, output, identity,
                      stop_after=a.stop_after, resume=a.resume, observe=observe)
    torch.save({'coeff': trainer.fourier_coeff.detach().cpu(), 'suppress': trainer.suppress_small.detach().cpu()},
               output / 'final_tensors.pt')
    # Bound export to the same current active coordinates the parent saves at step5.
    idx = trainer._active_idx
    params = dict(coords=[list(trainer.freq_candidate_coords[int(i)]) for i in idx.cpu().tolist()],
        fourier_coeff=trainer.fourier_coeff.detach()[idx].cpu(), suppress_small=trainer.suppress_small.detach().cpu(),
        method='legacy_kproto_ret', support_mode=trainer.support_mode, target_class_id=14, eps=trainer.eps,
        feature_layer=trainer.feature_layer)
    torch.save(params, output / 'global_params.pt')
    generator = S1CSBGenerator(cfg, mc, device, model, str(output / 'global_params.pt'),
                               budget_store=store, budget_mode=a.mode)
    exports = []
    for path in images[:8]:
        image = load_image_rgb_float(path)
        anns = read_yolo_annotations(label_path_for_image(path, str(label_dir)))
        budget = store.get(image, anns)
        target, other = domains(image.shape, anns)
        if a.mode == 'parent_fixed': budget = np.where(target & other, 4 / 255, np.where(target, 8 / 255, 0)).astype(np.float32)
        if a.mode == 'uniform_matched': budget = np.where(target, budget[target].mean(), 0).astype(np.float32)
        result = generator.generate(image, anns, 0, 24, trainer.eps, 'mask', image_path=path)
        clean = np.rint(image * 255).astype(np.uint8)
        poison = png_project(clean, result.poisoned_image, budget)
        diff = np.abs(poison.astype(np.int16) - clean.astype(np.int16)).max(2)
        if np.any(diff[~target] != 0) or np.any(diff / 255 > budget + 1e-7):
            raise ValueError('PNG spatial budget violation')
        with np.load(Path(a.cache) / 'maps' / (Path(path).stem + '.npz')) as maps: norm = maps['normalized']
        exports.append({'id': Path(path).stem, 'max_abs_delta_byte': int(diff.max()),
            'mean_abs_delta_byte_person': float(diff[target].mean()), 'changed_person_area_ratio': float((diff[target] > 0).mean()),
            'outside_max': int(diff[~target].max()) if (~target).any() else 0,
            'high_sensitivity_delta_byte': float(diff[target & (norm >= .9)].mean()) if np.any(target & (norm >= .9)) else None,
            'low_sensitivity_delta_byte': float(diff[target & (norm <= .1)].mean()) if np.any(target & (norm <= .1)) else None})
    (output / 'receipt.json').write_text(json.dumps({'state': 'complete', 'source_root': source,
        'production_entrypoint': 's1_optimization.optimize + S1CSBGenerator.generate', 'mode': a.mode,
        'status': status, 'diagnostics': records, 'export_audit': exports,
        'duration_seconds': time.perf_counter() - started, 'optimizer_steps': status['steps'],
        'checkpoint': status['checkpoint'], 'ids': selection['ids']}, indent=2))


def compare_tree(a, b):
    import numpy as np
    import torch
    if torch.is_tensor(a):
        return float((a.cpu().to(torch.float64) - b.cpu().to(torch.float64)).abs().max()) if a.numel() else 0.
    if isinstance(a, np.ndarray): return float(np.abs(a.astype(float) - b.astype(float)).max()) if a.size else 0.
    if isinstance(a, dict):
        if set(a) != set(b): raise ValueError('Replay dictionary keys differ')
        return max([compare_tree(a[k], b[k]) for k in a] or [0.])
    if isinstance(a, (tuple, list)):
        if len(a) != len(b): raise ValueError('Replay list length differs')
        return max([compare_tree(x, y) for x, y in zip(a, b)] or [0.])
    if a != b: raise ValueError(f'Replay scalar differs: {a} / {b}')
    return 0.


def main():
    p = argparse.ArgumentParser()
    for name in ['parent-root', 'config', 'prototypes', 'cache', 'output']: p.add_argument('--' + name, required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--mode', choices=['native', 'parent_fixed', 'csb', 'uniform_matched'])
    p.add_argument('--stop-after', type=int, default=5)
    p.add_argument('--resume')
    a = p.parse_args()
    if a.worker: worker(a); return
    import torch
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    for name in ['yolo', 'mpl', 'cache']: (out / 'runtime' / name).mkdir(parents=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        YOLO_CONFIG_DIR=str(out / 'runtime/yolo'), MPLCONFIGDIR=str(out / 'runtime/mpl'),
        XDG_CACHE_HOME=str(out / 'runtime/cache'), WANDB_MODE='disabled')
    for name in ['config', 'prototypes', 'cache']: setattr(a, name, str(Path(getattr(a, name)).resolve()))
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy: raise RuntimeError('GPU occupied; no concurrent S1 probe')
    def launch(mode, suffix, limit=5, resume=None):
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--parent-root', a.parent_root,
            '--config', a.config, '--prototypes', a.prototypes, '--cache', a.cache, '--mode', mode,
            '--output', str(out / suffix), '--stop-after', str(limit)]
        if resume: cmd += ['--resume', resume]
        subprocess.run(cmd, check=True, env=env, cwd=str(out))
    for mode in ['native', 'parent_fixed', 'csb', 'uniform_matched']: launch(mode, mode)
    launch('csb', 'interrupted', limit=3)
    interrupted = json.loads((out / 'interrupted/receipt.json').read_text())
    launch('csb', 'resumed', resume=interrupted['checkpoint'])
    native = torch.load(out / 'native/final_tensors.pt', map_location='cpu', weights_only=True)
    fixed = torch.load(out / 'parent_fixed/final_tensors.pt', map_location='cpu', weights_only=True)
    equivalence = compare_tree(native, fixed)
    receipts = {x: json.loads((out / x / 'receipt.json').read_text()) for x in ['parent_fixed', 'csb', 'uniform_matched', 'resumed']}
    full = torch.load(receipts['csb']['checkpoint'], map_location='cpu', weights_only=False)
    resumed = torch.load(receipts['resumed']['checkpoint'], map_location='cpu', weights_only=False)
    replay = compare_tree(full, resumed)
    with (out / 'native/diagnostics.csv').open() as f: native_rows = list(csv.DictReader(f))
    with (out / 'parent_fixed/diagnostics.csv').open() as f: fixed_rows = list(csv.DictReader(f))
    loss_diff = max(abs(float(x[k]) - float(y[k])) for x, y in zip(native_rows, fixed_rows)
        for k in x if k.startswith(('L_', 'weighted_', 'grad_')))
    passed = equivalence <= 1e-6 and loss_diff <= 1e-6 and replay == 0
    summary = dict(state='complete' if passed else 'failed', passed=passed, finite_loss=True,
        production_entrypoint_reached=True, official_metrics_disabled=True, artifact_ingest_disabled=True,
        formal_generation_enabled=False, victim_training_enabled=False, cleanup_policy='retain_all',
        checkpoint_cleanup_completed=False, deleted_paths=[], checkpoint_paths_remaining=[v['checkpoint'] for v in receipts.values()],
        original_loop_fixed_budget_max_difference=equivalence, original_loop_loss_max_difference=loss_diff,
        csb_checkpoint_rng_optimizer_cursor_replay_max_difference=replay, steps_per_arm=5, receipts=receipts)
    (out / 'smoke_summary.json').write_text(json.dumps(summary, indent=2))
    if not passed: raise ValueError('S1 numerical baseline/resume gate failed')


if __name__ == '__main__': main()
