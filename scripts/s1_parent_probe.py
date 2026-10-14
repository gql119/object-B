"""Bounded production-core equivalence; old source is imported in a separate process.

No victim, no validation images, no mutation of the historical source/artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def worker(args):
    sys.path.insert(0, args.source_root)
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.data_utils import load_image_rgb_float, read_yolo_annotations, label_path_for_image
    from ue_framework.methods.b2_semantic import B2SemanticTrainer
    from ue_framework.methods.legacy_kproto_ret import _snapshot_metrics

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    mc = cfg['methods']['legacy_kproto_ret']
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    model = YOLO(cfg['surrogate']['ckpt']).model.to(device)
    learner = YOLO(cfg['surrogate']['ckpt']).model.to(device)
    trainer = B2SemanticTrainer(cfg, mc, device, model, learner)
    images = trainer._collect_target_images(str(Path(cfg['data']['dataset_root']) / cfg['data']['train_images']),
                                            str(Path(cfg['data']['dataset_root']) / cfg['data']['train_labels']))[:16]
    labels = Path(cfg['data']['dataset_root']) / cfg['data']['train_labels']
    items = []
    for path in images:
        im = load_image_rgb_float(path)
        ann = read_yolo_annotations(label_path_for_image(path, str(labels)))
        support, _, _ = trainer._build_support(im.shape, ann, image_path=path)
        items.append((im, ann, support))
    prototypes = torch.load(args.prototypes, map_location=device, weights_only=True)['prototypes']
    opt = torch.optim.Adam([{'params': [trainer.fourier_coeff], 'lr': trainer.universal_lr_fourier}])
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    records, arrays = [], {}
    for step in range(5):
        trainer._epoch = 0
        trainer._active_idx = trainer._get_active_freq_indices(0, step)
        trainer._active_coords = [tuple(trainer.freq_candidate_coords[int(i)]) for i in trainer._active_idx.cpu().tolist()]
        opt.zero_grad(set_to_none=True)
        batch_metrics = []
        for eot in range(trainer.eot_samples):
            loss, metrics = trainer._batch_loss(items, prototypes)
            if not torch.isfinite(loss):
                raise FloatingPointError('Parent loss nonfinite')
            terms = trainer._semantic_terms
            target = terms['BG_sel'] + terms['rel_sel'] + trainer.lambda_adapt * metrics['L_adapt_target']
            utility = terms['ret'] + terms['orth'] + trainer.lambda_adapt * trainer.lambda_nt * metrics['L_adapt_nt']
            for name, term in [('target', target), ('utility', utility)]:
                g = torch.autograd.grad(term, trainer.fourier_coeff, retain_graph=True)[0]
                arrays[f'{step}.{eot}.g_{name}'] = g.detach().cpu()
            arrays[f'{step}.{eot}.poison'] = trainer._last_poison.detach().cpu()
            loss.backward()
            batch_metrics.append(_snapshot_metrics(metrics))
        trainer.fourier_coeff.grad.div_(len(batch_metrics))
        arrays[f'{step}.grad_total'] = trainer.fourier_coeff.grad.detach().cpu()
        opt.step()
        arrays[f'{step}.coeff'] = trainer.fourier_coeff.detach().cpu().clone()
        records.append(batch_metrics)
        print(json.dumps({'state': 'running', 'step': step + 1, 'loss': records[-1][-1]['L_total']}), flush=True)
    torch.save(arrays, out / 'trajectory.pt')
    source = Path(args.source_root)
    paths = ['ue_framework/methods/b2_semantic.py', 'ue_framework/methods/b2_adaptive.py',
             'ue_framework/methods/adaptive_learner.py', 'ue_framework/methods/legacy_kproto_ret.py',
             'ue_framework/methods/tausb_universal.py']
    (out / 'receipt.json').write_text(json.dumps({'records': records, 'ids': [Path(p).stem for p in images],
        'source_root': str(source), 'source_sha256': {p: sha(source / p) for p in paths},
        'reference_checkpoint_sha256': sha(cfg['surrogate']['ckpt']), 'prototypes_sha256': sha(args.prototypes),
        'config_sha256': sha(args.config), 'production_core': 'B2SemanticTrainer._batch_loss',
        'steps': 5, 'eot_samples': trainer.eot_samples}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source-root')
    p.add_argument('--config', required=True)
    p.add_argument('--prototypes', required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    if a.worker:
        worker(a); return
    import torch
    a.config = str(Path(a.config).resolve())
    a.prototypes = str(Path(a.prototypes).resolve())
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / 'runtime'
    for name in ['yolo', 'mpl', 'cache']:
        (runtime / name).mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               YOLO_CONFIG_DIR=str(runtime / 'yolo'), MPLCONFIGDIR=str(runtime / 'mpl'),
               XDG_CACHE_HOME=str(runtime / 'cache'), WANDB_MODE='disabled')
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError('GPU occupied; no parallel launch authorized for this S1 probe')
    start = time.perf_counter()
    for label, root in [('historical', a.parent_root), ('restored', str(Path(__file__).resolve().parents[1]))]:
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--source-root', root,
               '--parent-root', a.parent_root, '--config', a.config, '--prototypes', a.prototypes,
               '--output', str(out / label)]
        subprocess.run(cmd, check=True, env=env, cwd=str(runtime.parent))
    r = torch.load(out / 'historical/trajectory.pt', map_location='cpu', weights_only=True)
    c = torch.load(out / 'restored/trajectory.pt', map_location='cpu', weights_only=True)
    diffs = {k: float((r[k] - c[k]).abs().max()) for k in r}
    refs = [json.loads((out / x / 'receipt.json').read_text()) for x in ['historical', 'restored']]
    loss_diff = max(abs(x[k] - y[k]) for rr, cc in zip(refs[0]['records'], refs[1]['records'])
                    for x, y in zip(rr, cc) for k in x if isinstance(x[k], (float, int)))
    tolerance = 1e-6
    passed = max(diffs.values()) <= tolerance and loss_diff <= tolerance and refs[0]['ids'] == refs[1]['ids']
    summary = {'state': 'complete' if passed else 'failed', 'passed': passed, 'steps': 5,
        'finite_loss': True, 'production_entrypoint_reached': True,
        'official_metrics_disabled': True, 'artifact_ingest_disabled': True,
        'cleanup_policy': 'retain_all', 'checkpoint_cleanup_completed': False, 'deleted_paths': [],
        'max_loss_difference': loss_diff, 'max_tensor_difference': max(diffs.values()),
        'differences': diffs, 'tolerance': tolerance, 'receipts': refs,
        'duration_seconds': time.perf_counter() - start,
        'checkpoint_paths_remaining': [str(out / x / 'trajectory.pt') for x in ['historical', 'restored']]}
    (out / 'smoke_summary.json').write_text(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError('Historical parent numerical equivalence failed')


if __name__ == '__main__':
    main()
