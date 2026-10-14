"""Frozen train-only cohort/cache audit. No noise optimization or victim."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode = True


def fixed_selection(root, count):
    """Geometry-only categories selected before observing any sensitivity/result."""
    from PIL import Image
    from ue_framework.data_utils import list_images, read_yolo_annotations, label_path_for_image
    from ue_framework.methods.csb_budget import domains
    paths = list_images(str(root / 'images/train'))
    eligible = []
    categories = {k: [] for k in ['Person_only', 'Person_Dog', 'Person_Bicycle', 'Person_Horse',
                                 'high_overlap', 'adjacent_no_overlap', 'small_Person', 'large_Person']}
    for path in paths:
        annotations = read_yolo_annotations(label_path_for_image(path, str(root / 'labels/train')))
        persons = [a for a in annotations if int(a['cls']) == 14]
        if not persons:
            continue
        eligible.append(path)
        other = [a for a in annotations if int(a['cls']) != 14]
        key = hashlib.sha256(('s1_train_dev_seed0:' + Path(path).stem).encode()).hexdigest()
        def add(k): categories[k].append((key, path))
        if not other: add('Person_only')
        for cls, label in [(11, 'Person_Dog'), (1, 'Person_Bicycle'), (12, 'Person_Horse')]:
            if any(int(a['cls']) == cls for a in other): add(label)
        areas = [a['bbox'][2] * a['bbox'][3] for a in persons]
        if min(areas) <= .02: add('small_Person')
        if max(areas) >= .3: add('large_Person')
        if other:
            with Image.open(path) as im: w, h = im.size
            target, nt = domains((h, w), annotations)
            overlap = float((target & nt).sum()) / max(1, int(target.sum()))
            if overlap >= .3: add('high_overlap')
            if overlap == 0:
                def xyxy(a):
                    cx, cy, bw, bh = a['bbox']
                    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                gaps = []
                for a in persons:
                    ax1, ay1, ax2, ay2 = xyxy(a)
                    for b in other:
                        bx1, by1, bx2, by2 = xyxy(b)
                        gaps.append(max(0, bx1 - ax2, ax1 - bx2) + max(0, by1 - ay2, ay1 - by2))
                if min(gaps) <= .05: add('adjacent_no_overlap')
    missing = [k for k, v in categories.items() if not v]
    if missing:
        raise ValueError(f'Cannot form required train geometry cohort: {missing}')
    selected, coverage = [], {}
    for k, v in categories.items():
        path = sorted(v)[0][1]
        coverage[k] = Path(path).stem
        if path not in selected: selected.append(path)
    pool = sorted(eligible, key=lambda x: hashlib.sha256(('s1_train_dev_seed0:' + Path(x).stem).encode()).hexdigest())
    selected.extend(x for x in pool if x not in selected)
    return selected[:count], {'policy': 'frozen_geometry_strata_then_sha256_seed0_order',
                              'coverage': coverage, 'eligible_person_images': len(eligible)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--csb-config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--selection', help='Reuse a previously frozen train selection; never select by gradient')
    p.add_argument('--full-cache', action='store_true')
    p.add_argument('--resume-from', help='Existing partial cache; preserve it and recover into a fresh output')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    for name in ['yolo', 'mpl', 'cache']:
        (out / 'runtime' / name).mkdir(parents=True)
    os.environ.update(YOLO_CONFIG_DIR=str(out / 'runtime/yolo'), MPLCONFIGDIR=str(out / 'runtime/mpl'),
                      XDG_CACHE_HOME=str(out / 'runtime/cache'), WANDB_MODE='disabled',
                      PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy: raise RuntimeError('GPU occupied; no competing S1 job')
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.data_utils import load_image_rgb_float, read_yolo_annotations, label_path_for_image
    from ue_framework.methods.csb_budget import digest, config_hash, item_key, domains, construct_budget, spearman
    from ue_framework.methods.csb_sensitivity import gradients
    cfg = load_config(a.config)
    settings = json.loads(Path(a.csb_config).read_text())
    root = Path(cfg['data']['dataset_root'])
    paths, selection = fixed_selection(root, 6095 if a.full_cache else a.count) if not a.selection else (
        [str(root / 'images/train' / q) for q in json.loads(Path(a.selection).read_text())['filenames']],
        json.loads(Path(a.selection).read_text())['selection'])
    selection_file = dict(selection=selection, filenames=[Path(x).name for x in paths], ids=[Path(x).stem for x in paths])
    if a.full_cache and (selection['eligible_person_images'] != 6095 or len(paths) != 6095):
        raise ValueError('Full CSB cache must cover exactly all6095 Person images')
    (out / 'selection.json').write_text(json.dumps(selection_file, indent=2))
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0').eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    def state_hash():
        h = hashlib.sha256()
        for k, v in model.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before = state_hash()
    identity = {'teacher_sha256': digest(cfg['surrogate']['ckpt']), 'csb_config_sha256': config_hash(settings),
                'source_dataset_root': str(root), 'csb_source_sha256': digest(Path(__file__).resolve().parents[1] / 'ue_framework/methods/csb_budget.py'),
                'sensitivity_source_sha256': digest(Path(__file__).resolve().parents[1] / 'ue_framework/methods/csb_sensitivity.py')}
    (out / 'maps').mkdir(); (out / 'heatmaps').mkdir()
    manifest_rows, stats, object_rows, seconds = [], [], [], []
    recovered = {}
    if a.resume_from:
        prior = Path(a.resume_from)
        progress = json.loads((prior / 'progress.json').read_text())
        if progress['identity'] != identity or progress['selection'] != selection_file:
            raise ValueError('Sensitivity cache resume identity/selection mismatch')
        recovered = {r['id']: r for r in progress['images']}
        stats = progress['stats']; object_rows = progress['object_rows']
        for row in progress['images']:
            old = prior / row['map_file']; new = out / row['map_file']
            if digest(old) != row['map_sha256']: raise ValueError('Recovered map changed')
            shutil.copyfile(old, new)
            manifest_rows.append(row)
    for path in paths:
        im = load_image_rgb_float(path)
        label = Path(label_path_for_image(path, str(root / 'labels/train')))
        annotations = read_yolo_annotations(str(label))
        if Path(path).stem in recovered:
            row = recovered[Path(path).stem]
            if digest(path) != row['image_sha256'] or digest(label) != row['annotation_sha256']:
                raise ValueError('Recovered cache source bytes changed')
            continue
        target, other = domains(im.shape, annotations)
        torch.cuda.synchronize(); start = time.perf_counter()
        try:
            g, rows = gradients(model, im, annotations)
        except Exception as error:
            raise RuntimeError(f'CSB input {Path(path).stem}: {error}') from error
        budget, norm, nstats = construct_budget(g, target, settings, no_nt_gt=not any(int(x['cls']) != 14 for x in annotations))
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        seconds.append(elapsed)
        file = out / 'maps' / (Path(path).stem + '.npz')
        average_raw = np.stack(list(g.values())).mean(0) if g else np.zeros(target.shape, np.float32)
        np.savez_compressed(file, budget=budget, normalized=norm, raw=average_raw)
        v = budget[target]
        old = np.where(target & other, 4 / 255, np.where(target, 8 / 255, 0))
        overlap, exclusive = target & other, target & ~other
        def mean(z, mask): return float(z[mask].mean()) if mask.any() else None
        quantiles = np.quantile(v, [.1, .25, .5, .75, .9])
        row = {'id': Path(path).stem, 'nt_gt_count': len(rows), 'person_pixels': int(target.sum()),
            'overlap_pixels': int(overlap.sum()), 'epsilon_min': float(v.min()), 'epsilon_mean': float(v.mean()),
            'epsilon_max': float(v.max()), 'epsilon_q10': float(quantiles[0]), 'epsilon_q25': float(quantiles[1]),
            'epsilon_q50': float(quantiles[2]), 'epsilon_q75': float(quantiles[3]), 'epsilon_q90': float(quantiles[4]),
            'high_sensitivity_budget': mean(budget, target & (norm >= .9)),
            'low_sensitivity_budget': mean(budget, target & (norm <= .1)),
            'overlap_budget': mean(budget, overlap), 'nonoverlap_budget': mean(budget, exclusive),
            'delta_budget_mean': mean(budget - old, target), 'delta_budget_overlap': mean(budget - old, overlap),
            'delta_budget_nonoverlap': mean(budget - old, exclusive), 'zero_budget_in_person': int((v == 0).sum()),
            'normalization_degenerate_class_count': sum(x['degenerate'] for x in nstats), 'seconds': elapsed}
        row['sensitivity_budget_spearman'] = spearman(norm[target], budget[target])
        stats.append(row)
        for obj in rows: object_rows.append(dict(id=row['id'], **obj))
        manifest_rows.append({'id': row['id'], 'item_key': item_key(im, annotations), 'image_sha256': digest(path),
            'annotation_sha256': digest(label), 'map_file': str(file.relative_to(out)), 'map_sha256': digest(file),
            'normalization': nstats, 'no_nt_gt': not bool(rows)})
        progress = {'state': 'partial', 'identity': identity, 'selection': selection_file, 'images': manifest_rows,
                    'stats': stats, 'object_rows': object_rows}
        if not a.full_cache or len(stats) == 1 or len(stats) % 50 == 0 or len(stats) == len(paths):
            temporary = out / 'progress.writing'
            temporary.write_text(json.dumps(progress, indent=2)); temporary.replace(out / 'progress.json')
        if a.full_cache:
            print(json.dumps({'state': 'running', 'images': len(stats), 'id': row['id'], 'seconds': elapsed}), flush=True)
            continue
        # Heatmaps are retained server-side, not raw-data transfers.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        axes[0].imshow(im)
        for ann in annotations:
            cx, cy, bw, bh = ann['bbox']; h, w = im.shape[:2]
            axes[0].add_patch(plt.Rectangle(((cx-bw/2)*w, (cy-bh/2)*h), bw*w, bh*h,
                fill=False, color='red' if int(ann['cls']) == 14 else 'cyan', linewidth=1))
        for ax, value, title in zip(axes[1:], [average_raw, norm, budget * 255], ['Raw NT sensitivity', 'Normalized', 'Budget (byte levels)']):
            handle = ax.imshow(value, cmap='viridis'); ax.set_title(title); fig.colorbar(handle, ax=ax, fraction=.045)
        axes[0].set_title(row['id'] + ' train GT')
        for ax in axes: ax.axis('off')
        fig.tight_layout(); fig.savefig(out / 'heatmaps' / (row['id'] + '.png'), dpi=120); plt.close(fig)
        print(json.dumps({'state': 'running', 'images': len(stats), 'id': row['id'], 'seconds': elapsed}), flush=True)
    for filename, rows in [('spatial_budget.csv', stats), ('sensitivity_objects.csv', object_rows)]:
        if rows:
            with (out / filename).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    after = state_hash()
    if before != after: raise ValueError('Frozen teacher changed during sensitivity')
    manifest = {'state': 'complete', 'identity': identity, 'images': manifest_rows}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summary = {'state': 'complete', 'passed': True, 'computation_kind': 'training', 'finite_loss': True,
        'production_entrypoint_reached': True, 'official_metrics_disabled': True, 'artifact_ingest_disabled': True,
        'formal_generation_enabled': False, 'victim_training_enabled': False, 'cleanup_policy': 'retain_all',
        'checkpoint_cleanup_completed': False, 'deleted_paths': [], 'checkpoint_paths_remaining': [],
        'identity': identity, 'selection': selection_file, 'sensitivity_mode': settings['sensitivity_mode'],
        'stats': stats, 'teacher_unchanged': True, 'frozen_state_sha256': before,
        'mean_seconds_per_image': float(np.mean([s['seconds'] for s in stats])),
        'estimated_6095_seconds': float(np.mean([s['seconds'] for s in stats]) * 6095),
        'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(),
        'cache_bytes': sum(q.stat().st_size for q in (out / 'maps').iterdir()),
        'images_completed': len(stats), 'full_cache': a.full_cache,
        'recovered_images': len(recovered), 'map_manifest_sha256': digest(out / 'manifest.json')}
    (out / 'smoke_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__': main()
