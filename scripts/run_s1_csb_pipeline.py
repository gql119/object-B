"""One authorized S1: frozen cache -> parent optimization -> byte audit -> victim.

No legacy reset paths, no automatic deletion, no second full control experiment.
All outputs must remain under the explicit guoqinglong S1 candidate root.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode = True
ALLOWED = Path('/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014')
ROOT = Path(__file__).resolve().parents[1]


def summarize(box):
    import numpy as np
    ids = np.asarray(box.ap_class_index).astype(int).tolist()
    ap = np.asarray(box.all_ap, dtype=float)
    recall, precision = np.asarray(box.r, dtype=float), np.asarray(box.p, dtype=float)
    if ids != list(range(20)) or ap.shape != (20, 10) or recall.shape != (20,) or precision.shape != (20,):
        raise ValueError('Incomplete or unordered VOC20 validator output')
    if not all(np.isfinite(v).all() for v in [ap, recall, precision]):
        raise FloatingPointError('Nonfinite validator output')
    keep = [i for i in range(20) if i != 14]
    result = {'AP50_per_class': ap[:, 0].tolist(), 'AP75_per_class': ap[:, 5].tolist(),
        'AP50_95_per_class': ap.mean(1).tolist(), 'Recall_per_class': recall.tolist(), 'Precision_per_class': precision.tolist(),
        'Recall_target': float(recall[14]), 'Precision_target': float(precision[14]),
        'Recall_non_target': float(recall[keep].mean()), 'Precision_non_target': float(precision[keep].mean()),
        'precision_recall_definition': 'validator best-F1 operating point; AP uses full confidence curve'}
    for name, values in [('mAP50', ap[:, 0]), ('mAP75', ap[:, 5]), ('mAP50_95', ap.mean(1))]:
        result.update({name + '_all': float(values.mean()), name + '_target': float(values[14]),
                       name + '_non_target': float(values[keep].mean())})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--csb-config', required=True)
    p.add_argument('--parent-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--review')
    p.add_argument('--admission')
    p.add_argument('--admission-mode', action='store_true')
    p.add_argument('--cache-existing')
    p.add_argument('--cache-resume-from')
    p.add_argument('--resume-noise')
    a = p.parse_args()
    out = Path(a.output).resolve()
    if not out.is_relative_to(ALLOWED) or not ROOT.is_relative_to(ALLOWED):
        raise ValueError('S1 worktree/output outside authorized guoqinglong project root')
    if out.exists(): raise FileExistsError('Never relaunch/collide with an existing S1 output')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied; do not compete/interfere')
    if shutil.disk_usage(ALLOWED).free < 40 * 1024**3:
        raise RuntimeError('S1 needs 40GiB free; no deletion permitted')
    if a.admission_mode:
        if not a.cache_existing: raise ValueError('Admission needs a fixed train-only development cache')
        command = [sys.executable, str(ROOT / 'scripts/s1_mechanism_probe.py'),
            '--config', str(Path(a.config).resolve()), '--csb-config', str(Path(a.csb_config).resolve()),
            '--parent-root', a.parent_root, '--prototypes',
            str(Path(a.parent_root) / 'noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt'),
            '--cache', a.cache_existing, '--output', str(out)]
        subprocess.run(command, check=True, env=dict(os.environ))
        return
    if not a.review or not a.admission: raise ValueError('Formal pipeline requires review AND passing admission')
    review = json.loads(Path(a.review).read_text()); admission = json.loads(Path(a.admission).read_text())
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if review.get('result') != 'scientifically_correct' or review.get('decision') != 'allow_run' or review.get('code_snapshot') != commit:
        raise ValueError('Single scientific review has not allowed this exact source snapshot')
    if admission.get('passed') is not True or admission.get('formal_generation_enabled') is not False:
        raise ValueError('Missing bounded admission evidence')
    if admission.get('csb_checkpoint_rng_optimizer_cursor_replay_max_difference') != 0:
        raise ValueError('Exact recovery gate failed')
    out.mkdir(parents=True)
    for name in ['yolo', 'mpl', 'cache', 'torch']: (out / 'runtime' / name).mkdir(parents=True)
    os.environ.update(YOLO_CONFIG_DIR=str(out / 'runtime/yolo'), MPLCONFIGDIR=str(out / 'runtime/mpl'),
        XDG_CACHE_HOME=str(out / 'runtime/cache'), TORCH_HOME=str(out / 'runtime/torch'), WANDB_MODE='disabled',
        PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.data_utils import list_images, load_image_rgb_float, read_yolo_annotations, label_path_for_image
    from ue_framework.methods.csb_budget import digest, config_hash, BudgetStore, png_project, domains
    from ue_framework.methods.s1_csb import S1CSBTrainer, S1CSBGenerator
    from ue_framework.methods.s1_optimization import optimize
    from ue_framework.methods.semantic_coordinates import semantic_identity
    from scripts.audit_safe_bg_ablation_fingerprint_20261004 import split_fingerprint
    from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES, pus, collateral
    if review.get('csb_config_sha256') != digest(a.csb_config) or review.get('admission_sha256') != digest(a.admission):
        raise ValueError('Reviewed spatial config or admission receipt changed')
    cfg = load_config(a.config); mc = cfg['methods']['legacy_kproto_ret']
    parent = Path(a.parent_root).resolve()
    provenance = json.loads((ROOT / 'docs/s1_evidence/parent_remote_identity.json').read_text())
    if str(parent) != provenance['parent_root']: raise ValueError('Different Safe-Soft05 parent')
    for rel, expected in provenance['source_sha256'].items():
        if digest(ROOT / rel) != expected: raise ValueError('Canonical parent changed: ' + rel)
    if digest(a.config) != provenance['formal_status']['config_sha256']:
        raise ValueError('Original formal Safe-Soft05 config changed')
    if digest(cfg['surrogate']['ckpt']) != provenance['formal_status']['surrogate_sha256']:
        raise ValueError('Different frozen/Adaptive checkpoint')
    if semantic_identity(cfg, ROOT) != provenance['formal_status']['semantic_identity']:
        raise ValueError('Parent TSFL/orth/Safe-BG artifact or source identity changed')
    if torch.__version__ != provenance['environment']['torch']:
        raise ValueError('Parent PyTorch environment changed')
    import ultralytics
    if ultralytics.__version__ != provenance['environment']['ultralytics']:
        raise ValueError('Parent validator/training environment changed')
    source = Path(cfg['data']['dataset_root'])
    settings = json.loads(Path(a.csb_config).read_text())
    status = dict(state='running', stage='source_audit', started_at_unix=time.time(), run_id=out.name,
                  source_commit=commit, primary='last.pt@200', secondary='best.pt', noise_time_policy='finish_registered_24_epochs')
    status_path = out / 'pipeline_status.json'
    def mark(stage, **kwargs):
        status.update(stage=stage, **kwargs); status_path.write_text(json.dumps(status, indent=2)); print(json.dumps(status), flush=True)
    def dump(path, value): path.write_text(json.dumps(value, indent=2))
    try:
        import hashlib
        splits = {}
        for split, expected in [('train', 16551), ('val', 4952)]:
            images = [Path(x) for x in list_images(str(source / 'images' / split))]
            labels = list((source / 'labels' / split).glob('*.txt'))
            if len(images) != expected or len(labels) != expected or {x.stem for x in images} != {x.stem for x in labels}:
                raise ValueError('Full train/val source coverage mismatch')
            splits[split] = split_fingerprint(images + labels)
        combined = hashlib.sha256((splits['train'] + '\n' + splits['val']).encode()).hexdigest()
        if combined != '29b131dfca2e3636e9c47502961caa6297acf0450f646ff6409b7b3bfb21b3fa':
            raise ValueError('VOC source bytes differ from matched parent/clean')
        dump(out / 'source_audit.json', dict(state='passed', splits=splits, dataset_fingerprint=combined))
        expected_cache = {'teacher_sha256': digest(cfg['surrogate']['ckpt']), 'csb_config_sha256': config_hash(settings),
            'source_dataset_root': str(source), 'csb_source_sha256': digest(ROOT / 'ue_framework/methods/csb_budget.py'),
            'sensitivity_source_sha256': digest(ROOT / 'ue_framework/methods/csb_sensitivity.py')}
        if not a.cache_existing:
            mark('frozen_csb_cache')
            cache = out / 'csb_cache'
            cmd = [sys.executable, str(ROOT / 'scripts/s1_spatial_audit.py'), '--config', str(Path(a.config).resolve()),
                '--csb-config', str(Path(a.csb_config).resolve()), '--output', str(cache), '--full-cache']
            if a.cache_resume_from: cmd += ['--resume-from', a.cache_resume_from]
            subprocess.run(cmd, check=True, env=dict(os.environ))
        else:
            cache = Path(a.cache_existing).resolve()
            if not cache.is_relative_to(ALLOWED): raise ValueError('Cache outside S1')
        cache_manifest = json.loads((cache / 'manifest.json').read_text())
        store = BudgetStore(cache, expected_cache)
        if len(store.rows) != 6095 or len({r['id'] for r in cache_manifest['images']}) != 6095:
            raise ValueError('Frozen budget cache does not cover all6095 Person images')
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        model = YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
        learner = YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
        trainer = S1CSBTrainer(cfg, mc, torch.device('cuda:0'), model, learner, budget_store=store, budget_mode='csb')
        expected_channels = [0,6,10,29,41,68,77,86,89,112,116,125,128,133,140,167,173,184,220,229,237,238,239,240]
        if trainer.channels.tolist() != expected_channels or trainer.safe_bg_rank != 8 or trainer.safe_bg_removal_alpha != .5:
            raise ValueError('Exact Parent P5 channels/rank/alpha changed')
        images = trainer._collect_target_images(str(source / 'images/train'), str(source / 'labels/train'))
        if len(images) != 6095 or {Path(x).stem for x in images} != {r['id'] for r in cache_manifest['images']}:
            raise ValueError('Noise/cached image identity mismatch')
        prototypes_path = parent / 'noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt'
        if digest(prototypes_path) != provenance['noise_artifacts']['background_prototypes.pt']['sha256']:
            raise ValueError('Parent background prototypes changed')
        prototypes = torch.load(prototypes_path, map_location='cuda:0', weights_only=True)['prototypes']
        identity = dict(parent_config_sha256=digest(a.config), cache_manifest_sha256=digest(cache / 'manifest.json'),
                        budget_mode='csb', optimization_source_sha256=digest(ROOT / 'ue_framework/methods/s1_optimization.py'))
        mark('noise_generation')
        noise = out / 'noise'
        noise_status = optimize(trainer, images, source / 'labels/train', prototypes, noise, identity, resume=a.resume_noise)
        if noise_status.get('schedule_complete') is not True or noise_status['epochs_completed'] != 24 or noise_status['steps'] != 9144:
            raise ValueError('Incomplete registered 24epoch/9144step S1 generation')
        mark('materialization')
        data = out / 'dataset'
        for split in ['train', 'val']:
            for folder in ['images', 'labels']: (data / folder / split).mkdir(parents=True)
        gen = S1CSBGenerator(cfg, mc, torch.device('cuda:0'), model, str(noise / 'global_params.pt'), budget_store=store)
        rows = []
        for path in list_images(str(source / 'images/train')):
            path = Path(path); label = source / 'labels/train' / (path.stem + '.txt')
            annotations = read_yolo_annotations(str(label))
            has = any(int(q['cls']) == 14 for q in annotations)
            dest = data / 'images/train' / (path.stem + '.png' if has else path.name)
            if has:
                image = load_image_rgb_float(str(path)); b = store.get(image, annotations)
                result = gen.generate(image, annotations, 0, 24, trainer.eps, 'mask', image_path=str(path))
                clean = np.rint(image * 255).astype(np.uint8)
                poison = png_project(clean, result.poisoned_image, b)
                from PIL import Image
                Image.fromarray(poison).save(dest)
            else: shutil.copyfile(path, dest)
            shutil.copyfile(label, data / 'labels/train' / label.name)
            rows.append(dict(id=path.stem, has_target=has, source=str(path), output=str(dest)))
        for split in ['val']:
            for path in list_images(str(source / 'images' / split)):
                path = Path(path); label = source / 'labels' / split / (path.stem+'.txt')
                shutil.copyfile(path, data / 'images' / split / path.name)
                shutil.copyfile(label, data / 'labels' / split / label.name)
        dump(out / 'materialization_manifest.json', rows)
        mark('pixel_label_audit')
        changes = []
        for row in rows:
            original, dest = Path(row['source']), Path(row['output'])
            label = source / 'labels/train' / (row['id']+'.txt')
            if digest(label) != digest(data / 'labels/train' / label.name): raise ValueError('Training label bytes changed')
            if not row['has_target']:
                if digest(original) != digest(dest): raise ValueError('Non-Person image changed')
                continue
            clean = np.rint(load_image_rgb_float(str(original)) * 255).astype(np.uint8)
            poison = np.rint(load_image_rgb_float(str(dest)) * 255).astype(np.uint8)
            ann = read_yolo_annotations(str(label)); target, other = domains(clean.shape, ann)
            budget = store.get(clean.astype(np.float32)/255, ann)
            difference = np.abs(poison.astype(np.int16) - clean.astype(np.int16)).max(2)
            if np.any(difference[~target] != 0) or np.any(difference/255 > budget + 1e-7):
                raise ValueError('Decoded PNG violates frozen spatial budget: '+row['id'])
            per_object = []
            from ue_framework.methods.legacy_kproto_ret import _rectangle
            for q in ann:
                if int(q['cls']) == 14:
                    mask = _rectangle(clean.shape[:2], q['bbox'])[0,0].numpy().astype(bool)
                    per_object.append(bool(np.any(difference[mask] > 0)))
            if not all(per_object): raise ValueError('Person GT box unchanged: '+row['id'])
            changes.append(dict(id=row['id'], max_abs_delta_byte=int(difference.max()),
                mean_abs_delta_byte=float(difference[target].mean()), changed_person_gt_boxes=sum(per_object)))
        for path in list_images(str(source / 'images/val')):
            path = Path(path)
            if digest(path) != digest(data / 'images/val' / path.name): raise ValueError('Clean val image changed')
            label = source / 'labels/val' / (path.stem+'.txt')
            if digest(label) != digest(data / 'labels/val' / label.name): raise ValueError('Clean val label changed')
        if len(rows) != 16551 or len(changes) != 6095 or sum(x['changed_person_gt_boxes'] for x in changes) != 13256:
            raise ValueError('Full materialization/Person object count mismatch')
        dump(out / 'dataset_integrity.json', dict(state='passed', train=16551, val=4952, target_images=6095,
            person_gt=13256, changed_person_gt=13256, labels_byte_identical=True, clean_val_byte_identical=True,
            non_target_images_byte_identical=True, outside_target_zero=True, per_pixel_frozen_budget_passed=True, images=changes))
        data_yaml = out / 'train_data.yaml'
        names = '\n'.join(f'  {i}: {name}' for i,name in enumerate(NAMES))
        data_yaml.write_text(f'path: {data}\ntrain: images/train\nval: images/val\nnames:\n{names}\n')
        mark('victim_training')
        after_splits = {}
        for split in ['train', 'val']:
            paths = [Path(x) for x in list_images(str(source / 'images' / split))] + list((source / 'labels' / split).glob('*.txt'))
            after_splits[split] = split_fingerprint(paths)
        if after_splits != splits: raise ValueError('Original VOC bytes changed during generation/materialization')
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        args = dict(provenance['victim_spec']['train_args'])
        args.update(data=str(data_yaml), project=str(out / 'victim_run'), name='victim')
        init = ROOT / cfg['victim']['init']
        if digest(init) != provenance['victim_spec']['init_yaml_sha256']: raise ValueError('Victim architecture changed')
        dump(out / 'victim_run_spec.json', dict(initialization='random_yaml', init_yaml=str(init), init_yaml_sha256=digest(init), train_args=args))
        YOLO(str(init)).train(**args)
        victim = out / 'victim_run/victim'
        with (victim / 'results.csv').open() as f: history = list(csv.DictReader(f))
        if len(history) != 200 or int(float(list(history[-1].values())[0])) != 200:
            raise ValueError('Victim did not finish exactly200 epochs')
        mark('evaluation')
        eval_dir = out / 'evaluation'; eval_dir.mkdir()
        eval_args = dict(data=str(data_yaml), imgsz=640, batch=36, workers=16, device='0',
                         verbose=False, plots=False, seed=0)
        targets = {'S1_last': victim / 'weights/last.pt', 'S1_best_secondary': victim / 'weights/best.pt',
            'Safe_Soft05_last': parent / 'victim_b2_to_safe_soft05_eps8_person_seed0_20261004/victim/weights/last.pt'}
        expected_clean = Path('/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/baseline_dev_20260928/clean_baseline_seed0_20260928/victim/weights/last.pt')
        targets['Clean_last'] = expected_clean
        if digest(targets['Safe_Soft05_last']) != provenance['victim_last_sha256']:
            raise ValueError('Parent last.pt changed')
        if digest(expected_clean) != '834c68977d31dedc859e6e905a7d2d4259b73fb0077fb587e5a5636aa93e3bad':
            raise ValueError('Matched clean last.pt changed')
        metrics = {}
        for role, checkpoint in targets.items():
            result = YOLO(str(checkpoint)).val(**eval_args, project=str(eval_dir), name=role, exist_ok=False)
            metrics[role] = dict(summarize(result.box), checkpoint_role=role, checkpoint_sha256=digest(checkpoint), class_names=list(NAMES))
            dump(eval_dir / (role+'_metrics.json'), metrics[role])
        if abs(metrics['Safe_Soft05_last']['mAP50_target'] - provenance['last_metrics']['mAP50_target']) > 1e-6:
            raise ValueError('Parent reevaluation differs; pause formal comparison')
        clean = metrics['Clean_last']; baseline = metrics['Safe_Soft05_last']; s1 = metrics['S1_last']
        for role, value in metrics.items():
            value.update(selectivity=pus(clean, value), collateral=collateral(clean, value))
        report = dict(state='complete', primary='last.pt@200', secondary='best.pt', metrics=metrics,
            comparisons={'delta_target_ap50': s1['mAP50_target']-baseline['mAP50_target'],
                'delta_nt_map50': s1['mAP50_non_target']-baseline['mAP50_non_target'],
                'delta_nt_map75': s1['mAP75_non_target']-baseline['mAP75_non_target'],
                'target_below_5percent': s1['mAP50_target'] < .05},
            uniform_control='bounded diagnostics only, no full victim', causal_mechanism_established=False)
        dump(out / 'comparison.json', report)
        with (out / 'voc20_comparison.csv').open('w', newline='') as f:
            rows = []
            for i, name in enumerate(NAMES):
                row = {'class': name}
                for role,value in metrics.items():
                    for key in ['AP50', 'AP75', 'Recall', 'Precision']: row[role+'_'+key]=value[key+'_per_class'][i]
                rows.append(row)
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        mark('complete', state='complete', completed_at_unix=time.time())
    except BaseException as error:
        mark(status['stage'], state='failed', error=repr(error), failed_at_unix=time.time())
        raise


if __name__ == '__main__': main()
