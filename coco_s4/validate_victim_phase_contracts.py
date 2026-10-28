"""CPU-only matched-victim protocol contracts; no dataset or training access."""
import copy
import ast
import importlib
import importlib.util
import json
from pathlib import Path

from .phase_victim import (canonical_model_state_sha256, cfg_predecessor,
                           expected_data_root, validate_phase_config)
from .victim_release import READ_DEPENDENCY


def main():
    root = Path(__file__).resolve().parents[1]
    resolved_imports = []
    tree = ast.parse((root/'coco_s4/phase_victim.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_name = importlib.util.resolve_name('.'*node.level+(node.module or ''), 'coco_s4') if node.level else node.module
        module = importlib.import_module(module_name)
        for item in node.names:
            if item.name == '*' or not hasattr(module, item.name):
                raise AssertionError('Production import cannot resolve: '+module_name+'.'+item.name)
            resolved_imports.append(module_name+'.'+item.name)
    clean = json.loads((root/'coco_s4/configs/clean200.json').read_text(encoding='utf-8'))
    s4 = json.loads((root/'coco_s4/configs/s4_200.json').read_text(encoding='utf-8'))
    validate_phase_config(clean)
    validate_phase_config(s4)
    if clean['training'] != s4['training']:
        raise AssertionError('Clean and poisoned victim protocols differ')
    if expected_data_root('clean200') != Path('/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023'):
        raise AssertionError('Clean victim route differs from the accepted fixed dataset root')
    if expected_data_root('s4_200').name != 'poisoned':
        raise AssertionError('S4 victim route must consume only the accepted poisoned export')
    if (cfg_predecessor({'phase_config': clean}, 'clean200') != 'export25k'
            or cfg_predecessor({'phase_config': s4}, 's4_200') != 'clean200'):
        raise AssertionError('Matched-victim predecessor binding differs from serial order')
    try:
        cfg_predecessor({'phase_config': {**clean, 'predecessor_phase': 'teacher200'}}, 'clean200')
    except ValueError:
        pass
    else:
        raise AssertionError('Victim predecessor contract accepted an out-of-order phase')
    compile(READ_DEPENDENCY, '<victim-read-dependency>', 'exec')
    import torch
    first = {'z': torch.tensor(3, dtype=torch.int64), 'a': torch.tensor([1.0, 2.0])}
    reordered = {'a': torch.tensor([1.0, 2.0]), 'z': torch.tensor(3, dtype=torch.int64)}
    if canonical_model_state_sha256(first) != canonical_model_state_sha256(reordered):
        raise AssertionError('Initial model-state digest depends on mapping insertion order')
    changed = {'a': torch.tensor([1.0, 2.5]), 'z': torch.tensor(3, dtype=torch.int64)}
    if canonical_model_state_sha256(first) == canonical_model_state_sha256(changed):
        raise AssertionError('Initial model-state digest ignored a tensor value change')
    try:
        canonical_model_state_sha256({'bad': 'not a tensor'})
    except TypeError:
        pass
    else:
        raise AssertionError('Initial model-state digest accepted a non-tensor state entry')

    invalid = []
    for phase, base, edits in (
        ('clean200', clean, [('training', 'batch', 32), ('training', 'workers', 8),
                             ('training', 'epochs', 199), ('training', 'seed', 1),
                             ('training', 'imgsz', 512), ('training', 'amp', False),
                             ('validation', 'training_val', True), ('validation', 'official_metrics', True)]),
        ('s4_200', s4, [(None, 'target_class_id', 14), (None, 'image_count', 24999),
                        (None, 'initialization', 'COCO pretrained weights'),
                        (None, 'automatic_resume', True), (None, 'cleanup_policy', 'delete old outputs')]),
    ):
        for group, key, value in edits:
            bad = copy.deepcopy(base)
            if group is None:
                bad[key] = value
            else:
                bad[group][key] = value
            try:
                validate_phase_config(bad, phase)
            except ValueError:
                invalid.append(f'{phase}:{group or "root"}.{key}')
            else:
                raise AssertionError(f'Invalid matched-victim contract accepted: {phase}/{group}/{key}')
    if len(invalid) != 13:
        raise AssertionError('Expected all 13 invalid protocol cases to be rejected')
    if torch.cuda.is_initialized():
        raise AssertionError('CPU contract/import validation initialized CUDA')
    print(json.dumps(dict(state='passed', phases=['clean200', 's4_200'],
                          actual_production_imports_resolved=resolved_imports,
                          matched_training_protocol=True, invalid_contracts_rejected=len(invalid),
                          predecessor_bindings=2, canonical_state_hash_cases=3,
                          dataset_images_opened=0, GPU_units=0, official_metrics_enabled=False,
                          formal_registration=False), sort_keys=True))


if __name__ == '__main__':
    main()
