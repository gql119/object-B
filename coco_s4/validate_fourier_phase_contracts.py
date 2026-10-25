"""CPU-only contracts for the Original S4 COCO Fourier phase wrapper."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import yaml

from .phase_fourier import (MAIN_CONFIG, eligible_paths_from_manifest,
                            select_validation_images, validate_fourier_source_config,
                            validate_phase_config)
from .runtime import ROOT


def _expect_reject(fn, label):
    try:
        fn()
    except (ValueError, KeyError, TypeError):
        return
    raise AssertionError('contract accepted invalid input: '+label)


def run():
    phase = json.loads((ROOT/'coco_s4/configs/fourier24.json').read_text(encoding='utf-8'))
    validate_phase_config(phase)
    for key, value in (('epochs', 23), ('batch_size', 8), ('eot_samples', 1),
                       ('seed', 1), ('support_mode', 'bbox'), ('optimizer_steps', '9144'),
                       ('selected_space_dimension', 23), ('eps', 4/255),
                       ('validation_used', True), ('official_metrics_enabled', True),
                       ('artifact_ingest_enabled', True), ('optimizer_resume_claim', True),
                       ('checkpoint_selection_allowed', True), ('predecessor_phase', 'teacher200'),
                       ('workflow_run_id', 'other-run')):
        altered = deepcopy(phase); altered[key] = value
        _expect_reject(lambda x=altered: validate_phase_config(x), 'phase.'+key)

    original = yaml.safe_load((ROOT/MAIN_CONFIG).read_text(encoding='utf-8'))
    validate_fourier_source_config(original)
    mutations = (
        ('enable_adaptive_freq_basis', False), ('freq_candidate_num_bases', 32),
        ('freq_active_num_bases', 8), ('adaptive_freq_warmup_epochs', 4),
        ('universal_epochs', 23), ('universal_batch_size', 8), ('eot_samples', 1),
        ('jnd_floor', .5), ('universal_lr_fourier', .002), ('lambda_tv', 0),
    )
    for key, value in mutations:
        altered = deepcopy(original); altered['methods']['legacy_kproto_ret'][key] = value
        _expect_reject(lambda x=altered: validate_fourier_source_config(x), 'method.'+key)
    altered = deepcopy(original); altered['methods']['legacy_kproto_ret']['loss']['lambda_ret'] = 0
    _expect_reject(lambda: validate_fourier_source_config(altered), 'loss.lambda_ret')
    altered = deepcopy(original); altered['methods']['legacy_kproto_ret']['adaptive']['inner_steps'] = 2
    _expect_reject(lambda: validate_fourier_source_config(altered), 'adaptive.inner_steps')
    altered = deepcopy(original); altered['methods']['legacy_kproto_ret']['kproto']['max_steps'] = 1
    _expect_reject(lambda: validate_fourier_source_config(altered), 'formal.max_steps')

    with tempfile.TemporaryDirectory(prefix='coco-fourier-contract-') as temporary:
        root = Path(temporary)/'images'/'train'; root.mkdir(parents=True)
        names = ['000000000001.jpg', '000000000002.jpg', '000000000003.jpg']
        valid = dict(paths=[str(root/name) for name in names], count=3,
                     predicate='original bbox_half_overlap support.sum()>10, Person0',
                     optimizer_steps=24, epochs=24, batch=16)
        paths, steps = eligible_paths_from_manifest(valid, root)
        assert len(paths) == 3 and steps == 24
        for label, patch in (
            ('wrong_count', {'count': 2}), ('wrong_steps', {'optimizer_steps': 48}),
            ('wrong_epochs', {'epochs': 200}), ('wrong_batch', {'batch': 36}),
            ('wrong_predicate', {'predicate': 'all Person images'}),
            ('duplicate', {'paths': [str(root/names[0])]*3}),
            ('escaped_path', {'paths': [str(root/names[0]), str(root/names[1]), str(Path(temporary)/'outside.jpg')]}),
        ):
            altered = dict(valid); altered.update(patch)
            _expect_reject(lambda x=altered: eligible_paths_from_manifest(x, root), label)
        altered = dict(valid); altered['paths'] = list(reversed(valid['paths']))
        _expect_reject(lambda: eligible_paths_from_manifest(altered, root), 'unsorted')
        assert select_validation_images(paths, smoke_images=1) == paths[:1]
        assert select_validation_images(paths, cpu_probe_images=1) == paths[:1]
        full_list=[root/(str(i)+'.jpg') for i in range(20)]
        assert select_validation_images(full_list,smoke_images=16)==full_list
        assert select_validation_images(full_list,cpu_probe_images=16)==full_list
        _expect_reject(lambda: select_validation_images(paths,smoke_images=16),'too_few_batch16_images')
        _expect_reject(lambda: select_validation_images(full_list,smoke_images=2),'nonoriginal_batch')
        assert select_validation_images(paths) == paths
        _expect_reject(lambda: select_validation_images(paths, smoke_images=1, cpu_probe_images=1),
                       'combined_probe_kinds')
    return dict(state='passed', invalid_phase_contracts=15, invalid_source_protocols=13,
                eligible_manifest_cases=13, full_batch16_order_cases=4, dataset_images_opened=0, GPU_units=0,
                official_metrics_enabled=False, formal_registration=False)


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
