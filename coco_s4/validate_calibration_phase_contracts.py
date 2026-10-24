"""CPU-only checks for immutable calibration protocol and artifact containment."""
import copy
from pathlib import Path
import yaml

from .phase_calibration import (MAIN_CONFIG, ROOT, runtime_config, validate_original_config,
                                validate_phase_config)
from .runtime import load
from .calibration import verify_expected_hashes


def rejects(label, fn):
    try:
        fn()
    except (ValueError, KeyError):
        return label
    raise AssertionError('Invalid calibration contract was accepted: '+label)


def main():
    phase = load(ROOT/'coco_s4/configs/calibration.json')
    validate_phase_config(phase)
    experiment = yaml.safe_load((ROOT/MAIN_CONFIG).read_text(encoding='utf-8'))
    validate_original_config(experiment)
    resolved, checkpoint, artifact_root = runtime_config(
        phase, dict(config_path=MAIN_CONFIG, files={MAIN_CONFIG:'0'},
                    teacher_dependency={'checkpoint_path':'/accepted/last.pt'}))
    semantic = resolved['methods']['legacy_kproto_ret']['semantic']
    assert Path(semantic['artifact_dir']) == artifact_root/'method'/'coordinates'
    assert Path(semantic['safe_bg']['reference_artifact_dir']) == artifact_root/'method'/'reference'
    assert Path(semantic['strict_tsfl']['layer_selection_path']) == artifact_root/'fixed_p5_identity.json'
    assert set(resolved['data']) == {'dataset_root','train_images','train_labels','instance_mask_dir','allow_pseudo_mask_fallback'}
    assert checkpoint.name == 'last.pt'
    cases = []
    for field, value in [('selected_space_dimension', 23), ('prototype_count', 4),
                         ('support_mode', 'bbox'), ('workflow_run_id', 'wrong-run')]:
        bad = copy.deepcopy(phase); bad[field] = value
        cases.append(rejects('phase_'+field, lambda bad=bad: validate_phase_config(bad)))
    for field, value in [('rank', 4), ('removal_alpha', 1.0), ('active_k', 1)]:
        bad = copy.deepcopy(phase); bad['safe_bg'][field] = value
        cases.append(rejects('safe_bg_'+field, lambda bad=bad: validate_phase_config(bad)))
    for field, value in [('target_candidate_fraction', .5), ('nt_safety_quantile', .5),
                         ('selected_layer', 'P3')]:
        bad = copy.deepcopy(phase); bad['strict_tsfl'][field] = value
        cases.append(rejects('strict_tsfl_'+field, lambda bad=bad: validate_phase_config(bad)))
    bad = copy.deepcopy(phase); bad['artifact_relative_root'] = 'execution/../../outside'
    cases.append(rejects('artifact_path_traversal', lambda: validate_phase_config(bad)))
    for path, value in [(('experiment', 'target_class_id'), 14),
                        (('experiment', 'steps'), [9144]),
                        (('methods', 'legacy_kproto_ret', 'semantic', 'safe_bg', 'rank'), 4),
                        (('methods', 'legacy_kproto_ret', 'support_mode'), 'full_box')]:
        bad = copy.deepcopy(experiment); node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        cases.append(rejects('source_config_'+'_'.join(path), lambda bad=bad: validate_original_config(bad)))
    accepted = {'000000000025': ('a'*64, 'b'*64)}
    verify_expected_hashes('/train/000000000025.jpg', '/labels/000000000025.txt', 'a'*64, 'b'*64, accepted)
    cases.append(rejects('per_file_hash_mismatch',
                         lambda: verify_expected_hashes('/train/000000000025.jpg', '/labels/000000000025.txt',
                                                        'c'*64, 'b'*64, accepted)))
    cases.append(rejects('per_file_hash_missing',
                         lambda: verify_expected_hashes('/train/000000000026.jpg', '/labels/000000000026.txt',
                                                        'a'*64, 'b'*64, accepted)))
    print({'state': 'passed', 'valid_protocol': True, 'rejected_invalid_contracts': len(cases),
           'cases': cases, 'gpu_units': 0, 'data_images_opened': 0, 'formal_registration': False})


if __name__ == '__main__':
    main()
