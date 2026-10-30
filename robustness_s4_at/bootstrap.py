"""One-time local protocol extraction from the retained, accepted Original-S4 archive."""
import argparse
import shutil
import subprocess
from pathlib import Path
import yaml
from .common import ROOT, atomic, sha, write_once


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--reference', type=Path, required=True)
    a.add_argument('--authorization', type=Path, required=True)
    args = a.parse_args()
    import json
    qpath = args.reference / 'robustness_s4_extended/configs/QUEUE.json'
    q = json.loads(qpath.read_text(encoding='utf-8-sig'))
    b = q['baseline']
    st_path = args.reference / 'docs/S4_AMP_RECOVERY_20261018/S4_COMPARISON.json'
    st = json.loads(st_path.read_text(encoding='utf-8-sig'))
    import ultralytics
    defaults_path = Path(ultralytics.__file__).resolve().parent / 'cfg/default.yaml'
    defaults = yaml.safe_load(defaults_path.read_text(encoding='utf-8'))
    if sha(defaults_path) != q['native_sources']['cfg/default.yaml']['sha256']:
        raise ValueError('Installed native defaults differ from accepted native identity')
    base = 'ab626917f20917e58cf30033ac99df00938cb561'
    parent = 'b1e29510e7b1ed5fa0fe361a27132d8790e65c14'
    subprocess.run(['git', 'merge-base', '--is-ancestor', parent, base], cwd=ROOT, check=True)
    original = {}
    for rel in subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', base,
                                         'ue_framework', 'configs/voc_yolov8n_20cls.yaml'],
                                        cwd=ROOT, text=True).splitlines():
        data = subprocess.check_output(['git', 'show', base + ':' + rel], cwd=ROOT)
        import hashlib
        if (ROOT / rel).read_bytes() != data:
            raise ValueError('Original source bytes changed: ' + rel)
        original[rel] = hashlib.sha256(data).hexdigest()
    train = dict(defaults)
    train.update(b['victim_protocol']['train_args'])
    for key in ('data', 'project', 'name', 'device'):
        train.pop(key, None)
    train.update(model='configs/voc_yolov8n_20cls.yaml', mode='train', task='detect',
                 resume=False, exist_ok=True, classes=None, single_cls=False, fraction=1.)
    docs = ROOT / 'docs/S4_AT_ROBUSTNESS_20261027'
    docs.mkdir(parents=True, exist_ok=True)
    auth = docs / 'USER_AT_TASK_20261027.md'
    if auth.exists():
        raise FileExistsError(auth)
    shutil.copyfile(args.authorization, auth)
    accepted = args.reference / 'execution/EXTENDED_FINAL_LOCAL_ACCEPTANCE_20261024.json'
    shutil.copyfile(accepted, docs / 'REUSED_ORIGINAL_S4_ACCEPTANCE.json')
    config = dict(schema='s4.at.protocol.v1', method_name='SMIC-Det',
                  scientific_parent='Original S4 = S0 Safe-Soft05 + TFSS-Utility',
                  generation_parent_commit=parent, victim_parent_commit=base,
                  target_id=14, nc=20, train_count=16551, val_count=4952,
                  order=['S4AT4_Clean_seed0', 'S4AT4_Poison_seed0',
                         'S4AT8_Clean_seed0', 'S4AT8_Poison_seed0'],
                  pgd={str(e): dict(epsilon_pixels=e, step_pixels=s, steps=10, random_start=True)
                       for e, s in ((4, 1), (8, 2))},
                  pgd_definition='project-defined full-GT detection PGD10; not undisclosed MI-UE Table4 details',
                  inner_dtype='FP32', inner_BN='freeze running statistics; preserve affine and all module flags',
                  attack_input='actual native augmentation/LetterBox/normalization output; full image',
                  original_S4_budget_pixels=dict(exclusive=8, overlap=4, outside=0),
                  train_args=train, model_yaml_sha256=b['victim_protocol']['init_yaml_sha256'],
                  native_sources={k: v['sha256'] for k, v in q['native_sources'].items()},
                  native_versions=b['runtime_versions'], original_sources=original,
                  clean_root=b['clean_root'], poison_root=b['poisoned_dataset_path'],
                  parent_output=b['parent_output'],
                  reused_artifacts={
                      'materialization_manifest.json': b['parent_materialization_manifest_sha256'],
                      'dataset_integrity.json': b['parent_dataset_integrity_sha256'],
                      'noise/checkpoints/step_009144_epoch_24_cursor_00000.pt': b['parent_noise_checkpoint_sha256']},
                  S4_config_sha256=b['frozen_s4_config_sha256'],
                  reused_acceptance=dict(path='docs/S4_AT_ROBUSTNESS_20261027/REUSED_ORIGINAL_S4_ACCEPTANCE.json',
                                         sha256=sha(accepted), archive_commit='cebb7411c31d667c3748b6739a2f699c9f90fa7e'),
                  authorization=dict(path=auth.relative_to(ROOT).as_posix(), sha256=sha(auth)),
                  remote_owned_root='/project/guoqinglong/projects/s4_at_robustness_20261027',
                  existing_COCO_output='/project/guoqinglong/projects/s4_coco_experiment_a_20261025/outputs/s4_coco_experiment_a_seed0_20261027_risk_v1',
                  resources=dict(RAM_available_GiB=32, GPU_free_MiB=24000,
                                 disk_policy='per_filesystem_dynamic_v1',
                                 disk_output_budget_GiB=3, disk_safety_reserve_GiB=5,
                                 disk_dataset_reserve_GiB=2,
                                 all_three_IOwait_percent_at_least=20, own_concurrent_GPU_allowed=False),
                  primary='last.pt@200', secondary='best.pt', validation='full untouched VOC20 4952',
                  cleanup_policy='retain_all',
                  reference_ST_human_rounded=dict(clean_Person_AP50=.867943, clean_NT_AP50=.766462,
                                                  poison_Person_AP50=.028159, poison_NT_AP50=.693787))
    config['reference_ST_exact'] = dict(
        clean_Person_AP50=st['metrics']['Clean']['mAP50_target'],
        clean_NT_AP50=st['metrics']['Clean']['mAP50_non_target'],
        poison_Person_AP50=st['metrics']['S4']['mAP50_target'],
        poison_NT_AP50=st['metrics']['S4']['mAP50_non_target'])
    config['S4_ST_victim_last_sha256'] = st['source_provenance']['last_sha256']
    write_once(ROOT / 'robustness_s4_at/configs/PROTOCOL.json', config)
    atomic(docs / 'SOURCE_AND_PROTOCOL_AUDIT.json', dict(
        state='LOCAL_PROVENANCE_VERIFIED_REMOTE_RECHECK_PENDING',
        parent=parent, base=base, original_files_byte_exact=len(original),
        reused_reference_queue_sha256=sha(qpath), authorization_sha256=sha(auth),
        native_defaults_sha256=sha(defaults_path), no_GPU_job_launched=True))


if __name__ == '__main__':
    main()
