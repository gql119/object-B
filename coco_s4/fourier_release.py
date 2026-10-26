"""Prepare one Fourier24 smoke behind accepted Teacher-eval/calibration gates.

This is a finite RRCTL validation workload only. The existing serial controller
remains the only formal workflow owner; this module never registers READY.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .runtime import ROOT, DATA_FILES, PARENT, VICTIM_PARENT, commit, load, now, sha, write_once
from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT
from .release_transport import anchor_path, close_and_register, validate_anchor_outside_lifecycle
from .phase_fourier import CALIBRATION_ARTIFACT_SUFFIXES, MAIN_CONFIG, validate_phase_config

EX = ROOT/'execution'
PYTHON = '/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'
CALIBRATION_SOURCE = r'''
from pathlib import Path
import hashlib,json,os,sys
sys.path.insert(0,sys.argv[6]);from coco_s4.dimension_protocol import validate_receipt_dimension
o=Path(sys.argv[1]);s=json.loads((o/'pipeline_state.json').read_text())
assert os.getuid()==1014 and s['run_id']==sys.argv[2] and s['source_commit']==sys.argv[3]
ident=s['controller'];p=Path('/proc')/str(ident['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split()
assert p.stat().st_uid==1014 and int(f[19])==ident['start_ticks'] and f[0]!='Z'
if (s['state']!='WAITING_PHASE_RELEASE' or s['current_stage']!='fourier24'
    or s.get('active_job') is not None or s.get('active_child') is not None):
 print(json.dumps(dict(state='WAITING_CALIBRATION_DEPENDENCY',native_state=s['state'],current_stage=s['current_stage'])));sys.exit(0)
for name in ('teacher200','teacher_eval','calibration'):
 assert s['stages'][name]['state']=='ACCEPTED'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
t=o/'teacher200/accepted.json';tr=json.loads(t.read_text());last=o/'teacher200/train/weights/last.pt';full=o/'teacher200/train/state_snapshots/epoch_200_full_state.pt'
assert tr['state']=='ACCEPTED' and tr['completed_epochs']==200 and tr['zero_validation_forwards'] is True
assert sha(t)==s['stages']['teacher200']['receipt_sha256'] and sha(last)==tr['last_checkpoint_sha256'] and sha(full)==tr['full_state_sha256']
e=o/'teacher_eval/accepted.json';er=json.loads(e.read_text())
assert er['state']=='ACCEPTED' and er['phase']=='teacher_eval' and er['image_count']==5000 and er['classes']==80
assert er['teacher_checkpoint_sha256']==sha(last) and er['parameter_updates'] is False
assert sha(e)==s['stages']['teacher_eval']['receipt_sha256']
c=o/'calibration/accepted.json';cr=json.loads(c.read_text())
assert cr['state']=='ACCEPTED' and cr['phase']=='calibration' and cr['images']==25000
assert len(cr['selected_channels'])==cr['selected_channel_count'] and cr['dimension_protocol']=='coco80-strict-tsfl-selected-dimension-v1' and cr['validation_used'] is False and cr['parameter_updates'] is False
validate_receipt_dimension(cr)
assert cr['teacher_checkpoint_sha256']==sha(last) and cr['teacher_receipt_sha256']==sha(t)
assert sha(c)==s['stages']['calibration']['receipt_sha256']
spec=json.loads((o/'calibration/phase_spec.json').read_text());source=Path(spec['source_root']);artifact=Path(spec['artifact_root'])
owned=Path(sys.argv[4]).resolve();assert source.resolve().is_relative_to((owned/'releases').resolve())
assert artifact.resolve().is_relative_to(source.resolve())
assert sha(source/'source_release.json')==cr['release_sha256']
files={r['path']:r['sha256'] for r in cr['artifact_files']}
suffixes=json.loads(sys.argv[5]);selected={}
for suffix in suffixes:
 path=artifact/suffix;relative=path.relative_to(source).as_posix()
 assert relative in files and path.is_file() and sha(path)==files[relative],suffix
 selected[relative]=dict(path=str(path),sha256=files[relative],bytes=path.stat().st_size)
assert cr['eligible_person_image_count']>0
assert cr['original_fourier_optimizer_steps']==24*((cr['eligible_person_image_count']+15)//16)
sys.path.insert(0,sys.argv[6]);from coco_s4.serial import capacity
cap=capacity(o)
print(json.dumps(dict(state='READY_FOR_ISOLATED_PHASE_VALIDATION' if cap['admitted'] else 'WAITING_RESOURCES',
 capacity=cap,teacher_dependency=dict(accepted_path=str(t),accepted_sha256=sha(t),checkpoint_path=str(last),
 checkpoint_sha256=sha(last),full_state_path=str(full),full_state_sha256=sha(full)),
 calibration_dependency=dict(accepted_path=str(c),accepted_sha256=sha(c),release_sha256=cr['release_sha256'],
 source_root=str(source.resolve()),phase_spec=spec,artifact_files=selected,eligible_person_image_count=cr['eligible_person_image_count'],
 optimizer_steps=cr['original_fourier_optimizer_steps'],clean_train_manifest_sha256=cr['accepted_clean_train_manifest_sha256']),
 teacher_eval_dependency=dict(accepted_path=str(e),accepted_sha256=sha(e),image_count=5000,classes=80),controller=ident)))
'''


def services():
    from .control import services as existing
    return existing()


def candidate_name(tag):
    if not re.fullmatch(r'coco_fourier24_[a-z0-9_]{1,40}', tag):
        raise ValueError('Use a unique explicit coco_fourier24_... identifier')
    return tag


def current_source():
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty:
        raise ValueError('Tracked source must be committed and clean before Fourier validation')
    head = commit()
    remote = subprocess.check_output(['git', '-C', str(ROOT), 'ls-remote', 'origin',
        'refs/heads/codex/s4-coco80-dynamic-dimension-20261025'], text=True).split()
    if not remote or remote[0] != head:
        raise ValueError('Current Fourier candidate must be published and remotely verified')
    return head


def read_dependency(transport):
    active = OWNED/'repos'/WORKFLOW
    from .phase_fourier import CALIBRATION_ARTIFACT_SUFFIXES
    result = transport.run([PYTHON, '-B', '-c', CALIBRATION_SOURCE, WORKFLOW_OUTPUT.as_posix(), WORKFLOW,
                            load(EX/(WORKFLOW+'.json'))['source']['commit'], OWNED.as_posix(),
                            json.dumps(CALIBRATION_ARTIFACT_SUFFIXES), active.as_posix()])
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-2500:])
    return json.loads(result.stdout)


def build(tag, head, observation, *, validation_kind):
    if observation['state'] != 'READY_FOR_ISOLATED_PHASE_VALIDATION':
        raise ValueError('Accepted calibration dependency/fresh capacity is not ready')
    tag = candidate_name(tag)
    native = load(EX/'ACTUAL_TEACHER_NATIVE_SOURCE_AUDIT_20261025.json')
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
    selected = [name for name in names if name.startswith(('coco_s4/', 'ue_framework/', 'configs/')) or name in
                ('docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json', MAIN_CONFIG,
                 'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json')]
    required = ('coco_s4/phase_fourier.py', 'coco_s4/phase_calibration.py', 'coco_s4/calibration.py',
                'coco_s4/phase_contract.py', 'coco_s4/configs/fourier24.json', MAIN_CONFIG)
    if any(name not in selected for name in required):
        raise ValueError('Fourier production source/config closure is incomplete')
    previous = observation['calibration_dependency']
    teacher = observation['teacher_dependency']
    release = dict(schema='s4.coco.release.v1', phase='fourier24', workflow_run_id=WORKFLOW,
                   source_commit=head, scientific_parent=PARENT, victim_parent=VICTIM_PARENT,
                   files={name: sha(ROOT/name) for name in selected},
                   data_root='/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023',
                   data_files=DATA_FILES,
                   native_sources={name: row['sha256'] for name, row in native['files'].items()},
                   versions=native['versions'], config_path=MAIN_CONFIG,
                   phase_config_path='coco_s4/configs/fourier24.json',
                   teacher_dependency=teacher,
                   calibration_dependency=dict(accepted_path=previous['accepted_path'],
                       accepted_sha256=previous['accepted_sha256'], release_sha256=previous['release_sha256'],
                       source_root=previous['source_root'], artifact_files=previous['artifact_files'],
                       eligible_person_image_count=previous['eligible_person_image_count'],
                       optimizer_steps=previous['optimizer_steps'],
                       clean_train_manifest_sha256=previous['clean_train_manifest_sha256']),
                   teacher_eval_dependency=observation['teacher_eval_dependency'])
    validate_phase_config(load(ROOT/'coco_s4/configs/fourier24.json'))
    local = EX/tag/'source_release.json'
    write_once(local, release)
    template = load(EX/(WORKFLOW+'.json'))
    repo = (OWNED/'releases'/tag).as_posix(); output = (OWNED/'validation'/tag).as_posix()
    remote_release = anchor_path(tag)
    template.update(run_id=tag, project='s4-coco-fourier24-validation')
    template['source'] = dict(repo_root=str(ROOT), branch='codex/s4-coco80-dynamic-dimension-20261025',
                              commit=head, allowed_post_commit_paths=['execution/', 'Ultralytics/'])
    template['remote'].update(repo_root=repo, output_root=output, stage_root=(OWNED/'stage'/tag).as_posix(),
                              control_root=(OWNED/'control'/tag).as_posix())
    validate_anchor_outside_lifecycle(remote_release, template['remote'])
    template['session']['name'] = tag
    unit = 's4coco-fourier-'+hashlib.sha256(tag.encode()).hexdigest()[:24]
    phase_args = [PYTHON, '-B', '-m', 'coco_s4.phase_fourier', '--release', remote_release,
                  '--output', output+'/result']
    if validation_kind == 'gpu_smoke':
        phase_args.extend(['--smoke-images', '1'])
    elif validation_kind == 'cpu_probe':
        phase_args.extend(['--cpu-probe-images', '1'])
    else:
        raise ValueError('Unknown bounded Fourier validation kind')
    template['workload'] = dict(cwd='.', argv=['systemd-run', '--user', '--scope', '--quiet', '--unit='+unit,
        PYTHON, '-B', '-m', 'coco_s4.process', '--job', output+'/supervision', '--cwd', repo, '--',
        *phase_args])
    template['anchors'] = [dict(name='phase_release', local_path=str(local), remote_path=remote_release, sha256=sha(local))]
    adapter = "import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')!='FAILED',complete=s.get('state')=='passed',progress=s,artifacts=['result/status.json','result/smoke_summary.json'])))"
    for health in template['health'].values():
        health['adapter_argv'] = [PYTHON, '-B', '-c', adapter]
        health['gpu_utilization_policy'] = 'advisory'
        health['timeout_seconds'] = 900
    summary_name = 'cpu_probe_summary.json' if validation_kind == 'cpu_probe' else 'smoke_summary.json'
    template['artifacts'] = [dict(path='result/'+name, required=True) for name in
        (summary_name, 'phase_spec.json', 'diagnostics.csv', 'diagnostics.json',
         'TFSS_OPTIMIZATION_PROOF.json')]
    template['local_pull_root'] = str(EX/'p')
    template['metadata'] = dict(execution_purpose=('isolated_fourier24_cpu_full_loss_validation'
                                                   if validation_kind == 'cpu_probe'
                                                   else 'isolated_fourier24_pre_review_smoke'),
        computation_kind='fourier-gradient', candidate_commit=head, cleanup_policy='retain_all',
        official_metrics_disabled=True, artifact_ingest_disabled=True, validation_used=False,
        one_actual_eligible_training_image=True, optimizer_updates=1, eot_samples=2,
        raw_forward_maximum=100, no_formal_phase_registration=True, no_second_controller=True,
        no_parameter_or_AP_tuning=True, validation_kind=validation_kind,
        expected_device='cpu' if validation_kind == 'cpu_probe' else 'cuda:0')
    from remote_run_control.models import RunSpec
    RunSpec.from_dict(template)
    spec = EX/(tag+'.json')
    write_once(spec, template)
    write_once(EX/tag/'dependency_observation.json', observation)
    return spec


def prepare(tag):
    tag = candidate_name(tag)
    if (EX/(tag+'.json')).exists() or (EX/'rrctl_state'/tag).exists() or (EX/tag).exists():
        raise FileExistsError('Prepared/bound Fourier candidate exists: inspect only')
    _, transport = services()
    observation = read_dependency(transport)
    if observation['state'] != 'READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation)); return
    spec = build(tag, current_source(), observation, validation_kind='gpu_smoke')
    print(json.dumps(dict(prepared=str(spec), candidate_commit=commit(), launched=False, formal_registered=False)))


def prepare_cpu_probe(tag):
    tag = candidate_name(tag)
    if (EX/(tag+'.json')).exists() or (EX/'rrctl_state'/tag).exists() or (EX/tag).exists():
        raise FileExistsError('Prepared/bound Fourier CPU probe exists: inspect only')
    _, transport = services()
    observation = read_dependency(transport)
    if observation['state'] != 'READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation)); return
    spec = build(tag, current_source(), observation, validation_kind='cpu_probe')
    print(json.dumps(dict(prepared=str(spec), candidate_commit=commit(), launched=False,
                          validation_kind='cpu_probe', formal_registered=False)))


def _launch_candidate(tag, expected_kind):
    tag = candidate_name(tag); spec_path = EX/(tag+'.json')
    if not spec_path.is_file() or (EX/'rrctl_state'/tag).exists() or (EX/tag/'launch_intent.json').exists():
        raise FileExistsError('Missing prepared Fourier candidate or existing bound smoke; never relaunch')
    spec = load(spec_path)
    if spec.get('metadata', {}).get('validation_kind') != expected_kind:
        raise ValueError('Prepared Fourier validation kind differs; inspect only')
    if current_source() != spec['source']['commit']:
        raise ValueError('Prepared Fourier candidate changed')
    controller, transport = services()
    observation = read_dependency(transport)
    if observation['state'] != 'READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation)); return
    release = load(EX/tag/'source_release.json')
    previous = observation['calibration_dependency']
    if (previous['accepted_sha256'] != release['calibration_dependency']['accepted_sha256']
            or previous['release_sha256'] != release['calibration_dependency']['release_sha256']
            or previous['optimizer_steps'] != release['calibration_dependency']['optimizer_steps']):
        raise ValueError('Prepared Fourier calibration dependency changed')
    remote_paths = [spec['remote'][key] for key in ('repo_root', 'output_root', 'stage_root', 'control_root')]
    checked = transport.run([PYTHON, '-B', '-c',
        'import json,os,sys;from pathlib import Path;assert os.getuid()==1014;assert all(not Path(p).exists() for p in json.loads(sys.argv[1]));print("absent")',
        json.dumps(remote_paths)])
    if checked.returncode:
        raise ValueError('Existing remote Fourier candidate paths: no launch')
    from remote_run_control.models import RunSpec
    run_spec = RunSpec.from_path(spec_path); ready = controller.ready(run_spec)
    write_once(EX/tag/'ready.json', ready)
    if not ready.get('ready'):
        raise ValueError('RRCTL thin readiness rejected current Fourier smoke')
    temp = EX/'rrctl_temp'; temp.mkdir(exist_ok=True)
    os.environ['TMP'] = os.environ['TEMP'] = str(temp)
    write_once(EX/tag/'launch_intent.json', dict(UTC=now(), run_id=tag, source_commit=spec['source']['commit']))
    result = controller.launch(run_spec)
    write_once(EX/tag/'launch.json', result)
    print(json.dumps(dict(launched_once=True, run_id=tag, result=result,
                          validation_kind=expected_kind, formal_registered=False)))


def launch_cpu_probe(tag):
    return _launch_candidate(tag, 'cpu_probe')


def launch_smoke(tag):
    return _launch_candidate(tag, 'gpu_smoke')


def register(tag, review_path):
    tag = candidate_name(tag); spec_path = EX/(tag+'.json')
    if not spec_path.is_file() or not (EX/'rrctl_state'/tag).exists():
        raise ValueError('No RRCTL-bound Fourier CUDA smoke')
    spec = load(spec_path)
    if current_source() != spec['source']['commit']:
        raise ValueError('Fourier review source differs from the published candidate')
    summary_path = Path(spec['local_pull_root'])/tag/'result'/'smoke_summary.json'
    if not summary_path.is_file(): raise ValueError('RRCTL has not pulled the Fourier smoke evidence')
    smoke = load(summary_path); release_path = EX/tag/'source_release.json'; release = load(release_path)
    cfg_sha = sha(ROOT/release['phase_config_path'])
    expected = dict(state='passed', phase='fourier24', source_commit=spec['source']['commit'],
        release_sha256=sha(release_path), phase_config_sha256=cfg_sha,
        calibration_receipt_sha256=release['calibration_dependency']['accepted_sha256'],
        teacher_checkpoint_sha256=release['teacher_dependency']['checkpoint_sha256'],
        computation_kind='fourier-gradient', production_entrypoint_reached=True, finite_outputs=True,
        frozen_teacher_unchanged=True, validation_used=False, parameter_updates=True,
        actual_fourier_optimizer_updates=1, official_metrics_disabled=True,
        artifact_ingest_disabled=True, actual_device='cuda:0', actual_training_images=1)
    if any(smoke.get(k) != v for k,v in expected.items()) or not 1 <= smoke.get('production_forward_calls',0) <= 100:
        raise ValueError('Fourier candidate lacks the bounded native CUDA update evidence')
    review = load(review_path)
    if review.get('calibration_receipt_sha256') != release['calibration_dependency']['accepted_sha256']:
        raise ValueError('Fourier review is not bound to accepted calibration')
    _, transport = services(); observation = read_dependency(transport)
    if (observation.get('state') != 'READY_FOR_ISOLATED_PHASE_VALIDATION'
            or observation.get('calibration_dependency',{}).get('accepted_sha256')
               != release['calibration_dependency']['accepted_sha256']):
        raise ValueError('Serial calibration/capacity gate changed; no Fourier registration')
    result = close_and_register(transport, tag=tag, phase='fourier24', module='coco_s4.phase_fourier',
        spec=spec, release_path=release_path, smoke_path=summary_path, review_path=review_path,
        previous_stage_receipt_sha256=release['calibration_dependency']['accepted_sha256'],
        closure_extra={'calibration_receipt_sha256':release['calibration_dependency']['accepted_sha256']})
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'launch-smoke', 'prepare-cpu-probe', 'launch-cpu-probe','register'))
    parser.add_argument('--tag', required=True);parser.add_argument('--review',type=Path)
    args = parser.parse_args()
    {'prepare': prepare, 'launch-smoke': launch_smoke,
     'prepare-cpu-probe': prepare_cpu_probe, 'launch-cpu-probe': launch_cpu_probe,
     'register':register}[args.action](args.tag,*(([args.review] if args.action=='register' else [])))
