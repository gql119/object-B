"""Prepare one immutable poisoned-train export smoke behind accepted gates.

This finite RRCTL candidate does not register a formal phase. The existing
serial controller remains the only owner of Experiment A stages.
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
from .phase_export import MAIN_CONFIG, TFSS_LOCK, validate_phase_config

EX = ROOT/'execution'
PYTHON = '/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'
READ_DEPENDENCY = r'''
from pathlib import Path
import hashlib,json,os,sys
o=Path(sys.argv[1]);s=json.loads((o/'pipeline_state.json').read_text())
assert os.getuid()==1014 and s['run_id']==sys.argv[2] and s['source_commit']==sys.argv[3]
ident=s['controller'];p=Path('/proc')/str(ident['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split()
assert p.stat().st_uid==1014 and int(f[19])==ident['start_ticks'] and f[0]!='Z'
if (s['state']!='WAITING_PHASE_RELEASE' or s['current_stage']!='export25k'
    or s.get('active_job') is not None or s.get('active_child') is not None):
 print(json.dumps(dict(state='WAITING_PREVIOUS_PHASE',native_state=s['state'],current_stage=s['current_stage'],
                       active_job=s.get('active_job'))));sys.exit(0)
for name in ('teacher200','teacher_eval','calibration','fourier24'):
 assert s['stages'][name]['state']=='ACCEPTED'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def receipt(name,required):
 path=o/name/'accepted.json';r=json.loads(path.read_text())
 assert r['state']=='ACCEPTED' and s['stages'][name]['receipt_sha256']==sha(path)
 assert all(r.get(k)==v for k,v in required.items())
 return path,r
t,tr=receipt('teacher200',dict(completed_epochs=200,zero_validation_forwards=True))
last=o/'teacher200/train/weights/last.pt';full=o/'teacher200/train/state_snapshots/epoch_200_full_state.pt'
assert sha(last)==tr['last_checkpoint_sha256'] and sha(full)==tr['full_state_sha256']
e,er=receipt('teacher_eval',dict(phase='teacher_eval',image_count=5000,classes=80,parameter_updates=False,
 checkpoint_selection_allowed=False,method_statistics_allowed=False))
assert er['teacher_checkpoint_sha256']==sha(last)
c,cr=receipt('calibration',dict(phase='calibration',images=25000,person_ground_truth=55096,
 selected_channel_count=24,validation_used=False,parameter_updates=False,checkpoint_selection_allowed=False))
assert cr['teacher_checkpoint_sha256']==sha(last) and cr['teacher_receipt_sha256']==sha(t)
assert cr['teacher_eval_receipt_sha256']==sha(e)
f,fr=receipt('fourier24',dict(phase='fourier24',epochs_completed=24,validation_used=False,
 official_metrics_disabled=True,artifact_ingest_disabled=True,clean_train_eligible_files_unchanged=True))
assert fr['calibration_receipt_sha256']==sha(c) and fr['train_manifest_sha256']==cr['accepted_clean_train_manifest_sha256']
params=o/'fourier24/global_params.pt'
assert sha(params)==fr['Fourier_params_sha256'] and fr['optimizer_steps']==cr['original_fourier_optimizer_steps']
spec_path=o/'calibration/phase_spec.json';spec=json.loads(spec_path.read_text())
source=Path(spec['source_root']).resolve();artifact=Path(spec['artifact_root']).resolve();owned=Path(sys.argv[4]).resolve()
assert source.is_relative_to((owned/'releases').resolve()) and artifact.is_relative_to(source)
assert sha(source/'source_release.json')==cr['release_sha256']
files={r['path']:r['sha256'] for r in cr['artifact_files']}
selected={}
for suffix in json.loads(sys.argv[5]):
 path=artifact/suffix;relative=path.relative_to(source).as_posix()
 assert relative in files and path.is_file() and sha(path)==files[relative],suffix
 selected[relative]=dict(path=str(path),sha256=files[relative],bytes=path.stat().st_size)
assert cr['eligible_person_image_count']>0
assert fr['optimizer_steps']==24*((cr['eligible_person_image_count']+15)//16)
sys.path.insert(0,sys.argv[6]);from coco_s4.serial import capacity
cap=capacity(o)
print(json.dumps(dict(state='READY_FOR_ISOLATED_PHASE_VALIDATION' if cap['admitted'] else 'WAITING_RESOURCES',
 capacity=cap,teacher_dependency=dict(accepted_path=str(t),accepted_sha256=sha(t),checkpoint_path=str(last),
 checkpoint_sha256=sha(last),full_state_path=str(full),full_state_sha256=sha(full)),
 teacher_eval_dependency=dict(accepted_path=str(e),accepted_sha256=sha(e),teacher_checkpoint_sha256=sha(last),
 image_count=5000,classes=80),
 calibration_dependency=dict(accepted_path=str(c),accepted_sha256=sha(c),release_sha256=cr['release_sha256'],
 source_root=str(source),phase_spec=spec,artifact_files=selected,eligible_person_image_count=cr['eligible_person_image_count'],
 optimizer_steps=cr['original_fourier_optimizer_steps'],clean_train_manifest_sha256=cr['accepted_clean_train_manifest_sha256']),
 fourier_dependency=dict(accepted_path=str(f),accepted_sha256=sha(f),params_path=str(params),
 params_sha256=sha(params),optimizer_steps=fr['optimizer_steps'],train_manifest_sha256=fr['train_manifest_sha256'],
 calibration_receipt_sha256=fr['calibration_receipt_sha256']),controller=ident)))
'''


def services():
    from .control import services as existing
    return existing()


def candidate_name(tag):
    if not re.fullmatch(r'coco_export25k_[a-z0-9_]{1,40}', tag):
        raise ValueError('Use a unique explicit coco_export25k_... identifier')
    return tag


def current_source():
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty:
        raise ValueError('Tracked source must be committed and clean before export validation')
    head = commit()
    remote = subprocess.check_output(['git', '-C', str(ROOT), 'ls-remote', 'origin',
        'refs/heads/codex/s4-coco-minitrain25k-generalization'], text=True).split()
    if not remote or remote[0] != head:
        raise ValueError('Current export candidate must be published and remotely verified')
    return head


def read_dependency(transport):
    active = OWNED/'repos'/WORKFLOW
    from .phase_fourier import CALIBRATION_ARTIFACT_SUFFIXES
    result = transport.run([PYTHON, '-B', '-c', READ_DEPENDENCY, WORKFLOW_OUTPUT.as_posix(), WORKFLOW,
                            'ee0b4256fdd344c5a275e6d15efc78cb7636d6e1', OWNED.as_posix(),
                            json.dumps(CALIBRATION_ARTIFACT_SUFFIXES), active.as_posix()])
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-2500:])
    return json.loads(result.stdout)


def build(tag, head, observation):
    if observation['state'] != 'READY_FOR_ISOLATED_PHASE_VALIDATION':
        raise ValueError('Accepted Original S4 predecessor phases/fresh capacity are not ready')
    tag = candidate_name(tag)
    native = load(EX/'ACTUAL_TEACHER_NATIVE_SOURCE_AUDIT_20261025.json')
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
    selected = [name for name in names if name.startswith(('coco_s4/', 'ue_framework/', 'configs/')) or name in
                ('docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json', MAIN_CONFIG,
                 TFSS_LOCK)]
    required = ('coco_s4/phase_export.py', 'coco_s4/phase_fourier.py', 'coco_s4/phase_calibration.py',
                'coco_s4/export_release.py', 'coco_s4/configs/export25k.json', MAIN_CONFIG, TFSS_LOCK,
                'ue_framework/methods/s4_tfss.py', 'ue_framework/methods/tfss_spatial.py',
                'ue_framework/methods/tfss_clean_maps.py', 'ue_framework/methods/tfss_nt_sensitivity.py',
                'ue_framework/methods/s4_pixel_audit.py')
    if any(name not in selected for name in required):
        raise ValueError('Export source/config/TFSS closure is incomplete')
    teacher = observation['teacher_dependency']
    previous = observation['calibration_dependency']
    fourier = observation['fourier_dependency']
    release = dict(schema='s4.coco.release.v1', phase='export25k', workflow_run_id=WORKFLOW,
                   source_commit=head, scientific_parent=PARENT, victim_parent=VICTIM_PARENT,
                   files={name: sha(ROOT/name) for name in selected},
                   data_root='/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023',
                   data_files=DATA_FILES, native_sources={name: row['sha256'] for name, row in native['files'].items()},
                   versions=native['versions'], config_path=MAIN_CONFIG,
                   phase_config_path='coco_s4/configs/export25k.json',
                   teacher_dependency=teacher,
                   teacher_eval_dependency=observation['teacher_eval_dependency'],
                   calibration_dependency=dict(accepted_path=previous['accepted_path'],
                       accepted_sha256=previous['accepted_sha256'], release_sha256=previous['release_sha256'],
                       source_root=previous['source_root'], artifact_files=previous['artifact_files'],
                       eligible_person_image_count=previous['eligible_person_image_count'],
                       optimizer_steps=previous['optimizer_steps'],
                       clean_train_manifest_sha256=previous['clean_train_manifest_sha256']),
                   fourier_dependency=dict(accepted_path=fourier['accepted_path'],
                       accepted_sha256=fourier['accepted_sha256'], params_path=fourier['params_path'],
                       params_sha256=fourier['params_sha256'], optimizer_steps=fourier['optimizer_steps'],
                       train_manifest_sha256=fourier['train_manifest_sha256'],
                       calibration_receipt_sha256=fourier['calibration_receipt_sha256']))
    validate_phase_config(load(ROOT/'coco_s4/configs/export25k.json'))
    local = EX/tag/'source_release.json'
    write_once(local, release)
    template = load(EX/(WORKFLOW+'.json'))
    repo = (OWNED/'releases'/tag).as_posix(); output = (OWNED/'validation'/tag).as_posix()
    remote_release = anchor_path(tag)
    template.update(run_id=tag, project='s4-coco-export25k-validation')
    template['source'] = dict(repo_root=str(ROOT), branch='codex/s4-coco-minitrain25k-generalization',
                              commit=head, allowed_post_commit_paths=['execution/', 'Ultralytics/'])
    template['remote'].update(repo_root=repo, output_root=output, stage_root=(OWNED/'stage'/tag).as_posix(),
                              control_root=(OWNED/'control'/tag).as_posix())
    validate_anchor_outside_lifecycle(remote_release, template['remote'])
    template['session']['name'] = tag
    unit = 's4coco-export-'+hashlib.sha256(tag.encode()).hexdigest()[:24]
    phase_args = [PYTHON, '-B', '-m', 'coco_s4.phase_export', '--release', remote_release,
                  '--output', output+'/result', '--smoke-images', '1']
    template['workload'] = dict(cwd='.', argv=['systemd-run', '--user', '--scope', '--quiet', '--unit='+unit,
        PYTHON, '-B', '-m', 'coco_s4.process', '--job', output+'/supervision', '--cwd', repo, '--', *phase_args])
    template['anchors'] = [dict(name='phase_release', local_path=str(local), remote_path=remote_release,
                                 sha256=sha(local))]
    adapter = "import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')!='FAILED',complete=s.get('state')=='passed',progress=s,artifacts=['result/status.json','result/smoke_summary.json'])))"
    for health in template['health'].values():
        health['adapter_argv'] = [PYTHON, '-B', '-c', adapter]
        health['gpu_utilization_policy'] = 'advisory'
        health['timeout_seconds'] = 900
    template['artifacts'] = [dict(path='result/'+name, required=True) for name in
                             ('smoke_summary.json', 'phase_spec.json', 'status.json')]
    template['local_pull_root'] = str(EX/'p')
    template['metadata'] = dict(execution_purpose='isolated_export25k_pre_review_smoke',
        computation_kind='poisoned-export', candidate_commit=head, cleanup_policy='retain_all',
        official_metrics_disabled=True, artifact_ingest_disabled=True, validation_used=False,
        parameter_updates=False, actual_training_images=1, raw_forward_maximum=100,
        no_formal_phase_registration=True, no_second_controller=True, no_parameter_or_AP_tuning=True,
        expected_device='cuda:0')
    from remote_run_control.models import RunSpec
    RunSpec.from_dict(template)
    spec = EX/(tag+'.json')
    write_once(spec, template)
    write_once(EX/tag/'dependency_observation.json', observation)
    return spec


def prepare(tag):
    tag = candidate_name(tag)
    if (EX/(tag+'.json')).exists() or (EX/'rrctl_state'/tag).exists() or (EX/tag).exists():
        raise FileExistsError('Prepared/bound export candidate exists: inspect only')
    _, transport = services()
    observation = read_dependency(transport)
    if observation['state'] != 'READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation)); return
    spec = build(tag, current_source(), observation)
    print(json.dumps(dict(prepared=str(spec), candidate_commit=commit(), launched=False, formal_registered=False)))


def launch_smoke(tag):
    tag = candidate_name(tag); spec_path = EX/(tag+'.json')
    if not spec_path.is_file() or (EX/'rrctl_state'/tag).exists() or (EX/tag/'launch_intent.json').exists():
        raise FileExistsError('Missing prepared export candidate or existing bound smoke; never relaunch')
    spec = load(spec_path)
    if current_source() != spec['source']['commit']:
        raise ValueError('Prepared export source changed')
    controller, transport = services()
    observation = read_dependency(transport)
    if observation['state'] != 'READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation)); return
    release = load(EX/tag/'source_release.json')
    for key in ('accepted_sha256', 'params_sha256', 'optimizer_steps', 'train_manifest_sha256'):
        if observation['fourier_dependency'][key] != release['fourier_dependency'][key]:
            raise ValueError('Prepared export is bound to different accepted Fourier evidence')
    remote_paths = [spec['remote'][key] for key in ('repo_root', 'output_root', 'stage_root', 'control_root')]
    checked = transport.run([PYTHON, '-B', '-c',
        'import json,os,sys;from pathlib import Path;assert os.getuid()==1014;assert all(not Path(p).exists() for p in json.loads(sys.argv[1]));print("absent")',
        json.dumps(remote_paths)])
    if checked.returncode:
        raise ValueError('Existing remote export candidate paths: no launch')
    from remote_run_control.models import RunSpec
    run_spec = RunSpec.from_path(spec_path); ready = controller.ready(run_spec)
    write_once(EX/tag/'ready.json', ready)
    if not ready.get('ready'):
        raise ValueError('RRCTL thin readiness rejected current export smoke')
    temp = EX/'rrctl_temp'; temp.mkdir(exist_ok=True)
    os.environ['TMP'] = os.environ['TEMP'] = str(temp)
    write_once(EX/tag/'launch_intent.json', dict(UTC=now(), run_id=tag, source_commit=spec['source']['commit']))
    result = controller.launch(run_spec)
    write_once(EX/tag/'launch.json', result)
    print(json.dumps(dict(launched_once=True, run_id=tag, result=result, formal_registered=False)))


def register(tag, review_path):
    tag=candidate_name(tag);spec_path=EX/(tag+'.json')
    if not spec_path.is_file() or not (EX/'rrctl_state'/tag).exists():
        raise ValueError('No RRCTL-bound export CUDA smoke')
    spec=load(spec_path)
    if current_source()!=spec['source']['commit']:
        raise ValueError('Export review source differs from the published candidate')
    summary_path=Path(spec['local_pull_root'])/tag/'result'/'smoke_summary.json'
    if not summary_path.is_file():raise ValueError('RRCTL has not pulled export smoke evidence')
    smoke=load(summary_path);release_path=EX/tag/'source_release.json';release=load(release_path)
    cfg_sha=sha(ROOT/release['phase_config_path']);fourier_sha=release['fourier_dependency']['accepted_sha256']
    expected=dict(state='passed',phase='export25k',source_commit=spec['source']['commit'],
        release_sha256=sha(release_path),phase_config_sha256=cfg_sha,
        Fourier_receipt_sha256=fourier_sha,computation_kind='poisoned-export',finite_outputs=True,
        production_entrypoint_reached=True,decoded_budget_passed=True,labels_byte_identical=True,
        validation_used=False,parameter_updates=False,official_metrics_disabled=True,
        artifact_ingest_disabled=True,actual_device='cuda:0',actual_training_images=1)
    if any(smoke.get(k)!=v for k,v in expected.items()) or not 1<=smoke.get('production_forward_calls',0)<=100:
        raise ValueError('Export candidate lacks decoded 8/4/0, label identity, and bounded CUDA evidence')
    review=load(review_path)
    if review.get('Fourier_receipt_sha256')!=fourier_sha:
        raise ValueError('Export review is not bound to the accepted Fourier receipt')
    _,transport=services();observation=read_dependency(transport)
    if (observation.get('state')!='READY_FOR_ISOLATED_PHASE_VALIDATION'
            or observation.get('fourier_dependency',{}).get('accepted_sha256')!=fourier_sha):
        raise ValueError('Serial Fourier/capacity gate changed; no export registration')
    result=close_and_register(transport,tag=tag,phase='export25k',module='coco_s4.phase_export',
        spec=spec,release_path=release_path,smoke_path=summary_path,review_path=review_path,
        previous_stage_receipt_sha256=fourier_sha,closure_extra={'Fourier_receipt_sha256':fourier_sha})
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'launch-smoke','register'))
    parser.add_argument('--tag', required=True);parser.add_argument('--review',type=Path)
    args = parser.parse_args()
    if args.action=='register' and args.review is None:parser.error('register requires --review JSON')
    {'prepare': prepare, 'launch-smoke': launch_smoke,'register':register}[args.action](args.tag,*(([args.review] if args.action=='register' else [])))
