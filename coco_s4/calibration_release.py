"""Prepare one immutable calibration smoke after accepted Teacher evaluation.

This module never creates a second controller or registers the formal phase.
The current serial workflow remains the only owner of formal stage execution.
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

EX = ROOT/'execution'
PYTHON = '/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'
PHASE = 'calibration'
PREVIOUS = 'teacher_eval'


READ_DEPENDENCY = r'''
from pathlib import Path
import datetime,hashlib,json,os,sys
o=Path(sys.argv[1]);s=json.loads((o/'pipeline_state.json').read_text())
assert os.getuid()==1014 and s['run_id']==sys.argv[2] and s['source_commit']==sys.argv[3]
ident=s['controller'];p=Path('/proc')/str(ident['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split()
assert p.stat().st_uid==1014 and int(f[19])==ident['start_ticks'] and f[0]!='Z'
if s['state']!='WAITING_PHASE_RELEASE' or s['current_stage']!='calibration' or s.get('active_job') is not None or s.get('active_child') is not None:
 print(json.dumps(dict(state='WAITING_PREVIOUS_PHASE',native_state=s['state'],current_stage=s['current_stage'])));sys.exit(0)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
assert s['stages']['teacher200']['state']=='ACCEPTED' and s['stages']['teacher_eval']['state']=='ACCEPTED'
e=o/'teacher_eval/accepted.json';er=json.loads(e.read_text());t=o/'teacher200/accepted.json';tr=json.loads(t.read_text())
assert er['state']=='ACCEPTED' and er['phase']=='teacher_eval' and er['image_count']==5000 and er['classes']==80
assert er['teacher_checkpoint_sha256']==tr['last_checkpoint_sha256']
assert s['stages']['teacher_eval']['receipt_sha256']==sha(e)
assert er['parameter_updates'] is False and er['checkpoint_selection_allowed'] is False and er['method_statistics_allowed'] is False
assert tr['state']=='ACCEPTED' and tr['completed_epochs']==200 and tr['zero_validation_forwards'] is True
last=o/'teacher200/train/weights/last.pt';full=o/'teacher200/train/state_snapshots/epoch_200_full_state.pt'
assert sha(last)==tr['last_checkpoint_sha256'] and sha(full)==tr['full_state_sha256']
sys.path.insert(0,sys.argv[4]);from coco_s4.serial import capacity
r=capacity(o)
print(json.dumps(dict(state='READY_FOR_ISOLATED_PHASE_VALIDATION' if r['admitted'] else 'WAITING_RESOURCES',capacity=r,
 previous_dependency=dict(accepted_path=str(e),accepted_sha256=sha(e),teacher_checkpoint_sha256=sha(last),
 full_state_sha256=sha(full),teacher_accepted_sha256=sha(t)),controller=ident)))
'''


def services():
    from .control import services as existing
    return existing()


def candidate_name(tag):
    if not re.fullmatch(r'coco_calibration_[a-z0-9_]{1,40}', tag):
        raise ValueError('Use a unique explicit coco_calibration_... identifier')
    return tag


def read_dependency(transport):
    active=OWNED/'repos'/WORKFLOW
    result=transport.run([PYTHON,'-B','-c',READ_DEPENDENCY,WORKFLOW_OUTPUT.as_posix(),WORKFLOW,
                         'ee0b4256fdd344c5a275e6d15efc78cb7636d6e1',active.as_posix()])
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-2500:])
    return json.loads(result.stdout)


def current_source():
    dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True)
    if dirty:
        raise ValueError('Tracked source must be committed and clean before preparation')
    head=commit()
    remote=subprocess.check_output(['git','-C',str(ROOT),'ls-remote','origin',
                                    'refs/heads/codex/s4-coco-minitrain25k-generalization'],text=True).split()
    if not remote or remote[0]!=head:
        raise ValueError('Current candidate must be published and remotely verified')
    return head


def build(tag, head, observation):
    if observation['state']!='READY_FOR_ISOLATED_PHASE_VALIDATION':
        raise ValueError('Accepted Teacher-evaluation dependency/fresh capacity is not ready')
    tag=candidate_name(tag)
    native=load(EX/'ACTUAL_TEACHER_NATIVE_SOURCE_AUDIT_20261025.json')
    names=subprocess.check_output(['git','-C',str(ROOT),'ls-files'],text=True).splitlines()
    selected=[name for name in names if name.startswith(('coco_s4/','ue_framework/','configs/')) or name in
              ('docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json',
               'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml')]
    required=('coco_s4/phase_calibration.py','coco_s4/calibration.py','coco_s4/phase_contract.py',
              'coco_s4/configs/calibration.json',
              'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml')
    if any(name not in selected for name in required):
        raise ValueError('Calibration source/configuration closure is incomplete')
    prev=observation['previous_dependency']
    teacher=dict(accepted_path=str(WORKFLOW_OUTPUT/'teacher200/accepted.json'),
                 accepted_sha256=prev['teacher_accepted_sha256'],
                 checkpoint_path=str(WORKFLOW_OUTPUT/'teacher200/train/weights/last.pt'),
                 checkpoint_sha256=prev['teacher_checkpoint_sha256'],
                 full_state_path=str(WORKFLOW_OUTPUT/'teacher200/train/state_snapshots/epoch_200_full_state.pt'),
                 full_state_sha256=prev['full_state_sha256'])
    release=dict(schema='s4.coco.release.v1',phase=PHASE,workflow_run_id=WORKFLOW,source_commit=head,
                 scientific_parent=PARENT,victim_parent=VICTIM_PARENT,
                 files={name:sha(ROOT/name) for name in selected},data_root='/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023',
                 data_files=DATA_FILES,native_sources={name:record['sha256'] for name,record in native['files'].items()},
                 versions=native['versions'],config_path='docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml',
                 phase_config_path='coco_s4/configs/calibration.json',teacher_dependency=teacher,
                 teacher_eval_dependency=dict(accepted_path=prev['accepted_path'],accepted_sha256=prev['accepted_sha256'],
                                              teacher_checkpoint_sha256=prev['teacher_checkpoint_sha256'],
                                              image_count=5000,classes=80))
    local=EX/tag/'source_release.json'
    write_once(local,release)
    template=load(EX/(WORKFLOW+'.json'))
    repo=(OWNED/'releases'/tag).as_posix();output=(OWNED/'validation'/tag).as_posix()
    remote_release=repo+'/source_release.json'
    template.update(run_id=tag,project='s4-coco-calibration-validation')
    template['source']=dict(repo_root=str(ROOT),branch='codex/s4-coco-minitrain25k-generalization',commit=head,
                            allowed_post_commit_paths=['execution/','Ultralytics/'])
    template['remote'].update(repo_root=repo,output_root=output,stage_root=(OWNED/'stage'/tag).as_posix(),
                              control_root=(OWNED/'control'/tag).as_posix())
    template['session']['name']=tag
    unit='s4coco-cal-'+hashlib.sha256(tag.encode()).hexdigest()[:24]
    template['workload']=dict(cwd='.',argv=['systemd-run','--user','--scope','--quiet','--unit='+unit,
        PYTHON,'-B','-m','coco_s4.process','--job',output+'/supervision','--cwd',repo,'--',
        PYTHON,'-B','-m','coco_s4.phase_calibration','--release',remote_release,'--output',output+'/result',
        '--smoke-images','1'])
    template['anchors']=[dict(name='phase_release',local_path=str(local),remote_path=remote_release,sha256=sha(local))]
    adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state') not in ('FAILED','BLOCKED_PROTOCOL_COMPATIBILITY'),complete=s.get('state')=='passed',progress=s,artifacts=['result/status.json','result/smoke_summary.json'])))"
    for health in template['health'].values():
        health['adapter_argv']=[PYTHON,'-B','-c',adapter]
        health['gpu_utilization_policy']='advisory'
        health['timeout_seconds']=600
    template['artifacts']=[dict(path='result/'+name,required=True) for name in
                           ('smoke_summary.json','phase_spec.json','smoke_statistics/raw_statistics.pt',
                            'smoke_statistics/eligible_support_manifest.json')]
    template['local_pull_root']=str(EX/tag/'small_pull')
    template['metadata']=dict(execution_purpose='isolated_calibration_pre_review_smoke',
        computation_kind='gradient-calibration',candidate_commit=head,cleanup_policy='retain_all',
        official_metrics_disabled=True,artifact_ingest_disabled=True,validation_used=False,
        parameter_updates=False,selected_inputs='one sorted eligible Person0 official train image',
        actual_training_images=1,raw_forwards_maximum=100,no_formal_phase_registration=True,
        no_second_controller=True)
    from remote_run_control.models import RunSpec
    RunSpec.from_dict(template)
    spec=EX/(tag+'.json')
    write_once(spec,template)
    write_once(EX/tag/'dependency_observation.json',observation)
    return spec


def prepare(tag):
    tag=candidate_name(tag)
    if (EX/(tag+'.json')).exists() or (EX/'rrctl_state'/tag).exists() or (EX/tag).exists():
        raise FileExistsError('Prepared/bound calibration candidate exists: inspect only')
    _,transport=services()
    observation=read_dependency(transport)
    if observation['state']!='READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation));return
    spec=build(tag,current_source(),observation)
    print(json.dumps(dict(prepared=str(spec),candidate_commit=commit(),launched=False,formal_registered=False)))


def launch_smoke(tag):
    tag=candidate_name(tag)
    spec_path=EX/(tag+'.json')
    if not spec_path.is_file() or (EX/'rrctl_state'/tag).exists() or (EX/tag/'launch_intent.json').exists():
        raise FileExistsError('Missing prepared candidate or existing bound smoke; never relaunch')
    cfg=load(spec_path)
    if current_source()!=cfg['source']['commit']:
        raise ValueError('Prepared calibration candidate changed')
    ctl,transport=services()
    observation=read_dependency(transport)
    if observation['state']!='READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation));return
    release=load(EX/tag/'source_release.json')
    if (observation['previous_dependency']['accepted_sha256']!=release['teacher_eval_dependency']['accepted_sha256']
            or observation['previous_dependency']['teacher_checkpoint_sha256']!=release['teacher_dependency']['checkpoint_sha256']):
        raise ValueError('Prepared calibration Teacher dependency changed')
    paths=[cfg['remote'][key] for key in ('repo_root','output_root','stage_root','control_root')]
    result=transport.run([PYTHON,'-B','-c',
        'import json,os,sys;from pathlib import Path;assert os.getuid()==1014;assert all(not Path(p).exists() for p in json.loads(sys.argv[1]));print("absent")',
        json.dumps(paths)])
    if result.returncode:
        raise ValueError('Existing remote calibration candidate paths: no launch')
    from remote_run_control.models import RunSpec
    run_spec=RunSpec.from_path(spec_path)
    ready=ctl.ready(run_spec)
    write_once(EX/tag/'ready.json',ready)
    if not ready.get('ready'):
        raise ValueError('RRCTL thin readiness rejected current calibration smoke')
    temp=EX/'rrctl_temp';temp.mkdir(exist_ok=True)
    os.environ['TMP']=os.environ['TEMP']=str(temp)
    write_once(EX/tag/'launch_intent.json',dict(UTC=now(),run_id=tag,source_commit=cfg['source']['commit']))
    launched=ctl.launch(run_spec)
    write_once(EX/tag/'launch.json',launched)
    print(json.dumps(dict(launched_once=True,run_id=tag,result=launched,formal_registered=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','launch-smoke'))
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    {'prepare':prepare,'launch-smoke':launch_smoke}[args.action](args.tag)
