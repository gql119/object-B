"""Prepare/launch one isolated evaluation smoke after accepted Teacher200.

This is a finite RRCTL validation job, never a second training controller. It
does not register or launch the formal evaluation stage. That registration
requires the actual smoke and the one scientific review of the same release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .runtime import ROOT, DATA_ROOT, DATA_FILES, PARENT, VICTIM_PARENT, commit, load, now, sha, write_once
from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT, TEACHER_COMMIT, TEACHER_RELEASE
from .release_transport import anchor_path, close_and_register, validate_anchor_outside_lifecycle

EX=ROOT/'execution'
PYTHON='/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'

READ_DEPENDENCY=r'''
from pathlib import Path
import hashlib,json,os,sys
o=Path(sys.argv[1]);s=json.loads((o/'pipeline_state.json').read_text())
assert os.getuid()==1014 and s['run_id']==sys.argv[2] and s['source_commit']==sys.argv[3]
ident=s['controller'];p=Path('/proc')/str(ident['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split()
assert p.stat().st_uid==1014 and int(f[19])==ident['start_ticks'] and f[0]!='Z'
if (s['state']!='WAITING_PHASE_RELEASE' or s['current_stage']!='teacher_eval'
    or s['active_job'] is not None or s.get('active_child') is not None):
 print(json.dumps(dict(state='WAITING_TEACHER_DEPENDENCY',native_state=s['state'],current_stage=s['current_stage'])));sys.exit(0)
assert s['stages']['teacher200']['state']=='ACCEPTED'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
a=o/'teacher200/accepted.json';r=json.loads(a.read_text());last=o/'teacher200/train/weights/last.pt'
full=o/'teacher200/train/state_snapshots/epoch_200_full_state.pt'
assert r['state']=='ACCEPTED' and r['completed_epochs']==200 and r['source_commit']==sys.argv[3]
assert r['release_sha256']==sys.argv[4] and r['finite_loss'] and r['zero_validation_forwards']
assert sha(a)==s['stages']['teacher200']['receipt_sha256']
assert sha(last)==r['last_checkpoint_sha256'] and sha(full)==r['full_state_sha256']
sys.path.insert(0,sys.argv[5]);from coco_s4.serial import capacity
resources=capacity(o)
print(json.dumps(dict(state='READY_FOR_ISOLATED_PHASE_VALIDATION' if resources['admitted'] else 'WAITING_RESOURCES',
 capacity=resources,teacher_dependency=dict(accepted_path=str(a),accepted_sha256=sha(a),checkpoint_path=str(last),
 checkpoint_sha256=r['last_checkpoint_sha256'],full_state_path=str(full),full_state_sha256=r['full_state_sha256']),
 accepted_teacher=r,controller=ident)))
'''


def services():
    from .control import services as existing
    return existing()


def read_dependency(t):
    active=OWNED/'repos'/WORKFLOW
    r=t.run([PYTHON,'-B','-c',READ_DEPENDENCY,WORKFLOW_OUTPUT.as_posix(),WORKFLOW,TEACHER_COMMIT,
             TEACHER_RELEASE,active.as_posix()])
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-2500:])
    return json.loads(r.stdout)


def candidate_name(tag):
    if not re.fullmatch(r'coco_teacher_eval_[a-z0-9_]{1,40}',tag):
        raise ValueError('Use a unique explicit coco_teacher_eval_... identifier')
    return tag


def current_source():
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True):
        raise ValueError('Tracked source must be committed and clean before preparation')
    head=commit()
    remote=subprocess.check_output(['git','-C',str(ROOT),'ls-remote','origin',
                                    'refs/heads/codex/s4-coco-minitrain25k-generalization'],text=True).split()
    if not remote or remote[0]!=head:raise ValueError('Current candidate must be published and remotely verified')
    return head


def build(tag,head,observation):
    """Build only metadata; actual dependency and current source were read first."""
    if observation['state']!='READY_FOR_ISOLATED_PHASE_VALIDATION':
        raise ValueError('Teacher acceptance/fresh no-conflict capacity is still pending')
    tag=candidate_name(tag)
    actual=load(EX/'ACTUAL_TEACHER_NATIVE_SOURCE_AUDIT_20261025.json')
    evaluator=load(EX/'ISOLATED_EVALUATOR_CPU_VALIDATION_20261025.json')
    names=subprocess.check_output(['git','-C',str(ROOT),'ls-files'],text=True).splitlines()
    selected=[n for n in names if n.startswith(('coco_s4/','ue_framework/','configs/')) or n in
              ('docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json',
               'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml')]
    if any(n not in selected for n in ('coco_s4/phase_teacher_eval.py','coco_s4/phase_contract.py','coco_s4/configs/teacher_eval.json')):
        raise ValueError('Scientific evaluation entry/configuration must be committed')
    release=dict(schema='s4.coco.release.v1',phase='teacher_eval',workflow_run_id=WORKFLOW,source_commit=head,
                 scientific_parent=PARENT,victim_parent=VICTIM_PARENT,files={n:sha(ROOT/n) for n in selected},
                 data_root=DATA_ROOT.as_posix(),data_files=DATA_FILES,
                 native_sources={n:r['sha256'] for n,r in actual['files'].items()},versions=actual['versions'],
                 config_path='docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml',
                 phase_config_path='coco_s4/configs/teacher_eval.json',teacher_dependency=observation['teacher_dependency'],
                 evaluator=dict(directory=evaluator['addon'],files=evaluator['files'],wheel_sha256=evaluator['wheel_sha256']))
    local=EX/tag/'source_release.json';write_once(local,release)
    template=load(EX/(WORKFLOW+'.json'))
    repo=(OWNED/'releases'/tag).as_posix();output=(OWNED/'validation'/tag).as_posix()
    remote_release=anchor_path(tag)
    template.update(run_id=tag,project='s4-coco-teacher-evaluation-validation')
    template['source']=dict(repo_root=str(ROOT),branch='codex/s4-coco-minitrain25k-generalization',commit=head,
                            allowed_post_commit_paths=['execution/','Ultralytics/'])
    template['remote'].update(repo_root=repo,output_root=output,stage_root=(OWNED/'stage'/tag).as_posix(),
                              control_root=(OWNED/'control'/tag).as_posix())
    validate_anchor_outside_lifecycle(remote_release, template['remote'])
    template['session']['name']=tag
    unit='s4coco-eval-'+hashlib.sha256(tag.encode()).hexdigest()[:24]
    template['workload']=dict(cwd='.',argv=['systemd-run','--user','--scope','--quiet','--unit='+unit,
        PYTHON,'-B','-m','coco_s4.process','--job',output+'/supervision','--cwd',repo,'--',
        PYTHON,'-B','-m','coco_s4.phase_teacher_eval','--release',remote_release,'--output',output+'/result',
        '--smoke-batches','2'])
    template['anchors']=[dict(name='phase_release',local_path=str(local),remote_path=remote_release,sha256=sha(local))]
    adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')!='FAILED',complete=s.get('state')=='passed',progress=s,artifacts=['result/status.json','result/smoke_summary.json'])))"
    for h in template['health'].values():
        h['adapter_argv']=[PYTHON,'-B','-c',adapter];h['gpu_utilization_policy']='advisory';h['timeout_seconds']=600
    template['artifacts']=[dict(path='result/'+n,required=True) for n in
                           ('smoke_summary.json','prediction_audit.json','phase_spec.json')]
    template['local_pull_root']=str(EX/tag/'small_pull')
    template['metadata']=dict(execution_purpose='isolated_teacher_eval_pre_review_smoke',computation_kind='inference',
        candidate_commit=head,cleanup_policy='retain_all',official_metrics_disabled=True,artifact_ingest_disabled=True,
        selected_inputs='first72sorted officialval2017 IDs',inference_batches=4,raw_forwards_including_warmups_maximum=100,
        one_shot_no_scheduler=True,no_formal_phase_registration=True)
    from remote_run_control.models import RunSpec
    RunSpec.from_dict(template)
    spec=EX/(tag+'.json');write_once(spec,template)
    write_once(EX/tag/'dependency_observation.json',observation)
    return spec


def prepare(tag):
    tag=candidate_name(tag)
    if (EX/(tag+'.json')).exists() or (EX/'rrctl_state'/tag).exists() or (EX/tag).exists():
        raise FileExistsError('Prepared/bound candidate exists: inspect only')
    _,t=services();observation=read_dependency(t)
    if observation['state']!='READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation));return
    head=current_source();spec=build(tag,head,observation)
    print(json.dumps(dict(prepared=str(spec),candidate_commit=head,launched=False,formal_registered=False)))


def launch_smoke(tag):
    tag=candidate_name(tag);spec_path=EX/(tag+'.json')
    if (EX/'rrctl_state'/tag).exists() or (EX/tag/'launch_intent.json').exists():
        raise FileExistsError('Existing bound smoke is inspect-only; never relaunch')
    cfg=load(spec_path)
    if current_source()!=cfg['source']['commit']:raise ValueError('Prepared candidate changed')
    ctl,t=services();observation=read_dependency(t)
    if observation['state']!='READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation));return
    release=load(EX/tag/'source_release.json')
    if observation['teacher_dependency']!=release['teacher_dependency']:
        raise ValueError('Prepared accepted Teacher dependency changed')
    paths=[cfg['remote'][k] for k in ('repo_root','output_root','stage_root','control_root')]
    r=t.run([PYTHON,'-B','-c',
             'import json,os,sys;from pathlib import Path;assert os.getuid()==1014;assert all(not Path(p).exists() for p in json.loads(sys.argv[1]));print("absent")',
             json.dumps(paths)])
    if r.returncode:raise ValueError('Existing remote candidate paths: no launch')
    from remote_run_control.models import RunSpec
    spec=RunSpec.from_path(spec_path);ready=ctl.ready(spec)
    write_once(EX/tag/'ready.json',ready)
    if not ready.get('ready'):raise ValueError('RRCTL thin readiness rejected current smoke')
    temp=EX/'rrctl_temp';temp.mkdir(exist_ok=True);os.environ['TMP']=os.environ['TEMP']=str(temp)
    write_once(EX/tag/'launch_intent.json',dict(UTC=now(),run_id=tag,source_commit=cfg['source']['commit']))
    result=ctl.launch(spec);write_once(EX/tag/'launch.json',result)
    print(json.dumps(dict(launched_once=True,run_id=tag,result=result,formal_registered=False)))


def register(tag, review_path):
    """Register the exact reviewed fixed-last Teacher assessment candidate."""
    tag=candidate_name(tag);spec_path=EX/(tag+'.json')
    if not spec_path.is_file() or not (EX/'rrctl_state'/tag).exists():
        raise ValueError('No bound RRCTL Teacher-evaluation smoke')
    spec=load(spec_path)
    if current_source()!=spec['source']['commit']:
        raise ValueError('Review source is no longer the prepared published candidate')
    summary_path=Path(spec['local_pull_root'])/'result'/'smoke_summary.json'
    if not summary_path.is_file():
        raise ValueError('RRCTL has not pulled the completed inference smoke')
    smoke=load(summary_path);release_path=EX/tag/'source_release.json';release=load(release_path)
    cfg_sha=sha(ROOT/release['phase_config_path'])
    baseline=smoke.get('baseline_equivalence_probe',{})
    if (smoke.get('state')!='passed' or smoke.get('phase')!='teacher_eval'
            or smoke.get('source_commit')!=spec['source']['commit']
            or smoke.get('release_sha256')!=sha(release_path)
            or smoke.get('phase_config_sha256')!=cfg_sha
            or smoke.get('teacher_checkpoint_sha256')!=release['teacher_dependency']['checkpoint_sha256']
            or smoke.get('official_metrics_disabled') is not True
            or smoke.get('artifact_ingest_disabled') is not True
            or smoke.get('parameter_updates') is not False
            or smoke.get('checkpoint_selection_allowed') is not False
            or smoke.get('method_statistics_allowed') is not False
            or smoke.get('finite_outputs') is not True
            or not 1<=smoke.get('actual_inference_batches',0)<=100
            or not 1<=smoke.get('total_raw_forwards_including_warmup',0)<=100
            or baseline.get('raw_outputs_exact') is not True
            or baseline.get('decoded_predictions_exact') is not True
            or baseline.get('absolute_difference')!=0 or baseline.get('tolerance')!=0):
        raise ValueError('Teacher-evaluation smoke failed its fixed-checkpoint/read-only contract')
    _,transport=services();observation=read_dependency(transport)
    if observation.get('state')!='READY_FOR_ISOLATED_PHASE_VALIDATION':
        raise ValueError('The existing controller/dependency/capacity gate changed; no registration')
    result=close_and_register(transport,tag=tag,phase='teacher_eval',module='coco_s4.phase_teacher_eval',
        spec=spec,release_path=release_path,smoke_path=summary_path,review_path=review_path,
        previous_stage_receipt_sha256=release['teacher_dependency']['accepted_sha256'])
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch-smoke','register']);p.add_argument('--tag',required=True);p.add_argument('--review',type=Path)
    a=p.parse_args()
    if a.action=='register' and a.review is None:p.error('register requires --review JSON')
    {'prepare':prepare,'launch-smoke':launch_smoke,'register':register}[a.action](a.tag,*(([a.review] if a.action=='register' else [])))
