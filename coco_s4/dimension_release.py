"""Finite RRCTL reconstruction/full-loss probes; no controller or formal launch."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .runtime import ROOT, DATA_ROOT, DATA_FILES, PARENT, VICTIM_PARENT, commit, load, now, sha, write_once
from .phase_contract import OWNED, WORKFLOW, LEGACY_WORKFLOW, ORIGIN_OUTPUT, WORKFLOW_OUTPUT, PREVIOUS_DIMENSION_OUTPUT
from .dimension_protocol import PROTOCOL, validate_receipt_dimension
from .release_transport import anchor_path,validate_anchor_outside_lifecycle,ensure_remote_exact

EX=ROOT/'execution'
PYTHON='/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'
BRANCH='codex/s4-coco80-dynamic-dimension-20261025'
MAIN='docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml'
PREFLIGHT=r'''
import json,os,sys
from pathlib import Path
assert os.getuid()==1014
old=Path(sys.argv[1]);state=json.loads((old/'pipeline_state.json').read_text())
assert state['state']=='FAILED' and state['current_stage']=='calibration'
phase=json.loads((old/'calibration/status.json').read_text())
assert phase['state']=='BLOCKED_PROTOCOL_COMPATIBILITY'
def live(ident):
 if not ident:return False
 p=Path('/proc')/str(ident['pid'])
 if not p.exists():return False
 a=(p/'stat').read_text().rsplit(')',1)[1].split()
 return int(a[19])==ident['start_ticks'] and p.stat().st_uid==ident['uid'] and a[0]!='Z'
assert not live(state['controller']) and not live(state.get('active_child'))
for item in ('teacher200','teacher_eval'):assert state['stages'][item]['state']=='ACCEPTED'
previous=Path(sys.argv[3]);ps=json.loads((previous/'pipeline_state.json').read_text())
assert ps['state']=='FAILED' and ps['current_stage']=='fourier24'
assert not live(ps['controller']) and not live(ps.get('active_child'))
assert ps['source_commit']=='34fee63a232d489ae54d61eb37a42483091c40d4'
assert json.loads((previous/'fourier24/status.json').read_text())['state']=='FAILED'
assert json.loads((previous/'jobs/fourier24/exit.json').read_text())['returncode']!=0
proof=dict(run_id=ps['run_id'],source_commit=ps['source_commit'],
 files={name:__import__('hashlib').sha256((previous/name).read_bytes()).hexdigest()
 for name in ('pipeline_state.json','fourier24/status.json','jobs/fourier24/exit.json')})
sys.path.insert(0,sys.argv[2]);from coco_s4.serial import capacity
resources=capacity(old)
# GPU utility/CPU load are telemetry. Capacity checks exactly match the existing controller.
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  if p.stat().st_uid!=1014:continue
  argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
  modules=[argv[i+1] for i,v in enumerate(argv[:-1]) if v=='-m']
  if any(v.startswith('coco_s4.phase_') or v in ('coco_s4.serial_recovery','coco_s4.dimension_rebuild') for v in modules):
   raise ValueError('Another own scientific process or controller is active')
 except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(dict(state='READY' if resources['admitted'] else 'WAITING_RESOURCES',capacity=resources,
 old_state_sha256=__import__('hashlib').sha256((old/'pipeline_state.json').read_bytes()).hexdigest(),
 old_controller_absent=True,old_child_absent=True,failed_dimension_predecessor=proof)))
'''


def services():
    from .control import services as existing
    return existing()


def current_source():
    if subprocess.check_output(['git','-C',str(ROOT),'diff','HEAD','--name-only']):
        raise ValueError('Commit all tracked candidate changes before preparation')
    head=commit()
    remote=subprocess.check_output(['git','-C',str(ROOT),'ls-remote','origin','refs/heads/'+BRANCH],text=True).split()
    if not remote or remote[0]!=head:raise ValueError('Exact current candidate is not published')
    return head


def check_tag(tag):
    if not re.fullmatch(r'coco_dim_[a-z0-9_]{1,40}',tag):raise ValueError('Unique coco_dim_... tag required')
    return tag


def preflight(transport):
    r=transport.run([PYTHON,'-B','-c',PREFLIGHT,ORIGIN_OUTPUT.as_posix(),(OWNED/'repos'/LEGACY_WORKFLOW).as_posix(),PREVIOUS_DIMENSION_OUTPUT.as_posix()])
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-3000:])
    return json.loads(r.stdout)


def release(head,phase):
    old=load(EX/'CALIBRATION_PROTOCOL_BLOCK_REMOTE_AUDIT_20261025.json')
    native=load(EX/'ACTUAL_TEACHER_NATIVE_SOURCE_AUDIT_20261025.json')
    files=subprocess.check_output(['git','-C',str(ROOT),'ls-files'],text=True).splitlines()
    names=[n for n in files if n.startswith(('coco_s4/','ue_framework/','configs/')) or n in
           (MAIN,'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json','docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json',
            'docs/S4_COCO_MINITRAIN_20261023/COCO_SAFEBG_DIMENSION_CPU_NUMERICS_20261025.json',
            'docs/S4_COCO_MINITRAIN_20261023/COCO_SAFEBG_DIMENSION_REGRESSION_20261025.json',
            'docs/S4_COCO_MINITRAIN_20261023/COCO_SAFEBG_PRECISION_CANDIDATE_CPU_20261025.json',
            'docs/S4_COCO_MINITRAIN_20261023/USER_DYNAMIC_DIMENSION_AUTHORIZATION_20261025.md')]
    return dict(schema='s4.coco.release.v1',phase=phase,source_commit=head,workflow_run_id=WORKFLOW,
                scientific_parent=PARENT,victim_parent=VICTIM_PARENT,dimension_protocol=PROTOCOL,
                files={n:sha(ROOT/n) for n in names},data_root=DATA_ROOT.as_posix(),data_files=DATA_FILES,
                native_sources={n:x['sha256'] for n,x in native['files'].items()},versions=native['versions'],
                config_path=MAIN,teacher_dependency=old['release']['teacher_dependency'],
                teacher_eval_dependency=old['release']['teacher_eval_dependency'])


def finite_spec(tag,manifest,kind):
    tag=check_tag(tag);head=manifest['source_commit']
    local=EX/tag/'source_release.json';write_once(local,manifest)
    template=load(EX/(LEGACY_WORKFLOW+'.json'))
    repo=(OWNED/'releases'/tag).as_posix();out=(OWNED/'validation'/tag).as_posix();anchor=anchor_path(tag)
    template.update(run_id=tag,project='s4-coco-dimension-validation')
    template['source']=dict(repo_root=str(ROOT),branch=BRANCH,commit=head,allowed_post_commit_paths=['execution/','Ultralytics/'])
    template['remote'].update(repo_root=repo,output_root=out,stage_root=(OWNED/'stage'/tag).as_posix(),control_root=(OWNED/'control'/tag).as_posix())
    validate_anchor_outside_lifecycle(anchor,template['remote'])
    module='coco_s4.dimension_rebuild' if kind=='rebuild' else 'coco_s4.phase_fourier'
    argv=[PYTHON,'-B','-m',module,'--release',anchor,'--output',out+'/result']
    if kind=='cpu':argv+=['--cpu-probe-images','16']
    if kind=='gpu':argv+=['--smoke-images','16']
    unit='s4coco-dim-'+hashlib.sha256(tag.encode()).hexdigest()[:24]
    template['workload']=dict(cwd='.',argv=['systemd-run','--user','--scope','--quiet','--unit='+unit,
        PYTHON,'-B','-m','coco_s4.process','--job',out+'/supervision','--cwd',repo,'--',*argv])
    template['session']['name']=tag
    template['anchors']=[dict(name='dimension_release',local_path=str(local),remote_path=anchor,sha256=sha(local))]
    adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')!='FAILED',complete=s.get('state')=='passed',progress=s,artifacts=['result/status.json'])))"
    for h in template['health'].values():
        h['adapter_argv']=[PYTHON,'-B','-c',adapter];h['gpu_utilization_policy']='advisory';h['timeout_seconds']=1800
    names=['status.json','candidate_receipt.json','phase_spec.json'] if kind=='rebuild' else [
        'status.json','phase_spec.json','diagnostics.csv','diagnostics.json',
        'per_loss_fourier_gradients.json','categorical_diagnostics.jsonl','TFSS_OPTIMIZATION_PROOF.json',
        'cpu_probe_summary.json' if kind=='cpu' else 'smoke_summary.json']
    template['artifacts']=[dict(path='result/'+n,required=True) for n in names]
    template['local_pull_root']=str(EX/'p')
    template['metadata']=dict(validation_kind=kind,computation_kind='orchestration' if kind=='rebuild' else 'fourier-gradient',
        candidate_commit=head,cleanup_policy='retain_all',official_metrics_disabled=True,artifact_ingest_disabled=True,
        no_second_controller=True,formal_registered=False,actual_production_step_limit=0 if kind=='rebuild' else 1)
    from remote_run_control.models import RunSpec
    RunSpec.from_dict(template)
    spec=EX/(tag+'.json');write_once(spec,template)
    return spec


def prepare(tag,kind,rebuild_tag=None):
    tag=check_tag(tag)
    if (EX/tag).exists() or (EX/(tag+'.json')).exists() or (EX/'rrctl_state'/tag).exists():
        raise FileExistsError('Existing candidate identity; inspect only')
    _,t=services();observation=preflight(t)
    if observation['state']!='READY':print(json.dumps(observation));return
    head=current_source();m=release(head,'dimension_rebuild' if kind=='rebuild' else 'fourier24')
    m['failed_dimension_predecessor']=observation['failed_dimension_predecessor']
    if kind=='rebuild':
        m.update(retained_calibration=load(EX/'COCO_DYNAMIC_DIMENSION_REUSE_AUDIT_20261025.json'),
                 artifact_relative_root='execution/generated_calibration/coco_dimension_template')
        if rebuild_tag:
            previous=load(EX/(check_tag(rebuild_tag)+'.json'))
            folder=Path(previous['local_pull_root'])/rebuild_tag/'result'
            if load(folder/'status.json')['state']!='passed':raise ValueError('Prior coordinate component did not pass')
            m['reuse_reconstructed_coordinates']=dict(
                receipt_path=previous['remote']['output_root']+'/result/candidate_receipt.json',
                receipt_sha256=sha(folder/'candidate_receipt.json'),
                phase_spec_path=previous['remote']['output_root']+'/result/phase_spec.json',
                phase_spec_sha256=sha(folder/'phase_spec.json'))
    else:
        rebuild_tag=check_tag(rebuild_tag)
        prev=load(EX/(rebuild_tag+'.json'));folder=Path(prev['local_pull_root'])/rebuild_tag/'result'
        receipt=load(folder/'candidate_receipt.json');status=load(folder/'status.json')
        if status.get('state')!='passed' or status.get('receipt_sha256')!=sha(folder/'candidate_receipt.json'):
            raise ValueError('Retained-data reconstruction is not independently pulled/validated')
        validate_receipt_dimension(receipt)
        phase_spec=load(folder/'phase_spec.json')
        old_release=load(EX/rebuild_tag/'source_release.json')
        ensure_remote_exact(t,EX/rebuild_tag/'source_release.json',prev['remote']['repo_root']+'/source_release.json')
        # Bind semantic consumers exactly even if only scaffolding changed later.
        for name in ('ue_framework/methods/safebg.py','ue_framework/methods/b2_semantic.py',
                     'ue_framework/methods/semantic_coordinates.py','ue_framework/methods/orth_reference_design.py'):
            if m['files'][name]!=old_release['files'][name]:raise ValueError('Reconstruction semantic consumer drift')
        remote_out=prev['remote']['output_root']+'/result'
        m.update(phase_config_path='coco_s4/configs/fourier24.json',dimension_reconstruction_validation_allowed=True,
                 canonical_safebg_reference=dict(
                     path=load(EX/'COCO_DYNAMIC_DIMENSION_REUSE_AUDIT_20261025.json')['source_root']+'/ue_framework/methods/safebg.py',
                     sha256=load(ROOT/'docs/S4_COCO_MINITRAIN_20261023/ORIGINAL_S4_METHOD_SOURCE_LOCK.json')['files']['ue_framework/methods/safebg.py']),
                 dimension_reconstruction_dependency=dict(receipt_path=remote_out+'/candidate_receipt.json',
                     receipt_sha256=sha(folder/'candidate_receipt.json'),phase_spec_path=remote_out+'/phase_spec.json'),
                 calibration_dependency=dict(accepted_path=(WORKFLOW_OUTPUT/'calibration/accepted.json').as_posix(),
                     accepted_sha256=sha(folder/'candidate_receipt.json'),release_sha256=receipt['release_sha256'],
                     source_root=phase_spec['source_root'],artifact_files=receipt['artifact_files'],
                     eligible_person_image_count=receipt['eligible_person_image_count'],
                     optimizer_steps=receipt['original_fourier_optimizer_steps'],
                     clean_train_manifest_sha256=receipt['accepted_clean_train_manifest_sha256']))
    spec=finite_spec(tag,m,kind);write_once(EX/tag/'preparation_capacity.json',observation)
    print(json.dumps(dict(prepared=str(spec),source_commit=head,kind=kind,launched=False)))


def launch(tag):
    tag=check_tag(tag);specpath=EX/(tag+'.json');s=load(specpath)
    if (EX/'rrctl_state'/tag).exists() or (EX/tag/'launch_intent.json').exists():
        raise FileExistsError('Previously bound candidate; no relaunch')
    if current_source()!=s['source']['commit']:raise ValueError('Prepared source differs from current published candidate')
    ctl,t=services();observation=preflight(t)
    if observation['state']!='READY':print(json.dumps(observation));return
    paths=[s['remote'][k] for k in ('repo_root','output_root','stage_root','control_root')]
    r=t.run([PYTHON,'-B','-c','import json,sys;from pathlib import Path;assert all(not Path(p).exists() for p in json.loads(sys.argv[1]))',json.dumps(paths)])
    if r.returncode:raise ValueError('Remote candidate path collision; no relaunch')
    from remote_run_control.models import RunSpec
    spec=RunSpec.from_path(specpath);ready=ctl.ready(spec);write_once(EX/tag/'ready.json',ready)
    if not ready.get('ready'):raise ValueError('RRCTL thin readiness failed')
    temp=EX/'rrctl_temp';temp.mkdir(exist_ok=True);os.environ['TMP']=os.environ['TEMP']=str(temp)
    write_once(EX/tag/'launch_intent.json',dict(UTC=now(),source_commit=s['source']['commit']))
    result=ctl.launch(spec);write_once(EX/tag/'launch.json',result)
    print(json.dumps(dict(tag=tag,launched_once=True,result=result,formal_registered=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','launch'));p.add_argument('--tag',required=True)
    p.add_argument('--kind',choices=('rebuild','cpu','gpu'));p.add_argument('--rebuild-tag')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.tag,a.kind,a.rebuild_tag)
    else:launch(a.tag)
