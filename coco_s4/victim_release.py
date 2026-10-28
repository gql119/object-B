"""Prepare one bounded candidate for a matched COCO victim phase.

The existing serial workflow remains the sole formal controller. This helper
can only make an immutable release and launch one isolated smoke per unique tag.
It never launches a formal victim or writes a READY receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT
from .release_transport import anchor_path, close_and_register, validate_anchor_outside_lifecycle
from .phase_victim import PHASES, validate_phase_config
from .runtime import ROOT, DATA_FILES, PARENT, VICTIM_PARENT, commit, load, now, sha, write_once

EX = ROOT/'execution'
PYTHON = '/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'

READ_DEPENDENCY = r'''
from pathlib import Path
import hashlib,json,os,sys
sys.path.insert(0,sys.argv[6]);from coco_s4.dimension_protocol import validate_receipt_dimension
o=Path(sys.argv[1]);phase=sys.argv[2];run=sys.argv[3];source=sys.argv[4];owned=Path(sys.argv[5]).resolve()
s=json.loads((o/'pipeline_state.json').read_text())
assert os.getuid()==1014 and s['run_id']==run and s['source_commit']==source
ident=s['controller'];p=Path('/proc')/str(ident['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split()
assert p.stat().st_uid==1014 and int(f[19])==ident['start_ticks'] and f[0]!='Z'
if s.get('state')!='WAITING_PHASE_RELEASE' or s.get('current_stage')!=phase or s.get('active_job') is not None or s.get('active_child') is not None:
 print(json.dumps(dict(state='WAITING_SERIAL_SLOT',native_state=s.get('state'),current_stage=s.get('current_stage'),active_job=s.get('active_job'))));sys.exit(0)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
required=['teacher200','teacher_eval','calibration','fourier24','export25k']
if phase=='s4_200':required.append('clean200')
deps={}
for name in required:
 stage=s.get('stages',{}).get(name,{})
 if stage.get('state')!='ACCEPTED':raise ValueError('Serial predecessor is not accepted: '+name)
 accepted=o/name/'accepted.json';r=json.loads(accepted.read_text());actual=sha(accepted)
 if stage.get('receipt_sha256')!=actual or r.get('state')!='ACCEPTED':raise ValueError('Accepted predecessor receipt drift: '+name)
 row=dict(accepted_path=str(accepted),accepted_sha256=actual)
 if name=='teacher200':
  last=o/'teacher200/train/weights/last.pt';full=o/'teacher200/train/state_snapshots/epoch_200_full_state.pt'
  assert r.get('completed_epochs')==200 and r.get('zero_validation_forwards') is True
  assert sha(last)==r['last_checkpoint_sha256'] and sha(full)==r['full_state_sha256']
  row.update(checkpoint_path=str(last),checkpoint_sha256=sha(last),full_state_path=str(full),full_state_sha256=sha(full))
 elif name=='teacher_eval':
  assert r.get('image_count')==5000 and r.get('classes')==80 and r.get('parameter_updates') is False
  assert r.get('checkpoint_selection_allowed') is False and r.get('method_statistics_allowed') is False
  row.update(teacher_checkpoint_sha256=r['teacher_checkpoint_sha256'])
 elif name=='calibration':
  validate_receipt_dimension(r)
  assert r.get('images')==25000 and r.get('selected_channel_count')==len(r.get('selected_channels',[])) and r.get('dimension_protocol')=='coco80-strict-tsfl-selected-dimension-v1' and r.get('validation_used') is False
  assert r.get('parameter_updates') is False and r.get('checkpoint_selection_allowed') is False
 elif name=='fourier24':
  assert r.get('epochs_completed')==24 and r.get('validation_used') is False
  row.update(Fourier_params_sha256=r.get('Fourier_params_sha256'))
 elif name=='export25k':
  assert r.get('image_count')==25000 and r.get('labels_byte_identical') is True
  assert r.get('decoded_budget_passed') is True and r.get('validation_used') is False
  m=o/'export25k/poisoned_manifest.jsonl';assert sha(m)==r['manifest_sha256']
  row.update(manifest_path=str(m),manifest_sha256=sha(m),original_train_manifest_sha256=r['original_train_manifest_sha256'])
 elif name=='clean200':
  assert r.get('completed_epochs')==200 and r.get('zero_validation_forwards') is True
  assert r.get('pretrained') is False and r.get('initial_state_sha256')
  row.update(checkpoint_path=str(o/'clean200/train/weights/last.pt'),
             checkpoint_sha256=r['primary_checkpoint_sha256'],initial_state_sha256=r['initial_state_sha256'])
 deps[name]=row
sys.path.insert(0,str(Path(sys.argv[6]).resolve()))
from coco_s4.serial import capacity
cap=capacity(o)
print(json.dumps(dict(state='READY_FOR_ISOLATED_PHASE_VALIDATION' if cap.get('admitted') else 'WAITING_RESOURCES',
 phase=phase,dependencies=deps,capacity=cap,controller=ident)))
'''


def services():
    from .control import services as existing
    return existing()


def candidate_name(phase, tag):
    prefix = 'coco_victim_clean200_' if phase == 'clean200' else 'coco_victim_s4_200_'
    if phase not in PHASES or not tag.startswith(prefix) or not re.fullmatch(re.escape(prefix)+r'[a-z0-9_]{1,36}', tag):
        raise ValueError('Use one unique phase-specific coco_victim_... candidate tag')
    return tag


def current_source():
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty:
        raise ValueError('Tracked source must be committed and clean before victim validation')
    head = commit()
    remote = subprocess.check_output(['git', '-C', str(ROOT), 'ls-remote', 'origin',
        'refs/heads/codex/s4-coco-risk-fallback-20261027'], text=True).split()
    if not remote or remote[0] != head:
        raise ValueError('Current matched-victim candidate must be published and remotely verified')
    return head


def read_dependency(transport, phase):
    from .workflow_admission import local_controller_binding
    controller = local_controller_binding()
    active = Path(controller['remote_repo_root'])
    return_code = transport.run([PYTHON, '-B', '-c', READ_DEPENDENCY, WORKFLOW_OUTPUT.as_posix(), phase,
                                 WORKFLOW, controller['source_commit'], OWNED.as_posix(),
                                 active.as_posix()])
    if return_code.returncode:
        raise RuntimeError(return_code.stderr.decode(errors='replace')[-3000:])
    return json.loads(return_code.stdout)


def build(phase, tag, head, observation, *, smoke_batches=2):
    if observation.get('state') != 'READY_FOR_ISOLATED_PHASE_VALIDATION':
        raise ValueError('Serial predecessor chain/fresh no-conflict capacity is not ready')
    tag = candidate_name(phase, tag)
    if isinstance(smoke_batches, bool) or not isinstance(smoke_batches, int) or not 1 <= smoke_batches <= 100:
        raise ValueError('Victim probe budget must be 1..100 native batches')
    native = load(EX/'ACTUAL_TEACHER_NATIVE_SOURCE_AUDIT_20261025.json')
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
    selected = [n for n in names if n.startswith(('coco_s4/', 'ue_framework/', 'configs/')) or n in
                ('docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json',
                 'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml')]
    config_path = f'coco_s4/configs/{phase}.json'
    config = load(ROOT/config_path)
    validate_phase_config(config, phase)
    required = ('coco_s4/phase_victim.py', 'coco_s4/teacher.py', 'coco_s4/phase_teacher.py',
                'coco_s4/phase_contract.py', 'coco_s4/configs/'+phase+'.json',
                'ue_framework/s4_victim.py', 'ue_framework/methods/s4_tfss.py',
                'docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json',
                'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml')
    if any(n not in selected for n in required):
        raise ValueError('Victim native/source/config closure is incomplete')
    deps = observation['dependencies']
    teacher = deps['teacher200']
    release = dict(schema='s4.coco.release.v1', phase=phase, workflow_run_id=WORKFLOW,
                   source_commit=head, scientific_parent=PARENT, victim_parent=VICTIM_PARENT,
                   files={name: sha(ROOT/name) for name in selected},
                   data_root='/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023',
                   data_files=DATA_FILES, native_sources={name: row['sha256'] for name,row in native['files'].items()},
                   versions=native['versions'], config_path='docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml',
                   phase_config_path=config_path, phase_config=config,
                   teacher_dependency=dict(accepted_path=teacher['accepted_path'],accepted_sha256=teacher['accepted_sha256'],
                     checkpoint_path=teacher['checkpoint_path'],checkpoint_sha256=teacher['checkpoint_sha256'],
                     full_state_path=teacher['full_state_path'],full_state_sha256=teacher['full_state_sha256']),
                   phase_dependencies=deps)
    from .workflow_admission import attach
    attach(release)
    local = EX/tag/'source_release.json'; write_once(local, release)
    template = load(EX/(WORKFLOW+'.json'))
    repo = (OWNED/'releases'/tag).as_posix(); output = (OWNED/'validation'/tag).as_posix()
    remote_release = anchor_path(tag)
    template.update(run_id=tag, project='s4-coco-'+phase+'-validation')
    template['source'] = dict(repo_root=str(ROOT), branch='codex/s4-coco-risk-fallback-20261027',
                              commit=head, allowed_post_commit_paths=['execution/','Ultralytics/'])
    template['remote'].update(repo_root=repo, output_root=output,
                              stage_root=(OWNED/'stage'/tag).as_posix(), control_root=(OWNED/'control'/tag).as_posix())
    validate_anchor_outside_lifecycle(remote_release, template['remote'])
    template['session']['name'] = tag
    unit = 's4coco-victim-'+hashlib.sha256(tag.encode()).hexdigest()[:24]
    template['workload'] = dict(cwd='.', argv=['systemd-run','--user','--scope','--quiet','--unit='+unit,
        PYTHON,'-B','-m','coco_s4.process','--job',output+'/supervision','--cwd',repo,'--',
        PYTHON,'-B','-m','coco_s4.phase_victim','--release',remote_release,'--output',output+'/result',
        '--smoke-batches',str(smoke_batches)])
    template['anchors'] = [dict(name='phase_release',local_path=str(local),remote_path=remote_release,sha256=sha(local))]
    adapter = "import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')!='FAILED',complete=s.get('state')=='passed',progress=s,artifacts=['result/status.json','result/smoke_summary.json'])))"
    for health in template['health'].values():
        health['adapter_argv']=[PYTHON,'-B','-c',adapter]
        health['gpu_utilization_policy']='advisory'; health['timeout_seconds']=900
    template['artifacts']=[dict(path='result/'+n,required=True) for n in
                           ('smoke_summary.json','baseline_equivalence.json','phase_spec.json','status.json')]
    template['local_pull_root'] = str(EX/'p')
    template['metadata'] = dict(execution_purpose='isolated_matched_victim_pre_review_smoke',
        computation_kind='native-train-only-victim',candidate_commit=head,cleanup_policy='retain_all',
        official_metrics_disabled=True,artifact_ingest_disabled=True,validation_used=False,
        actual_native_batches=smoke_batches,actual_updates_required=True,no_formal_phase_registration=True,
        no_second_controller=True,checkpoint_selection_allowed=False)
    from remote_run_control.models import RunSpec
    RunSpec.from_dict(template)
    spec = EX/(tag+'.json'); write_once(spec,template)
    write_once(EX/tag/'dependency_observation.json',observation)
    return spec


def prepare(phase, tag, smoke_batches=2):
    candidate_name(phase, tag)
    if (EX/(tag+'.json')).exists() or (EX/'rrctl_state'/tag).exists() or (EX/tag).exists():
        raise FileExistsError('Prepared/bound victim smoke exists; inspect only')
    _,transport = services(); observation=read_dependency(transport,phase)
    if observation.get('state') != 'READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation)); return
    spec=build(phase,tag,current_source(),observation,smoke_batches=smoke_batches)
    print(json.dumps(dict(prepared=str(spec),candidate_commit=commit(),launched=False,formal_registered=False)))


def launch_smoke(phase, tag):
    candidate_name(phase,tag); spec_path=EX/(tag+'.json')
    if not spec_path.is_file() or (EX/'rrctl_state'/tag).exists() or (EX/tag/'launch_intent.json').exists():
        raise FileExistsError('Missing prepared victim candidate or existing bound smoke; never relaunch')
    spec=load(spec_path)
    if current_source()!=spec['source']['commit']:
        raise ValueError('Prepared victim candidate source changed')
    controller,transport=services(); observation=read_dependency(transport,phase)
    if observation.get('state')!='READY_FOR_ISOLATED_PHASE_VALIDATION':
        print(json.dumps(observation));return
    release=load(EX/tag/'source_release.json')
    if observation['dependencies']!=release['phase_dependencies']:
        raise ValueError('Prepared victim accepted dependencies changed')
    remote_paths=[spec['remote'][k] for k in ('repo_root','output_root','stage_root','control_root')]
    check=transport.run([PYTHON,'-B','-c',
        'import json,os,sys;from pathlib import Path;assert os.getuid()==1014;assert all(not Path(p).exists() for p in json.loads(sys.argv[1]));print("absent")',
        json.dumps(remote_paths)])
    if check.returncode: raise ValueError('Remote candidate paths already exist; no launch')
    from remote_run_control.models import RunSpec
    run_spec=RunSpec.from_path(spec_path);ready=controller.ready(run_spec);write_once(EX/tag/'ready.json',ready)
    if not ready.get('ready'): raise ValueError('RRCTL thin readiness rejected current victim smoke')
    temp=EX/'rrctl_temp';temp.mkdir(exist_ok=True);os.environ['TMP']=os.environ['TEMP']=str(temp)
    write_once(EX/tag/'launch_intent.json',dict(UTC=now(),run_id=tag,source_commit=spec['source']['commit']))
    result=controller.launch(run_spec);write_once(EX/tag/'launch.json',result)
    print(json.dumps(dict(launched_once=True,run_id=tag,result=result,formal_registered=False)))


def register(phase, tag, review_path):
    """Close one passed review and register this exact smoke with the sole controller."""
    candidate_name(phase,tag); spec_path=EX/(tag+'.json')
    if not spec_path.is_file() or not (EX/'rrctl_state'/tag).exists():
        raise ValueError('Victim smoke has not been RRCTL-bound; no registration')
    spec=load(spec_path)
    if current_source()!=spec['source']['commit']:
        raise ValueError('Victim review source differs from the current published commit')
    summary_path=Path(spec['local_pull_root'])/tag/'result'/'smoke_summary.json'
    if not summary_path.is_file():
        raise ValueError('RRCTL has not pulled the completed victim smoke summary')
    smoke=load(summary_path)
    release_path=EX/tag/'source_release.json';release=load(release_path)
    config_path=ROOT/release['phase_config_path']; config_sha=sha(config_path)
    if (smoke.get('state')!='passed' or smoke.get('phase')!=phase
            or smoke.get('source_commit')!=spec['source']['commit']
            or smoke.get('release_sha256')!=sha(release_path)
            or smoke.get('phase_config_sha256')!=config_sha
            or smoke.get('official_metrics_disabled') is not True
            or smoke.get('artifact_ingest_disabled') is not True
            or smoke.get('validation_forwards')!=0
            or smoke.get('finite_outputs') is not True
            or smoke.get('production_entrypoint_reached') is not True
            or smoke.get('teacher_checkpoint_sha256')!=release['teacher_dependency']['checkpoint_sha256']
            or not 1<=smoke.get('actual_native_batches',0)<=100
            or smoke.get('actual_native_updates',0)<1):
        raise ValueError('Current candidate does not have a complete bounded smoke')
    if phase=='s4_200':
        clean_initial=release['phase_dependencies']['clean200'].get('initial_state_sha256')
        baseline=smoke.get('baseline_equivalence',{})
        if not clean_initial or baseline.get('initial_state_sha256')!=clean_initial:
            raise ValueError('S4 victim smoke did not match the accepted clean victim initialization')
    review=load(review_path)
    expected=dict(phase=phase,source_commit=spec['source']['commit'],release_sha256=sha(release_path),
                  phase_config_sha256=config_sha,smoke_sha256=sha(summary_path),
                  reviewer_count=1,pre_run_result='pass',scientific_verdict='allow_run',
                  reviewer_independence=True)
    if any(review.get(k)!=v for k,v in expected.items()) or review.get('blockers') not in ([],None):
        raise ValueError('One independent, blocker-free current-victim review is required')
    _,transport=services(); observation=read_dependency(transport,phase)
    if observation.get('state')!='READY_FOR_ISOLATED_PHASE_VALIDATION':
        raise ValueError('Existing serial phase slot/dependencies/resources changed; no registration')
    if observation['dependencies']!=release['phase_dependencies']:
        raise ValueError('Victim predecessor receipts changed after smoke')
    pred_name='export25k' if phase=='clean200' else 'clean200'
    result=close_and_register(transport,tag=tag,phase=phase,module='coco_s4.phase_victim',
        spec=spec,release_path=release_path,smoke_path=summary_path,review_path=review_path,
        previous_stage_receipt_sha256=release['phase_dependencies'][pred_name]['accepted_sha256'])
    print(json.dumps(result))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch-smoke','register']);p.add_argument('--phase',required=True,choices=sorted(PHASES));p.add_argument('--tag',required=True);p.add_argument('--review',type=Path)
    p.add_argument('--smoke-batches',type=int,default=2)
    a=p.parse_args()
    if a.action=='register' and a.review is None: p.error('register requires --review JSON')
    if a.action=='prepare':prepare(a.phase,a.tag,a.smoke_batches)
    else:{'launch-smoke':launch_smoke,'register':register}[a.action](a.phase,a.tag,*(([a.review] if a.action=='register' else [])))


if __name__ == '__main__': main()
