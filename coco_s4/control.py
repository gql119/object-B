"""Exact-once RRCTL deployment for the single authorized COCO workflow."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .runtime import DATA_FILES, DATA_ROOT, PARENT, VICTIM_PARENT, ROOT, atomic, commit, load, now, sha, write_once

EX = ROOT/'execution'
BASE = ROOT.parent
RUN_ID = 's4_coco_experiment_a_seed0_20261025_v1'
REMOTE_ROOT = '/project/guoqinglong/projects/s4_coco_experiment_a_20261025'
PYTHON = '/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'
PROFILES = BASE/'s4_robustness_evaluation_20261019/execution/profiles.json'
SPEC = EX/(RUN_ID+'.json')


def services():
    from remote_run_control.controller import Controller
    from remote_run_control.profiles import ProfileStore
    from remote_run_control.transport import transport_for
    return Controller(profiles_path=PROFILES, state_root=EX/'rrctl_state'), transport_for(ProfileStore(PROFILES).load('school-s1'))


def remote(t, code, payload):
    r = t.run([PYTHON, '-B', '-c', code, json.dumps(payload)], input_data=None)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors='replace')[-3500:])
    return json.loads(r.stdout)


def prepare():
    if SPEC.exists() or (EX/'rrctl_state'/RUN_ID).exists():
        raise FileExistsError('Existing prepared/bound RunID: inspect only')
    actual = load(EX/'ACTUAL_TEACHER_NATIVE_SOURCE_AUDIT_20261025.json')
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
    selected = [n for n in names if n.startswith(('coco_s4/', 'ue_framework/', 'configs/'))
                or n in ('docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json',
                         'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml')]
    if not all(n in selected for n in ('coco_s4/phase_teacher.py','coco_s4/teacher.py','coco_s4/serial.py')):
        raise ValueError('Production source must be committed before preparation')
    changes = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if changes:
        raise ValueError('Tracked source must be clean')
    release = dict(schema='s4.coco.release.v1', source_commit=commit(), scientific_parent=PARENT, victim_parent=VICTIM_PARENT,
                   files={n: sha(ROOT/n) for n in selected}, data_root=str(DATA_ROOT), data_files=DATA_FILES,
                   native_sources={n: r['sha256'] for n, r in actual['files'].items()}, versions=actual['versions'],
                   config_path='docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml',
                   implemented_phases=['teacher200'], later_phase_implementation_acceptance_not_implied=True)
    rel_local=EX/'TEACHER_SOURCE_RELEASE_20261025.json'
    write_once(rel_local, release)
    template = load(BASE/'s4_robustness_extended_20261022/execution/s4_robustness_extended_selected_seed0_20261022_recovery03.json')
    prev = dict(output=template['remote']['output_root'],run_id=template['run_id'],source_commit=template['source']['commit'])
    remote_paths = {k:REMOTE_ROOT+'/'+d+'/'+RUN_ID for k,d in (('stage_root','stage'),('repo_root','repos'),('control_root','control'),('output_root','outputs'))}
    binding_dir=REMOTE_ROOT+'/bindings/'+RUN_ID
    rel_remote=binding_dir+'/source_release.json'
    cfg_remote=binding_dir+'/workflow.json'
    acc_local=BASE/'s4_robustness_extended_20261022/execution/EXTENDED_FINAL_LOCAL_ACCEPTANCE_20261024.json'
    config=dict(schema='s4.coco.controller-config.v1',run_id=RUN_ID,source_commit=commit(),release_sha256=sha(rel_local),
                predecessor=prev,predecessor_acceptance=binding_dir+'/predecessor_acceptance.json',
                predecessor_acceptance_sha256=sha(acc_local),owned_release_root=REMOTE_ROOT+'/releases',
                future_phases_require_separate_immutable_validated_release=True,OP_enabled=False,
                user_authorization_sha256=sha(ROOT/'docs/S4_COCO_MINITRAIN_20261023/USER_AUTOSTART_TASK_20261025.md'))
    cfg_local=EX/'COCO_WORKFLOW_BOUND_CONFIG_20261025.json';write_once(cfg_local,config)
    template.update(run_id=RUN_ID,project='s4-coco-experiment-a')
    template['source']=dict(repo_root=str(ROOT),branch='codex/s4-coco-minitrain25k-generalization',commit=commit(),
                            allowed_post_commit_paths=['execution/','Ultralytics/'])
    template['remote'].update(remote_paths)
    template['session']['name']=RUN_ID
    template['workload']=dict(argv=[PYTHON,'-B','-m','coco_s4.serial','--config',cfg_remote,'--release',rel_remote,
                                   '--output',remote_paths['output_root']],cwd='.')
    template['anchors']=[dict(name=n,local_path=str(local),remote_path=dest,sha256=sha(local)) for n,local,dest in
                         [('source_release',rel_local,rel_remote),('workflow',cfg_local,cfg_remote),
                          ('predecessor_acceptance',acc_local,config['predecessor_acceptance'])]]
    adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'pipeline_state.json';d=json.loads(p.read_text()) if p.exists() else {};print(json.dumps({'protocol':'rrctl.adapter.v1','healthy':d.get('state')!='FAILED','complete':d.get('state')=='COMPLETED_AND_ACCEPTED','progress':d,'artifacts':['pipeline_state.json']}))"
    for phase in template['health'].values():
        phase['adapter_argv']=[PYTHON,'-B','-c',adapter]
        phase['timeout_seconds']=3600;phase['gpu_utilization_policy']='advisory'
    template['artifacts']=[dict(path='pipeline_state.json',required=True),dict(path='teacher_smoke/smoke_summary.json',required=False)]
    template['local_pull_root']=str(BASE/'s4_coco_small_results_20261025')
    template['metadata']=dict(execution_purpose='single_serial_COCO_workflow',computation_kind='orchestration',
                              candidate_commit=commit(),cleanup_policy='retain_all',official_metrics_disabled_during_smoke=True,
                              conditional_formal_authorization='current phase CPU/GPU/single scientific closure; sequential actual dependencies',
                              no_second_scheduler=True,AT_authorized=False,OP_authorized=False)
    from remote_run_control.models import RunSpec
    RunSpec.from_dict(template)
    write_once(SPEC,template)
    print(json.dumps(dict(prepared=str(SPEC),source_commit=commit(),run_id=RUN_ID,launched=False)),flush=True)


PREFLIGHT = r'''
from pathlib import Path
import json,os,sys
p=json.loads(sys.argv[1]);assert os.getuid()==1014
for path in p['paths']:
 assert not Path(path).exists(),('COLLISION',path)
print(json.dumps(dict(paths_absent=True,uid=os.getuid())))
'''


def launch():
    if (EX/'rrctl_state'/RUN_ID).exists() or (EX/(RUN_ID+'_launch_intent.json')).exists():
        raise FileExistsError('Bound RunID or launch intent: inspect only')
    specdata=load(SPEC)
    if specdata['source']['commit'] != commit():
        raise ValueError('Prepared source differs')
    ctl,t=services()
    paths=list(specdata['remote'][k] for k in ('repo_root','output_root','stage_root','control_root'))
    remote(t,PREFLIGHT,dict(paths=paths))
    from remote_run_control.models import RunSpec
    spec=RunSpec.from_path(SPEC)
    ready=ctl.ready(spec);atomic(EX/(RUN_ID+'_ready.json'),ready)
    if not ready.get('ready'):
        raise ValueError('RRCTL not ready: '+json.dumps(ready))
    remote(t,PREFLIGHT,dict(paths=paths))
    temp=EX/'rrctl_temp';temp.mkdir(exist_ok=True);os.environ['TMP']=os.environ['TEMP']=str(temp)
    write_once(EX/(RUN_ID+'_launch_intent.json'),dict(run_id=RUN_ID,source_commit=commit(),UTC=now()))
    result=ctl.launch(spec);write_once(EX/(RUN_ID+'_launch.json'),result)
    print(json.dumps(dict(run_id=RUN_ID,launched_once=True,result=result)),flush=True)


def status():
    ctl,t=services();spec=load(SPEC)
    r=remote(t,"from pathlib import Path;import json,sys;p=Path(json.loads(sys.argv[1])['output']);s=json.loads((p/'pipeline_state.json').read_text());print(json.dumps(s))",
             dict(output=spec['remote']['output_root']))
    atomic(EX/'COCO_SERVER_STATUS_LATEST.json',r)
    print(json.dumps(r),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','status']);a=p.parse_args()
    dict(prepare=prepare,launch=launch,status=status)[a.action]()
