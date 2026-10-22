"""Local RRCTL-only preparation, exact-once deploy, and read-only query."""
import argparse,datetime,json,os,sys,time
from pathlib import Path
from .common import ROOT,CONFIG,load,sha,atomic,commit

def services(profiles):
    from remote_run_control.controller import Controller
    from remote_run_control.profiles import ProfileStore
    from remote_run_control.transport import transport_for
    return Controller(profiles_path=profiles,state_root=ROOT/'execution/state'),transport_for(ProfileStore(profiles).load('school-s1'))

def remote(t,python,code,payload):
    r=t.run([python,'-B','-',json.dumps(payload,separators=(',',':'))],input_data=code.encode())
    if r.returncode:raise RuntimeError(json.dumps(dict(returncode=r.returncode,stderr=r.stderr.decode(errors='replace'),stdout=r.stdout.decode(errors='replace'))))
    return json.loads(r.stdout)

def prepare(cpu):
    c=load(CONFIG);rid=c.get('cpu_scope_smoke_run_id','s4_robustness_extended_cpu_scope_smoke_seed0_20261022_v1') if cpu else c['pipeline_run_id']
    if (ROOT/'execution/state'/rid/'run_ref.json').exists():raise ValueError('Already bound: inspect only')
    template=load(ROOT.parent/'s4_robustness_evaluation_20261019/execution/s4_robustness_group1_train32_smoke_seed0_20261019_v2.json')
    template.update(run_id=rid,project='s4-robustness-extended')
    template['source']=dict(repo_root=str(ROOT),branch='codex/s4-robustness-extended-defenses-20261022',commit=commit(),allowed_post_commit_paths=['execution/','Ultralytics/','research_workspace/STATE.md','robustness_s4_extended/reports/CPU_TEST_RESULT.json','robustness_s4_extended/reports/QUEUE_STATUS.json'])
    for key,folder in [('stage_root','stage'),('repo_root','repos'),('control_root','control'),('output_root','outputs')]:template['remote'][key]=c['root']+'/'+folder+'/'+rid
    template['session']['name']=rid;python=c['python']
    command=[python,'-B','-m','robustness_s4_extended.cpu_server_smoke','--output',template['remote']['output_root']] if cpu else [python,'-B','-m','robustness_s4_extended.successor','--expected-source',commit()]
    template['workload']=dict(argv=command,cwd='.')
    artifacts=['smoke_summary.json','CPU_TEST_CONSOLE.txt'] if cpu else ['pipeline_state.json','pipeline_progress.json','pipeline_events.jsonl','reports/EXTENDED_RESULTS.csv','reports/EXTENDED_PER_CLASS_RESULTS.csv','reports/EXTENDED_METRICS.csv','reports/METRICS_ALL_CHECKPOINTS.json','reports/EXTENDED_REPORT.md','reports/REPORT_STATE.json','reports/ARCHIVE_MANIFEST.json','S4_FULL_DECISION.json','S4_FULL_COMPARISON.csv','S4_FULL_COMPARISON.md','PIPELINE_STATUS.json','PREDECESSOR_ACCEPTANCE.json','reports/EXTENDED_ROBUSTNESS_RESULTS.csv','reports/EXTENDED_ROBUSTNESS_PER_CLASS.csv','reports/EXTENDED_ROBUSTNESS_REPORT.md']
    adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root']);f=p/"+repr('smoke_summary.json' if cpu else 'pipeline_state.json')+";d=json.loads(f.read_text()) if f.exists() else {};done=d.get('state') in ('passed','BLOCKED_PROTOCOL_APPROVAL');print(json.dumps({'protocol':'rrctl.adapter.v1','healthy':d.get('state') not in ('FAILED','failed'),'complete':done,'progress':d,'artifacts':"+repr(artifacts)+" if done else []}))"
    for name,h in template['health'].items():
        h['adapter_argv']=[python,'-B','-c',adapter];h['gpu_utilization_policy']='advisory';h['timeout_seconds']=180 if name=='completion' else 3600
    template['artifacts']=[dict(path=x,required=(x in ['smoke_summary.json','CPU_TEST_CONSOLE.txt','pipeline_state.json','reports/EXTENDED_RESULTS.csv','reports/EXTENDED_REPORT.md'])) for x in artifacts]
    template['local_pull_root']=str(ROOT.parent/'s4_extended_small_20261022')
    template['metadata']=dict(execution_purpose='CPU_supervision_smoke' if cpu else 'server_successor_waiter',computation_kind='orchestration',candidate_commit=commit(),config_sha256=sha(CONFIG),GPU_work_now=False,conditional_GPU_authorization='only after Full completed AND independently accepted; source/data/CPU/native identity checks; capacity; actual bounded GPU smoke; one scientific review closure',OP_formal_authorized=False,AT_authorized=False,cleanup_policy='retain_all',user_authorization='2026-10-22 direct extended defenses task stagesA-E; Full priority; no immediate new GPU work')
    p=ROOT/'execution'/(rid+'.json');atomic(p,template)
    from remote_run_control.models import RunSpec
    RunSpec.from_path(p)
    print(json.dumps(dict(prepared=str(p),run_id=rid,source_commit=commit(),launched=False)))

PREFLIGHT=r'''
import json,sys,os
from pathlib import Path
p=json.loads(sys.argv[1]);paths=p['paths'];full=Path(p['full'])
assert all(not Path(x).exists() for x in paths.values()),paths
s=json.loads((full/'pipeline_status.json').read_text())
assert s['run_id']==p['predecessor_id'] and s['source_commit']==p['predecessor_source']
print(json.dumps({'paths_absent':True,'current_Full_state':s,'no_training_or_process_mutation':True}))
'''

def launch(cpu,profiles):
    c=load(CONFIG);rid=c.get('cpu_scope_smoke_run_id','s4_robustness_extended_cpu_scope_smoke_seed0_20261022_v1') if cpu else c['pipeline_run_id'];p=ROOT/'execution'/(rid+'.json');specdata=load(p)
    if specdata['source']['commit']!=commit():raise ValueError('Prepared source changed')
    if (ROOT/'execution/state'/rid).exists() or (ROOT/'execution'/(rid+'_launch_intent.json')).exists():raise ValueError('Existing binding/intent: inspect only')
    ctl,t=services(profiles)
    if not cpu:
        smoke=load(ROOT/'execution/CPU_SERVER_SMOKE_VERIFIED.json')
        if smoke.get('state')!='passed' or smoke.get('source_commit')!=commit() or smoke.get('config_sha256')!=sha(CONFIG):raise ValueError('Current-source CPU supervision smoke required')
    payload=dict(paths={k:specdata['remote'][k] for k in ['stage_root','repo_root','control_root','output_root']},full=c['predecessor']['output'],predecessor_id=c['predecessor']['run_id'],predecessor_source=c['predecessor']['source_commit'])
    atomic(ROOT/'execution'/(rid+'_preflight.json'),remote(t,c['python'],PREFLIGHT,payload))
    from remote_run_control.models import RunSpec
    spec=RunSpec.from_path(p);ready=ctl.ready(spec);atomic(ROOT/'execution'/(rid+'_ready.json'),ready)
    if not ready.get('ready'):raise ValueError('RRCTL not ready: '+json.dumps(ready))
    remote(t,c['python'],PREFLIGHT,payload)
    tmp=ROOT/'execution/tmp';tmp.mkdir(exist_ok=True);os.environ['TMP']=os.environ['TEMP']=str(tmp)
    atomic(ROOT/'execution'/(rid+'_launch_intent.json'),dict(run_id=rid,source_commit=commit(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    answer=ctl.launch(spec);atomic(ROOT/'execution'/(rid+'_launch.json'),answer)
    print(json.dumps(dict(run_id=rid,launched_once=True,CPU_only_now=True)))

def inspect(cpu,profiles):
    c=load(CONFIG);rid=c.get('cpu_scope_smoke_run_id','s4_robustness_extended_cpu_scope_smoke_seed0_20261022_v1') if cpu else c['pipeline_run_id'];ctl,t=services(profiles);i=ctl.inspect(rid)
    h=ctl.health(rid,phase='completion' if i['status']['state'] in ['completed','workload_complete','failed','aborted'] else 'periodic')
    answer=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),inspect=i,health=h)
    if h.get('healthy') and h.get('complete'):
        answer['wait']=ctl.wait(rid);answer['pull']=ctl.pull(rid)
    if not cpu:
        answer['server']=remote(t,c['python'],"from pathlib import Path;import json,sys;p=json.loads(sys.argv[1]);r=Path(p['output']);f=r/'pipeline_state.json';x=json.loads(f.read_text());z=r/'pipeline_progress.json';print(json.dumps({'state':x,'progress':json.loads(z.read_text()) if z.exists() else None}))",dict(output=c['output_root']))
    atomic(ROOT/'execution'/('CPU_SERVER_CHECK_LATEST.json' if cpu else 'EXTENDED_CHECK_LATEST.json'),answer)
    print(json.dumps(dict(run_id=rid,state=i['status']['state'],health=h.get('status'),server=answer.get('server'),pull=answer.get('pull'))))

def review_evidence(profiles):
    c=load(CONFIG);ctl,t=services(profiles)
    s=remote(t,c['python'],"from pathlib import Path;import json,sys;p=json.loads(sys.argv[1]);s=json.loads((Path(p['output'])/'pipeline_state.json').read_text());assert s['state']=='PENDING_SCIENTIFIC_REVIEW';print(json.dumps(s))",dict(output=c['output_root']))
    if s['config_sha256']!=sha(CONFIG):raise ValueError('Review config differs')
    target=ROOT/'execution/review_evidence';target.mkdir(exist_ok=True);manifest={}
    for name in ['cutout','cutmix','mixup']:
        for rel in ['smoke_summary.json','clean/baseline_equivalence.json','clean/trial_summary.json','poison/trial_summary.json','clean/data_manifest.json','poison/data_manifest.json','clean/victim_run_spec.json','poison/victim_run_spec.json']:
            remote_path=c['output_root']+'/experiments/smoke_'+name+'/'+rel
            data=t.download(remote_path);local=target/name/rel;local.parent.mkdir(parents=True,exist_ok=True)
            if local.exists() and local.read_bytes()!=data:raise ValueError('Previously downloaded evidence changed')
            local.write_bytes(data);manifest[name+'/'+rel]=sha(local)
    atomic(target/'DOWNLOAD_MANIFEST.json',dict(state=s,artifacts=manifest,weights_or_images_transferred=False))
    print(json.dumps(dict(artifacts=len(manifest),source_commit=s['source_commit'],state=s['state'],reviewer_called=False)))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare-cpu','launch-cpu','check-cpu','prepare','launch','status','review-evidence']);p.add_argument('--profiles',type=Path);a=p.parse_args();cpu=a.action.endswith('cpu')
    if a.action.startswith('prepare'):prepare(cpu);return
    if not a.profiles:raise ValueError('Explicit existing profiles path required')
    if a.action=='review-evidence':review_evidence(a.profiles)
    elif a.action.startswith('launch'):launch(cpu,a.profiles)
    else:inspect(cpu,a.profiles)
if __name__=='__main__':main()
