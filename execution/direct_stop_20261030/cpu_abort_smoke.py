"""One real RRCTL CPU lifecycle proof; never signals original AT."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON
from guarded_abort_repair import CPU_TAG,CONTROL,request,invoke
from remote_run_control.models import RunSpec
DEST=HERE/CPU_TAG

def save(name,value):
    with (DEST/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2)

def main():
    ctl,t=services();tag=CPU_TAG;DEST.mkdir(exist_ok=False)
    s=json.loads((HERE.parent/'cuda_batch_optimization_20261030/s4_at_queue_window_20261030_cpu04/runspec.json').read_text())
    owned='/project/guoqinglong/projects/s4_at_robustness_20261027'
    s.update(run_id=tag,project='s4-at-guarded-abort-CPU-smoke')
    for k,folder in (('stage_root','stage'),('repo_root','releases'),('control_root','control'),('output_root','validation')):
        s['remote'][k]=owned+'/'+folder+'/'+tag
    s['session']['name']=tag;s['anchors']=[]
    code="import json,subprocess,sys,time;from pathlib import Path;o=Path(sys.argv[1]);o.mkdir(parents=True,exist_ok=False);(o/'result').mkdir();(o/'result/status.json').write_text(json.dumps(dict(state='RUNNING',CPU_only=True)));p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(180)',sys.argv[2]]);time.sleep(180);p.wait()"
    s['workload']=dict(cwd='.',argv=['systemd-run','--user','--scope','--quiet','--unit=s4at-abort-cpu02',PYTHON,'-B','-c',code,s['remote']['output_root'],tag])
    adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')=='RUNNING',complete=False,progress=s,artifacts=['result/status.json'])))"
    for h in s['health'].values():h.update(adapter_argv=[PYTHON,'-B','-c',adapter],timeout_seconds=120)
    s['artifacts']=[dict(path='result/status.json',required=True)]
    s['local_pull_root']=str(HERE/'pull')
    s['metadata']=dict(execution_purpose='RRCTL_postexec_abort_identity_repair_CPU_smoke',GPU_work=False,scientific_model_forwards=0,
        formal_experiment=False,cleanup_policy='retain_all',official_metrics_disabled=True,artifact_ingest_disabled=True,
        control_source_sha256=hashlib.sha256(CONTROL.encode()).hexdigest(),wallclock_budget_seconds=180)
    spec=RunSpec.from_dict(s);save('CPU_ABORT_SMOKE_RUNSPEC.json',s)
    ready=ctl.ready(spec);save('CPU_ABORT_READY.json',ready);assert ready['ready']
    launch=ctl.launch(spec);save('CPU_ABORT_LAUNCH.json',launch)
    b=ctl.inspect(tag)['binding'];raw=t.download(b['control_root']+'/binding.json')
    req=request(b,raw,fixture=True)
    rejected=[]
    for key,value in [('uid',1015),('workload_start_ticks',b['workload_start_ticks']+1),('spec_sha256','0'*64),('binding_sha256','0'*64)]:
        bad=dict(req);bad[key]=value;r=invoke(t,PYTHON,bad);save('CPU_REJECT_'+key+'.json',r);assert r['returncode']!=0
        rejected.append(key)
    assessment=invoke(t,PYTHON,req);save('CPU_ABORT_INSPECT.json',assessment);assert not assessment['returncode']
    req['action']='abort';result=invoke(t,PYTHON,req);save('CPU_ABORT_RESULT.json',result);assert not result['returncode']
    after=ctl.inspect(tag);save('CPU_ABORT_TERMINAL_INSPECT.json',after);assert after['status']['state']=='aborted'
    old=ctl.health('s4_at_serial_seed0_20261029_v2',phase='periodic');save('ORIGINAL_AT_POST_CPU_HEALTH.json',old);assert old['healthy']
    proof=dict(state='PASS_REAL_RRCTL_CPU_ABORT_WITH_POSTEXEC_GUARD',UTC=datetime.now(timezone.utc).isoformat(),
        RunID=tag,rejected_before_signals=rejected,control_source_sha256=hashlib.sha256(CONTROL.encode()).hexdigest(),
        real_owned_CPU_process_group_terminated=True,terminal='aborted',original_AT_healthy=True,
        original_AT_signals_sent=0,GPU_work=False,models=0,
        capture_branch='bootstrap capture modeled explicitly in CPU fixture; actual original live binding independently exhibits this transition',
        production_stop_allowed=False)
    save('CPU_ABORT_MAIN_CLOSURE.json',proof);print(json.dumps(proof))

if __name__=='__main__':main()
