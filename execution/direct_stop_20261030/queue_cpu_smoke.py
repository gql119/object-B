"""Finite production-launcher CPU proof for the fresh queue; no models/GPU."""
import json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON
from remote_run_control.models import RunSpec
TAG='s4_at_fresh_serial_control_20261030_cpu01'
DEST=HERE/TAG

def save(name,value):
    with (DEST/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2)

def main():
    DEST.mkdir(exist_ok=False);ctl,t=services()
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/s4-at-robustness-20261027'],cwd=ROOT,text=True).split()[0]
    assert source==remote
    s=json.loads((HERE.parent/'cuda_batch_optimization_20261030/s4_at_queue_window_20261030_cpu04/runspec.json').read_text())
    owned='/project/guoqinglong/projects/s4_at_robustness_20261027'
    s.update(run_id=TAG,project='s4-at-fresh-serial-control-CPU-smoke');s['source']['commit']=source
    for k,folder in (('stage_root','stage'),('repo_root','releases'),('control_root','control'),('output_root','validation')):s['remote'][k]=owned+'/'+folder+'/'+TAG
    s['session']['name']=TAG;s['anchors']=[]
    entry='execution/direct_stop_20261030/fresh_serial_queue.py'
    startup="import runpy,sys,time;time.sleep(3);sys.argv=['"+entry+"']+sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    s['workload']=dict(cwd='.',argv=['systemd-run','--user','--scope','--quiet','--unit=s4at-fresh-cpu01',PYTHON,'-B','-c',startup,
        '--out',s['remote']['output_root']+'/result','--run-id',TAG,'--source-commit',source,'--cpu-control-smoke'])
    adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/pipeline_state.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')!='FAILED',complete=s.get('state')=='CPU_CONTROL_PASSED',progress=s,artifacts=['result/pipeline_state.json','result/CPU_SERIAL_CONTROL_PROOF.json'])))"
    for h in s['health'].values():h.update(adapter_argv=[PYTHON,'-B','-c',adapter],timeout_seconds=120)
    s['artifacts']=[dict(path='result/'+x,required=True) for x in ('pipeline_state.json','CPU_SERIAL_CONTROL_PROOF.json')]
    s['metadata']=dict(execution_purpose='finite_fresh_queue_lifecycle_CPU_only',GPU_work=False,scientific_model_forwards=0,
        formal_experiment=False,cleanup_policy='retain_all',official_metrics_disabled=True,artifact_ingest_disabled=True)
    s['local_pull_root']=str(ROOT/'execution/d30p')
    spec=RunSpec.from_dict(s);save('runspec.json',s)
    ready=ctl.ready(spec);save('ready.json',ready);assert ready['ready']
    launch=ctl.launch(spec);save('launch.json',launch)
    wait=ctl.wait(TAG,poll_seconds=1);save('wait.json',wait)
    pull=ctl.pull(TAG);save('pull.json',pull)
    proof=json.loads(t.download(s['remote']['output_root']+'/result/CPU_SERIAL_CONTROL_PROOF.json'))
    assert proof['state']=='PASS_REAL_QUEUE_CONTROL_ONLY' and proof['three_stages_serial'] and proof['duplicate_controller_rejected'] and proof['old_global_lock_SHA_unchanged']
    save('MAIN_CLOSURE.json',proof);print(json.dumps(proof))

if __name__=='__main__':main()
