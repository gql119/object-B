"""Finite local RRCTL client suite: existing equality then eight serial ABBA jobs.

No remote long-running controller/watcher is created, and no formal run starts.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services
ctl,_=services()
SOURCE='dbdc05f938d1e553fbd62af32b2fa36f908c68d0'
deadline=datetime.fromisoformat('2026-10-30T05:25:17.243900+00:00').timestamp()+7200
log=HERE/'FINITE_IMPLEMENTATION_SUITE_EVENTS.jsonl'

def event(**v):
    v['UTC']=datetime.now(timezone.utc).isoformat()
    with log.open('a',encoding='utf8') as f:f.write(json.dumps(v)+'\n')
    print(json.dumps(v),flush=True)

def cli(*args):
    r=subprocess.run([sys.executable,'-B',str(HERE/'run_probe.py'),*args],capture_output=True,text=True,encoding='utf8')
    event(command=list(args),returncode=r.returncode,stdout=r.stdout[-7000:],stderr=r.stderr[-4000:])
    if r.returncode:raise RuntimeError('Finite suite command failed; no retry/new dispatch: '+repr(args))

def finish(label):
    d=HERE/'probes'/label;s=json.loads((d/'runspec.json').read_text());start=time.monotonic()
    while True:
        v=ctl.inspect(s['run_id']);phase=v['status']['state']
        event(label=label,run_id=s['run_id'],observed_state=phase)
        if phase in ('workload_complete','completed'):
            if not (d/'collect.json').exists():cli('collect','--label',label)
            summary=json.loads((d/'summary.json').read_text())
            assert summary['source_commit']==SOURCE and (summary['state'].startswith('PASS_') or summary['state']=='passed')
            return summary
        if phase in ('failed','aborted'):
            cli('diagnose','--label',label)
            raise RuntimeError('Probe terminal '+phase)
        if time.monotonic()-start>900 or time.time()>deadline:
            raise TimeoutError('Bounded probe did not finish by dispatch deadline; no further jobs, inspect existing run')
        time.sleep(20)

def main():
    startfile=HERE/'FINITE_IMPLEMENTATION_SUITE_STARTED.json'
    with startfile.open('x',encoding='utf8') as f:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),finite=True,formal_controller=False,remote_watcher_created=False,deadline_UTC=datetime.fromtimestamp(deadline,timezone.utc).isoformat()),f,indent=2)
    assert ctl.inspect('s4_at_serial_seed0_20261029_v2')['status']['state']=='aborted'
    eq=finish('eq03c4');assert eq['exact_step_and_state_hashes'] and eq['actual_native_SGD_per_side']>=1
    eq02=json.loads((HERE/'probes/eq02c4/summary.json').read_text());assert eq02['exact_step_and_state_hashes']
    order=[('bo1','original',1),('b21','02',1),('b22','02',2),('bo2','original',2),
           ('bo3','original',3),('b31','03',1),('b32','03',2),('bo4','original',4)]
    for label,impl,repeat in order:
        if time.time()+900+120>deadline:raise TimeoutError('Insufficient conservative remaining campaign budget for next finite job')
        filename=f's4_at_speed_bench_c{impl}_clean4_b36_r{repeat}_20261030_gpu01.json'
        tag=f's4_at_b36_{impl}_r{repeat}_20261030_direct01'
        cli('launch','--label',label,'--prepared',filename,'--tag',tag)
        finish(label)
    for impl,labels in [('02',['bo1','b21','b22','bo2']),('03',['bo3','b31','b32','bo4'])]:
        out=HERE/f'ABBA_CANDIDATE_{impl}_B36.json'
        args=[sys.executable,'-B',str(ROOT/'execution/cuda_batch_optimization_20261030/assess_benchmarks.py'),'--summaries',*[str(HERE/'probes'/x/'summary.json') for x in labels],'--out',str(out)]
        r=subprocess.run(args,capture_output=True,text=True,encoding='utf8');event(assessment=impl,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        assert r.returncode==0
    event(state='FINITE_IMPLEMENTATION_ABBA_SUITE_FINISHED',formal_controller_started=False)

try:main()
except BaseException as exc:
    event(state='FINITE_SUITE_STOPPED_NO_FURTHER_DISPATCH',error=repr(exc));raise
