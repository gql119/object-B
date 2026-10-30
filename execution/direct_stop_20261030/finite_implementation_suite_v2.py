"""Fresh ABBA series after transient external GPU competition invalidated bo1."""
import json,subprocess,sys,time
from datetime import datetime,timezone
from finite_implementation_suite import HERE,ROOT,deadline,event,finish,cli,ctl

def main():
    with (HERE/'FINITE_IMPLEMENTATION_SUITE_V2_STARTED.json').open('x',encoding='utf8') as f:
        json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),reason='bo1 invalidated by actual transient GPU PID3546352; no sampling error; no threshold change',original_failed_run_retained=True,same_campaign_deadline_UTC=datetime.fromtimestamp(deadline,timezone.utc).isoformat()),f,indent=2)
    assert ctl.inspect('s4_at_serial_seed0_20261029_v2')['status']['state']=='aborted'
    for label in ('eq02c4','eq03c4'):
        v=json.loads((HERE/'probes'/label/'summary.json').read_text());assert v['exact_step_and_state_hashes']
    order=[('bo1v2','original',1,'direct02'),('b21','02',1,'direct01'),('b22','02',2,'direct01'),('bo2','original',2,'direct01'),
           ('bo3','original',3,'direct01'),('b31','03',1,'direct01'),('b32','03',2,'direct01'),('bo4','original',4,'direct01')]
    for label,impl,repeat,generation in order:
        if time.time()+1020>deadline:raise TimeoutError('Conservative campaign budget insufficient')
        cli('launch','--label',label,'--prepared',f's4_at_speed_bench_c{impl}_clean4_b36_r{repeat}_20261030_gpu01.json',
            '--tag',f's4_at_b36_{impl}_r{repeat}_20261030_{generation}')
        finish(label)
    for impl,labels in [('02',['bo1v2','b21','b22','bo2']),('03',['bo3','b31','b32','bo4'])]:
        out=HERE/f'ABBA_CANDIDATE_{impl}_B36.json'
        r=subprocess.run([sys.executable,'-B',str(ROOT/'execution/cuda_batch_optimization_20261030/assess_benchmarks.py'),
            '--summaries',*[str(HERE/'probes'/x/'summary.json') for x in labels],'--out',str(out)],capture_output=True,text=True,encoding='utf8')
        event(assessment=impl,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);assert not r.returncode
    event(state='FINITE_IMPLEMENTATION_ABBA_SUITE_FINISHED',formal_controller_started=False)

if __name__=='__main__':
    try:main()
    except BaseException as exc:event(state='FINITE_SUITE_V2_STOPPED_NO_FURTHER_DISPATCH',error=repr(exc));raise
