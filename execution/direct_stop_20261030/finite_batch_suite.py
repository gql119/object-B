"""One selected implementation, one frozen new-batch ABBA, finite RRCTL only."""
import argparse,json,subprocess,sys,time
from datetime import datetime,timezone
from finite_implementation_suite import HERE,ROOT,deadline,event,finish,cli,ctl


def main():
    p=argparse.ArgumentParser();p.add_argument('--implementation',choices=['original','02','03'],required=True)
    p.add_argument('--batch',type=int,choices=[48,64,72],required=True);a=p.parse_args()
    stem=f'BATCH_ABBA_C{a.implementation}_B{a.batch}'
    order=[(36,1),(a.batch,1),(a.batch,2),(36,2)]
    plan=dict(UTC=datetime.now(timezone.utc).isoformat(),source='dbdc05f938d1e553fbd62af32b2fa36f908c68d0',
        implementation=a.implementation,batch=a.batch,order=order,formal=False,old_AT='ABORTED',
        deadline_UTC=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),new_protocol_not_trajectory_equivalence=True)
    with (HERE/(stem+'_PLAN.json')).open('x',encoding='utf8') as f:json.dump(plan,f,indent=2)
    assert ctl.inspect('s4_at_serial_seed0_20261029_v2')['status']['state']=='aborted'
    labels=[]
    for batch,repeat in order:
        if time.time()+1020>deadline:raise TimeoutError('No conservative finite campaign margin remains')
        label=f'bc{a.implementation}_{a.batch}_b{batch}r{repeat}';labels.append(label)
        cli('launch','--label',label,'--prepared',f's4_at_speed_bench_c{a.implementation}_clean4_b{batch}_r{repeat}_20261030_gpu01.json',
            '--tag',f's4_at_bc{a.implementation}_b{a.batch}_sample{batch}r{repeat}_20261030_direct01')
        finish(label)
    r=subprocess.run([sys.executable,'-B',str(HERE/'assess_batch_abba.py'),'--summaries',
        *[str(HERE/'probes'/label/'summary.json') for label in labels],'--out',str(HERE/(stem+'_ASSESSMENT.json'))],
        capture_output=True,text=True,encoding='utf8')
    event(batch_assessment=a.batch,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    assert not r.returncode


if __name__=='__main__':
    try:main()
    except BaseException as exc:event(state='FINITE_BATCH_SUITE_STOPPED_NO_FURTHER_DISPATCH',error=repr(exc));raise
