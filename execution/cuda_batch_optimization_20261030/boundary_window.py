"""Reserve a natural between-arm window using the existing AT supervisor only.

Default assessment is read-only. Reservation requires the final CPU control
closure, the exact original binding, near-completion progress, and a ready
published bounded benchmark plan. It never signals a training child.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from at_cuda_rrctl_20261029 import services, PYTHON
from queue_window_control import CONTROL, HERE

TAG="s4_at_serial_seed0_20261029_v2"


def load(p):return json.loads(Path(p).read_text(encoding='utf8'))


def journal(label,value):
    d=HERE/'boundary_window_journal';d.mkdir(exist_ok=True)
    p=d/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+label+'.json')
    with p.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    return p


def request(ctl):
    closure=load(HERE/'s4_at_queue_window_20261030_cpu04/MAIN_CONTROL_CLOSURE.json')
    if closure.get('state')!='PASS_REAL_RRCTL_CPU_QUEUE_BOUNDARY_SMOKE' or closure.get('control_source_sha256')!=hashlib.sha256(CONTROL.encode()).hexdigest():
        raise ValueError('Current control code lacks its exact CPU launcher smoke closure')
    bound=ctl.inspect(TAG)['binding']
    if bound['commit']!='51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb' or bound['run_spec_sha256']!='e9f80f82d4288a9b08ed9877b900911f813854e390479eb6a453176c907c87b3':
        raise ValueError('Original AT source or frozen RunSpec changed')
    discovered=load(HERE/'QUEUE_SUPERVISOR_DISCOVERY.json')['queue_supervisors']
    if len(discovered)!=1:raise ValueError('Original queue discovery was ambiguous')
    q=discovered[0]
    if q['pid']!=bound['workload_pid'] or q['start_ticks']!=bound['workload_start_ticks']:
        raise ValueError('Original queue identity differs from its RRCTL binding')
    if len(q['children'])!=1:raise ValueError('Unexpected scientific child structure')
    child=q['children'][0]
    return dict(scope='original_AT_queue_supervisor',run_id=TAG,control_root=bound['control_root'],
        pid=q['pid'],uid=q['uid'],start_ticks=q['start_ticks'],argv=q['argv'],cwd=q['cwd'],
        child_pid=child['pid'],child_start_ticks=child['start_ticks'])


def invoke(t,q,action):
    r=t.run([PYTHON,'-B','-c',CONTROL,json.dumps(dict(q,action=action))])
    v=json.loads(r.stdout);journal(action,dict(request=dict(q,action=action),result=v,returncode=r.returncode))
    if r.returncode:raise RuntimeError('Control guard refused operation: '+repr(v))
    return v


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('assess','reserve','release'))
    p.add_argument('--benchmark-plan',type=Path)
    a=p.parse_args();ctl,t=services();q=request(ctl)
    ticket=HERE/'ORIGINAL_QUEUE_WINDOW_TICKET.json'
    if a.action=='release':
        # The scientific child may already be reaped. Do not route release
        # through an inspection that requires a live child.
        if not ticket.exists():raise FileNotFoundError('No owned boundary hold ticket')
        intent=load(ticket)
        if intent.get('queue')!=q or intent.get('state')!='QUEUE_SUPERVISOR_HELD_CHILD_UNTOUCHED':
            raise ValueError('Release does not match the owned hold ticket')
        result=invoke(t,q,'release');intent.update(state='SAME_ORIGINAL_SUPERVISOR_RELEASED',release_receipt=result,
            released_UTC=datetime.now(timezone.utc).isoformat())
        ticket.write_text(json.dumps(intent,indent=2),encoding='utf8');print(json.dumps(intent))
        return
    identity=invoke(t,q,'inspect')
    assessment=invoke(t,q,'assess')
    if a.action=='assess':
        print(json.dumps(dict(state='READ_ONLY_BOUNDARY_ASSESSMENT',identity=identity,assessment=assessment,
            production_signals_sent=0,formal_worker_or_source_modified=False,
            earliest_window='after this CleanAT4 child naturally completes training plus final evaluation and exits',
            policy='hold queue supervisor only at completed epochs195..199; release after bounded GPU tests have exited')))
        return
    if a.action=='reserve':
        if not assessment['eligible_for_hold']:raise ValueError('Not yet near the natural boundary; no hold')
        if a.benchmark_plan is None:raise ValueError('Ready published benchmark plan required before reserving')
        plan=load(a.benchmark_plan)
        if plan.get('state')!='READY_FOR_EXCLUSIVE_GPU_WINDOW' or not plan.get('source_publication_verified') or not plan.get('bounded_RRCTL_RunSpecs'):
            raise ValueError('Benchmark entry is not ready; preserve normal next-stage start')
        if ticket.exists():raise FileExistsError('Existing boundary ticket is inspect-only')
        if not ctl.health(TAG,phase='periodic').get('healthy'):raise ValueError('Original AT is not healthy')
        intent=dict(state='HOLD_INTENT',queue=q,control_sha256=hashlib.sha256(CONTROL.encode()).hexdigest(),
            plan_path=str(a.benchmark_plan.resolve()),purpose='reserve natural arm boundary without interrupting training',
            max_validation_window_seconds=3600,UTC=datetime.now(timezone.utc).isoformat())
        with ticket.open('x',encoding='utf8') as f:json.dump(intent,f,indent=2)
        try:
            result=invoke(t,q,'hold');intent.update(state='QUEUE_SUPERVISOR_HELD_CHILD_UNTOUCHED',receipt=result)
        except BaseException as exc:
            intent.update(state='HOLD_NOT_CONFIRMED_DO_NOT_ASSUME',error=repr(exc));raise
        finally:ticket.write_text(json.dumps(intent,indent=2),encoding='utf8')
        print(json.dumps(intent))


if __name__=='__main__':main()
