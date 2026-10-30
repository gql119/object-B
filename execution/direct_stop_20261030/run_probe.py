"""Finite CUDA probes after verified old-run termination, existing RRCTL only."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,subprocess
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON,admission
from remote_run_control.models import RunSpec
AUTH_SHA='e46fc5aa728c294a4ce803a8ba23cacf5e80191173b411579bb3325d39581db8'

def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,default=str)

def record_state(s,label,phase,summary=None):
    path=ROOT/'docs/S4_AT_ROBUSTNESS_20261027/CURRENT_STATE.json'
    state=json.loads(path.read_text(encoding='utf8'))
    state.update(UTC=datetime.now(timezone.utc).isoformat(),active_formal_controllers=0,current_arm=None,
                 state='BOUNDED_CUDA_OPTIMIZATION_'+phase,formal_run_allowed=False)
    state['active_CUDA_probe']=None if phase=='COLLECTED' else dict(run_id=s['run_id'],label=label,source=s['source']['commit'],phase=phase,
        receipts=str((HERE/'probes'/label).relative_to(ROOT)),formal_training=False)
    if summary is not None:
        state.setdefault('completed_direct_CUDA_probes',{})[label]=dict(run_id=s['run_id'],state=summary['state'],
            summary=str((HERE/'probes'/label/'summary.json').relative_to(ROOT)))
    path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def termination():
    path=sorted(HERE.glob('*_OLD_AT_TERMINATED.json'))[-1]
    p=json.loads(path.read_text());assert p['state']=='OLD_AT_TERMINATED' and p['RRCTL']['state']=='aborted'
    assert not p['verification']['own_GPU_pids'] and not p['verification']['original_live_processes']
    return path,p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['launch','observe','collect','diagnose'])
    ap.add_argument('--label',required=True);ap.add_argument('--prepared');ap.add_argument('--tag');ap.add_argument('--source-commit')
    a=ap.parse_args();assert all(c.isalnum() or c in '_-' for c in a.label)
    d=HERE/'probes'/a.label;ctl,t=services()
    if a.action=='launch':
        proof,_=termination();assert ctl.inspect('s4_at_serial_seed0_20261029_v2')['status']['state']=='aborted'
        d.mkdir(parents=True,exist_ok=False)
        s=json.loads((ROOT/'execution/cuda_batch_optimization_20261030/prepared_specs'/a.prepared).read_text())
        if a.source_commit:
            if s['metadata']['case']['kind']!='resume':raise ValueError('Source override is only the isolated resume adapter repair')
            oldsource=s['source']['commit']
            changed=subprocess.check_output(['git','diff','--name-only',oldsource,a.source_commit],cwd=ROOT,text=True).splitlines()
            if any(not p.startswith(('execution/','docs/S4_AT_ROBUSTNESS_20261027/')) for p in changed):
                raise ValueError('Resume adapter source override cannot change scientific package')
            s['source']['commit']=a.source_commit
            s['workload']['argv']=[a.source_commit if v==oldsource else v for v in s['workload']['argv']]
            s['metadata'].update(candidate_commit=a.source_commit,retained_original_prepared_source=oldsource,
                source_rebinding_reason='Isolated batch-dependent restored-phase assertions; no scientific package change')
        old=s['remote']['output_root'];tag=a.tag;assert tag and tag!=s['run_id']
        owned='/project/guoqinglong/projects/s4_at_robustness_20261027'
        s['run_id']=tag;s['session']['name']=tag
        for k,folder in (('stage_root','stage'),('repo_root','releases'),('control_root','control'),('output_root','validation')):
            s['remote'][k]=owned+'/'+folder+'/'+tag
        argv=s['workload']['argv']
        s['workload']['argv']=[v.replace(old,s['remote']['output_root']) if not v.startswith('--unit=') else '--unit=s4at-direct-'+hashlib.sha256(tag.encode()).hexdigest()[:20] for v in argv]
        s['local_pull_root']=str(ROOT/'execution/d30p')
        s['metadata'].update(requires_natural_boundary_hold_and_accepted_CleanAT4=False,
            original_formal_run_unchanged=False,not_a_launch_authorization_now=False,
            direct_human_authorization_sha256=AUTH_SHA,old_AT_terminated_proof=str(proof),
            execution_purpose='finite_CUDA_optimization_after_authorized_old_AT_termination')
        # The archived generator advertised a union of all phase summaries,
        # making equality demand a nonexistent resume summary at completion.
        # Bind each new transport adapter to its already-declared phase files.
        artifacts=[x['path'] for x in s['artifacts']]
        adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};z=s.get('state','');print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=z!='FAILED',complete=z=='passed' or z.startswith('PASS_'),progress={k:s.get(k) for k in ('state','implementation','arm','epsilon_pixels','actual_outer_batches','error')},artifacts="+repr(artifacts)+")))"
        for h in s['health'].values():h['adapter_argv']=[PYTHON,'-B','-c',adapter]
        spec=RunSpec.from_dict(s);save(d/'runspec.json',s)
        capacity=admission(t);save(d/'admission.json',capacity);assert capacity['admitted'],capacity
        ready=ctl.ready(spec);save(d/'ready.json',ready);assert ready['ready'],ready
        try:
            r=ctl.launch(spec);save(d/'launch.json',r);record_state(s,a.label,'RUNNING');print(json.dumps(dict(label=a.label,tag=tag,launch=r)))
        except Exception as e:
            save(d/'launch_error.json',dict(error=repr(e),details=getattr(e,'details',None)))
            raise
        return
    s=json.loads((d/'runspec.json').read_text());tag=s['run_id']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if a.action=='observe':
        v=ctl.health(tag,phase='periodic');save(d/(stamp+'_health.json'),v)
        phase='COMPLETE_PENDING_COLLECTION' if v.get('adapter',{}).get('complete') and v.get('observations',{}).get('state') in ('workload_complete','completed') else ('RUNNING' if v['healthy'] else 'FAILED')
        record_state(s,a.label,phase)
        print(json.dumps(dict(label=a.label,tag=tag,health=v)));return
    if a.action=='diagnose':
        b=ctl.inspect(tag);save(d/(stamp+'_inspect.json'),b)
        code="from pathlib import Path;import json,sys;p=Path(sys.argv[1]);o=Path(sys.argv[2]);v={};f=p/'console.log';f=f.open('rb');f.seek(max(0,f.seek(0,2)-12000));v['console_tail']=f.read().decode(errors='replace');s=o/'result/status.json';v['status']=json.loads(s.read_text()) if s.exists() else None;print(json.dumps(v))"
        r=t.run([PYTHON,'-B','-c',code,s['remote']['control_root'],s['remote']['output_root']])
        result=json.loads(r.stdout);save(d/(stamp+'_diagnostic.json'),result)
        print(json.dumps(dict(status=result['status'],console_tail=result['console_tail'][-1800:])));return
    if (d/'collect.json').exists():raise FileExistsError('Already collected once; inspect retained receipt')
    h=ctl.health(tag,phase='completion');save(d/'completion_health.json',h);assert h['healthy'] and h['complete'],h
    try:wait=ctl.wait(tag,poll_seconds=1)
    except Exception as e:
        save(d/'wait_error.json',dict(error=repr(e),details=getattr(e,'details',None)));raise
    save(d/'wait.json',wait)
    pull=ctl.pull(tag);save(d/'collect.json',pull)
    raw=t.download(s['remote']['output_root']+'/result/summary.json') if any(x['path']=='result/summary.json' for x in s['artifacts']) else t.download(s['remote']['output_root']+'/result/candidate_resume_summary.json')
    (d/'summary.json').write_bytes(raw)
    result=json.loads(raw)
    record_state(s,a.label,'COLLECTED',result)
    keys=('state','implementation','arm','epsilon_pixels','actual_native_outer_batches','actual_native_SGD_per_side',
          'actual_AMP_skips_per_side','actual_outer_batches','actual_SGD','actual_AMP_skips','native_batch','timing','cuda_peak_allocated_MiB','error')
    print(json.dumps(dict(label=a.label,tag=tag,summary={k:result[k] for k in keys if k in result},summary_path=str(d/'summary.json'))))

if __name__=='__main__':main()
