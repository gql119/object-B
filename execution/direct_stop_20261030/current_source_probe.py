"""Thin current-source native/baseline/resume probes; finite, serial, retain all."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,admission,PYTHON,current
from robustness_s4_at.common import load,write_once,protocol,now
from remote_run_control.models import RunSpec


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','observe','collect']);p.add_argument('--tag',required=True)
    p.add_argument('--kind',choices=['native','baseline','resume']);p.add_argument('--arm',choices=['clean','poison'],default='clean')
    p.add_argument('--epsilon',type=int,choices=[4,8],default=4);a=p.parse_args()
    d=HERE/'current_probes'/a.tag;ctl,t=services()
    if a.action=='launch':
        source=current();d.mkdir(parents=True,exist_ok=False);batch=protocol()['train_args']['batch']
        if ctl.inspect('s4_at_serial_seed0_20261029_v2')['status']['state']!='aborted':raise ValueError('Old formal not aborted')
        capacity=admission(t);write_once(d/'capacity.json',capacity)
        if not capacity['admitted']:raise ValueError('Current source GPU admission refused')
        if a.kind=='baseline':s=load(ROOT/'execution/s4_at_cuda_baseline_20261029_gpu03.json')
        elif a.kind=='resume':s=load(ROOT/f'execution/cuda_batch_optimization_20261030/prepared_specs/s4_at_speed_resume_coriginal_{a.arm}{a.epsilon}_b{batch}_r1_20261030_gpu01.json')
        else:s=load(ROOT/'execution/s4_at_cuda_native_20261029_gpu03.json')
        oldroot=s['remote']['output_root'];oldsource=s['source']['commit'];owned=protocol()['remote_owned_root']
        s['run_id']=a.tag;s['session']['name']=a.tag;s['source']['commit']=source;s['anchors']=[]
        for key,folder in (('stage_root','stage'),('repo_root','releases'),('control_root','control'),('output_root','validation')):s['remote'][key]=owned+'/'+folder+'/'+a.tag
        out=s['remote']['output_root']+'/result';unit='--unit=s4at-current-'+hashlib.sha256(a.tag.encode()).hexdigest()[:20]
        if a.kind=='native':
            s['workload']=dict(cwd='.',argv=['systemd-run','--user','--scope','--quiet',unit,PYTHON,'-B','execution/direct_stop_20261030/validate_selected.py','--out',out,'--source-commit',source])
            files=['status.json','smoke_summary.json','GT_and_budget.json'];summary='smoke_summary.json';budget=64
        else:
            s['workload']['argv']=[unit if x.startswith('--unit=') else source if x==oldsource else x.replace(oldroot,s['remote']['output_root']) for x in s['workload']['argv']]
            if a.kind=='baseline':
                # Diagnostic setting follows the actual frozen batch; the
                # reference/candidate computation and zero tolerance are exact.
                index=s['workload']['argv'].index('-c')+1
                inline=s['workload']['argv'][index]
                if inline.count('parameters=dict(batch=36')!=1 or inline.count('sample_count=36')!=1:raise ValueError('Unexpected baseline setting schema')
                s['workload']['argv'][index]=inline.replace('parameters=dict(batch=36',"parameters=dict(batch=load(PROTOCOL)['train_args']['batch']").replace('sample_count=36',"sample_count=load(PROTOCOL)['train_args']['batch']")
                files=['status.json','smoke_summary.json','baseline_equivalence.json'];summary='smoke_summary.json';budget=16
            else:files=['status.json','candidate_resume_summary.json'];summary='candidate_resume_summary.json';budget=64
        artifacts=['result/'+x for x in files]
        adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};z=s.get('state','');print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=z!='FAILED',complete=z=='passed',progress={k:s.get(k) for k in ('state','source_commit','current_run','completed_bounded_arms','actual_outer_batches','error')},artifacts="+repr(artifacts)+")))"
        for h in s['health'].values():h.update(adapter_argv=[PYTHON,'-B','-c',adapter],timeout_seconds=900)
        s['artifacts']=[dict(path=x,required=True) for x in artifacts];s['local_pull_root']=str(ROOT/'execution/d30p')
        s['metadata']=dict(execution_purpose='current_selected_production_source_finite_'+str(a.kind),candidate_commit=source,
            actual_selected_batch=batch,arm=a.arm,epsilon=a.epsilon,max_total_native_outer_batches=budget,
            summary_filename=summary,official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all')
        spec=RunSpec.from_dict(s);write_once(d/'runspec.json',s);ready=ctl.ready(spec);write_once(d/'READY.json',ready)
        if not ready['ready']:raise ValueError('Current source thin readiness refused')
        r=ctl.launch(spec);write_once(d/'launch.json',r);print(json.dumps(dict(tag=a.tag,source=source,kind=a.kind,launch=r)));return
    s=load(d/'runspec.json')
    if a.action=='observe':
        print(json.dumps(ctl.health(a.tag,phase='periodic')));return
    if (d/'pull.json').exists():raise FileExistsError('Already collected; use retained evidence')
    h=ctl.health(a.tag,phase='completion');write_once(d/'completion_health.json',h)
    if not h['healthy'] or not h['complete']:raise ValueError('Current source smoke did not pass completion health')
    write_once(d/'wait.json',ctl.wait(a.tag,poll_seconds=1));write_once(d/'pull.json',ctl.pull(a.tag))
    raw=t.download(s['remote']['output_root']+'/result/'+s['metadata']['summary_filename']);(d/'summary.json').write_bytes(raw)
    v=json.loads(raw)
    if v['state']!='passed' or v['source_commit']!=s['source']['commit']:raise ValueError('Current-source numerical evidence failed')
    print(json.dumps(dict(tag=a.tag,state=v['state'],source=v['source_commit'],actual_outer_batches=v.get('actual_outer_batches'),path=str(d/'summary.json'))))


if __name__=='__main__':main()
