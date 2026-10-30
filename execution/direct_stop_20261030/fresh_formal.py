"""Explicit, gated new-generation binding; never reuses the abandoned RunID.

The selected runtime and actual scientific gate must exist before preparation.
This is transport plumbing, not scientific acceptance or automatic selection.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON,admission
from robustness_s4_at.common import load,sha,write_once,protocol,now
from remote_run_control.models import RunSpec

ADAPTER="""
import json,sys
from pathlib import Path
c=json.load(sys.stdin);o=Path(c['output_root'])/'result';f=o/'pipeline_state.json';s=json.loads(f.read_text()) if f.exists() else {};a=s.get('current_run');progress={}
if a:
 p=o/a/'audit/progress.json'
 if p.exists():
  v=json.loads(p.read_text());progress={k:v.get(k) for k in ('run_id','epoch','completed_epochs','at_batches','actual_updates','legal_amp_skips','loss','UTC')}
print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')!='FAILED',complete=s.get('state')=='COMPLETED_AND_ACCEPTED',progress=dict(state=s.get('state'),source_commit=s.get('source_commit'),current_run=a,stages=s.get('stages'),training=progress,error=s.get('error')),artifacts=['result/pipeline_state.json','result/AT_MAIN_RESULTS.csv','result/AT_PER_CLASS_RESULTS.csv','result/AT_PAIRED_SCORES.json','result/ST_REFERENCE_ONLY.json'])))
"""


def main():
    q=argparse.ArgumentParser();q.add_argument('action',choices=['prepare','launch']);q.add_argument('--selection',type=Path,required=True)
    a=q.parse_args();v=load(a.selection);source=v['source_commit'];tag=v['new_formal_RunID'];owned=protocol()['remote_owned_root']
    if tag=='s4_at_serial_seed0_20261029_v2' or not tag.startswith('s4_at_'):raise ValueError('New formal RunID required')
    if v['state']!='CURRENT_SOURCE_SCIENTIFICALLY_ACCEPTED' or not v['fresh_epoch0_all_four_arms']:raise ValueError('Actual scientific acceptance missing')
    if v['protocol_sha256']!=sha(ROOT/'robustness_s4_at/configs/PROTOCOL.json'):raise ValueError('Selected protocol changed')
    # Compare actual committed scientific source and the queue entry, rather
    # than confusing a later evidence-only archive HEAD with runtime source.
    code=list((ROOT/'robustness_s4_at').glob('*.py'))+[ROOT/'robustness_s4_at/configs/PROTOCOL.json',HERE/'fresh_serial_queue.py']
    for p in code:
        blob=subprocess.check_output(['git','show',source+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=sha(p):raise ValueError('Selected runtime source drift: '+str(p))
    gate=load(v['preflight_local']);gate_sha=sha(v['preflight_local'])
    if gate['state']!='PREFLIGHT_ACCEPTED' or gate['source_commit']!=source:raise ValueError('Wrong gate source')
    ctl,t=services();d=HERE/tag
    if ctl.inspect('s4_at_serial_seed0_20261029_v2')['status']['state']!='aborted':raise ValueError('Old original is not terminal aborted')
    capacity=admission(t)
    if not capacity['admitted']:raise ValueError('Fresh capacity refused: '+str(capacity))
    if hashlib.sha256(t.download(v['preflight_remote'])).hexdigest()!=gate_sha:raise ValueError('Actual remote gate bytes differ')
    if a.action=='prepare':
        d.mkdir(exist_ok=False);s=load(ROOT/'execution/s4_at_cuda_native_20261029_gpu03.json')
        s.update(run_id=tag,project='original-s4-fresh-four-arm-AT-after-authorized-abandonment')
        s['source'].update(commit=source,allowed_post_commit_paths=['execution/','docs/S4_AT_ROBUSTNESS_20261027/'])
        for key,folder in (('stage_root','stage'),('repo_root','releases'),('control_root','control'),('output_root','outputs')):s['remote'][key]=owned+'/'+folder+'/'+tag
        s['session']['name']=tag;s['anchors']=[];entry='execution/direct_stop_20261030/fresh_serial_queue.py'
        s['workload']=dict(cwd='.',argv=['systemd-run','--user','--scope','--quiet','--unit=s4at-fresh-'+hashlib.sha256(tag.encode()).hexdigest()[:20],
            PYTHON,'-B',entry,'--out',s['remote']['output_root']+'/result','--run-id',tag,'--source-commit',source,'--preflight',v['preflight_remote']])
        for h in s['health'].values():h.update(adapter_argv=[PYTHON,'-B','-c',ADAPTER],gpu_utilization_policy='advisory',timeout_seconds=1800)
        names=['pipeline_state.json','CONTROLLER_LOCK.json','CONTROLLER_TERMINAL.json','AT_MAIN_RESULTS.csv','AT_PER_CLASS_RESULTS.csv','AT_PAIRED_SCORES.json','ST_REFERENCE_ONLY.json']
        for arm in protocol()['order']:names += [arm+'/accepted.json',arm+'/identity.json',arm+'/audit/initialization.json',arm+'/last_primary/metrics.json',arm+'/best_secondary/metrics.json']
        s['artifacts']=[dict(path='result/'+n,required=True) for n in names];s['local_pull_root']=str(ROOT/'execution/d30p')
        s['metadata']=dict(execution_purpose='authorized_fresh_epoch0_four_serial_200epoch_AT_arms',candidate_commit=source,
            protocol_sha256=v['protocol_sha256'],preflight_sha256=gate_sha,reviewed_once=True,cleanup_policy='retain_all',
            old_original='ABORTED_NOT_ACCEPTED',old_checkpoint_resume=False,no_own_concurrent_GPU=True,
            selection_sha256=sha(a.selection),official_metrics_disabled=False,artifact_ingest_disabled=False)
        RunSpec.from_dict(s);write_once(d/'runspec.json',s);write_once(d/'prepare_capacity.json',capacity)
        print(json.dumps(dict(prepared=tag,launched=False,source=source)));return
    if (d/'launch_intent.json').exists():raise FileExistsError('Intended or bound new formal is inspect-only')
    s=load(d/'runspec.json');spec=RunSpec.from_dict(s)
    if s['source']['commit']!=source or s['metadata']['preflight_sha256']!=gate_sha:raise ValueError('Prepared formal binding changed')
    code="import json,sys;from pathlib import Path;paths=json.loads(sys.argv[1]);assert all(not Path(p).exists() for p in paths),'New formal path collision';print('all_new_paths_absent')"
    check=t.run([PYTHON,'-B','-c',code,json.dumps([s['remote'][k] for k in ('stage_root','repo_root','control_root','output_root')])])
    if check.returncode:raise ValueError('New formal collision, retained all paths')
    ready=ctl.ready(spec);write_once(d/'READY.json',ready)
    if not ready['ready']:raise ValueError('RRCTL formal readiness refused')
    write_once(d/'launch_intent.json',dict(UTC=now(),source_commit=source,preflight_sha256=gate_sha))
    result=ctl.launch(spec);write_once(d/'launch.json',result)
    print(json.dumps(dict(new_formal_started=True,run_id=tag,source=source,result=result)))


if __name__=='__main__':main()
