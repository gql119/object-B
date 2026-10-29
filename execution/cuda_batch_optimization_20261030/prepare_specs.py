"""Build immutable thin RRCTL specs after exact source publication; never launch."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from bounded_common import ROOT
from robustness_s4_at.common import commit,sha,write_once

HERE=Path(__file__).resolve().parent
PYTHON='/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'
BRANCH='codex/s4-at-robustness-20261027'


def cases():
    result=[]
    for implementation in ('02','03'):
        for arm in ('clean','poison'):
            for epsilon in (4,8):
                result.append(dict(kind='eq',implementation=implementation,arm=arm,epsilon=epsilon,budget=32))
    for implementation in ('original','02','03'):
        # Do not run all choices blindly: only the measured best valid one
        # needs these per-arm continuation cases within the one-hour window.
        for batch in (36,48,64,72):
            for arm in ('clean','poison'):
                for epsilon in (4,8):
                    result.append(dict(kind='resume',implementation=implementation,arm=arm,epsilon=epsilon,batch=batch,repeat=1,budget=64))
            for repeat in range(1,5):
                result.append(dict(kind='bench',implementation=implementation,arm='clean',epsilon=4,
                    batch=batch,repeat=repeat,pool='common720',budget=100))
            if batch!=36:
                result.append(dict(kind='tail',implementation=implementation,arm='poison',epsilon=8,
                    batch=batch,repeat=1,pool='formal-tail',budget=100))
                for arm,epsilon in (('clean',8),('poison',4),('poison',8)):
                    result.append(dict(kind='correctness',implementation=implementation,arm=arm,epsilon=epsilon,
                        batch=batch,repeat=1,pool='common720',budget=100))
        result.append(dict(kind='profile',implementation=implementation,arm='clean',epsilon=4,
            batch=36,repeat=1,pool='common720',budget=100))
    return result


def build(template,binding,source,case):
    s=deepcopy(template)
    label=case['kind']+'_c'+case['implementation']+'_'+case['arm']+str(case['epsilon'])
    if 'batch' in case:label+='_b'+str(case['batch'])+'_r'+str(case['repeat'])
    tag='s4_at_speed_'+label+'_20261030_gpu01'
    owned='/project/guoqinglong/projects/s4_at_robustness_20261027'
    s['run_id']=tag;s['project']='original-s4-bounded-speed-validation'
    s['source']=dict(repo_root=str(ROOT),branch=BRANCH,commit=source,
        allowed_post_commit_paths=['execution/','docs/S4_AT_ROBUSTNESS_20261027/'])
    s['remote'].update(stage_root=owned+'/stage/'+tag,repo_root=owned+'/releases/'+tag,
        control_root=owned+'/control/'+tag,output_root=owned+'/validation/'+tag)
    s['session']['name']=tag;s['anchors']=[]
    entry={'eq':'cuda_equivalence.py','resume':'cuda_resume.py'}.get(case['kind'],'cuda_benchmark.py')
    argv=[PYTHON,'-B','execution/cuda_batch_optimization_20261030/'+entry,
        '--out',s['remote']['output_root']+'/result','--source-commit',source,
        '--implementation',case['implementation'],'--arm',case['arm'],'--epsilon',str(case['epsilon']),
        '--device','0','--fixture-root',str(Path(binding['path']).parent/case['arm']).replace('\\','/'),
        '--fixture-manifest',binding['path'],'--fixture-manifest-sha256',binding['sha256']]
    if case['kind']=='eq':argv+=['--max-batches','16']
    elif case['kind']=='resume':argv+=['--batch',str(case['batch'])]
    else:
        argv+=['--batch',str(case['batch']),'--measure-batches','12','--pool-mode',case['pool']]
        if case['kind']=='profile':argv+=['--profile']
    s['workload']=dict(cwd='.',argv=['systemd-run','--user','--scope','--quiet',
        '--unit=s4at-speed-'+hashlib.sha256(tag.encode()).hexdigest()[:20]]+argv)
    adapter="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};z=s.get('state','');print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=z!='FAILED',complete=z=='passed' or z.startswith('PASS_'),progress={k:s.get(k) for k in ('state','implementation','arm','epsilon_pixels','actual_outer_batches','error')},artifacts=['result/status.json','result/summary.json','result/candidate_resume_summary.json'])))"
    for h in s['health'].values():h.update(adapter_argv=[PYTHON,'-B','-c',adapter],timeout_seconds=900,gpu_utilization_policy='advisory')
    summary='candidate_resume_summary.json' if case['kind']=='resume' else 'summary.json'
    s['artifacts']=[dict(path='result/'+n,required=True) for n in ('status.json',summary)]
    s['local_pull_root']=str(HERE/'pull')
    s['metadata']=dict(execution_purpose='bounded_scientific_speed_validation_not_formal',candidate_commit=source,
        case=case,computation_kind='training',max_total_native_outer_batches=case['budget'],PGD_steps_per_batch=10,
        official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all',
        requires_natural_boundary_hold_and_accepted_CleanAT4=True,requires_fresh_exclusive_GPU=True,
        no_own_concurrent_GPU=True,original_formal_run_unchanged=True,not_a_launch_authorization_now=True)
    return s


def main():
    q=argparse.ArgumentParser();q.add_argument('--source-commit',required=True);a=q.parse_args()
    if a.source_commit!=commit():raise ValueError('Prepare only the exact current committed source')
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=ROOT,text=True).split()
    if not remote or remote[0]!=a.source_commit:raise ValueError('Exact source publication is not verified')
    from remote_run_control.models import RunSpec
    template=json.loads((ROOT/'execution/s4_at_cuda_native_20261029_gpu03.json').read_text())
    binding=json.loads((HERE/'RETAINED_GPU_FIXTURE_BINDING.json').read_text())
    directory=HERE/'prepared_specs';directory.mkdir(exist_ok=False)
    index=[]
    for case in cases():
        value=build(template,binding,a.source_commit,case);RunSpec.from_dict(value)
        path=directory/(value['run_id']+'.json');write_once(path,value)
        index.append(dict(run_id=value['run_id'],path=str(path.relative_to(ROOT)),sha256=sha(path),case=case))
    write_once(directory/'INDEX.json',dict(state='PREPARED_UNLAUNCHED',source_commit=a.source_commit,
        source_publication_verified=True,count=len(index),RunSpecs=index,not_all_cases_should_run=True,
        one_hour_window_limit_remains_mandatory=True))
    print(json.dumps(dict(prepared=len(index),launched=False,source=a.source_commit)))


if __name__=='__main__':main()
