"""Current-source production worker smoke on retained, hash-bound VOC inputs.

No full data census/calibration/export is repeated. Each child reaches the real
worker/Trainer/PGD/native AMP SGD path, then exits and releases its CUDA context.
The completed-epoch resume proof remains a separate finite invocation.
"""
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'execution/cuda_batch_optimization_20261030'))
from robustness_s4_at.common import atomic,commit,load,now,protocol,sha,scientific_source,write_once
from bounded_common import fixture_view

MANIFEST_SHA='ac31138cd158e5888317ff7cc7023c21565b8090b780f3ff6ef962e934c98ed6'
OWNED=Path('/project/guoqinglong/projects/s4_at_robustness_20261027')
FIXTURE=OWNED/'validation/s4_at_cuda_native_20261029_gpu03/result/fixtures'
GT=OWNED/'validation/s4_at_cuda_native_20261029_gpu03/result/GT_and_budget.json'
GT_SHA='a26066386002176abab08bb30e2f17f0a8d06473a82c9460c41034425103f5e1'


def run(args):
    p=protocol();source=commit();batch=p['train_args']['batch']
    if args.source_commit!=source or scientific_source()['uncommitted_files']:raise ValueError('Wrong/uncommitted current production source')
    if os.getuid()!=1014 or not args.out.resolve().is_relative_to(OWNED):raise ValueError('Unowned smoke scope')
    rel=Path(__file__).resolve().relative_to(ROOT).as_posix();blob=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
    if hashlib.sha256(blob).hexdigest()!=sha(__file__):raise ValueError('Uncommitted production-reaching smoke entry')
    if sha(GT)!=GT_SHA or sha(FIXTURE/'selected_train_identity.json')!=MANIFEST_SHA:raise ValueError('Accepted retained data evidence changed')
    args.out.mkdir(parents=True,exist_ok=False)
    status=dict(state='RUNNING',source_commit=source,started_UTC=now(),native_batch=batch,
        production_entrypoint='robustness_s4_at.worker.run -> ATTrainer -> native loop',
        official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all',deleted_paths=[])
    atomic(args.out/'status.json',status)
    try:
        fixture_records={}
        for arm in ('clean','poison'):
            bound=argparse.Namespace(device='0',fixture_root=FIXTURE/arm,fixture_manifest=FIXTURE/'selected_train_identity.json',arm=arm)
            _,records=fixture_view(bound,args.out/'fixtures'/arm)
            fixture_records[arm]=dict(count=len(records),selected_identity_sha256=sha(args.out/'fixtures'/arm/'selected_identity.json'))
        write_once(args.out/'GT_and_budget.json',dict(state='passed',retained_full_16551_4952_evidence=str(GT),
            retained_full_evidence_sha256=GT_SHA,retained_fixture_manifest_sha256=MANIFEST_SHA,actual_fixture_byte_checks=fixture_records,
            full_accepted_census_repeated=False,original_S4_and_labels_changed=False))
        summaries={}
        for run_id in p['order']:
            arm='clean' if '_Clean_' in run_id else 'poison';dest=args.out/'arms'/run_id
            env=dict(os.environ,YOLO_CONFIG_DIR=str(args.out/'settings'/run_id),MPLCONFIGDIR=str(args.out/'mpl'),WANDB_MODE='disabled')
            with (args.out/(run_id+'.console.log')).open('xb') as log:
                r=subprocess.run([sys.executable,'-B','-m','robustness_s4_at.worker','--run-id',run_id,'--device','0',
                    '--fixture-root',str(args.out/'fixtures'/arm),'--out',str(dest),'--max-batches','16'],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise ValueError('Production native arm failed: '+run_id)
            s=load(dest/'smoke_summary.json')
            if s['source_commit']!=source or s['state']!='PASSED_BOUNDED_NATIVE_TRAINING' or s['actual_updates']<1 or s['actual_input_batch']!=batch:
                raise ValueError('Current production worker did not pass actual declared batch')
            if s['actual_PGD_forwards']!=10*s['outer_batches'] or not s['finite_loss'] or s['validation_forwards']!=0:raise ValueError('PGD/native finite/no-validation gate failed')
            if s['latest_PGD']['actual_raw_devices']!=[['cuda:0',0]]:raise ValueError('Raw actual GPU tensors missing')
            summaries[run_id]=s;status.update(current_run=run_id,completed_bounded_arms=list(summaries));atomic(args.out/'status.json',status)
        initials={s['initial_model_sha256'] for s in summaries.values()};steps=sum(s['outer_batches'] for s in summaries.values())
        if len(initials)!=1 or not 1<=steps<=64:raise ValueError('Fresh initialization or finite total native budget mismatch')
        result=dict(status,state='passed',completed_UTC=now(),arms=summaries,initial_model_sha256=next(iter(initials)),
            actual_outer_batches=steps,step_budget=64,actual_device='cuda:0',finite_loss=True,source_closure=scientific_source(),
            production_entrypoint_reached=True,formal_queue_launched=False,full_GT_and_static_budget_passed=True,
            resume_evidence_is_separate=True,checkpoint_cleanup_completed=False,
            checkpoint_paths_remaining=[str(args.out/'arms'/run_id/'retained_bounded_state.pt') for run_id in p['order']])
        write_once(args.out/'smoke_summary.json',result);atomic(args.out/'status.json',result)
    except BaseException as exc:
        status.update(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),ended_UTC=now());atomic(args.out/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source-commit',required=True);run(p.parse_args())
