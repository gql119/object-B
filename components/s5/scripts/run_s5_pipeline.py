"""Single authorized S5: current-source smoke -> reviewed full serial schedule."""
import argparse
import copy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BOUNDARY=Path('/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018')
from scripts.s5_report import sha,write,table,comparison_tables


def verify_lock(lock):
    from scripts.run_s4_tfss_pipeline import sha as original_sha
    s=lock['s5'];parent=json.loads((ROOT/'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json').read_text())
    if lock['schema']!='s5.formal-lock.v1' or s['selected_retention_layer']!='P3':raise ValueError('Selected layer identity')
    allowed={'schema','method','parent','run_id','authorization_sha256','s5','limits','success_gate'}
    if any(lock[k]!=v for k,v in parent.items() if k not in allowed):raise ValueError('Original S4 science/protocol changed')
    checks={'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json':s['S4_config_sha256'],
        s['calibration_path']:s['calibration_sha256'],s['smoke_protocol_path']:s['smoke_protocol_sha256'],
        s['S4_comparison_path']:s['S4_comparison_sha256'],
        'docs/S5_CUP_S4/P3_P5_LAYER_DECISION.md':s['layer_decision_sha256'],
        'docs/S5_CUP_S4/CUP_S4_CALIBRATION_RULE.md':s['rule_sha256'],
        'docs/S5_CUP_S4/USER_TASK_LAYER_SELECTION_SOURCE.md':lock['authorization_sha256']}
    if any(original_sha(ROOT/p)!=value for p,value in checks.items()):raise ValueError('Frozen S5 evidence identity changed')
    return checks


def victim(cfg,out,data,provenance,lock,mark):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ue_framework.s4_victim import S4VictimTrainer,attach_safety_callbacks
    from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES
    from scripts.run_s4_tfss_pipeline import summarize
    # Writable cache view only; the fully audited generated data stays read-only.
    from scripts.s4_victim_recovery import dataset_view
    view=dataset_view(data,out)
    yaml=out/'train_data.yaml';yaml.write_text(f'path: {view}\ntrain: images/train\nval: images/val\nnames:\n'+'\n'.join(f'  {i}: {n}' for i,n in enumerate(NAMES))+'\n')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    args=dict(provenance['victim_spec']['train_args']);args.update(data=str(yaml),project=str(out/'victim_run'),name='victim',patience=0)
    init=ROOT/cfg['victim']['init']
    if (sha(init)!=provenance['victim_spec']['init_yaml_sha256'] or args['epochs']!=200 or args['seed']!=0
            or args['batch']!=36 or args['workers']!=16 or args['amp'] is not True or args.get('pretrained') is not False):
        raise ValueError('Matched S5 victim protocol changed')
    write(out/'victim_run_spec.json',dict(initialization='fresh_random_yaml',init_yaml_sha256=sha(init),train_args=args))
    model=YOLO(str(init));attach_safety_callbacks(model,amp_log_dir=out/'victim_AMP_audit')
    model.train(trainer=S4VictimTrainer,**args)
    run=out/'victim_run/victim'
    with (run/'results.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=200 or [int(float(list(row.values())[0])) for row in rows]!=list(range(1,201)):
        raise ValueError('Incomplete200 continuous victim rows')
    if not all(np.isfinite(float(value)) for row in rows for key,value in row.items() if 'loss' in key):
        raise FloatingPointError('Nonfinite recorded victim losses')
    write(out/'victim_completion.json',dict(epochs=200,rows=200,last_sha256=sha(run/'weights/last.pt'),best_sha256=sha(run/'weights/best.pt')))
    history=json.loads((ROOT/lock['s5']['S4_comparison_path']).read_text())['metrics']
    metrics={name:copy.deepcopy(history[name]) for name in ('Clean','S0','S1','S2','S3','S4')}
    evaluation=out/'evaluation';evaluation.mkdir(exist_ok=False)
    for role,name in (('S5','last.pt'),('S5_best_secondary','best.pt')):
        mark('evaluation_'+role)
        path=run/'weights'/name
        result=YOLO(str(path)).val(data=str(yaml),imgsz=640,batch=36,workers=16,device='0',verbose=False,plots=False,seed=0,
            project=str(evaluation),name=role,exist_ok=False)
        metrics[role]=dict(summarize(result.box),checkpoint_sha256=sha(path),
            checkpoint_role='last.pt@200' if role=='S5' else 'best.pt secondary',class_names=list(NAMES))
        write(out/(role+'_evaluation.json'),metrics[role])
    comparison,aggregate,perclass=comparison_tables(metrics)
    write(out/'comparison.json',comparison)
    table(out/'S5_S0_S3_S4_COMPARISON.csv',aggregate);table(out/'S5_VOC20_PER_CLASS.csv',perclass)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('smoke','formal'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--lock',default=str(ROOT/'docs/S5_CUP_S4/S5_CONFIG_LOCK.json'))
    p.add_argument('--review');a=p.parse_args();out=Path(a.output).resolve();lock=json.loads(Path(a.lock).read_text())
    if not out.is_relative_to(BOUNDARY/'outputs') or not ROOT.is_relative_to(BOUNDARY):raise ValueError('S5 write boundary')
    if out.exists():raise FileExistsError('Never relaunch an existing output')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True,cwd=ROOT).strip():
        raise ValueError('Remote scientific source dirty')
    verify_lock(lock)
    if a.phase=='formal':
        review=json.loads(Path(a.review).read_text()) if a.review else {}
        if (review.get('source_commit')!=commit or review.get('pre_run_result')!='pass'
                or review.get('config_sha256')!=sha(a.lock) or out.name!=lock['run_id']):raise ValueError('Current review/config/unique RunID gate')
        smoke=json.loads((BOUNDARY/'outputs'/lock['s5']['smoke_run_id']/'smoke_summary.json').read_text())
        if (smoke['state']!='complete' or smoke['source_commit']!=commit or smoke['formal_config_sha256']!=sha(a.lock)
                or smoke['original_S4_vs_P3_noop_max_difference']>1e-7 or smoke['native_AMP']['state']!='passed'
                or len(smoke['PNG_checks'])!=32 or not all(x['passed'] for x in smoke['resume'])):
            raise ValueError('Current-source production smoke not passed')
    from scripts.run_s4_tfss_pipeline import resource_gate,setup,materialize_and_audit,_resource_snapshot,_resource_blockers,_sustained_cpu_or_io_contention
    resource=resource_gate();out.mkdir(parents=True,exist_ok=False)
    for key,name in (('YOLO_CONFIG_DIR','yolo'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','cache'),('TORCH_HOME','torch')):
        path=out/'runtime'/name;path.mkdir(parents=True);os.environ[key]=str(path)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')
    state=dict(state='running',stage='source_preflight',run_id=out.name,source_commit=commit,config_lock_sha256=sha(a.lock),started_at_unix=time.time(),resource=resource,
        optimizer_updates=0,official_metrics_disabled=a.phase=='smoke',artifact_ingest_disabled=a.phase=='smoke')
    def mark(stage,**fields):
        state.update(stage=stage,**fields);write(out/'pipeline_status.json',state);print(json.dumps(state),flush=True)
    mark('source_preflight')
    try:
        write(out/'config_lock_snapshot.json',lock)
        if a.phase=='smoke':
            from scripts.s5_cup_probe import run
            protocol_path=ROOT/lock['s5']['smoke_protocol_path'];run(out,state,json.loads(protocol_path.read_text()),sha(protocol_path))
            summary=json.loads((out/'smoke_summary.json').read_text());summary['formal_config_sha256']=sha(a.lock)
            write(out/'smoke_summary.json',summary);mark('complete',state='complete',completed_at_unix=time.time());return
        write(out/'review_snapshot.json',review)
        cfg,initial,prototypes,provenance,splits=setup(lock,out);identity_cache=initial.s4_cache.identity
        del initial;gc.collect()
        import torch
        torch.cuda.empty_cache()
        from ue_framework.methods.s5_runtime import create_trainer
        from ue_framework.methods.s5_optimization import optimize_s5
        t=create_trainer(cfg,'P3_CUP',ROOT/lock['s5']['calibration_path'],out/'clean_maps',identity_cache)
        source=Path(cfg['data']['dataset_root']);images=t._collect_target_images(str(source/'images/train'),str(source/'labels/train'))
        if len(images)!=6095:raise ValueError('Full target support missing')
        identity=dict(config_lock_sha256=sha(a.lock),source_commit=commit,parent_method='S4',initialization='original_S0_seed0',
            method='TFSS-U+CUP-S4(P3)',eta=.85,dataset_fingerprint=lock['dataset_fingerprint'],cache_identity=identity_cache,
            calibration_sha256=lock['s5']['calibration_sha256'])
        mark('noise_generation');noise=out/'noise'
        result=optimize_s5(t,images,source/'labels/train',prototypes,noise,identity,formal_lock=lock)
        if result.get('schedule_complete') is not True:raise ValueError('Noise schedule incomplete')
        table(out/'S5_TFSS_SPATIAL_STATS.csv',[json.loads(line) for line in (noise/'epoch_summary.jsonl').read_text().splitlines()])
        mark('materialization_and_pixel_audit');data=materialize_and_audit(cfg,t,noise,out,splits)
        if json.loads((out/'dataset_integrity.json').read_text())['state']!='passed':raise ValueError('Dataset audit admission failed')
        t.s5_close_observer();del t,prototypes;gc.collect();torch.cuda.empty_cache()
        while True:
            samples=[]
            for i in range(3):
                samples.append(_resource_snapshot(10.))
                if i<2:time.sleep(30)
            blockers=[_resource_blockers(s,allow_own_gpu=True) for s in samples]
            if not any(blockers) and not _sustained_cpu_or_io_contention(samples):break
            mark('waiting_victim_resources',resource_window=samples,resource_blockers=blockers)
            time.sleep(60)
        mark('victim_training');victim(cfg,out,data,provenance,lock,mark)
        mark('reporting')
        from scripts.s5_report import finalize
        acceptance=finalize(out,ROOT,commit,lock)
        if not acceptance['accepted']:raise ValueError('Final evidence not accepted')
        mark('complete',state='complete',completed_at_unix=time.time(),accepted=True)
    except BaseException as error:
        mark(state['stage'],state='failed',error=repr(error),failed_at_unix=time.time());raise


if __name__=='__main__':main()
