"""Native Victim-only recovery and bounded actual-path AMP checks.

No noise optimization/generation or spatial-map computation is reachable here.
"""
import argparse
import csv
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
CONFIG=ROOT/'docs/S4_AMP_RECOVERY_20261018/VICTIM_RECOVERY_CONFIG.json'

def baseline_prefix_without_update_is_valid(name,rows):
    """Permit only fully logged native AMP skips in the bounded S0 pair."""
    return name in ('native_S0','repaired_S0') and bool(rows) and all(
        r.get('native_scaler_nonfinite_skip') is True
        and r.get('backoff') is True
        and r.get('optimizer_update_executed') is False
        for r in rows
    )

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(p,value):
    Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def prepare_output(out,cfg):
    if not out.is_relative_to(Path(cfg['remote_root'])/'outputs'):raise ValueError('Output outside authorized recovery boundary')
    out.mkdir(parents=True,exist_ok=False)
    for key,name in [('YOLO_CONFIG_DIR','yolo'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','cache'),('TORCH_HOME','torch')]:
        d=out/'runtime'/name;d.mkdir(parents=True);os.environ[key]=str(d)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')

def dataset_view(dataset,out):
    # Only new-root cache metadata is writable. Images/labels remain the same
    # original files, in exactly the same sorted image-ID order.
    view=out/'data_view'
    for kind in ('images','labels'):
        (view/kind).mkdir(parents=True,exist_ok=False)
        for split in ('train','val'):
            directory=view/kind/split
            directory.mkdir(exist_ok=False)
            allowed=('.jpg','.jpeg','.png') if kind=='images' else ('.txt',)
            for path in sorted((dataset/kind/split).iterdir()):
                if path.is_file() and path.suffix.lower() in allowed:
                    (directory/path.name).symlink_to(path)
    return view

def check_admission(cfg):
    parent=Path(cfg['parent_output'])
    assert sha(parent/'noise/checkpoints/step_009144_epoch_24_cursor_00000.pt')==cfg['parent_noise_checkpoint_sha256']
    assert sha(parent/'dataset_integrity.json')==cfg['parent_dataset_integrity_sha256']
    assert sha(parent/'materialization_manifest.json')==cfg['parent_materialization_manifest_sha256']
    assert sha(ROOT/'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json')==cfg['frozen_s4_config_sha256']
    audit=Path(cfg['remote_root'])/'outputs'/cfg['audit_run_id']/'dataset_reuse_audit.json'
    result=json.loads(audit.read_text());assert result['state']=='passed'
    assert result['poisoned_dataset_manifest_sha256']==cfg['parent_materialization_manifest_sha256']
    assert result['audit_script_sha256']==sha(ROOT/'scripts/s4_reuse_dataset_audit.py')
    for rel,expected in json.loads((ROOT/'docs/tfss/S0_CANONICAL_SOURCE_SHA256.json').read_text()).items():
        assert sha(ROOT/rel)==expected
    import importlib.metadata
    environment=json.loads((audit.parent/'runtime_and_S0_provenance.json').read_text())
    assert importlib.metadata.version('torch')==environment['torch']
    assert importlib.metadata.version('ultralytics')==environment['ultralytics']
    native=Path(sys.prefix)/'lib/python3.11/site-packages/ultralytics/engine/trainer.py'
    assert sha(native)==environment['native_trainer_sha256']
    return {'audit_path':str(audit),'audit_sha256':sha(audit),'audit':result}

class SmokeBoundary(Exception):pass

def trial(name,out,cfg):
    prepare_output(out,cfg)
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ue_framework.s4_victim import S4VictimTrainer,attach_safety_callbacks
    from ue_framework.s4_amp_observer import attach_amp_observer,tree_hash
    from ue_framework.methods.s4_tfss import rng_state
    from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES
    assert S4VictimTrainer.optimizer_step is DetectionTrainer.optimizer_step
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    s0=json.loads((ROOT/'docs/tfss/SOURCE_METADATA.json').read_text())['s0_poison_data']['path']
    dataset=Path(s0 if name in ('native_S0','repaired_S0') else cfg['poisoned_dataset_path'])
    view=dataset_view(dataset,out)
    yaml=out/'train_data.yaml';yaml.write_text(f'path: {view}\ntrain: images/train\nval: images/val\nnames:\n'+'\n'.join(f'  {i}: {n}' for i,n in enumerate(NAMES))+'\n')
    args=dict(cfg['victim_protocol']['train_args']);args.update(data=str(yaml),project=str(out/'victim_run'),name='victim',patience=0)
    init=ROOT/'configs/voc_yolov8n_20cls.yaml';assert sha(init)==cfg['victim_protocol']['init_yaml_sha256']
    model=YOLO(str(init))
    reference=name=='native_S0'
    holder=attach_amp_observer(model,out/'amp_audit',enforce=False,debug_hashes=True) if reference else attach_safety_callbacks(model,amp_log_dir=out/'amp_audit',debug_hashes=True)
    limit=5 if name in ('native_S0','repaired_S0') else (16 if name=='real_S4' else 32)
    synthetic={'forced_batch':None,'forced_after_actual_updates':None,'multiplier':1e20,'recovered':False}
    if name=='synthetic_S4':
        def inject(t):
            original=t.model.loss
            def isolated_loss(batch,preds=None):
                result=original(batch,preds)
                observer=holder['observer']
                if synthetic['forced_batch'] is None and observer.updates>=1:
                    synthetic['forced_batch']=observer.total_batches
                    synthetic['forced_after_actual_updates']=observer.updates
                    return result[0]*synthetic['multiplier'],result[1]
                return result
            t.model.loss=isolated_loss
        model.add_callback('on_train_start',inject)
    def boundary(t):
        o=holder['observer']
        if name=='synthetic_S4' and synthetic['forced_batch'] is not None:
            forced=o.rows[synthetic['forced_batch']-1]
            if not forced['native_scaler_nonfinite_skip'] or not forced['backoff'] or not forced['parameters_unchanged_on_skip']:
                raise AssertionError('Isolated forced overflow did not take unchanged native skip/backoff')
            synthetic['recovered']=o.updates>synthetic['forced_after_actual_updates']
        done=o.total_batches>=limit or (name=='synthetic_S4' and o.total_batches>=5 and synthetic['recovered'])
        if done:
            if not o.updates and not baseline_prefix_without_update_is_valid(name,o.rows):
                raise AssertionError('No actual native optimizer update observed outside an exactly logged S0 AMP-skip equivalence prefix')
            if name=='synthetic_S4' and not synthetic['recovered']:raise AssertionError('No subsequent normal update after forced overflow')
            assert all(r['raw_loss_finite'] and r['model_finite'] and r['EMA_finite'] and r['optimizer_finite'] for r in o.rows)
            snapshot=out/'retained_smoke_state.pt'
            torch.save({'model':t.model.state_dict(),'optimizer':t.optimizer.state_dict(),'scheduler':t.scheduler.state_dict(),
                        'scaler':t.scaler.state_dict(),'EMA':t.ema.ema.state_dict(),'rng':rng_state(),'actual_updates':o.updates},snapshot)
            write(out/'trial_summary.json',{'state':'passed','trial':name,'trainer':type(t).__module__+'.'+type(t).__name__,
                'native_optimizer_inherited':S4VictimTrainer.optimizer_step is DetectionTrainer.optimizer_step,'initial_state':o.initial,
                'rows':o.rows,'actual_updates':o.updates,'skips':o.skips,'batches':o.total_batches,
                'synthetic':synthetic if name=='synthetic_S4' else None,'train_args':args,'checkpoint_retained':str(snapshot),'checkpoint_sha256':sha(snapshot),
                'official_metrics_disabled':True,'validation_forward_batches':0})
            raise SmokeBoundary('Registered bounded actual victim prefix completed')
    model.add_callback('on_train_batch_end',boundary)
    try:model.train(trainer=DetectionTrainer if reference else S4VictimTrainer,**args)
    except SmokeBoundary:return
    raise AssertionError('Bounded trial returned without boundary receipt')

def compare_smoke(out):
    trials={n:json.loads((out/n/'trial_summary.json').read_text()) for n in ('native_S0','repaired_S0','real_S4','synthetic_S4')}
    a,b=trials['native_S0'],trials['repaired_S0']
    assert a['initial_state']==b['initial_state'],'Different native/candidate initialization state'
    assert len(a['rows'])==len(b['rows'])==5
    for x,y in zip(a['rows'],b['rows']):
        assert x['input_tensor_sha256']==y['input_tensor_sha256'] and x['input_ids']==y['input_ids'],'Different actual transformed batch input'
        assert x['raw_loss']==y['raw_loss'],'Loss differs'
        assert x['state']==y['state'],'Model/optimizer/scaler/scheduler/EMA differs'
        for k in ('optimizer_update_executed','native_scaler_nonfinite_skip','amp_scale_before','amp_scale_after'):
            assert x[k]==y[k],k
    return {'state':'passed','baseline_equivalence_max_difference':0.,'input_hashes_exact':True,
            'native_reference':'actual installed DetectionTrainer on identical S0 inputs; historical installed source unknown',
            'REAL_S4_VICTIM_BATCH_TEST':{k:trials['real_S4'][k] for k in ('batches','actual_updates','skips')},
            'SYNTHETIC_OVERFLOW_TEST':{k:trials['synthetic_S4'][k] for k in ('batches','actual_updates','skips','synthetic')},
            'total_actual_batches':sum(t['batches'] for t in trials.values()),'finite_loss':True,'finite_models_EMA_optimizer':True,
            'production_entrypoint_reached':True,'checkpoints_retained':[t['checkpoint_retained'] for t in trials.values()],
            'official_metrics_disabled':True,'artifact_ingest_disabled':True,'deleted_paths':[]}

def final_reports(out,cfg,commit):
    result=json.loads((out/'comparison.json').read_text());completion=json.loads((out/'victim_completion.json').read_text())
    reports=out/'reports';reports.mkdir(exist_ok=False)
    keys=['Method','Person_AP50','NT_mAP50','NT_mAP75','S','P','U','WCD','TCD3']
    with (reports/'S4_S0_S1_S2_S3_COMPARISON.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader()
        for name in ('S0','S1','S2','S3','S4'):
            m=result['metrics'][name];w.writerow(dict(zip(keys,[name,m['mAP50_target'],m['mAP50_non_target'],m['mAP75_non_target'],m['selectivity']['S'],m['selectivity']['P'],m['selectivity']['U'],m['collateral']['WCD'],m['collateral']['TCD3']])))
    for source,dest in [('S4_evaluation.json','S4_LAST_EVALUATION.json'),('voc20_comparison.csv','S4_VOC20_PER_CLASS.csv')]:
        (reports/dest).write_bytes((out/source).read_bytes())
    (reports/'S4_VICTIM_FINAL_REPORT.md').write_text('# S4 Victim recovery\n\n'+json.dumps({'recovery_run_id':out.name,'repair_source_commit':commit,'parent_noise_run_id':cfg['parent_noise_run_id'],'completion':completion,'outcome':result['outcome'],'delta':result['delta']},indent=2)+'\n\nFresh seed0 random initialization on the preserved S4 dataset; not an exact interrupted-batch resume. Last@200 primary, best secondary. Original native AMP/SGD/EMA/scheduler and matched200epoch protocol. Single-seed descriptive results; no significance or causal claims.\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','formal','trial'],required=True);p.add_argument('--output',required=True);p.add_argument('--trial');a=p.parse_args()
    cfg=json.loads(CONFIG.read_text());out=Path(a.output)
    if a.phase=='trial':trial(a.trial,out,cfg);return
    prepare_output(out,cfg);admission=check_admission(cfg)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    state={'state':'running','stage':a.phase,'run_id':out.name,'source_commit':commit,'parent_noise_run_id':cfg['parent_noise_run_id'],'started_at_unix':time.time()}
    def mark(**fields):state.update(fields);write(out/'pipeline_status.json',state);print(json.dumps(state),flush=True)
    mark()
    try:
        write(out/'RECOVERY_PROVENANCE.json',dict(cfg,repair_commit_sha=commit,admission=admission,original_failure_report='docs/S4_AMP_RECOVERY_20261018/AMP_FAILURE_ROOT_CAUSE.md'))
        if a.phase=='smoke':
            for name in ('native_S0','repaired_S0','real_S4','synthetic_S4'):
                mark(stage=name)
                subprocess.run([sys.executable,'-B',__file__,'--phase','trial','--trial',name,'--output',str(out/name)],check=True,timeout=900,cwd=ROOT)
            summary=compare_smoke(out);write(out/'smoke_summary.json',summary);mark(state='complete',stage='actual_victim_smoke_complete',completed_at_unix=time.time());return
        assert out.name==cfg['formal_run_id']
        smoke=Path(cfg['remote_root'])/'outputs'/cfg['smoke_run_id']/'smoke_summary.json'
        assert json.loads(smoke.read_text())['state']=='passed'
        smoke_status=json.loads((smoke.parent/'pipeline_status.json').read_text())
        assert smoke_status['state']=='complete' and smoke_status['source_commit']==commit
        # Re-read every retained image/label hash immediately before formal use.
        manifest=json.loads((Path(cfg['parent_output'])/'materialization_manifest.json').read_text())
        for row in manifest:
            assert sha(row['output'])==row['output_sha256']
            label=Path(cfg['poisoned_dataset_path'])/'labels/train'/(row['id']+'.txt');assert sha(label)==row['label_sha256']
        clean_root=Path(manifest[0]['source']).parents[2]
        for image in (clean_root/'images/val').iterdir():
            if image.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
            assert sha(image)==sha(Path(cfg['poisoned_dataset_path'])/'images/val'/image.name)
            label=clean_root/'labels/val'/(image.stem+'.txt')
            assert sha(label)==sha(Path(cfg['poisoned_dataset_path'])/'labels/val'/label.name)
        data=dataset_view(Path(cfg['poisoned_dataset_path']),out)
        from scripts.run_s4_tfss_pipeline import train_evaluate
        from ue_framework.config import load_config
        lock=json.loads((ROOT/'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json').read_text())
        parent=json.loads((ROOT/'docs/S4_TFSS_UTILITY/PARENT_IDENTITY.json').read_text())
        original=load_config(str(ROOT/lock['parent_config_path']))
        import torch
        torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
        mark(stage='victim_training_and_evaluation')
        train_evaluate(original,out,data,parent,lock)
        final_reports(out,cfg,commit)
        mark(state='complete',stage='complete',completed_at_unix=time.time())
    except BaseException as error:
        mark(state='failed',error=repr(error),failed_at_unix=time.time());raise

if __name__=='__main__':main()
