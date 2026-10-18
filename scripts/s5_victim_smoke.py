"""Same native S4 AMP trial, bounded to20 synthetic batches for the S5 smoke budget.

The legacy real_S4/synthetic_S4 trial labels use the supplied S5 development PNGs.
No native trainer/optimizer/loss is changed (synthetic injection is isolated only).
"""
from pathlib import Path
import json,random
from scripts.s4_victim_recovery import prepare_output,dataset_view,sha,write,baseline_prefix_without_update_is_valid,SmokeBoundary
ROOT=Path(__file__).resolve().parents[1]

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
    limit=5 if name in ('native_S0','repaired_S0') else (16 if name=='real_S4' else 20)
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
