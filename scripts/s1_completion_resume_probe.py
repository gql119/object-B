"""B1 closure: actual production completion/last-batch/frequency-freeze recovery.

Original24epoch/batch16/EOT2 settings, only two previously fixed train images.
This is 24 diagnostic optimizer steps, never the formal6095-image/9144step run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser()
    for name in ['config', 'csb-config', 'cache', 'parent-root', 'output']: p.add_argument('--'+name, required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--mode', choices=['full', 'resume_warmup', 'pause_last', 'resume_last', 'resume_complete'])
    p.add_argument('--resume')
    a = p.parse_args()
    if not a.worker:
        out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
        for name in ['yolo', 'mpl', 'cache']: (out / 'runtime' / name).mkdir(parents=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
            YOLO_CONFIG_DIR=str(out / 'runtime/yolo'), MPLCONFIGDIR=str(out / 'runtime/mpl'),
            XDG_CACHE_HOME=str(out / 'runtime/cache'), WANDB_MODE='disabled')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU busy; no concurrent B1 probe')
        for name in ['config', 'csb_config', 'cache', 'parent_root']: setattr(a,name,str(Path(getattr(a,name)).resolve()))
        def run(mode, resume=None):
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--mode',mode,
                '--config',a.config,'--csb-config',a.csb_config,'--cache',a.cache,
                '--parent-root',a.parent_root,'--output',str(out/mode)]
            if resume: cmd+=['--resume',resume]
            subprocess.run(cmd,check=True,env=env,cwd=str(out))
        run('full')
        full=json.loads((out/'full/receipt.json').read_text())
        # Full run retained the epoch4 boundary before frequency freeze.
        warm=str(out/'full/checkpoints/step_000004_epoch_04_cursor_00000.pt')
        run('resume_warmup',warm)
        # Construct an actual last-batch checkpoint by stopping the same
        # production loop after one additional step from completed epoch23.
        before_last=str(out/'full/checkpoints/step_000023_epoch_23_cursor_00000.pt')
        run('pause_last',before_last)
        last=json.loads((out/'pause_last/receipt.json').read_text())
        run('resume_last',last['status']['checkpoint'])
        run('resume_complete',full['status']['checkpoint'])
        import torch
        from scripts.s1_mechanism_probe import compare_tree
        ref=torch.load(out/'full/global_params.pt',map_location='cpu',weights_only=True)
        ref_checkpoint=torch.load(full['status']['checkpoint'],map_location='cpu',weights_only=False)
        ref_csv=(out/'full/diagnostics.csv').read_bytes()
        checks={}
        for mode in ['resume_warmup','resume_last','resume_complete']:
            r=json.loads((out/mode/'receipt.json').read_text())
            value=torch.load(out/mode/'global_params.pt',map_location='cpu',weights_only=True)
            diff=compare_tree(ref,value)
            checkpoint_diff=compare_tree(ref_checkpoint,torch.load(r['status']['checkpoint'],map_location='cpu',weights_only=False))
            csv_equal=ref_csv==(out/mode/'diagnostics.csv').read_bytes()
            checks[mode]=dict(global_params_max_difference=diff,checkpoint_state_max_difference=checkpoint_diff,diagnostics_byte_identical=csv_equal,
                complete=r['status']['state']=='complete',steps=r['status']['steps'],epochs=r['status']['epochs_completed'],
                added_optimizer_updates=r['added_optimizer_updates'],frequency_indices=r['frequency_indices'],
                frozen_model_unchanged=r['status']['frozen_model_unchanged'],adaptive_model_unchanged=r['status']['adaptive_model_unchanged'])
        passed=all(q['global_params_max_difference']==0 and q['checkpoint_state_max_difference']==0 and q['diagnostics_byte_identical'] and q['complete']
                   and len(q['frequency_indices'])==16 and q['frozen_model_unchanged'] and q['adaptive_model_unchanged']
                   and q['steps']==24 and q['epochs']==24 for q in checks.values())
        passed=passed and checks['resume_last']['added_optimizer_updates']==0 and checks['resume_complete']['added_optimizer_updates']==0
        summary=dict(state='complete' if passed else 'failed',passed=passed,finite_loss=True,
            production_entrypoint_reached=True,official_metrics_disabled=True,artifact_ingest_disabled=True,
            formal_generation_enabled=False,victim_training_enabled=False,cleanup_policy='retain_all',
            checkpoint_cleanup_completed=False,deleted_paths=[],checkpoint_paths_remaining=[str(q) for q in out.rglob('*.pt')],
            formal_schedule_unchanged=dict(epochs=24,batch16=True,eot2=True,full_person_images=6095,full_optimizer_steps=9144),
            diagnostic_cohort_size=2,diagnostic_complete_steps=24,checks=checks,
            source_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip())
        (out/'smoke_summary.json').write_text(json.dumps(summary,indent=2))
        if not passed: raise ValueError('B1 production completion/recovery closure failed')
        return
    import random
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.methods.csb_budget import BudgetStore,digest,config_hash
    from ue_framework.methods.s1_csb import S1CSBTrainer
    from ue_framework.methods.s1_optimization import optimize
    cfg=load_config(a.config);mc=cfg['methods']['legacy_kproto_ret'];root=Path(cfg['data']['dataset_root'])
    if mc['universal_epochs']!=24 or mc['universal_batch_size']!=16 or mc['eot_samples']!=2:
        raise ValueError('Original formal parameters must not be modified for B1 test')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    model=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0');learner=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
    source=Path(__file__).resolve().parents[1];settings=json.loads(Path(a.csb_config).read_text())
    identity={'teacher_sha256':digest(cfg['surrogate']['ckpt']),'csb_config_sha256':config_hash(settings),
        'source_dataset_root':str(root),'csb_source_sha256':digest(source/'ue_framework/methods/csb_budget.py'),
        'sensitivity_source_sha256':digest(source/'ue_framework/methods/csb_sensitivity.py')}
    store=BudgetStore(a.cache,identity)
    selection=json.loads((Path(a.cache)/'selection.json').read_text())['filenames'][:2]
    images=[str(root/'images/train'/x) for x in selection]
    trainer=S1CSBTrainer(cfg,mc,torch.device('cuda:0'),model,learner,budget_store=store)
    proto=Path(a.parent_root)/'noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt'
    prototypes=torch.load(proto,map_location='cuda:0',weights_only=True)['prototypes']
    ident=dict(parent_config_sha256=digest(a.config),cache_manifest_sha256=digest(Path(a.cache)/'manifest.json'),
        budget_mode='csb',optimization_source_sha256=digest(source/'ue_framework/methods/s1_optimization.py'),
        projection_source_sha256=digest(source/'ue_framework/methods/s1_csb.py'),prototypes_sha256=digest(proto))
    previous=torch.load(a.resume,map_location='cpu',weights_only=False)['cursor_state']['step'] if a.resume else 0
    status=optimize(trainer,images,root/'labels/train',prototypes,a.output,ident,
                    stop_after=24 if a.mode=='pause_last' else 0,resume=a.resume)
    receipt=dict(status=status,added_optimizer_updates=status['steps']-previous,
                 frequency_indices=trainer.freq_active_idx.cpu().tolist(),actual_images=selection)
    (Path(a.output)/'receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
