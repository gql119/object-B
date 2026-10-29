"""Bounded native CUDA entrypoints and a real completed-epoch resume probe.

This never launches the formal queue, evaluates validation images, or changes
the accepted Original S4 data. All checkpoints and failed probes are retained.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

from .common import ROOT, PROTOCOL, atomic, commit, load, now, protocol, sha, write_once


class ProbeComplete(Exception):
    pass


def fixtures(out, checked, p, count=720):
    manifest = {r['id']: r for r in load(Path(p['parent_output']) / 'materialization_manifest.json')}
    selected = sorted(checked['train'])[:count]
    if len(selected) != count:
        raise ValueError('CUDA probe needs 720 distinct accepted training images')
    rows = []
    for arm, images in (('clean', checked['train']), ('poison', checked['poison'])):
        root = out / arm
        (root / 'images/train').mkdir(parents=True)
        (root / 'labels/train').mkdir(parents=True)
        for stem in selected:
            image = images[stem]
            label = Path(p['clean_root'] if arm == 'clean' else p['poison_root']) / 'labels/train' / (stem + '.txt')
            if sha(label) != manifest[stem]['label_sha256']:
                raise ValueError('CUDA fixture GT differs from accepted static data')
            digest = sha(image)
            if arm == 'poison' and digest != manifest[stem]['output_sha256']:
                raise ValueError('CUDA poison fixture bytes differ from the accepted manifest')
            (root / 'images/train' / image.name).symlink_to(image.resolve())
            (root / 'labels/train' / label.name).symlink_to(label.resolve())
            rows.append(dict(arm=arm, stem=stem, image_sha256=digest, label_sha256=sha(label)))
    write_once(out / 'selected_train_identity.json', dict(count_per_arm=count, selection='first 720 sorted accepted training IDs', records=rows))
    return selected


def close_loaders(trainer):
    for name in ('train_loader', 'test_loader'):
        loader = getattr(trainer, name, None)
        if loader is not None and hasattr(loader, 'close'):
            loader.close()


def resume_probe(out, fixture_root, expected_initial):
    """Reach native epoch 1, then the production resume method and another SGD."""
    import numpy as np
    import torch
    from ue_framework.s4_amp_observer import tree_hash
    from .checkpoint import restore
    from .common import run_config
    from .data import data_view
    from .trainer import ATTrainer
    p = protocol(); out.mkdir(parents=True, exist_ok=False)
    cfg = run_config('S4AT4_Clean_seed0')
    data = data_view(out, 'clean', fixture_root=fixture_root)
    identity = dict(run_id='CUDA_CONTROLLED_SAME_RUN_RESUME', source_commit=commit(),
                    protocol_sha256=sha(PROTOCOL), data_audit_sha256=sha(out / 'data_audit.json'),
                    attack=cfg['attack'], arm='clean', validation_only=True)
    overrides = dict(p['train_args'], model=str(ROOT / 'configs/voc_yolov8n_20cls.yaml'),
                     data=str(data), project=str(out), name='training', device='0',
                     epochs=200, batch=36, imgsz=640, workers=16, amp=True,
                     val=False, save=True, plots=False, cache=False)
    def seed():
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    seed()
    first = ATTrainer(at_config=cfg['attack'], identity=identity, audit_root=out/'audit',
                      validation_only=True, overrides=overrides)
    def no_val(t):raise AssertionError('CUDA resume probe must not run validation')
    first.add_callback('on_val_start', no_val)
    def started(t):
        if not t.amp or not t.scaler.state_dict():
            raise ValueError('CUDA resume probe did not enable the actual native AMP scaler')
        if tree_hash(dict(t.model.named_parameters())) != expected_initial:
            raise ValueError('Resume probe did not start from the same fresh random YAML')
    first.add_callback('on_train_start', started)
    def completed_epoch(t):
        if t.epoch != 0 or t.at_holder['observer'].updates < 1:
            raise ValueError('Resume probe requires an actual completed epoch and native SGD')
        raise ProbeComplete()
    first.add_callback('on_fit_epoch_end', completed_epoch)
    try:
        first.train()
    except ProbeComplete:
        pass
    finally:
        close_loaders(first)
    checkpoint=Path(first.save_dir)/'full_state/epoch_001.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['completed_epochs'] != 1 or not payload['resume_safe_optimizer_boundary']:
        raise ValueError('Native epoch checkpoint is not a safe cleared optimizer boundary')
    first_batches=first.at_holder['observer'].total_batches
    if not 1 <= first_batches <= 20:
        raise ValueError('Completed-epoch resume fixture exceeded 20 native batches')
    del first
    seed()
    resumed=ATTrainer(at_config=cfg['attack'],identity=identity,audit_root=out/'resume_audit',
                      validation_only=True,resume_fullstate=checkpoint,overrides=overrides)
    resumed.add_callback('on_val_start',no_val)
    proof={}
    def verify_restored(t):
        if t.start_epoch != 1 or not t.amp or not t.scaler.state_dict():
            raise ValueError('Production Trainer resume did not consume completed epoch 1')
        for name,actual,expected in (
            ('model',t.model.state_dict(),payload['model']),
            ('optimizer',t.optimizer.state_dict(),payload['optimizer']),
            ('scaler',t.scaler.state_dict(),payload['scaler']),
            ('EMA',t.ema.ema.state_dict(),payload['ema'])):
            if tree_hash(actual)!=tree_hash(expected):
                raise ValueError('CUDA production restore changed '+name)
        if not torch.equal(t.attack.generator.get_state().cpu(),payload['pgd_rng'].cpu()):
            raise ValueError('Production resume did not restore the PGD RNG')
        before=tree_hash(t.model.state_dict())
        try:
            restore(payload,t.model,t.optimizer,t.scheduler,t.scaler,t.attack,
                    dict(identity,run_id='DIFFERENT_ARM'),t.ema,t.train_loader)
        except ValueError:
            proof['cross_arm_restore_rejected']=True
        else:raise ValueError('Cross-arm CUDA restore was accepted')
        if tree_hash(t.model.state_dict())!=before:
            raise ValueError('Rejected restore changed the model')
        proof['restored_before_next_batch_exact']=True
    resumed.add_callback('on_train_start',verify_restored)
    def resumed_step(t):
        observer=t.at_holder['observer']
        if observer.updates>=1:
            if (not bool(torch.isfinite(t.loss).all()) or t.at_latest['actual_input_shape'][0]!=36
                    or t.at_latest['actual_raw_devices'] != [('cuda:0',0)]):
                raise ValueError('Resumed CUDA next update is not finite native batch36')
            proof.update(actual_resumed_updates=observer.updates,resumed_batches=observer.total_batches,
                         latest_PGD=t.at_latest,finite_loss=True)
            raise ProbeComplete()
        if observer.total_batches>=12:
            raise ValueError('No resumed native SGD update within 12 batches')
    resumed.add_callback('on_train_batch_end',resumed_step)
    try:
        resumed.train()
    except ProbeComplete:
        pass
    finally:
        close_loaders(resumed)
    if not proof.get('actual_resumed_updates'):
        raise ValueError('CUDA same-run continuation was not reached')
    proof.update(state='passed',checkpoint_sha256=sha(checkpoint),completed_epochs=1,
                 initial_batches=first_batches,actual_device=str(resumed.device),
                 native_batch=36,imgsz=640,workers=16,AMP=True,
                 data_loader_worker_prefetch_rng_verified=False,
                 bitwise_native_loader_trajectory_claim=False,
                 CUDA_peak_allocated_MiB=torch.cuda.max_memory_allocated(0)/2**20)
    write_once(out/'resume_summary.json',proof)
    return proof


def run(args):
    from .data import audit_accepted_inputs
    from .queue import resource_status
    from .common import native_identity,scientific_source
    p=protocol();source=commit()
    if source!=args.source_commit or scientific_source()['uncommitted_files']:
        raise ValueError('CUDA validation requires its exact committed source')
    if not args.out.resolve().is_relative_to(Path(p['remote_owned_root']).resolve()):
        raise ValueError('CUDA validation escaped its owned output root')
    resources=resource_status()
    if not resources['admitted']:raise ValueError('AT CUDA resources/COCO closure blocked: '+str(resources))
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ['YOLO_CONFIG_DIR']=str(args.out/'settings/parent')
    os.environ['MPLCONFIGDIR']=str(args.out/'mpl')
    os.environ['WANDB_MODE']='disabled'
    status=dict(state='RUNNING',source_commit=source,started_UTC=now(),official_metrics_disabled=True,
                artifact_ingest_disabled=True,cleanup_policy='retain_all',deleted_paths=[],resources=resources)
    atomic(args.out/'status.json',status)
    try:
        native=native_identity();checked=audit_accepted_inputs(p)
        audit={k:v for k,v in checked.items() if k not in ('train','val','poison')}
        audit.update(state='passed',train_count=len(checked['train']),val_count=len(checked['val']),
                     all_training_labels_byte_unchanged=True,original_S4_unchanged=True)
        write_once(args.out/'GT_and_budget.json',audit)
        fixtures(args.out/'fixtures',checked,p)
        summaries={}
        for run_id in p['order']:
            arm='clean' if '_Clean_' in run_id else 'poison';out=args.out/'arms'/run_id
            env=dict(os.environ,YOLO_CONFIG_DIR=str(args.out/'settings'/run_id),MPLCONFIGDIR=str(args.out/'mpl'),WANDB_MODE='disabled')
            log=args.out/(run_id+'.console.log')
            with log.open('xb') as stream:
                result=subprocess.run([sys.executable,'-B','-m','robustness_s4_at.worker','--run-id',run_id,
                    '--device','0','--fixture-root',str(args.out/'fixtures'/arm),'--out',str(out),
                    '--max-batches','16'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Bounded AT arm failed: '+run_id)
            summary=load(out/'smoke_summary.json')
            if summary['state']!='PASSED_BOUNDED_NATIVE_TRAINING' or summary['actual_updates']<1 or summary['actual_input_batch']!=36:
                raise ValueError('AT native CUDA arm did not pass: '+run_id)
            import torch
            state=torch.load(out/'retained_bounded_state.pt',map_location='cpu',weights_only=False)
            if not state['scaler'] or state['scaler'].get('scale',0)<=0:
                raise ValueError('Bounded CUDA arm did not use the actual native AMP scaler')
            summary=dict(summary,actual_AMP_enabled=True,scaler_state=state['scaler'],retained_checkpoint_sha256=sha(out/'retained_bounded_state.pt'))
            summaries[run_id]=summary
            status.update(completed_bounded_arms=list(summaries),current_run=run_id);atomic(args.out/'status.json',status)
        initial={s['initial_model_sha256'] for s in summaries.values()}
        if len(initial)!=1:raise ValueError('Four CUDA AT arms did not match their fresh initialization')
        resume=resume_probe(args.out/'same_run_resume',args.out/'fixtures/clean',next(iter(initial)))
        batches=sum(s['outer_batches'] for s in summaries.values())+resume['initial_batches']+resume['resumed_batches']
        if not 1<=batches<=96:raise ValueError('CUDA preflight exceeded its total 96 native-batch budget')
        summary=dict(status,state='passed',completed_UTC=now(),native_sources=native,source_closure=scientific_source(),
                     arms=summaries,initial_model_sha256=next(iter(initial)),resume=resume,
                     actual_outer_batches=batches,total_outer_batch_budget=96,full_GT_and_static_budget_passed=True,
                     actual_device='cuda:0',formal_queue_launched=False,official_metrics_generated=0)
        write_once(args.out/'smoke_summary.json',summary);atomic(args.out/'status.json',summary)
    except BaseException as exc:
        status.update(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),ended_UTC=now())
        atomic(args.out/'status.json',status);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-commit',required=True);args=parser.parse_args();run(args)


if __name__=='__main__':main()
