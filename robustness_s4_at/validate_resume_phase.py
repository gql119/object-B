"""Bounded fixed-input native CUDA proof for AT-SCI-001, not formal results."""
import argparse
from copy import deepcopy
from pathlib import Path
import random
import traceback
import numpy as np
import torch
from ue_framework.s4_amp_observer import tree_hash
from .common import ROOT, PROTOCOL, atomic, commit, load, now, protocol, sha, write_once
from .data import audit_accepted_inputs, data_view
from .queue import resource_status
from .trainer import ATTrainer
from .validate_cuda import fixtures, close_loaders


class Complete(Exception): pass


def run(args):
    source=commit();p=protocol()
    if source!=args.source_commit:raise ValueError('Wrong committed phase-fix source')
    resources=resource_status()
    if not resources['admitted']:raise ValueError('Resource/own-GPU conflict')
    args.out.mkdir(parents=True,exist_ok=False)
    status=dict(state='RUNNING',source_commit=source,blocker='AT-SCI-001',UTC=now(),
        official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all',deleted_paths=[])
    atomic(args.out/'status.json',status)
    try:
        checked=audit_accepted_inputs(p);fixtures(args.out/'fixtures',checked,p,count=684)
        firstroot=args.out/'same_run';firstroot.mkdir()
        data=data_view(firstroot,'clean',fixture_root=args.out/'fixtures/clean')
        identity=dict(run_id='CUDA_NATIVE_ACCUMULATE2_PHASE',source_commit=source,
            protocol_sha256=sha(PROTOCOL),data_audit_sha256=sha(firstroot/'data_audit.json'),validation_only=True)
        overrides=dict(p['train_args'],model=str(ROOT/'configs/voc_yolov8n_20cls.yaml'),data=str(data),
            project=str(firstroot),name='training',device='0',epochs=200,batch=36,imgsz=640,
            workers=16,amp=True,val=False,save=True,plots=False,cache=False)
        def seed():
            random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.set_device(0);torch.cuda.manual_seed_all(0)
            torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.use_deterministic_algorithms(True)
        raw_batches=[];reference=[];replayed=[]
        def fingerprint(t):
            return dict(model=tree_hash(t.model.state_dict()),optimizer=tree_hash(t.optimizer.state_dict()),
                scheduler=tree_hash(t.scheduler.state_dict()),scaler=tree_hash(t.scaler.state_dict()),
                EMA=tree_hash(t.ema.ema.state_dict()),EMA_updates=t.ema.updates,
                PGD_RNG=tree_hash(t.attack.generator.get_state()),
                gradients=tree_hash([v.grad for v in t.model.parameters()]),
                last_opt_step=t.at_last_opt_step,accumulate=int(t.accumulate))
        class FixedInputs(ATTrainer):
            def preprocess_batch(self,batch):
                if self.epoch==3:
                    if getattr(self,'replay',False):
                        batch=deepcopy(raw_batches[len(replayed)])
                    else:
                        raw_batches.append(deepcopy(batch))
                return super().preprocess_batch(batch)
        def no_val(t):raise AssertionError('No validation in accumulation phase probe')
        seed();first=FixedInputs(at_config=p['pgd']['4'],identity=identity,audit_root=firstroot/'audit',
            validation_only=True,overrides=overrides)
        first.add_callback('on_val_start',no_val)
        def stop_reference(t):
            if t.at_holder['observer'].total_batches>61:raise ValueError('Phase probe budget exceeded')
            if t.epoch==3:
                reference.append(fingerprint(t))
                if len(reference)==2:raise Complete()
        first.add_callback('on_train_batch_end',stop_reference)
        try:first.train()
        except Complete:pass
        finally:close_loaders(first)
        checkpoint=Path(first.save_dir)/'full_state/epoch_003.pt'
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if not payload['resume_safe_optimizer_boundary'] or payload['native_loop_phase']!=dict(last_opt_step=56,accumulate=2,batches_per_epoch=19):
            raise ValueError('Probe did not reach the real safe post-warmup accumulation2 boundary')
        if len(raw_batches)!=2 or len(reference)!=2 or [v['last_opt_step'] for v in reference]!=[56,58]:
            raise ValueError('Uninterrupted actual native update phase was not [noattempt57,attempt58]')
        write_once(args.out/'reference.json',reference)
        with (args.out/'fixed_transformed_batches.pt').open('xb') as f:torch.save(raw_batches,f)
        initial_batches=first.at_holder['observer'].total_batches
        del first
        seed();resumed=FixedInputs(at_config=p['pgd']['4'],identity=identity,audit_root=firstroot/'resume_audit',
            validation_only=True,resume_fullstate=checkpoint,overrides=overrides)
        resumed.replay=True
        resumed.add_callback('on_val_start',no_val)
        def restored(t):
            if t.at_last_opt_step!=56 or t.accumulate!=2 or t.start_epoch!=3:
                raise ValueError('Native loop phase not restored before actual production continuation')
        resumed.add_callback('on_train_start',restored)
        def compare(t):
            actual=fingerprint(t);expected=reference[len(replayed)]
            if actual!=expected:raise ValueError('Actual uninterrupted/resumed native state mismatch: '+str([k for k in actual if actual[k]!=expected[k]]))
            replayed.append(actual)
            if len(replayed)==2:raise Complete()
        resumed.add_callback('on_train_batch_end',compare)
        try:resumed.train()
        except Complete:pass
        finally:close_loaders(resumed)
        if len(replayed)!=2 or resumed.at_holder['observer'].updates!=1:
            raise ValueError('Expected two resumed batches and one actual SGD')
        result=dict(status,state='passed',completed_UTC=now(),checkpoint_sha256=sha(checkpoint),
            fixed_input_sha256=sha(args.out/'fixed_transformed_batches.pt'),completed_epochs=3,
            native_loop_phase=payload['native_loop_phase'],uninterrupted_last_opt_steps=[56,58],resumed_last_opt_steps=[v['last_opt_step'] for v in replayed],
            first_next_batch_accumulates_without_attempt=True,second_next_batch_actual_SGD=True,
            exact_model_optimizer_scaler_scheduler_EMA_gradients_PGD_RNG=True,absolute_difference=0,tolerance=0,
            actual_outer_batches=initial_batches+2,step_budget=64,actual_device='cuda:0',native_batch=36,imgsz=640,workers=16,AMP=True,
            actual_resumed_updates=resumed.at_holder['observer'].updates,finite_loss=bool(torch.isfinite(resumed.loss).all()),
            worker_prefetch_bitwise_trajectory_claim=False,validation_forwards=0)
        if result['actual_outer_batches']>64:raise ValueError('Combined native phase proof exceeded64batches')
        write_once(args.out/'smoke_summary.json',result);atomic(args.out/'status.json',result)
    except BaseException as exc:
        status.update(state='FAILED',error=repr(exc),traceback=traceback.format_exc());atomic(args.out/'status.json',status);raise


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--source-commit',required=True);run(a.parse_args())
