"""Finite native-loop benchmark; never imported by the formal worker.

Each invocation is ONE alternating-order sample. It reaches native warmup
normally on a retained train-only pool. It neither forces a stable scaler nor
changes native update arithmetic. A passed invocation alone proves no gain.
"""
import argparse
import json
import random
import statistics
import time
import traceback
from pathlib import Path
from bounded_common import ROOT,BASE_PGD_SHA,add_arguments,admission,fixture_view
from robustness_s4_at.common import PROTOCOL,atomic,now,protocol,sha,write_once
from gpu_telemetry import GPUTrace


class Complete(Exception):pass


def run(args):
    resources=admission(args,__file__);gpu=args.device=='0';p=protocol()
    if args.batch not in (36,48,64,72):raise ValueError('Undeclared future batch proposal')
    if not 2<=args.measure_batches<=12:raise ValueError('Frozen finite measurement budget is2..12')
    if args.profile and not gpu:raise ValueError('CUDA-event decomposition requires real CUDA')
    args.out.mkdir(parents=True,exist_ok=False)
    status=dict(schema='s4.at.native-finite-benchmark.v1',state='RUNNING',UTC=now(),source_commit=args.source_commit,
        implementation=args.implementation,arm=args.arm,epsilon_pixels=args.epsilon,requested_batch=args.batch,
        resources=resources,entry_sha256=sha(__file__),official_metrics_disabled=True,artifact_ingest_disabled=True,
        cleanup_policy='retain_all',full_scientific_gate=False,formal_ready=False,CUDA_performance_measured=False)
    atomic(args.out/'status.json',status)
    try:
        import numpy as np
        import torch
        from candidate_scalar_readback import build_candidate
        from robustness_s4_at.common import native_identity
        from robustness_s4_at.trainer import ATTrainer
        from robustness_s4_at.validate_cuda import close_loaders
        from ue_framework.s4_amp_observer import tree_hash
        native=native_identity()
        count=2*args.batch+(16551%args.batch) if gpu and args.pool_mode=='formal-tail' else None
        data,selected=fixture_view(args,args.out/'fixtures',count=count)
        batch,imgsz,workers=(args.batch,640,16) if gpu else (2,160,0)
        policy=json.loads((Path(__file__).parent/f'PROPOSED_TEST_PROTOCOL_B{args.batch}.json').read_text())
        if policy['state']!='UNLAUNCHED_NOT_SCIENTIFICALLY_VALIDATED':raise ValueError('Unexpected batch proposal state')
        nominal=policy['train_args']['weight_decay'] if gpu else p['train_args']['weight_decay']
        overrides=dict(p['train_args'],model=str(ROOT/'configs/voc_yolov8n_20cls.yaml'),data=str(data),
            project=str(args.out),name='native_training',device=args.device,batch=batch,imgsz=imgsz,workers=workers,
            amp=gpu,epochs=200,val=False,plots=False,save=True,cache=True,weight_decay=nominal)
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1)
        if gpu:torch.cuda.set_device(0);torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        identity=dict(run_id='BOUNDED_BENCH_'+args.implementation+'_'+args.arm+'_'+str(args.epsilon),
            source_commit=args.source_commit,protocol_sha256=sha(PROTOCOL),
            selected_train_identity_sha256=sha(args.out/'fixtures/selected_identity.json'),validation_only=True)
        Candidate=None
        if args.implementation!='original':
            Candidate,generated=build_candidate(BASE_PGD_SHA,vectorize_counts=args.implementation=='03')
            status['generated_candidate_sha256']=__import__('hashlib').sha256(generated.encode()).hexdigest()
        measured=[];epochs=[];events=[];all_batch_shapes=[];all_GT=[];starts={};stable_begin={};initial={}
        class BenchTrainer(ATTrainer):
            def __init__(self,**kw):
                super().__init__(**kw)
                if Candidate is not None:
                    prior=self.attack.generator.get_state();self.attack=Candidate(self.attack.config,self.device,0)
                    self.attack.generator.set_state(prior)
                if args.profile:
                    underlying=self.attack
                    class TimedAttack:
                        def __getattr__(self,name):return getattr(underlying,name)
                        def __call__(self,*a,**kw):
                            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            begin.record();result=underlying(*a,**kw);end.record();events.append([begin,end,None])
                            return result
                    self.attack=TimedAttack()
        t=BenchTrainer(at_config=p['pgd'][str(args.epsilon)],identity=identity,audit_root=args.out/'audit',
            validation_only=True,overrides=overrides)
        def started(t):
            if t.batch_size!=batch or bool(t.amp)!=gpu:raise ValueError('Native batch/AMP declaration mismatch')
            observer=t.at_holder['observer']
            if not observer.enforce or observer.max_consecutive_skips!=16:raise ValueError('Production finite/AMP guards changed')
            # Only validation trace hashing is disabled; this matches the live
            # formal observer. All finite/budget/GT/AMP enforcement stays on.
            observer.debug_hashes=False;observer.rows=None
            initial.update(model=tree_hash(t.model.state_dict()),optimizer=tree_hash(t.optimizer.state_dict()),
                scaler=tree_hash(t.scaler.state_dict()),EMA=tree_hash(t.ema.ema.state_dict()),
                parameter_strides={n:list(v.stride()) for n,v in t.model.named_parameters()})
            nb=len(t.train_loader);nw=t._get_warmup_iterations(nb)
            # Native _do_train updates accumulate only while ni<nw. The last
            # interpolated integer persists thereafter; a tiny CPU fixture
            # need not reach round(nbs/batch). Never force the nominal value.
            terminal_accumulate=max(1,int(np.interp(nw-1,[0,nw],[1,t.args.nbs/batch]).round())) if nw else max(round(t.args.nbs/batch),1)
            max_total=nw+4+args.measure_batches
            if not 1<=max_total<=100:raise ValueError('Native warmup plus measurement exceeds100outer budget')
            decay=[g['weight_decay'] for g in t.optimizer.param_groups if g['weight_decay']]
            if gpu and decay!=[0.0005625000000000001]:raise ValueError('Declared held actual native weight decay changed')
            status.update(native_sources=native,native_batch=batch,imgsz=imgsz,workers=workers,AMP=gpu,
                actual_selected_train_count=len(selected),native_batches_per_epoch=nb,native_warmup_batches=nw,
                pool_mode=args.pool_mode,fixture_identity_sha256=sha(args.out/'fixtures/selected_identity.json'),
                native_post_warmup_accumulate=terminal_accumulate,
                max_outer_batch_budget=max_total,nominal_weight_decay=nominal,actual_weight_decay=decay,
                initial_state=initial,original_observer_enforcement=True,validation_debug_hashes=False,
                dataset_is_bounded_fixture=True,full16551_warmup_batches=round(3*((16551+args.batch-1)//args.batch)),
                warmup_was_not_forced_or_shortened_in_native_fixture_protocol=True)
            atomic(args.out/'status.json',status)
        def no_val(t):raise AssertionError('Finite benchmark must never evaluate validation images')
        def beginning(t):
            ni=t.at_batches
            if ni>=status['native_warmup_batches']+4:
                if not stable_begin:
                    # Outside timed endpoints: record common model/AMP state.
                    stable_begin.update(model=tree_hash(t.model.state_dict()),optimizer=tree_hash(t.optimizer.state_dict()),
                        scaler=tree_hash(t.scaler.state_dict()),EMA=tree_hash(t.ema.ema.state_dict()),
                        PGD_RNG=tree_hash(t.attack.generator.get_state()),last_opt_step=t.at_last_opt_step)
                    if gpu:torch.cuda.synchronize(0)
                starts['batch']=time.perf_counter()
                starts['epoch']=t.epoch;starts['ni']=ni
        def ended(t):
            stamp=time.perf_counter();ledger=t.at_latest;observer=t.at_holder['observer']
            if args.profile:
                event=torch.cuda.Event(enable_timing=True);event.record();events[-1][2]=event
            shape=ledger['actual_input_shape'];expected=min(batch,len(selected)-((t.at_batches-1)%len(t.train_loader))*batch)
            if shape!=[expected,3,imgsz,imgsz] or ledger['actual_inner_forwards']!=10:
                raise ValueError('Native batch/tail/PGD10 mismatch')
            if t.batch_size!=batch:raise ValueError('Silent OOM batch fallback refused')
            all_batch_shapes.append(shape);all_GT.append(ledger['GT_sha256'])
            if 'batch' in starts:
                if gpu and ledger['actual_raw_devices']!=[('cuda:0',0)]:raise ValueError('Native raw tensors not on cuda0')
                if int(t.accumulate)!=status['native_post_warmup_accumulate']:raise ValueError('Native post-warmup recurrence changed')
                measured.append(dict(native_iteration=starts['ni'],epoch=t.epoch+1,shape=shape,
                    start=starts['batch'],end=stamp,callback_compute_wall_seconds=stamp-starts['batch'],
                    actual_updates=observer.updates,legal_AMP_skips=observer.skips,loss=float(t.loss.detach()),
                    gradient_budget=ledger['linf'],GT_sha256=ledger['GT_sha256'],PGD_steps=ledger['pgd_steps'],
                    input_ids=observer.input_ids,accumulate=int(t.accumulate),last_opt_step=t.at_last_opt_step))
                if len(measured)==args.measure_batches:raise Complete()
            if observer.total_batches>=status['max_outer_batch_budget']:raise ValueError('Finite native loop exceeded declared budget')
        def epoch_training_end(t):starts['epoch_finish']=time.perf_counter()
        def epoch_checkpoint_end(t):
            epochs.append(dict(epoch=t.epoch+1,post_training_checkpoint_callback_seconds=time.perf_counter()-starts['epoch_finish'],
                native_checkpoint_sha256=sha(t.last)))
        t.add_callback('on_train_start',started);t.add_callback('on_val_start',no_val)
        t.add_callback('on_train_batch_start',beginning);t.add_callback('on_train_batch_end',ended)
        t.add_callback('on_train_epoch_end',epoch_training_end);t.add_callback('on_fit_epoch_end',epoch_checkpoint_end)
        telemetry=GPUTrace() if gpu else None
        if telemetry is not None:telemetry.start()
        try:t.train()
        except Complete:pass
        finally:
            close_loaders(t)
            telemetry_rows=telemetry.stop() if telemetry is not None else []
            write_once(args.out/'bounded_telemetry.json',telemetry_rows)
        if len(measured)!=args.measure_batches or t.at_holder['observer'].updates<1:raise ValueError('No complete native measurement/SGD')
        if gpu:torch.cuda.synchronize(0)
        if any(v.get('sampling_error') or v.get('other_GPU_pids') for v in telemetry_rows):
            raise ValueError('Benchmark resource sampling failed or another GPU job competed; no fair gain claim')
        # Same-epoch end-to-end intervals include native DataLoader and logging.
        # Epoch transitions are explicitly excluded and separately retained.
        intervals=[dict(seconds=b['end']-a['end'],images=b['shape'][0],
            between_callbacks_seconds=b['start']-a['end']) for a,b in zip(measured,measured[1:]) if a['epoch']==b['epoch']]
        if not intervals and gpu:raise ValueError('No within-epoch full training interval')
        timing=dict(same_epoch_complete_intervals=len(intervals),seconds=sum(v['seconds'] for v in intervals),
            images=sum(v['images'] for v in intervals),intervals=intervals,
            seconds_per_batch_mean=statistics.mean(v['seconds'] for v in intervals) if gpu else None,
            images_per_second=sum(v['images'] for v in intervals)/sum(v['seconds'] for v in intervals) if gpu else None,
            between_callbacks_is_not_pure_DataLoader_time=True,unmodified_native_audit_and_guards_included=True,
            profile_instrumented=bool(args.profile),paired_comparison_required=True,stable_speedup_percent=None)
        if args.profile:
            timing['CUDA_timeline_ms']=[dict(PGD=a.elapsed_time(b),post_PGD_to_batch_end=b.elapsed_time(c)) for a,b,c in events]
        result=dict(status,state='PASS_FINITE_NATIVE_BENCHMARK_SAMPLE',completed_UTC=now(),stable_entry_state=stable_begin,
            actual_outer_batches=t.at_holder['observer'].total_batches,actual_PGD_forwards=10*t.at_holder['observer'].total_batches,
            actual_SGD=t.at_holder['observer'].updates,actual_AMP_skips=t.at_holder['observer'].skips,
            full_native_PGD10_and_original_guards=True,measured_rows=measured,timing=timing,epoch_checkpoint_costs=epochs,
            observed_native_shapes=all_batch_shapes,augmented_GT_sequence=all_GT,
            final_model=tree_hash(t.model.state_dict()),final_optimizer=tree_hash(t.optimizer.state_dict()),
            final_scaler=tree_hash(t.scaler.state_dict()),final_EMA=tree_hash(t.ema.ema.state_dict()),
            CUDA_initialized=torch.cuda.is_initialized(),CUDA_performance_measured=gpu,
            resource_telemetry=telemetry_rows,
            cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated(0)/2**20 if gpu else None,
            full_formal_200epoch_performance_claim=False,full_scientific_gate=False,formal_ready=False)
        write_once(args.out/'summary.json',result);atomic(args.out/'status.json',result)
    except BaseException as exc:
        status.update(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),ended_UTC=now())
        atomic(args.out/'status.json',status);raise


if __name__=='__main__':
    q=argparse.ArgumentParser();add_arguments(q);q.add_argument('--batch',type=int,required=True)
    q.add_argument('--measure-batches',type=int,default=12);q.add_argument('--profile',action='store_true')
    q.add_argument('--pool-mode',choices=('common720','formal-tail'),default='common720');run(q.parse_args())
