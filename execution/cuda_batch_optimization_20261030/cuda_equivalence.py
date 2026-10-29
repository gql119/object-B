"""Bounded native-loop equality entry. No throughput or formal-run claims.

GPU mode requires an exclusive admitted window and hash-bound retained real
training fixtures. CPU mode validates this new launcher/trace plumbing only.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'execution/speed_audit_20261029'))

from robustness_s4_at.common import PROTOCOL,atomic,commit,now,protocol,sha,write_once

BASE_PGD_SHA='63bfa27086953566efb771e8bddf4aba5cc7363b8198dd3d73a28ba93df08578'


class Complete(Exception):pass


def run(args):
    args.out=args.out.resolve();args.fixture_root=args.fixture_root.resolve();args.fixture_manifest=args.fixture_manifest.resolve()
    p=protocol();source=commit()
    if source!=args.source_commit:raise ValueError('Wrong exact source commit')
    if sha(ROOT/'robustness_s4_at/pgd.py')!=BASE_PGD_SHA:raise ValueError('Reference PGD source drift')
    gpu=args.device=='0'
    if gpu:
        for path in (Path(__file__).resolve(),ROOT/'execution/speed_audit_20261029/candidate_scalar_readback.py'):
            blob=subprocess.check_output(['git','show','HEAD:'+path.relative_to(ROOT).as_posix()],cwd=ROOT)
            if hashlib.sha256(blob).hexdigest()!=sha(path):raise ValueError('GPU entry is not the exact committed source')
        from robustness_s4_at.queue import resource_status
        resources=resource_status()
        if not resources['admitted']:raise ValueError('Exclusive GPU admission refused: '+str(resources))
        if not args.out.resolve().is_relative_to(Path(p['remote_owned_root']).resolve()):raise ValueError('GPU output outside owned root')
        if args.fixture_manifest_sha256 is None or sha(args.fixture_manifest)!=args.fixture_manifest_sha256:
            raise ValueError('GPU fixture manifest must be prebound before execution')
        if args.max_batches!=16:raise ValueError('Frozen GPU component budget is16 native batches per side')
    else:
        resources={'CPU_diagnostic_only':True};os.environ['CUDA_VISIBLE_DEVICES']=''
        if args.max_batches!=1:raise ValueError('CPU plumbing check uses one native batch per side')
    args.out.mkdir(parents=True,exist_ok=False)
    status=dict(schema='s4.at.native-fixed-input-equality.v1',state='RUNNING',source_commit=source,
        implementation=args.implementation,arm=args.arm,epsilon_pixels=args.epsilon,
        device=args.device,started_UTC=now(),atol=0,rtol=0,max_native_outer_batches=2*args.max_batches,
        official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all',
        throughput_measured=False,production_resume_verified=False,formal_ready=False,resources=resources)
    status['entry_sha256']=sha(__file__)
    atomic(args.out/'status.json',status)
    try:
        import numpy as np
        import torch
        from candidate_scalar_readback import build_candidate
        from robustness_s4_at.checkpoint import capture,rng_state,restore_rng
        from robustness_s4_at.common import native_identity
        from robustness_s4_at.data import data_view,image_index
        from robustness_s4_at.pgd import PGDAttack
        from robustness_s4_at.trainer import ATTrainer
        from robustness_s4_at.validate_cuda import close_loaders
        from ue_framework.s4_amp_observer import tree_hash
        native=native_identity()
        manifest=json.loads(args.fixture_manifest.read_text(encoding='utf8'))
        images=image_index(args.fixture_root/'images/train')
        if gpu:
            rows=[r for r in manifest['records'] if r['arm']==args.arm]
            if len(images)<612 or len(rows)!=len(images):raise ValueError('GPU proof needs retained distinct training fixtures, not repeated two-image inputs')
            byid={r['stem']:r for r in rows}
            if set(images)!=set(byid):raise ValueError('GPU fixture identity set mismatch')
            for stem,image in images.items():
                label=args.fixture_root/'labels/train'/(stem+'.txt');r=byid[stem]
                if sha(image)!=r['image_sha256'] or sha(label)!=r['label_sha256']:raise ValueError('Real retained training fixture bytes changed')
        else:
            for r in manifest['records']:
                for kind in ('image','label'):
                    if sha(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Local accepted CPU fixture bytes changed')
        view=args.out/'fixture_view';view.mkdir()
        data=data_view(view,args.arm,fixture_root=args.fixture_root)
        batch,imgsz,workers=(36,640,16) if gpu else (2,160,0)
        identity=dict(run_id='BOUNDED_EQ_'+args.arm+'_'+str(args.epsilon),source_commit=source,
            protocol_sha256=sha(PROTOCOL),data_audit_sha256=sha(view/'data_audit.json'),validation_only=True)
        overrides=dict(p['train_args'],model=str(ROOT/'configs/voc_yolov8n_20cls.yaml'),
            data=str(data),project=str(args.out),device=args.device,batch=batch,imgsz=imgsz,workers=workers,
            amp=gpu,epochs=200,val=False,save=False,plots=False,cache=False)
        Candidate,generated=build_candidate(BASE_PGD_SHA,vectorize_counts=args.implementation=='03')
        status['candidate_generated_call_sha256']=hashlib.sha256(generated.encode()).hexdigest()
        raw_inputs=[];fixed_RNG=[];reference=[];candidate=[];initial={};holders=[]
        # Convert values to SHA only after exact dtype/shape-preserving CPU copy.
        # Equality traces are intentionally NOT performance measurements.
        def digest_tree(v):
            h=hashlib.sha256()
            def add(x):
                if isinstance(x,np.ndarray):
                    h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
                elif isinstance(x,dict):
                    for k in sorted(x,key=str):h.update(repr(k).encode());add(x[k])
                elif isinstance(x,(list,tuple)):
                    h.update(type(x).__name__.encode())
                    for y in x:add(y)
                else:h.update(tree_hash(x).encode())
            add(v);return h.hexdigest()
        def seed():
            random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1)
            if gpu:torch.cuda.set_device(0);torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.use_deterministic_algorithms(True)
        class FixedInputTrace(ATTrainer):
            def __init__(self,side,**kw):
                self.side=side;self.outer_gradient_hashes={};self.inner_traces=[];self.assignment_traces=[]
                self.trace_handles=[];super().__init__(**kw)
                if side=='candidate':
                    rng=self.attack.generator.get_state();self.attack=Candidate(self.attack.config,self.device,0)
                    self.attack.generator.set_state(rng)
            def preprocess_batch(self,incoming):
                index=len(reference) if self.side=='reference' else len(candidate)
                observed_RNG=rng_state(self.device)
                write_once(args.out/(self.side+'_loader_RNG_'+str(index)+'.json'),
                    {k:digest_tree(v) for k,v in observed_RNG.items()})
                if self.side=='reference':
                    raw_inputs.append(deepcopy(incoming));fixed_RNG.append(deepcopy(observed_RNG))
                else:
                    # A fixed-input component test starts both compute paths at
                    # the same recorded input AND RNG boundary. Native loader
                    # setup/cache timing is outside this equality claim.
                    incoming=deepcopy(raw_inputs[index]);restore_rng(fixed_RNG[index],self.device)
                self.input_RNG={k:digest_tree(v) for k,v in rng_state(self.device).items()}
                self.outer_gradient_hashes={};self.inner_traces=[];self.assignment_traces=[]
                base=torch.autograd.grad;handles=[];attached=[False]
                def grad(output,inputs,*a,**k):
                    value=base(output,inputs,*a,**k)
                    self.inner_traces.append(dict(input=tree_hash(inputs),input_stride=list(inputs.stride()),
                        gradient=tree_hash(value[0]),loss=tree_hash(output)))
                    return value
                def pre(module,inputs):
                    if attached[0]:return
                    attached[0]=True
                    def assigned(module,inputs,output):self.assignment_traces.append(tree_hash(output))
                    handles.append(module.criterion.assigner.register_forward_hook(assigned))
                handles.append(self.model.register_forward_pre_hook(pre));torch.autograd.grad=grad
                try:
                    result=super().preprocess_batch(incoming)
                    self.final_adversarial_hash=tree_hash(result['img'])
                    return result
                finally:
                    torch.autograd.grad=base
                    for h in handles:h.remove()
        def initial_callback(t):
            if t.batch_size!=batch or bool(t.amp)!=gpu:raise ValueError('Native actual batch/AMP mismatch')
            initial[t.side]=dict(model=tree_hash(t.model.state_dict()),optimizer=tree_hash(t.optimizer.state_dict()),
                scaler=tree_hash(t.scaler.state_dict()),EMA=tree_hash(t.ema.ema.state_dict()),
                PGD_RNG=tree_hash(t.attack.generator.get_state()),
                parameter_strides={n:list(v.stride()) for n,v in t.model.named_parameters()})
            write_once(args.out/(t.side+'_initial_RNG.json'),{k:digest_tree(v) for k,v in rng_state(t.device).items()})
            if t.side=='candidate' and initial['candidate']!=initial['reference']:raise ValueError('Different native initial model/layout/optimizer/scaler')
            for name,param in t.model.named_parameters():
                if param.requires_grad:
                    def observe(g,name=name):t.outer_gradient_hashes[name]=tree_hash(g)
                    t.trace_handles.append(param.register_hook(observe))
        def no_val(t):raise AssertionError('Bounded equality is not an official validator or performance benchmark')
        def ended(t):
            observer=t.at_holder['observer'];ledger=t.at_latest
            if len(t.inner_traces)!=10 or len(t.assignment_traces)!=10 or ledger['actual_inner_forwards']!=10:
                raise ValueError('PGD10/TAL production sink not reached')
            if ledger['actual_input_shape']!=[batch,3,imgsz,imgsz]:raise ValueError('Wrong actual native input shape')
            if gpu and ledger['actual_raw_devices']!=[('cuda:0',0)]:raise ValueError('Wrong actual raw CUDA tensor device')
            row=dict(batch=len(reference)+1 if t.side=='reference' else len(candidate)+1,
                inner=t.inner_traces,TAL=t.assignment_traces,PGD_audit={k:v for k,v in ledger.items() if k not in ('UTC','run_id')},
                final_adversarial_input=t.final_adversarial_hash,outer_loss_parts=tree_hash(t.loss_items),
                outer_scaled_gradient_contributions=t.outer_gradient_hashes,
                model=tree_hash(t.model.state_dict()),optimizer=tree_hash(t.optimizer.state_dict()),
                scheduler=tree_hash(t.scheduler.state_dict()),scaler=tree_hash(t.scaler.state_dict()),
                EMA=tree_hash(t.ema.ema.state_dict()),EMA_updates=t.ema.updates,
                pending_gradients=tree_hash([p.grad for p in t.model.parameters()]),
                input_global_RNG=t.input_RNG,
                global_RNG={k:digest_tree(v) for k,v in rng_state(t.device).items()},PGD_RNG=tree_hash(t.attack.generator.get_state()),
                last_opt_step=t.at_last_opt_step,accumulate=int(t.accumulate),
                actual_SGD=observer.updates,AMP_skips=observer.skips,outer_loss=float(t.loss.detach()))
            if t.side=='reference':reference.append(deepcopy(row))
            else:
                expected=reference[len(candidate)]
                if row!=expected:
                    fields=[k for k in row if row[k]!=expected[k]]
                    write_once(args.out/'mismatch.json',dict(batch=len(candidate)+1,fields=fields,reference=expected,candidate=row))
                    raise ValueError('Zero-tolerance native CUDA/component mismatch: '+repr(fields))
                candidate.append(deepcopy(row))
            if observer.total_batches>=args.max_batches:
                if observer.updates<1:raise ValueError('No real native SGD within bounded equality budget')
                payload=capture(t.model,t.optimizer,t.scheduler,t.scaler,t.attack,t.at_identity,0,t.ema,t.train_loader)
                with (args.out/(t.side+'_bounded_state.pt')).open('xb') as f:torch.save(payload,f)
                raise Complete()
        for side in ('reference','candidate'):
            seed();t=FixedInputTrace(side,at_config=p['pgd'][str(args.epsilon)],identity=identity,
                audit_root=args.out/(side+'_audit'),validation_only=True,
                overrides=dict(overrides,name=side+'_native_training'))
            holders.append(t);t.add_callback('on_train_start',initial_callback)
            t.add_callback('on_val_start',no_val);t.add_callback('on_train_batch_end',ended)
            try:t.train()
            except Complete:pass
            finally:
                close_loaders(t)
                for handle in t.trace_handles:handle.remove()
        if len(reference)!=args.max_batches or len(candidate)!=args.max_batches:raise ValueError('Incomplete bounded native comparison')
        with (args.out/'fixed_transformed_train_batches.pt').open('xb') as f:torch.save(raw_inputs,f)
        write_once(args.out/'REFERENCE_TRACE.json',reference);write_once(args.out/'CANDIDATE_TRACE.json',candidate)
        status.update(state='PASS_NATIVE_FIXED_INPUT_EQUALITY_COMPONENT',completed_UTC=now(),
            native_sources=native,initial_state=initial,actual_native_outer_batches=len(reference)+len(candidate),
            actual_PGD_forwards=10*(len(reference)+len(candidate)),actual_native_SGD_per_side=reference[-1]['actual_SGD'],
            actual_AMP_skips_per_side=reference[-1]['AMP_skips'],exact_step_and_state_hashes=True,
            equality_method='exact dtype/shape/value SHA and passive native state fingerprints, no relaxed tolerance',
            fixed_input_RNG_replay=True,native_loader_trajectory_equivalence_claim=False,
            maximum_numeric_difference=None,CUDA_initialized=torch.cuda.is_initialized(),
            input_classes=sorted({i for r in reference for i,n in enumerate(r['PGD_audit']['class_histogram']) if n}),
            resume_phase='PENDING_SEPARATE_CURRENT_IMPLEMENTATION_PRODUCTION_PROBE',
            performance='NOT_MEASURED_TRACE_SYNCHRONIZATION_INTENTIONALLY_PRESENT',full_scientific_gate=False)
        write_once(args.out/'summary.json',status);atomic(args.out/'status.json',status)
    except BaseException as exc:
        status.update(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),ended_UTC=now())
        atomic(args.out/'status.json',status);raise


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--out',type=Path,required=True);q.add_argument('--source-commit',required=True)
    q.add_argument('--implementation',choices=('02','03'),required=True);q.add_argument('--arm',choices=('clean','poison'),required=True)
    q.add_argument('--epsilon',type=int,choices=(4,8),required=True);q.add_argument('--device',choices=('0','cpu'),default='0')
    q.add_argument('--fixture-root',type=Path,required=True);q.add_argument('--fixture-manifest',type=Path,required=True)
    q.add_argument('--fixture-manifest-sha256');q.add_argument('--max-batches',type=int,required=True)
    run(q.parse_args())
