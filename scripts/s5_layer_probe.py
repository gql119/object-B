"""Stage B: fixed TRAIN32, original S4/P3-noop/P5 independent five-step probes.

No official metrics or victim training. Reuses S4's original loop/AMP-independent
Fourier optimizer, and observes GT proxies from the same Frozen forwards.
"""
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BOUNDARY = Path('/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(ROOT/'configs/s5_layer_probe.json'))
    args = parser.parse_args(); out = Path(args.output).resolve()
    protocol=json.loads(Path(args.config).read_text())
    if (protocol['schema']!='s5.layer-probe.v1' or protocol['seed']!=0
            or protocol['steps_per_independent_trial']!=5 or protocol['eot_samples']!=2
            or protocol['lambda_ret']!=.25 or protocol['CUP_active']
            or protocol['roles']!=['S4_reference','P3_base','P5_base']
            or protocol['batch_partitions']!=[[0,16],[16,32]]):
        raise ValueError('Stage B declared scope changed')
    if BOUNDARY not in out.parents or out.exists():
        raise ValueError('Diagnostic output must be new and inside the S5 boundary')
    out.mkdir(parents=True, exist_ok=False)
    (out/'ultralytics_runtime').mkdir(exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['YOLO_CONFIG_DIR'] = str(out/'ultralytics_runtime')
    status = dict(state='running', stage='parent_setup', optimizer_updates=0,
                  official_metrics_disabled=True, artifact_ingest_disabled=True)
    write(out/'pipeline_status.json', status)
    try:
        run(out, status, protocol, sha(args.config))
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), failed_at_unix=time.time())
        write(out/'pipeline_status.json', status)
        raise


def run(out, status, protocol, config_hash):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    from ue_framework.methods.s4_tfss import S4Trainer, CleanMapCache, model_hash
    from ue_framework.methods.s4_optimization import optimize_s4
    from ue_framework.methods.s5_retention_probe import S5LayerTrainer
    from ue_framework.methods.s5_gt_loss_probe import SameFrozenGTProxy
    from scripts.run_s4_tfss_pipeline import setup, resource_guard

    resource_guard('S5 bounded Stage B admission')
    lock = json.loads((ROOT/'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json').read_text())
    manifest_path = ROOT/'docs/tfss/FIXED_TRAIN_SAMPLE_MANIFEST.json'
    if sha(manifest_path) != protocol['fixed_manifest_sha256']:
        raise ValueError('Fixed original TRAIN32 changed')
    manifest = json.loads(manifest_path.read_text()); records = manifest['records']
    assert len(records) == 32 and manifest['split'] == 'train'
    for row in records:
        if sha(row['image_path']) != row['image_sha256'] or sha(row['label_path']) != row['label_sha256']:
            raise ValueError('Fixed train bytes changed')
    cfg, initial, prototypes, parent, splits = setup(lock, out)
    if initial.lambda_ret!=protocol['lambda_ret'] or initial.eot_samples!=protocol['eot_samples']:
        raise ValueError('Parent retention weight/EOT changed')
    steps=protocol['steps_per_independent_trial']; tolerance=protocol['equivalence_absolute_tolerance']
    cache_identity = initial.s4_cache.identity
    del initial; gc.collect(); torch.cuda.empty_cache()
    traces, object_rows, class_rows, gt_rows, summaries = [], [], [], [], []
    equivalence = []
    values_keys = ['L_BG','L_rel','L_ret','L_orth_objective','L_adaptive','L_total',
                   'inner_update_norm','target_score_poison_after','target_score_clean_after']

    def journal(name, rows):
        with (out/name).open('a') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')

    def seed():
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

    def create(role):
        seed()
        frozen = YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
        adaptive = YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
        cls = S4Trainer if role == 'S4_reference' else S5LayerTrainer
        t = cls(cfg, cfg['methods']['legacy_kproto_ret'], torch.device('cuda:0'), frozen, adaptive)
        t.tfss_mode = 'TFSS-U'
        t.s4_cache = CleanMapCache(out/'shared_clean_maps', cache_identity)
        if role != 'S4_reference':
            t.s5_ret_layer = 'P3' if role == 'P3_base' else 'P5'
        return t

    def cosine(a,b):
        den = a.double().norm()*b.double().norm()
        return float((a.double()*b.double()).sum()/den) if den > 0 else None

    def compare_tree(a,b):
        if isinstance(a,np.ndarray):
            assert a.shape==b.shape and a.dtype==b.dtype
            return float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0.
        if isinstance(a,torch.Tensor):
            assert a.shape == b.shape and a.dtype == b.dtype
            return float((a.double()-b.double()).abs().max()) if a.numel() else 0.
        if isinstance(a,dict):
            assert a.keys()==b.keys();return max([compare_tree(a[k],b[k]) for k in a] or [0.])
        if isinstance(a,(list,tuple)):
            assert len(a)==len(b);return max([compare_tree(x,y) for x,y in zip(a,b)] or [0.])
        if isinstance(a,(int,float)) and not isinstance(a,bool):return abs(a-b)
        assert a==b;return 0.

    for batch,(start,end) in enumerate(protocol['batch_partitions']):
        fixed = records[start:end]
        images = [r['image_path'] for r in fixed]
        for role in protocol['roles']:
            t = create(role); trial = out/f'batch{batch}_{role}'; last_items = {}
            before_models = (model_hash(t.surrogate),model_hash(t.adaptive.model))
            reference = out/f'batch{batch}_S4_reference'
            trial_started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            status.update(stage=role, batch=batch); write(out/'pipeline_status.json',status)
            proxy = SameFrozenGTProxy(v8DetectionLoss(t.adaptive.model)) if role!='S4_reference' else None

            def capture(trainer, items, loss, metrics, step, eot, post=False):
                last_items['items']=items
                object_start,class_start,gt_start=len(object_rows),len(class_rows),len(gt_rows)
                gtotal=torch.autograd.grad(loss,t.fourier_coeff,retain_graph=True)[0]
                values={k:float(metrics[k].detach()) for k in values_keys}
                delta=((t.tfss_input+t.tfss_delta).clamp(0,1)-t.tfss_input).detach()
                state=dict(gradient=gtotal.detach().cpu(),delta=delta.cpu(),
                           support=t.tfss_support.detach().cpu(),W=t.tfss_applied_weight.detach().cpu(),values=values)
                key=f'{step:02d}_{eot}_{int(post)}.pt'
                common=dict(batch=batch,role=role,step=step+int(not post),eot=eot,
                            post_five_updates=post,image_ids=[Path(p).stem for p in t.s4_item_paths])
                with (trial/key).open('wb') as handle:torch.save(state,handle)
                if role=='P3_base':
                    ref=torch.load(reference/key,map_location='cpu',weights_only=True)
                    error=compare_tree(state,ref); equivalence.append(error)
                    if error>tolerance:raise ValueError('Actual S4/P3-noop loss/Fourier/delta/W equivalence failed: '+str(error))
                trace=dict(common,**values,actual_RMS=float(delta.double().square().mean().sqrt()),
                           actual_Linf=float(delta.abs().max()),finite_gradient=bool(torch.isfinite(gtotal).all()))
                if not trace['finite_gradient']:raise FloatingPointError('Nonfinite actual Fourier gradient')
                if role!='S4_reference':
                    protect=metrics['weighted_L_BG']+metrics['weighted_L_rel']+t.lambda_adapt*metrics['L_adapt_target']
                    gp=torch.autograd.grad(protect,t.fourier_coeff,retain_graph=True)[0]
                    trace['protection_proxy']=float(protect.detach());trace['protect_grad_norm']=float(gp.norm())
                    trace['P3_mapping_error']=t.s5_original_mapping_error
                    trace['P5_missing_original_ROIs']=len(t.s5_missing_P5_rois)
                    trace['P5_additional_ROIs']=len(t.s5_additional_P5_rois)
                    for layer in ('P3','P5'):
                        ret=t.s5_ret_losses[layer]
                        gr=torch.autograd.grad(ret,t.fourier_coeff,retain_graph=True)[0]
                        dot=(gp.double()*gr.double()).sum();pn=gp.double().square().sum()
                        active=(gp!=0)&(gr!=0)
                        trace.update({layer+'_raw_ret':float(ret.detach()),layer+'_ret_grad_norm':float(gr.norm()),
                            layer+'_weighted_ret_grad_norm':float(gr.norm())*t.lambda_ret,
                            layer+'_ret_to_protect_norm_ratio':float(gr.norm()*t.lambda_ret/gp.norm()) if gp.norm()>0 else None,
                            layer+'_protect_cosine':cosine(gp,gr),
                            layer+'_elementwise_conflict_fraction':float(((gp*gr<0)&active).sum()/active.sum()) if active.any() else None,
                            layer+'_opposing_protect_projection_ratio':float((-t.lambda_ret*dot/pn).clamp_min(0)) if pn>0 else None})
                        for k,v in t.s5_class_losses[layer].items():
                            gk=torch.autograd.grad(v,t.fourier_coeff,retain_graph=True)[0]
                            pi=t.s5_class_pi[layer][k]
                            class_rows.append(dict(common,layer=layer,cls=k,pi=pi,raw_loss=float(v.detach()),
                                grad_norm=float(gk.norm()),weighted_contribution_grad_norm=float(gk.norm())*pi*t.lambda_ret,
                                protect_cosine=cosine(gp,gk)))
                        for r in t.s5_retention_records[layer]:
                            row={k:v for k,v in r.items() if k!='loss'}
                            row.update(common,layer=layer,raw_loss=float(r['loss'].detach()) if r['loss'] is not None else None)
                            area=r['bbox'][2]*r['original_shape'][1]*r['bbox'][3]*r['original_shape'][0]
                            row['size']='small' if area<32**2 else ('medium' if area<96**2 else 'large')
                            object_rows.append(row)
                    journal('retention_objects.jsonl',object_rows[object_start:])
                    journal('retention_classes.jsonl',class_rows[class_start:])
                    journal('pre_GT_trace.jsonl',[trace])
                    with torch.no_grad():
                        gtbatch=t.adaptive._batch(t.tfss_input.detach(),items)
                        gtbatch['image_ids']=[Path(p).stem for p in t.s4_item_paths]
                        c,p,evidence=proxy.query_loss(t.s5_heads['clean'],t.s5_heads['poison'],gtbatch)
                        ds=[float((p[k]-c[k]).detach()) for k in c]
                        trace['GT_detection_signed_classmean_delta']=sum(ds)/len(ds) if ds else None
                        trace['GT_detection_observed_classes']=len(c)
                        trace['GT_native_box_cls_dfl_gains']=evidence['gains'].tolist()
                        for row in evidence['objects']:
                            gt_rows.append(dict(common,**{k:(v.tolist() if torch.is_tensor(v) else v) for k,v in row.items()}))
                    journal('GT_detection_proxy.jsonl',gt_rows[gt_start:])
                traces.append(trace)
                with (out/'trace.jsonl').open('a') as stream:stream.write(json.dumps(trace,allow_nan=False)+'\n')

            identity=dict(stage='S5_STAGE_B',role=role,batch=batch,seed=0,manifest_sha256=sha(manifest_path))
            result=optimize_s4(t,images,Path(fixed[0]['label_path']).parent,prototypes,trial,identity,
                stop_after=steps,observe=capture)
            status['optimizer_updates']+=result['steps']; write(out/'pipeline_status.json',status)
            if role=='P3_base':
                with (reference/'latest_continuation.pt').open('rb') as handle:ref=torch.load(handle,map_location='cpu',weights_only=False)
                with (trial/'latest_continuation.pt').open('rb') as handle:now=torch.load(handle,map_location='cpu',weights_only=False)
                for field in ('trainer','optimizer','scheduler','rng'):
                    error=compare_tree(ref[field],now[field]);equivalence.append(error)
                    if error>tolerance:raise ValueError('S4/P3 5-update state mismatch '+field+': '+str(error))
            # Independent final-state observation: no sixth optimizer update.
            seed()
            for eot in range(protocol['post_update_observation_eot_samples']):
                loss,metrics=t._batch_loss(last_items['items'],prototypes)
                capture(t,last_items['items'],loss,metrics,steps,eot,post=True)
            if before_models != (model_hash(t.surrogate),model_hash(t.adaptive.model)):
                raise ValueError('Frozen/Adaptive base state changed during bounded observations')
            summaries.append(dict(batch=batch,role=role,optimizer_updates=result['steps'],
                elapsed_seconds=time.perf_counter()-trial_started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                frozen_model_unchanged=result['frozen_model_unchanged'],adaptive_model_unchanged=result['adaptive_model_unchanged'],
                runtime_layers=t.s5_model_shapes if role!='S4_reference' else None,
                runtime_modules={name:type(dict(t.surrogate.named_modules())[name]).__name__
                                 for name in ('model.15','model.21')},
                timing_scope='full trial including clean-map cache/diagnostic observers, not native production throughput'))
            write(out/'trials.json',summaries)
            if hasattr(t,'s5_close_observer'):t.s5_close_observer()
            del t,proxy;gc.collect();torch.cuda.empty_cache()
    write(out/'retention_objects.json',object_rows); write(out/'retention_classes.json',class_rows)
    write(out/'GT_detection_proxy.json',gt_rows)
    summary=dict(state='complete',stage='P3_P5_TRAIN32_5STEP',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        parent_config_sha256=sha(ROOT/'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json'),config_hash=config_hash,
        manifest_sha256=sha(manifest_path),manifest_coverage=manifest['coverage'],trials=summaries,
        original_S4_vs_P3_noop_max_difference=max(equivalence),optimizer_updates=30,
        official_metrics_disabled=True,artifact_ingest_disabled=True,CUP_active=False,
        selection='NOT_DECIDED; analyze paired evidence and GT coverage before Stage C',
        cleanup_policy='retain_all',deleted_paths=[],victim_training=False)
    if status['optimizer_updates']!=30:raise ValueError('Stage B bounded update accounting mismatch')
    write(out/'layer_probe_summary.json',summary)
    status.update(state='complete',stage=summary['stage'],summary='layer_probe_summary.json')
    write(out/'pipeline_status.json',status)
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
