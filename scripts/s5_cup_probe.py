"""Stage E: selected-P3 CUP, original S4/off equivalence, resume/PNG/native AMP.

No official metrics or formal victim. Includes bounded native AMP prefixes.
Reuses S4's original loop/AMP-independent
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
    parser.add_argument('--config', default=str(ROOT/'configs/s5_cup_probe.json')); parser.add_argument('--amp-trial'); parser.add_argument('--amp-config')
    args = parser.parse_args(); out = Path(args.output).resolve()
    if args.amp_trial:
        from scripts.s5_victim_smoke import trial
        trial(args.amp_trial,out,json.loads(Path(args.amp_config).read_text()));return
    protocol=json.loads(Path(args.config).read_text())
    if (protocol['schema']!='s5.cup-probe.v1' or protocol['seed']!=0
            or protocol['steps_per_independent_trial']!=5 or protocol['eot_samples']!=2
            or protocol['lambda_ret']!=.25 or not protocol['CUP_active']
            or protocol['roles']!=['S4_reference','P3_off','P3_CUP']
            or protocol['batch_partitions']!=[[0,16],[16,32]]):
        raise ValueError('Stage E declared scope changed')
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
    from ue_framework.methods.s5_cup_trainer import S5CUPTrainer
    from ue_framework.methods.s5_cup_history import load_cup_config
    from ue_framework.methods.s5_optimization import optimize_s5
    from scripts.s5_cup_probe_checks import export_fixed, resume_check, run_amp_checks
    from ue_framework.methods.s5_gt_loss_probe import SameFrozenGTProxy
    from scripts.run_s4_tfss_pipeline import setup, resource_gate

    resource_gate()
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
    equivalence = [];png_rows=[];resume_rows=[];cup_rows=[]
    values_keys = ['L_BG','L_rel','L_ret','L_orth_objective','L_adaptive','L_total',
                   'inner_update_norm','target_score_poison_after','target_score_clean_after']

    def journal(name, rows):
        with (out/name).open('a') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')

    def seed():
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

    def create(role):
        from ue_framework.methods.s5_runtime import create_trainer
        return create_trainer(cfg,role,ROOT/protocol['cup_calibration'],out/'shared_clean_maps',cache_identity)

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
                if role=='P3_off' or (role=='P3_CUP' and step==0 and not post):
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
                    cup_rows.extend([dict(common,**row) for row in t.cup_audit])
                    trace.update(CUP_base_ret=float(t.cup_base_ret.detach()),CUP_ret=float(t.cup_weighted_ret.detach()),
                        CUP_history_step=t.cup.steps,CUP_entropy=t.cup_weight_summary.get('entropy',0.),
                        CUP_allocation_residual=abs(t.cup_weight_summary['effective_weighted_sum']-1.) if t.cup_audit else 0.)
                    gb=torch.autograd.grad(t.cup_base_ret,t.fourier_coeff,retain_graph=True)[0]
                    gw=torch.autograd.grad(t.cup_weighted_ret,t.fourier_coeff,retain_graph=True)[0]
                    factor=float(gw.norm()/gb.norm()) if gb.norm()>0 else None
                    matched=gb*factor if factor is not None else gb
                    uniform=torch.stack(list(t.s5_class_losses['P3'].values())).mean() if t.s5_class_losses['P3'] else loss*0
                    gu=torch.autograd.grad(uniform,t.fourier_coeff,retain_graph=True)[0]
                    trace.update(CUP_ret_grad_norm=float(gw.norm()),base_ret_grad_norm=float(gb.norm()),
                        CUP_to_base_ret_gradient_norm_ratio=factor,CUP_protect_cosine=cosine(gp,gw),
                        norm_matched_base_protect_cosine=cosine(gp,matched),class_uniform_protect_cosine=cosine(gp,gu),
                        class_uniform_raw_ret=float(uniform.detach()),class_uniform_ret_grad_norm=float(gu.norm()),
                        norm_matched_only_diagnostic=True,
                        CUP_ret_to_protection_norm=float(gw.norm()*t.lambda_ret/gp.norm()) if gp.norm()>0 else None,
                        CUP_protection_opposing_projection_ratio=float((-t.lambda_ret*(gp.double()*gw.double()).sum()/gp.double().square().sum()).clamp_min(0)) if gp.norm()>0 else None)
                    if (trace['CUP_ret_to_protection_norm'] is not None and trace['CUP_ret_to_protection_norm']>1
                            or trace['CUP_protection_opposing_projection_ratio'] is not None and trace['CUP_protection_opposing_projection_ratio']>1):
                        raise ValueError('CUP retention overwhelms the protection gradient in bounded development probe')
                    t.cup_training_weight_hashes=getattr(t,'cup_training_weight_hashes',{})
                    for i,path in enumerate(t.s4_item_paths):
                        h,w=items[i][0].shape[:2]
                        digest=hashlib.sha256(t.tfss_applied_weight[i,0,:h,:w].detach().cpu().numpy().tobytes()).hexdigest()
                        old=t.cup_training_weight_hashes.setdefault(Path(path).stem,digest)
                        if old!=digest:raise ValueError('Clean Utility weights changed across optimizer observations')
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
                                CUP_weighted_contribution_grad_norm=float(gk.norm())*pi*t.lambda_ret*next((r['w'] for r in t.cup_audit if r['cls']==k),1.) if layer=='P3' else None,
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

            identity=dict(stage='S5_STAGE_E',role=role,batch=batch,seed=0,manifest_sha256=sha(manifest_path))
            loop=optimize_s4 if role=='S4_reference' else optimize_s5
            result=loop(t,images,Path(fixed[0]['label_path']).parent,prototypes,trial,identity,
                stop_after=steps,observe=capture)
            status['optimizer_updates']+=result['steps']; write(out/'pipeline_status.json',status)
            if role=='P3_off':
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
            if role=='P3_CUP':
                png_rows.extend(export_fixed(t,cfg,fixed,out,batch))
                expected=result['checkpoint']
                t.s5_close_observer();del t,proxy;gc.collect();torch.cuda.empty_cache()
                resume_rows.append(resume_check(create,images,fixed,prototypes,out,batch,identity,expected,compare_tree))
                status['optimizer_updates']+=5;write(out/'pipeline_status.json',status)
                continue
            if hasattr(t,'s5_close_observer'):t.s5_close_observer()
            del t,proxy;gc.collect();torch.cuda.empty_cache()
    write(out/'retention_objects.json',object_rows); write(out/'retention_classes.json',class_rows)
    write(out/'GT_detection_proxy.json',gt_rows);write(out/'CUP_CLASSWISE_WEIGHTS.json',cup_rows)
    status.update(stage='native_AMP_checks');write(out/'pipeline_status.json',status)
    amp=run_amp_checks(out,ROOT,protocol)
    summary=dict(state='complete',stage='SELECTED_P3_CUP_TRAIN32_5STEP',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        parent_config_sha256=sha(ROOT/'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json'),config_hash=config_hash,
        manifest_sha256=sha(manifest_path),manifest_coverage=manifest['coverage'],trials=summaries,
        original_S4_vs_P3_noop_max_difference=max(equivalence),optimizer_updates=40,
        official_metrics_disabled=True,artifact_ingest_disabled=True,CUP_active=True,
        selected_layer='P3',
        cleanup_policy='retain_all',deleted_paths=[],formal_victim_training=False,bounded_native_victim_training=True)
    if status['optimizer_updates']!=40:raise ValueError('Stage E bounded update accounting mismatch')
    summary.update(native_AMP=amp,resume=resume_rows,PNG_checks=png_rows,calibration_sha256=sha(ROOT/protocol['cup_calibration']))
    reference_metric=sum(x['L_total'] for x in traces if x['role']=='S4_reference')/sum(x['role']=='S4_reference' for x in traces)
    candidate_metric=sum(x['L_total'] for x in traces if x['role']=='P3_off')/sum(x['role']=='P3_off' for x in traces)
    summary.update(schema='prerun.pre-review-smoke.v1',computation_kind='training',
        production_entrypoint_reached=True,finite_loss=all(__import__('math').isfinite(x['L_total']) and x['finite_gradient'] for x in traces),
        step_budget=98,production_step_count=52+amp['total_actual_batches'],
        step_unit='40 Fourier Adam updates +12 non-update EOT observations +at most46 actual native victim training batches',
        baseline_equivalence_required=True,baseline_equivalence_probe=dict(
            reference_id='actual unchanged S4Trainer/optimize_s4 from parent686eaac; no S5 core instantiated',
            reference_weights_path=cfg['surrogate']['ckpt'],candidate_weights_path=cfg['surrogate']['ckpt'],
            metric_name='mean actual production L_total over matched training/EOT/post-five observations',
            reference_metric=reference_metric,candidate_metric=candidate_metric,
            absolute_difference=abs(candidate_metric-reference_metric),tolerance=1e-7,
            evaluation_setting=dict(dataset='original fixed TRAIN32, split train',seed=0,sample_count=32,
                parameters=dict(steps_per_trial=5,EOT2=True,independent_batches=2),
                post_processing='identical original preprocessing and same clean Utility map cache'),
            evidence_paths=['trace.jsonl','trials.json','smoke_summary.json']),
        checkpoint_cleanup_completed=False,checkpoint_paths_remaining=[str(p.relative_to(out)) for p in out.rglob('*.pt')])
    if not summary['finite_loss'] or summary['production_step_count']>98:raise ValueError('Current-source smoke numerical/budget failure')

    write(out/'smoke_summary.json',summary)
    status.update(state='complete',stage=summary['stage'],summary='smoke_summary.json')
    write(out/'pipeline_status.json',status)
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
