"""Replay retained states and compare Utility to a static same-base control.

No optimizer.step, new training trajectory, dataset export or victim entrypoint.
All S0/U model/loss/gradient computations call the unchanged production classes.
"""
import argparse
import copy
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
from scripts.tfss_short_probe import sha,write,resource_gate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',default=str(ROOT/'configs/tfss_utility_static_control_20261017.json'))
    p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['smoke','measure'],required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--s3-control',required=True);p.add_argument('--s3-output',required=True)
    p.add_argument('--s3-acceptance',required=True);p.add_argument('--review-receipt')
    a=p.parse_args();conf=json.loads(Path(a.protocol).read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('Current computation must be committed and clean')
    if a.phase=='measure':
        r=json.loads(Path(a.review_receipt).read_text()) if a.review_receipt else {}
        if r.get('source_commit')!=commit or r.get('pre_run_result')!='pass':
            raise ValueError('Current static-control review/closure receipt required')
    resource=resource_gate(a.gpu,a.s3_control,a.s3_output,a.s3_acceptance)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    out=Path(a.output).resolve();boundary=Path(conf['remote_output_boundary']).resolve()
    if not out.is_relative_to(boundary) or out==boundary:raise ValueError('Output boundary')
    out.mkdir(exist_ok=False,parents=True)
    for k in ['YOLO_CONFIG_DIR','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:os.environ[k]=str(out/'runtime'/k)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import ultralytics
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations
    from ue_framework.methods.b2_semantic import B2SemanticTrainer
    from ue_framework.methods.legacy_kproto_ret import _snapshot_metrics
    from ue_framework.methods.tfss_spatial import SoftSpatialProjection,spatial_weights,realized_rms,match_realized_rms
    from ue_framework.methods.tfss_probe_metrics import collect
    from ue_framework.methods.tfss_nt_sensitivity import gradients as nt_gradients
    from ue_framework.methods.tfss_static_diagnostics import feasibility,spatial_selectivity
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    assert ultralytics.__version__=='8.4.150'
    device=torch.device('cuda:0')
    identity=json.loads((ROOT/'docs/tfss/S0_CANONICAL_SOURCE_SHA256.json').read_text())
    meta=json.loads((ROOT/'docs/tfss/SOURCE_METADATA.json').read_text())
    for rel,expected in identity.items():assert sha(ROOT/rel)==sha(Path(meta['parent_root'])/rel)==expected,rel
    manifest_path=ROOT/'docs/tfss/FIXED_TRAIN_SAMPLE_MANIFEST.json'
    oldconf=ROOT/'configs/tfss_short_probe.json'
    assert sha(manifest_path)==conf['manifest_sha256'] and sha(oldconf)==conf['original_protocol_sha256']
    manifest=json.loads(manifest_path.read_text());assert manifest['count']==32 and manifest['split']=='train'
    cfgpath=ROOT/'ue_framework/configs/exp_voc_person_b2_to_safe_soft05_eps8_seed0_20261004.yaml'
    assert sha(cfgpath)==meta['config_sha256'];cfg=load_config(str(cfgpath));mc=cfg['methods']['legacy_kproto_ret']
    checkpoint=Path(cfg['surrogate']['ckpt']);assert sha(checkpoint)==meta['channel_identity']['checkpoint_sha256']
    protopath=Path(meta['parent_root'])/'noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt'
    assert sha(protopath)==json.loads(oldconf.read_text())['prototype_sha256']
    prototypes=torch.load(protopath,map_location=device,weights_only=True)['prototypes']

    def digest(value):
        h=hashlib.sha256()
        def add(v):
            if isinstance(v,torch.Tensor):
                x=v.detach().cpu().contiguous();h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.numpy().tobytes())
            elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
            elif isinstance(v,dict):
                for k in sorted(v,key=str):add(k);add(v[k])
            elif isinstance(v,(list,tuple)):
                h.update(type(v).__name__.encode())
                for x in v:add(x)
            else:h.update(repr(v).encode())
        add(value);return h.hexdigest()
    def rng():return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    def restore(r):
        random.setstate(r[0]);np.random.set_state(r[1]);torch.set_rng_state(r[2]);torch.cuda.set_rng_state_all(r[3])
    def seed():random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)

    class Candidate(SoftSpatialProjection,B2SemanticTrainer):pass
    class Canonical(B2SemanticTrainer):
        def _compose_delta_batched(self,image,support,ring,*args,**kwargs):
            result=super()._compose_delta_batched(image,support,ring,*args,**kwargs)
            self.tfss_input=image;self.tfss_support=support;self.tfss_s0_delta=result[1];self.tfss_delta=result[1]
            self.tfss_applied_weight=torch.ones_like(support)
            self.tfss_matching_errors=[];self.tfss_matching_relative_errors=[]
            return result

    def create(klass):
        seed();frozen=YOLO(str(checkpoint)).model.to(device);adaptive=YOLO(str(checkpoint)).model.to(device)
        t=klass(copy.deepcopy(cfg),copy.deepcopy(mc),device,frozen,adaptive)
        assert t.channels.tolist()==meta['selected_channels'] and t.feature_layer=='model.21'
        assert (t.universal_batch_size,t.eot_samples,t.safe_bg_rank,t.safe_bg_removal_alpha,t.safe_bg_active_k)==(16,2,8,.5,3)
        return t,frozen,adaptive

    inputs=conf['inputs'];batches={};source_files={};started=time.time()
    for key,info in inputs.items():
        b=int(key);src=Path(info['output_root']);control=Path(info['control_root'])
        assert json.loads((control/'status.json').read_text())['state']=='failed'
        assert json.loads((control/'binding.json').read_text())['commit']==info['commit']
        for rel in list(identity)+['ue_framework/methods/tfss_spatial.py','ue_framework/methods/tfss_probe_metrics.py','ue_framework/methods/tfss_nt_sensitivity.py']:
            assert sha(ROOT/rel)==sha(Path(info['repo_root'])/rel),rel
        mapsfile=src/f'maps_batch{b}.npz';assert sha(mapsfile)==info['maps_sha256']
        setup,frozen,adaptive=create(Canonical);records=manifest['records'][b*16:b*16+16];items=[]
        for rec in records:
            assert sha(rec['image_path'])==rec['image_sha256'] and sha(rec['label_path'])==rec['label_sha256']
            image=load_image_rgb_float(rec['image_path']);anns=read_yolo_annotations(rec['label_path'])
            support,_,_=setup._build_support(image.shape,anns,image_path=rec['image_path']);items.append((image,anns,support))
        initial=rng();del setup,frozen,adaptive
        with np.load(mapsfile) as npz:
            maps=[tuple(torch.from_numpy(npz[f'{k}_{i}'].copy()) for k in ['q','nt_risk','q_valid']) for i in range(16)]
        ph=(max(im.shape[0] for im,_,_ in items)+31)//32*32;pw=(max(im.shape[1] for im,_,_ in items)+31)//32*32
        weights=[]
        for (im,_,support),(q,r,valid) in zip(items,maps):
            w=spatial_weights(q,r,torch.as_tensor(support),mode='TFSS-U',eta=.85)
            weights.append(F.pad(w,(0,pw-w.shape[1],0,ph-w.shape[0]),value=1))
        oldrows=json.loads((src/f'batch{b}_TFSS-U/trace.json').read_text());assert len(oldrows)==10
        batches[b]={'items':items,'records':records,'maps':maps,'weight':torch.stack(weights)[:,None].to(device),'initial':initial,'oldrows':oldrows}
        source_files[str(mapsfile)]=sha(mapsfile)

    def prepared(b,step,klass,method='TFSS-U'):
        t,frozen,adaptive=create(klass);pack=None
        optimizer=torch.optim.Adam([{'params':[t.fourier_coeff],'lr':t.universal_lr_fourier}])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=t.universal_epochs,eta_min=1e-5)
        if step>1:
            path=Path(inputs[str(b)]['output_root'])/f'batch{b}_{method}'/f'step{step-1}.pt'
            source_files[str(path)]=sha(path);pack=torch.load(path,map_location='cpu',weights_only=False)
            assert pack['step']==step-1 and pack['scheduler']['last_epoch']==0
            with torch.no_grad():
                t.fourier_coeff.copy_(pack['fourier_coeff']);t.freq_score.copy_(pack['freq_score']);t.freq_usage.copy_(pack['freq_usage'])
            optimizer.load_state_dict(copy.deepcopy(pack['optimizer']));scheduler.load_state_dict(pack['scheduler'])
            t._adaptive_calls=pack['adaptive_calls'];t._semantic_calls=pack['semantic_calls']
        t._epoch=0;t._active_idx=t._get_active_freq_indices(0,step-1)
        t._active_coords=[tuple(t.freq_candidate_coords[int(i)]) for i in t._active_idx.cpu().tolist()]
        if isinstance(t,SoftSpatialProjection):
            t.tfss_uniform_absolute_tolerance=conf['absolute_RMS_tolerance'];t.tfss_uniform_relative_tolerance=conf['relative_RMS_tolerance']
        restore(pack['rng'] if pack else batches[b]['initial'])
        return t,frozen,adaptive,optimizer,scheduler

    # Exact original failing composition: resume its own step4 and let the
    # unchanged Uniform-U computation raise its original guard before a forward.
    t,frozen,adaptive,optimizer,scheduler=prepared(1,5,Candidate,'Uniform-U')
    t.tfss_mode='Uniform-U';t.tfss_uniform_targets=batches[1]['oldrows'][8]['actual_person_rms']
    before=digest(optimizer.state_dict());before_rng=digest(rng());failure=None
    try:t._batch_loss(batches[1]['items'],prototypes)
    except ValueError as exc:
        if 'exceeds own S0 maximum' not in str(exc):raise
        failure=str(exc)
    assert failure and digest(optimizer.state_dict())==before
    ownbase=t.tfss_s0_delta.detach().clone();support=t.tfss_support.detach().clone();image=t.tfss_input
    failrows=[]
    for i,(x,d,m,target) in enumerate(zip(image,ownbase,support,t.tfss_uniform_targets)):
        row=feasibility(x,d,m,target,conf['absolute_RMS_tolerance'],conf['relative_RMS_tolerance'])
        row.update(image_index=i,image_id=batches[1]['records'][i]['image_id']);failrows.append(row)
    failed=[r for r in failrows if r['unreachable_at_locked_tolerance']];assert failed
    torch.save({'s0_base_delta':ownbase.cpu(),'actual_base_delta':((image+ownbase).clamp(0,1)-image).cpu(),'support':support.cpu(),
                'fourier_coeff':t.fourier_coeff.detach().cpu(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'before_composition_rng':torch.load(Path(inputs['1']['output_root'])/'batch1_Uniform-U/step4.pt',map_location='cpu',weights_only=False)['rng'],
                'active_indices':t._active_idx.cpu(),'active_coords':t._active_coords},out/'RECOVERED_FAILURE_STATE.pt')
    owncoeff=t.fourier_coeff.detach().clone();ownidx=t._active_idx.clone()
    recovery={'original_exception':failure,'replayed_exception_identical':True,'optimizer_step':4,
              'optimizer_before_after_digest_equal':True,'optimizer_digest':before,'Uniform_rng_before':before_rng,
              'first_failing_image':failed[0],'all_images':failrows,'GPU_tensor_state':'retained remotely, no RGB export'}
    del t,frozen,adaptive,optimizer,scheduler
    t,frozen,adaptive,optimizer,scheduler=prepared(1,5,Candidate,'TFSS-U')
    recovery['Utility_rng_before']=digest(rng());recovery['same_rng_before_composition']=digest(rng())==before_rng
    recovery['same_active_indices']=torch.equal(t._active_idx,ownidx)
    recovery['Fourier_coefficient_max_difference']=float((t.fourier_coeff-owncoeff).abs().max())
    t.tfss_mode='TFSS-U';t.tfss_weight=batches[1]['weight']
    # Direct actual composition only, exactly the first production call in _batch_loss.
    ph,pw=support.shape[-2:];tensor=torch.stack([F.pad(torch.from_numpy(im).permute(2,0,1).float(),(0,pw-im.shape[1],0,ph-im.shape[0]),value=.5) for im,_,_ in batches[1]['items']]).to(device)
    t._compose_delta_batched(tensor,support,torch.zeros_like(support),t._active_coords,t.fourier_coeff[t._active_idx],t.suppress_small,current_epoch=0)
    recovery['Utility_target_replay_max_difference']=max(abs(float(realized_rms(x,d,m))-target) for x,d,m,target in zip(tensor,t.tfss_delta,support,batches[1]['oldrows'][8]['actual_person_rms']))
    assert recovery['Utility_target_replay_max_difference']<=1e-7
    recovery['base_delta_max_difference']=float((t.tfss_s0_delta-ownbase).abs().max())
    recovery['same_Utility_base_feasibility']=[feasibility(x,d,m,target) for x,d,m,target in zip(tensor,t.tfss_s0_delta,support,batches[1]['oldrows'][8]['actual_person_rms'])]
    assert not any(r['unreachable_at_locked_tolerance'] for r in recovery['same_Utility_base_feasibility'])
    torch.save({'base_delta':t.tfss_s0_delta.detach().cpu(),'actual_base_delta':((tensor+t.tfss_s0_delta).clamp(0,1)-tensor).detach().cpu(),'Utility_delta':t.tfss_delta.detach().cpu(),'actual_Utility_delta':((tensor+t.tfss_delta).clamp(0,1)-tensor).detach().cpu(),'support':support.cpu(),'fourier_coeff':t.fourier_coeff.detach().cpu(),'optimizer':optimizer.state_dict()},out/'RECOVERED_UTILITY_STEP5_BASE.pt')
    write(out/'FAILURE_RECOVERY.json',recovery);del t,frozen,adaptive,optimizer,scheduler,tensor,image,ownbase,owncoeff,ownidx,support
    gc.collect();torch.cuda.empty_cache()

    cells=[(1,1),(1,5)] if a.phase=='smoke' else [(b,s) for b in [0,1] for s in range(1,6)]
    results=[];comparisons=[];noops=[];model_hashes=[];max_replay=0.;finite_losses=[]
    for b,step in cells:
        batch=batches[b];case=out/f'batch{b}_step{step}';case.mkdir();cache={};rng_records={}
        arms=['same_state_S0','Utility_replay','Static_Uniform_U']+(['S0_noop'] if a.phase=='smoke' else [])
        for arm in arms:
            t,frozen,adaptive,optimizer,scheduler=prepared(b,step,Canonical if arm=='same_state_S0' else Candidate)
            original_models=(digest(frozen.state_dict()),digest(adaptive.state_dict()));state_before=digest(optimizer.state_dict());coeff_before=digest(t.fourier_coeff)
            if arm!='same_state_S0':
                t.tfss_mode='S0' if arm=='S0_noop' else ('TFSS-U' if arm=='Utility_replay' else 'Uniform-U')
                t.tfss_weight=batch['weight']
            for eot in range(1,(2 if a.phase=='smoke' else 3)):
                idx=(step-1)*2+eot-1;old=batch['oldrows'][idx];target=old['actual_person_rms']
                if arm=='Static_Uniform_U':
                    target=next(r['actual_person_rms'] for r in results if (r['batch'],r['step'],r['eot'],r['arm'])==(b,step,eot,'Utility_replay'))
                    t.tfss_uniform_targets=target
                rng_before=digest(rng())
                if arm=='same_state_S0':rng_records[eot]=rng_before
                else:assert rng_before==rng_records[eot],(b,step,eot,arm,'random draws differ')
                result=t._batch_loss(batch['items'],prototypes);assert result is not None
                loss,metrics=result;assert torch.isfinite(loss);finite_losses.append(float(loss))
                row=_snapshot_metrics(metrics);row.update(batch=b,step=step,eot=eot,arm=arm,anchor='same Utility Fourier/optimizer pre-step state, no update',input_rng_sha256=rng_before)
                ref=case/f'same_state_reference_eot{eot}.pt'
                row['per_image']=collect(t,metrics,batch['items'],batch['maps'],mode='S0' if arm=='same_state_S0' else ('S0-noop' if arm=='S0_noop' else arm),reference_path=ref,output_prefix=case/f'{arm}_eot{eot}')
                row['actual_person_rms']=[float(realized_rms(x,d,m)) for x,d,m in zip(t.tfss_input,t.tfss_delta,t.tfss_support)]
                row['uniform_errors']=t.tfss_matching_errors;row['uniform_relative_errors']=t.tfss_matching_relative_errors
                row['fixed_image_ids']=[r['image_id'] for r in batch['records']]
                actual=(t.tfss_input+t.tfss_delta).clamp(0,1)-t.tfss_input
                assert not bool((actual.abs()>t.tfss_support*t.eps+1e-7).any()) and float((actual*(t.tfss_support==0)).abs().max())==0
                if arm=='Utility_replay':
                    keys=['L_BG','L_rel','L_ret','L_total','L_adapt_target','inner_update_norm']
                    diff=max([abs(row[k]-old[k]) for k in keys]+[abs(x-y) for x,y in zip(row['actual_person_rms'],target)])
                    row['original_U_replay_max_difference']=diff;max_replay=max(max_replay,diff);assert diff<=1e-7,(b,step,eot,diff)
                    row['original_reference_source']=inputs[str(b)]['commit']
                if arm=='S0_noop':
                    refrow=next(r for r in results if (r['batch'],r['step'],r['eot'],r['arm'])==(b,step,eot,'same_state_S0'))
                    diff=max(abs(row[k]-refrow[k]) for k in ['L_BG','L_rel','L_ret','L_total','L_adapt_target'])
                    assert diff<=1e-7;noops.append(diff)
                cache[arm,eot]={'base':t.tfss_s0_delta.detach().cpu(),'delta':t.tfss_delta.detach().cpu(),'support':t.tfss_support.detach().cpu()}
                if arm=='Static_Uniform_U':
                    base=cache['same_state_S0',eot]['delta'].to(device);u=cache['Utility_replay',eot]['delta'].to(device)
                    assert torch.equal(base,t.tfss_s0_delta.detach()),'Static control is not on the SAME incoming perturbation'
                    assert torch.equal(base,cache['Utility_replay',eot]['base'].to(device)), 'Utility base differs'
                    assert max(t.tfss_matching_errors)<=1e-7 and max(t.tfss_matching_relative_errors)<=1e-4
                    for i,((im,_,support),(q,r,valid)) in enumerate(zip(batch['items'],batch['maps'])):
                        h,w=im.shape[:2]
                        stats=spatial_selectivity(t.tfss_input[i,:,:h,:w],base[i,:,:h,:w],u[i,:,:h,:w],t.tfss_delta[i,:,:h,:w],torch.as_tensor(support,device=device),q,r,valid)
                        comparisons.append({'batch':b,'step':step,'eot':eot,'image_id':batch['records'][i]['image_id'],'regions':stats,'absolute_RMS_error':t.tfss_matching_errors[i],'relative_RMS_error':t.tfss_matching_relative_errors[i]})
                    del base,u
                results.append(row);write(out/'TRACE.json',results);write(out/'STATIC_SPATIAL_PAIRS.json',comparisons)
                if step==5:
                    np.savez_compressed(case/f'{arm}_eot{eot}_derived.npz',delta=t.tfss_delta.detach().cpu().numpy(),actual_delta=actual.detach().cpu().numpy(),base_delta=t.tfss_s0_delta.detach().cpu().numpy(),actual_base_delta=((t.tfss_input+t.tfss_s0_delta).clamp(0,1)-t.tfss_input).detach().cpu().numpy(),support=t.tfss_support.detach().cpu().numpy(),weight=t.tfss_applied_weight.detach().cpu().numpy())
                    if a.phase=='measure' and eot==2:
                        records=[];poison=t.tfss_input.detach()+actual.detach()
                        for i,(im,anns,support) in enumerate(batch['items']):
                            h,w=im.shape[:2];_,clean=nt_gradients(frozen,im,anns,imgsz=t.imgsz,coverage_policy='independent_same_TAL_for_uncovered_valid_GT')
                            _,dirty=nt_gradients(frozen,poison[i,:,:h,:w].permute(1,2,0).cpu().numpy(),anns,imgsz=t.imgsz,coverage_policy='independent_same_TAL_for_uncovered_valid_GT')
                            records.append({'image_id':batch['records'][i]['image_id'],'clean':clean,'poison':dirty,'scope':'recorded pre-update step5/EOT2; same Utility pre-step state; no optimizer update','assignment':'native allGT, uncovered sameTAL alone; clean/poison assignments may differ'})
                        write(case/f'{arm}_NT_GT.json',records)
            assert digest(optimizer.state_dict())==state_before and digest(t.fourier_coeff)==coeff_before
            assert original_models==(digest(frozen.state_dict()),digest(adaptive.state_dict()))
            model_hashes.append({'batch':b,'step':step,'arm':arm,'models_before_after':list(original_models),'optimizer_unchanged':True,'Fourier_unchanged':True,'scheduler_last_epoch':scheduler.last_epoch})
            del t,frozen,adaptive,optimizer,scheduler,loss,metrics,result,actual
            gc.collect();torch.cuda.empty_cache()
        del cache
    assert all(sha(path)==expected for path,expected in source_files.items())
    status={'state':'complete','phase':a.phase,'source_commit':commit,'new_optimizer_updates':0,'original_schedules_changed':False,'arms':'same_state_S0/Utility_replay/Static_Uniform_U, no independent control optimization','cells':cells,'recorded_forward_rows':len(results),'static_pair_cases':len(comparisons),'finite_loss':all(np.isfinite(finite_losses)),'max_original_U_replay_difference':max_replay,'S0_noop_max_difference':max(noops) if noops else None,'model_optimizer_checks':model_hashes,'protocol_sha256':sha(a.protocol),'input_file_sha256_unchanged':source_files,'official_metrics_disabled':True,'artifact_ingest_disabled':True,'cleanup_policy':'retain_all','deleted_paths':[],'elapsed_seconds':time.time()-started,'resource':resource,'reference_checkpoint_sha256':sha(checkpoint),'environment':{'torch':torch.__version__,'ultralytics':ultralytics.__version__,'deterministic':True,'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']}}
    write(out/'status.json',status);print(json.dumps({'state':'complete','phase':a.phase,'rows':len(results),'pairs':len(comparisons),'new_optimizer_updates':0}))


if __name__=='__main__':main()
