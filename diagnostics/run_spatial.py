"""CPU-only isolated mechanism diagnostic. Never train/export a dataset/victim.

Canonical S0 code is imported from its recorded parent directory, not rewritten.
The literal S1 projection is loaded unchanged from this locked source snapshot.
All generated files belong exclusively to the diagnostic output root.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--metadata',required=True)
    ap.add_argument('--config',required=True)
    ap.add_argument('--limit-images',type=int,default=0)
    a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=Path(a.output).resolve()
    allowed=Path('/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/tsfl_spatial_diagnostics_20261016')
    if not out.is_relative_to(allowed): raise ValueError('Output outside isolated diagnostic root')
    out.mkdir(parents=True,exist_ok=False)
    for k,v in {'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','YOLO_CONFIG_DIR':str(out/'runtime/yolo'),'MPLCONFIGDIR':str(out/'runtime/mpl'),'XDG_CACHE_HOME':str(out/'runtime/cache'),'TORCH_HOME':str(out/'runtime/torch'),'WANDB_MODE':'disabled'}.items():os.environ[k]=v
    sys.dont_write_bytecode=True
    if hasattr(os,'nice'):os.nice(19)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import cv2
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0,str(Path(__file__).parent))
    from spatial_math import (corr,top_mask,share,mean,rms,norm,cosine,normalized_attribution,center_weight,smoothness,vector_stats,operator_jacobian,gradient_energy,validate_math)
    meta=json.loads(Path(a.metadata).read_text());conf=json.loads(Path(a.config).read_text());manifest=json.loads(Path(a.manifest).read_text())
    if sha(a.manifest)!=conf['manifest_sha256']:raise ValueError('Manifest changed')
    if manifest['count']!=32 or manifest['split']!='train':raise ValueError('Not fixed32 train')
    parent=Path(meta['parent_root']);sys.path.insert(0,str(parent))
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.methods.b2_semantic import B2SemanticTrainer
    from ue_framework.methods.legacy_kproto_ret import LegacyKProtoGenerator,_rectangle,_snapshot_metrics,select_prototype
    from ue_framework.methods.safebg import safe_background_direction
    from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations
    import ue_framework.methods.b2_semantic as canonical
    if not Path(canonical.__file__).resolve().is_relative_to(parent):raise ValueError('Did not execute historical canonical Parent')
    provenance=json.loads((root/'docs/s1_evidence/parent_remote_identity.json').read_text())
    for rel,expected in provenance['source_sha256'].items():
        if sha(parent/rel)!=expected:raise ValueError('Canonical Parent source changed: '+rel)
    cfg=load_config(meta['config_path']);mc=cfg['methods']['legacy_kproto_ret']
    if sha(meta['config_path'])!=meta['config_sha256'] or sha(cfg['surrogate']['ckpt'])!=meta['channel_identity']['checkpoint_sha256']:raise ValueError('Parent config/reference changed')
    if sha(meta['global_params']['path'])!=meta['global_params']['sha256']:raise ValueError('S0 Fourier export changed')
    if sha(root/'ue_framework/methods/s1_csb.py')!=meta['source_sha256']['ue_framework/methods/s1_csb.py']:raise ValueError('Literal S1 projection changed')
    module_spec=importlib.util.spec_from_file_location('ue_framework.methods.diag_literal_s1',root/'ue_framework/methods/s1_csb.py')
    sm=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(sm)
    random.seed(conf['seed']);np.random.seed(conf['seed']);torch.manual_seed(conf['seed'])
    model=YOLO(cfg['surrogate']['ckpt']).model.to('cpu')
    adaptive=YOLO(cfg['surrogate']['ckpt']).model.to('cpu')
    pack=torch.load(meta['global_params']['path'],map_location='cpu',weights_only=True)
    proto_path=parent/'noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt'
    if sha(proto_path)!=provenance['noise_artifacts']['background_prototypes.pt']['sha256']:raise ValueError('Prototypes changed')
    prototypes=torch.load(proto_path,map_location='cpu',weights_only=True)['prototypes']
    cache=Path(meta['s1_root'])/'csb_cache';cm=json.loads((cache/'manifest.json').read_text());cache_by_id={r['id']:r for r in cm['images']}
    if sha(cache/'manifest.json')!=meta['s1_cache']['sha256']:raise ValueError('Frozen cache manifest changed')
    rows=manifest['records'][:a.limit_images] if a.limit_images else manifest['records']
    progress={'state':'running','stage':'setup','device':'cpu','threads':1,'official_metrics_disabled':True,'artifact_ingest_disabled':True,'sample_ids':[r['image_id'] for r in rows],'completed_batches':0,'started_at':time.time()}
    dump(out/'status.json',progress);dump(out/'DIAGNOSTIC_CONFIG.json',conf);dump(out/'FIXED_TRAIN_SAMPLE_MANIFEST.json',manifest)

    class BoundS1(sm.SpatialProjection):
        def __init__(self, original, budget, eps):self.original=original;self._batch_budget=budget;self.eps=eps
    class KnownParent:
        def _compose_delta_batched(self,*args,**kw):return self.original
    class LiteralS1(BoundS1,KnownParent):pass

    class DiagnosticTrainer(B2SemanticTrainer):
        def _forward_raw(self,image):
            result=super()._forward_raw(image);self.diag_decoded=result
            return result
        def _compose_delta_batched(self,*args,**kw):
            original=super()._compose_delta_batched(*args,**kw)
            img,support=args[:2];raw,old_delta,old_poison,binary,jnd=original
            base=old_delta/torch.where(support>0,support,torch.ones_like(support))
            if self.mode=='S0':delta=old_delta;poison=old_poison
            elif self.mode=='S1_literal':
                op=LiteralS1(original,self.diag_budget,self.eps)
                raw,delta,poison,binary,jnd=op._compose_delta_batched(*args,**kw)
                manual=torch.maximum(torch.minimum(base,self.diag_budget),-self.diag_budget)*(support>0)
                legacy=(self.diag_budget==support*self.eps).flatten(1).all(1)[:,None,None,None]
                manual=torch.where(legacy,base*(self.diag_budget/self.eps),manual)
                if float((manual-delta).detach().abs().max())!=0:raise ValueError('Literal S1/operator discrepancy')
            else:
                delta=old_delta*self.diag_weight;poison=(img+delta).clamp(0,1)
            self.diag_input=img;self.diag_support=support;self.diag_base=base
            self.diag_delta=delta;self.diag_raw=raw;self._last_poison=poison
            return raw,delta,poison,binary,jnd
        def _feature(self,image,grad):
            feature=super()._feature(image,grad)
            if grad:self.diag_poison_feature=feature;self.diag_poison_decoded=self.diag_decoded
            else:self.diag_clean_feature=feature;self.diag_clean_decoded=self.diag_decoded
            return feature

    trainer=DiagnosticTrainer(cfg,mc,torch.device('cpu'),model,adaptive)
    if trainer.channels.tolist()!=meta['selected_channels'] or trainer.feature_layer!='model.21':raise ValueError('Actual TSFL channels/layer changed')
    if trainer.universal_batch_size!=16 or trainer.eot_samples!=2:raise ValueError('Original batch/EOT protocol changed')
    index={tuple(c):i for i,c in enumerate(trainer.freq_candidate_coords)}
    active=torch.tensor([index[tuple(c)] for c in pack['coords']],dtype=torch.long)
    with torch.no_grad():
        trainer.fourier_coeff.zero_();trainer.fourier_coeff[active]=pack['fourier_coeff'];trainer.suppress_small.copy_(pack['suppress_small'])
    trainer.freq_active_idx=active;trainer._active_idx=active;trainer._active_coords=[tuple(c) for c in pack['coords']];trainer._epoch=23;trainer.mode='S0'
    def reset_calls():
        trainer._adaptive_calls=9144*2;trainer._semantic_calls=9144*2
    def model_hash(m):
        h=hashlib.sha256()
        for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    initial_model_hashes=[model_hash(model),model_hash(adaptive)]
    initial_coeff=trainer.fourier_coeff.detach().clone()
    generator=LegacyKProtoGenerator(cfg,mc,torch.device('cpu'),model,meta['global_params']['path'])
    groups={k:[] for k in ['TSFL_ATTRIBUTION','TSFL_SPATIAL_OVERLAP','S0_S1_SPATIAL_OPERATOR_AUDIT','CSB_CLIPPING_GRADIENT_AUDIT','TARGET_NONTARGET_SENSITIVITY_OVERLAP','CENTER_SCALING_FEASIBILITY','TSFL_SCALING_FEASIBILITY','FUSION_SCALING_FEASIBILITY','UNIFORM_B_CONTROL','PAIR_BATCH_LOSSES','FOURIER_COMMON_GOUT_AUDIT','MODE_PROXIES','MODE_PERTURBATION','OBJECT_SPATIAL_OVERLAP']}
    all_maps={};start=time.perf_counter();equivalence=[]

    def domains(im,anns):
        target=np.zeros(im.shape[:2],bool);other=target.copy()
        for ann in anns:
            mask=_rectangle(im.shape[:2],ann['bbox'])[0,0].numpy().astype(bool)
            if ann['cls']==14:target|=mask
            else:other|=mask
        return target,other

    def weighted_proxies(items):
        # Measured frozen P5 per-image diagnostics, not victim AP or new loss terms.
        clean=trainer.diag_clean_feature.detach();poison=trainer.diag_poison_feature.detach();ans=[]
        for b,(image,anns,support) in enumerate(items):
            targets,contexts,nts=trainer._masks(anns,support,image.shape[:2],poison[b:b+1],trainer.diag_delta.shape[-2:])
            nt0=trainer._pooled(clean[b:b+1],nts);ntp=trainer._pooled(poison[b:b+1],nts)
            nt=float((torch.stack(ntp)-torch.stack(nt0)).square().sum(-1).mean()) if nt0 else None
            bg=[];rel=[]
            for tm,ct in zip(targets,contexts):
                f0=trainer._pooled(clean[b:b+1],[tm]);fp=trainer._pooled(poison[b:b+1],[tm]);fc=trainer._pooled(clean[b:b+1],[ct])
                if not(f0 and fp and fc):continue
                _,pr,_=select_prototype(torch.stack(fc),prototypes)
                safe,_=safe_background_direction(torch.stack(f0)[:,trainer.channels],pr[:,trainer.channels],trainer.safe_bases,active_k=trainer.safe_bg_active_k,projector_rtol=trainer.safe_bg_projector_rtol,eps=trainer.orth_eps,mode=trainer.safe_bg_mode,safe_rank=trainer.safe_bg_rank,triplet_cache=trainer.safe_bg_cache,removal_alpha=trainer.safe_bg_removal_alpha)
                hp=torch.stack(fp)[:,trainer.channels];h0=torch.stack(f0)[:,trainer.channels]
                cs=F.cosine_similarity(hp,safe,dim=-1);cc=F.cosine_similarity(hp,h0,dim=-1)
                bg.append(float((1-cs).mean()));rel.append(float(F.relu(trainer.margin+cc-cs).mean()))
            ph,pw=trainer.diag_delta.shape[-2:];ratio=min(trainer.imgsz/ph,trainer.imgsz/pw)
            nh,nw=round(ph*ratio),round(pw*ratio);left=(trainer.imgsz-nw)//2;top=(trainer.imgsz-nh)//2
            evidence=[];clean_evidence=[]
            for ann in anns:
                cls=int(ann['cls'])
                if cls==14:continue
                ih,iw=image.shape[:2];cx,cy,bw,bh=ann['bbox']
                gt=poison.new_tensor([[cx*iw*ratio+left,cy*ih*ratio+top,bw*iw*ratio,bh*ih*ratio]])
                for decoded,values in [(trainer.diag_poison_decoded,evidence),(trainer.diag_clean_decoded,clean_evidence)]:
                    pred=decoded[b].detach();iou=trainer.adaptive._iou(pred[:4].T,gt).flatten()
                    aligned=pred[4+cls].clamp(1e-6,1).sqrt()*iou.clamp_min(1e-6).pow(6)
                    values.append(float(aligned.topk(min(100,len(aligned))).values.mean()))
            ans.append(dict(background_misdirection_proxy=float(np.mean(bg)) if bg else None,target_relation_proxy=float(np.mean(rel)) if rel else None,nt_feature_drift=nt,valid_target_proxy_instances=len(bg),nt_detection_proxy=float(np.mean(evidence)) if evidence else None,nt_detection_proxy_clean=float(np.mean(clean_evidence)) if clean_evidence else None,nt_detection_proxy_status='FROZEN_GT_TASK_ALIGNED_EVIDENCE_NOT_AP' if evidence else 'LEGAL_NO_NT_GT'))
        return ans

    for batch_idx in range(0,len(rows),16):
        batch=rows[batch_idx:batch_idx+16];items=[];maps=[];hp=wp=0
        for row in batch:
            if sha(row['image_path'])!=row['image_sha256'] or sha(row['label_path'])!=row['label_sha256']:raise ValueError('Fixed train source changed')
            image=load_image_rgb_float(row['image_path']);anns=read_yolo_annotations(row['label_path']);support=trainer._bbox_half_support(image.shape,anns)
            items.append((image,anns,support));hp=max(hp,image.shape[0]);wp=max(wp,image.shape[1])
        hp=(hp+31)//32*32;wp=(wp+31)//32*32
        progress.update(stage='attribution',batch=batch_idx//16);dump(out/'status.json',progress)
        for row,(image,anns,support) in zip(batch,items):
            h,w=image.shape[:2];target,other=domains(image,anns);q=np.zeros((h,w),np.float32);g_sum=q.copy();cent=q.copy()
            x=torch.from_numpy(image).permute(2,0,1)[None].requires_grad_(True)
            inp=F.pad(x,(0,wp-w,0,hp-h),value=.5)
            feature=trainer._feature(inp,grad=True);targets,_,_=trainer._masks(anns,support,(h,w),feature,(hp,wp))
            people=[ann for ann in anns if ann['cls']==14];object_maps=[]
            for obj,(ann,mask) in enumerate(zip(people,targets)):
                region=_rectangle((h,w),ann['bbox'])[0,0].numpy().astype(bool)
                mass=float(mask.sum());selected=feature[:,trainer.channels]
                if mass<=0:raise ValueError('Zero GT feature mask')
                z=(selected.square()*mask).sum()/(len(trainer.channels)*mask.sum())
                grad=torch.autograd.grad(z,x,retain_graph=True)[0][0].norm(dim=0).numpy()
                if not np.isfinite(grad).all() or not grad.max()>0:raise ValueError('Invalid TSFL attribution')
                qi,valid=normalized_attribution(grad,region)
                if qi is None:raise ValueError('Degenerate TSFL attribution '+row['image_id'])
                cw,rr=center_weight((h,w),ann['bbox'],region,conf['eta_tfss'],conf['sigma'])
                q=np.maximum(q,qi);g_sum+=grad;cent=np.maximum(cent,cw)
                rowo=dict(image_id=row['image_id'],object_index=obj,scale=row['annotations'][next(i for i,a in enumerate(row['annotations']) if a['bbox']==ann['bbox'] and a['cls']==14)]['scale'],person_gt_count=len(people),feature_mask_mass=mass,parent_pool_min_mass_pass=mass>=1,energy_scalar=float(z.detach()),attribution_inside_box_share=share(grad,region),attribution_center_share=share(grad,region&(rr<=.5),region),attribution_edge_share=share(grad,region&(rr>=.8),region),input_gradient_zero_fraction=float((grad==0).mean()),**valid)
                groups['TSFL_ATTRIBUTION'].append(rowo)
                object_maps.append(dict(object_index=obj,region=region,raw=grad,q=qi,radius=rr))
            cr=cache_by_id[row['image_id']];file=cache/cr['map_file']
            if sha(file)!=cr['map_sha256']:raise ValueError('Frozen CSB map changed')
            with np.load(file,allow_pickle=False) as saved:budget=saved['budget'].copy();gnt=saved['raw'].copy();gnt_norm=saved['normalized'].copy()
            s0path=Path(meta['s0_poison_data']['path'])/'images/train'/(row['image_id']+'.png');s1path=Path(meta['s1_root'])/'dataset/images/train'/(row['image_id']+'.png')
            s0=load_image_rgb_float(str(s0path))-image;s1=load_image_rgb_float(str(s1path))-image
            actual0=s0.transpose(2,0,1);actual1=s1.transpose(2,0,1)
            with torch.no_grad():
                xt=generator._to_tensor(image);it=torch.from_numpy(support)[None,None]
                _,de,_,_,_=generator._compose_delta_batched(xt,it,torch.zeros_like(it),generator.coords,generator.fourier_coeff,generator.suppress_small,current_epoch=23)
                export_base=(de/torch.where(it>0,it,torch.ones_like(it)))[0].numpy();pattern=generator._build_global_freq_pattern(h,w,generator.coords,generator.fourier_coeff)[0].norm(dim=0).numpy()
            wa=cent;wb=(conf['eta_tfss']+(1-conf['eta_tfss'])*q)*target
            wc=(conf['eta_tfss']+(1-conf['eta_tfss'])*(conf['rho']*q+(1-conf['rho'])*cent))*target
            uniform=np.where(target,float(wb[target].mean()),0).astype(np.float32)
            maps.append(dict(object_maps=object_maps,id=row['image_id'],target=target,other=other,q=q,g_tsfl=g_sum,g_nt=gnt,g_nt_normalized=gnt_norm,budget=budget,a0=support,delta0=actual0,delta1=actual1,export_base=export_base,pattern=pattern,weights={'A':wa,'B':wb,'C':wc,'UniformB':uniform},strata=row['strata']))
        progress.update(stage='paired_parent_gradients');dump(out/'status.json',progress)
        mode_outputs={};base_inputs=[];g_outputs=[];attack_outputs=[];param_s0=[];literal_jacobians=[]
        for mode in ['S0','S1_literal','A','B','C','UniformB']:
            trainer.mode=mode;reset_calls();results=[]
            if mode=='UniformB':
                for b,m in enumerate(maps):
                    h,w=m['target'].shape;dm=m['target']
                    eb=sum(float((x['delta'][b,:,:h,:w][:,dm]**2).sum()) for x in mode_outputs['B'])
                    e0=sum(float((x['delta'][b,:,:h,:w][:,dm]**2).sum()) for x in mode_outputs['S0'])
                    if e0<=0:raise ValueError('Zero S0 perturbation energy')
                    scale=math.sqrt(eb/e0)
                    m['weights']['UniformB']=np.where(dm,scale,0).astype(np.float32)
            trainer.diag_budget=torch.stack([F.pad(torch.from_numpy(m['budget'])[None],(0,wp-m['target'].shape[1],0,hp-m['target'].shape[0])) for m in maps])
            if mode not in ['S0','S1_literal']:
                trainer.diag_weight=torch.stack([F.pad(torch.from_numpy(m['weights'][mode].astype(np.float32))[None],(0,wp-m['target'].shape[1],0,hp-m['target'].shape[0])) for m in maps])
            for eot in range(2):
                seed=conf['seed']+1000*(batch_idx//16)+eot
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                loss,metrics=trainer._batch_loss(items,prototypes)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite original protection loss')
                gout=torch.autograd.grad(loss,trainer.diag_delta,retain_graph=True)[0]
                gf=torch.autograd.grad(loss,trainer.fourier_coeff,retain_graph=True)[0]
                if not(torch.isfinite(gout).all() and torch.isfinite(gf).all()):raise FloatingPointError('Nonfinite protection gradient')
                scalars={k:float(metrics[k].detach()) for k in ['L_total','L_BG','L_rel','L_ret','L_adapt_target','L_adapt_nt']}
                groups['PAIR_BATCH_LOSSES'].append(dict(batch=batch_idx//16,mode=mode,eot=eot,**scalars,fourier_norm=float(gf.norm())))
                results.append(dict(effective=(trainer._last_poison-trainer.diag_input).detach().numpy(),delta=trainer.diag_delta.detach().numpy(),poison=trainer._last_poison.detach().numpy(),gout=gout.detach().numpy(),gf=gf.numpy(),proxy=weighted_proxies(items),loss=scalars))
                if mode=='S0':
                    attack=metrics['weighted_L_BG']+metrics['weighted_L_rel']+trainer.lambda_adapt*metrics['L_adapt_target']
                    ga=torch.autograd.grad(attack,trainer.diag_delta,retain_graph=True)[0]
                    base_inputs.append(trainer.diag_base.detach().numpy());g_outputs.append(gout.detach().numpy());attack_outputs.append(ga.numpy());param_s0.append(gf.numpy())
                    base=trainer.diag_base;aa=trainer.diag_support;bb=trainer.diag_budget
                    literal=torch.maximum(torch.minimum(base,bb),-bb)*(aa>0)
                    legacy=(bb==aa*trainer.eps).flatten(1).all(1)[:,None,None,None]
                    literal=torch.where(legacy,base*(bb/trainer.eps),literal)
                    jclip=torch.autograd.grad(literal.sum(),base,retain_graph=True)[0]
                    literal_jacobians.append(jclip.detach().numpy())
                    gps=torch.autograd.grad(base,trainer.fourier_coeff,grad_outputs=gout*aa,retain_graph=True)[0]
                    gpc=torch.autograd.grad(base,trainer.fourier_coeff,grad_outputs=gout*jclip,retain_graph=True)[0]
                    groups['FOURIER_COMMON_GOUT_AUDIT'].append(dict(batch=batch_idx//16,eot=eot,scale_norm=float(gps.norm()),clip_norm=float(gpc.norm()),clip_to_scale_ratio=float(gpc.norm()/gps.norm()) if gps.norm()>0 else None,cosine=cosine(gps.numpy(),gpc.numpy()),definition='operator-only common g_out VJP through same S0 base; excludes direct raw TV/budget path; actual total Fourier gradients separately recorded'))
            mode_outputs[mode]=results
            if mode=='S0' and batch_idx==0:
                ref=B2SemanticTrainer(cfg,mc,torch.device('cpu'),YOLO(cfg['surrogate']['ckpt']).model.to('cpu'),YOLO(cfg['surrogate']['ckpt']).model.to('cpu'))
                with torch.no_grad():
                    ref.fourier_coeff.zero_();ref.fourier_coeff[active]=pack['fourier_coeff'];ref.suppress_small.copy_(pack['suppress_small'])
                ref.freq_active_idx=active;ref._active_idx=active;ref._active_coords=[tuple(c) for c in pack['coords']];ref._epoch=23;ref._adaptive_calls=9144*2;ref._semantic_calls=9144*2
                for eot in range(2):
                    seed=conf['seed']+eot;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                    lr,mr=ref._batch_loss(items,prototypes);gr=torch.autograd.grad(lr,ref.fourier_coeff)[0]
                    dif=dict(eot=eot,reference_module=str(canonical.__file__),samples=len(items),loss_difference=abs(float(lr)-results[eot]['loss']['L_total']),poison_max_difference=float(np.abs(ref._last_poison.detach().numpy()-results[eot]['poison']).max()),fourier_gradient_max_difference=float(np.abs(gr.numpy()-results[eot]['gf']).max()),tolerance=1e-7)
                    equivalence.append(dif)
                    if max(dif['loss_difference'],dif['poison_max_difference'],dif['fourier_gradient_max_difference'])>dif['tolerance']:raise ValueError('Canonical no-op equivalence failed')
        for b,(row,m) in enumerate(zip(batch,maps)):
            h,w=m['target'].shape;target=m['target'];over=target&m['other'];exclusive=target&~m['other'];q=m['q'];g=m['g_tsfl'];highq=top_mask(q,target,.2)
            gp=np.mean([z[b,:,:h,:w] for z in g_outputs],axis=0);gpmag=np.linalg.norm(gp,axis=0);attackmag=np.linalg.norm(np.mean([z[b,:,:h,:w] for z in attack_outputs],axis=0),axis=0);highp=top_mask(gpmag,target,.2);ng=m['g_nt'];highn=top_mask(ng,target,.2)
            actualmag=np.linalg.norm(m['delta0'],axis=0);highd=top_mask(actualmag,target,.2)
            groups['TSFL_SPATIAL_OVERLAP'].append(dict(image_id=m['id'],spearman_tsfl_actual_delta=corr(q,actualmag,target),spearman_tsfl_protection=corr(q,gpmag,target),top20_tsfl_delta_overlap=float((highq&highd).sum()/highq.sum()),top20_tsfl_protection_overlap=float((highq&highp).sum()/highq.sum()),top10_actual_delta_energy_share=share((m['delta0']**2).sum(0),top_mask(q,target,.1),target),top20_actual_delta_energy_share=share((m['delta0']**2).sum(0),highq,target),target_attribution_share=share(g,target),person_pixels=int(target.sum()),overlap_pixels=int(over.sum())))
            groups['TARGET_NONTARGET_SENSITIVITY_OVERLAP'].append(dict(image_id=m['id'],no_nt_gt=not m['other'].any(),spearman_raw_nt_tsfl=corr(ng,q,target),spearman_raw_nt_protection=corr(ng,gpmag,target),top20_nt_tsfl_overlap=float((highn&highq).sum()/highn.sum()) if ng[target].max()>0 else None,top20_nt_protection_overlap=float((highn&highp).sum()/highn.sum()) if ng[target].max()>0 else None,high_tsfl_budget_byte=mean(m['budget']*255,highq),high_protection_budget_byte=mean(m['budget']*255,highp),high_protection_budget_delta_byte=mean((m['budget']-m['a0']*trainer.eps)*255,highp),low_protection_budget_delta_byte=mean((m['budget']-m['a0']*trainer.eps)*255,target&~highp),overlap_budget_byte=mean(m['budget']*255,over),exclusive_budget_byte=mean(m['budget']*255,exclusive),s0_actual_rms=rms(m['delta0'],target),s1_actual_rms=rms(m['delta1'],target)))
            for om in m['object_maps']:
                reg=om['region'];oq=om['q'];hq=top_mask(oq,reg,.2);hg=top_mask(gpmag,reg,.2)
                groups['OBJECT_SPATIAL_OVERLAP'].append(dict(image_id=m['id'],object_index=om['object_index'],pixels=int(reg.sum()),spearman_tsfl_protection=corr(oq,gpmag,reg),top20_tsfl_protection_overlap=float((hq&hg).sum()/hq.sum()),tsfl_center_share=share(om['raw'],reg&(om['radius']<=.5),reg),protection_center_energy_share=share(gpmag**2,reg&(om['radius']<=.5),reg),protection_edge_energy_share=share(gpmag**2,reg&(om['radius']>=.8),reg),s0_delta_center_energy_share=share((m['delta0']**2).sum(0),reg&(om['radius']<=.5),reg),s0_delta_edge_energy_share=share((m['delta0']**2).sum(0),reg&(om['radius']>=.8),reg)))
            for mode,draws in mode_outputs.items():
                for eot,draw in enumerate(draws):
                    groups['MODE_PROXIES'].append(dict(image_id=m['id'],mode=mode,eot=eot,**draw['proxy'][b]))
                    for domain_name,dm in [('person',target),('overlap',over),('exclusive',exclusive)]:
                        req=draw['delta'][b,:,:h,:w];eff=draw['effective'][b,:,:h,:w]
                        groups['MODE_PERTURBATION'].append(dict(image_id=m['id'],mode=mode,eot=eot,domain=domain_name,pixels=int(dm.sum()),requested_rms=rms(req,dm),effective_rms=rms(eff,dm),requested_mean_abs=mean(np.abs(req).mean(0),dm),effective_mean_abs=mean(np.abs(eff).mean(0),dm),requested_energy=float((req[:,dm]**2).sum()),effective_energy=float((eff[:,dm]**2).sum()),effective_budget_excess=float(np.maximum(np.abs(eff)-(m['budget'][None] if mode=='S1_literal' else m['a0'][None]*trainer.eps),0).max()),effective_budget_excess_vs_s0=float(np.maximum(np.abs(eff)-m['a0'][None]*trainer.eps,0).max()),budget_reference='actual_CSB' if mode=='S1_literal' else 'S0_fixed8_4_0',effective_outside_max=float(np.abs(eff[:,~target]).max()) if (~target).any() else 0.))
            for domain_name,dm in [('person',target),('overlap',over),('exclusive',exclusive)]:
                low=dm&(m['budget']<m['a0']*trainer.eps-1e-9);hq=top_mask(q,dm,.2);hg=top_mask(gpmag,dm,.2);ha=top_mask(attackmag,dm,.2)
                groups['TARGET_NONTARGET_SENSITIVITY_OVERLAP'].append(dict(image_id=m['id'],domain=domain_name,pixels=int(dm.sum()),spearman_normalized_nt_tsfl=corr(m['g_nt_normalized'],q,dm),spearman_normalized_nt_protection=corr(m['g_nt_normalized'],gpmag,dm),spearman_nt_attack=corr(ng,attackmag,dm),high_tsfl_in_reduced_budget_fraction=float((hq&low).sum()/hq.sum()) if hq.any() else None,high_protection_in_reduced_budget_fraction=float((hg&low).sum()/hg.sum()) if hg.any() else None,high_attack_budget_delta_byte=mean((m['budget']-m['a0']*trainer.eps)*255,ha)))
            for eot in range(2):
                base=base_inputs[eot][b,:,:h,:w];go=g_outputs[eot][b,:,:h,:w];scale=go*m['a0'][None]
                for op,budget,literal in [('literal_s1_csb',m['budget'],True),('hard_clip_s0_budget',m['a0']*trainer.eps,False),('soft_same_csb_cap',m['budget'],False)]:
                    if op=='soft_same_csb_cap':jac=np.broadcast_to((budget/trainer.eps)[None],base.shape);od=base*jac;legacy=False
                    else:od,jac,legacy=operator_jacobian(base,m['a0'],budget,eps=trainer.eps,literal_s1=literal)
                    jdiff=float(np.abs(jac-literal_jacobians[eot][b,:,:h,:w]).max()) if literal else None
                    if literal and jdiff!=0:raise ValueError('Literal S1 Jacobian differs from production autograd')
                    transformed=go*jac;zero=np.all(jac==0,axis=0)
                    groups['CSB_CLIPPING_GRADIENT_AUDIT'].append(dict(image_id=m['id'],eot=eot,operator=op,literal_autograd_jacobian_max_difference=jdiff,configured_epsilon=trainer.eps,legacy_image_branch=legacy,strict_clip_scalar_ratio=float((np.abs(base[:,target])>budget[target][None]).mean()) if op!='soft_same_csb_cap' else 0.,jacobian_zero_scalar_ratio=float((jac[:,target]==0).mean()),jacobian_zero_pixel_ratio=mean(zero,target),overlap_zero_pixel_ratio=mean(zero,over),high_tsfl_zero_pixel_ratio=mean(zero,highq),high_protection_zero_pixel_ratio=mean(zero,highp),lost_gout_energy_share=gradient_energy(go,jac,target)['blocked_scalar_energy_share'],**gradient_energy(go,jac,target),high_tsfl_blocked_energy_share=gradient_energy(go,jac,highq)['blocked_scalar_energy_share'],high_protection_blocked_energy_share=gradient_energy(go,jac,highp)['blocked_scalar_energy_share'],overlap_blocked_energy_share=gradient_energy(go,jac,over)['blocked_scalar_energy_share'],exclusive_blocked_energy_share=gradient_energy(go,jac,exclusive)['blocked_scalar_energy_share'],**vector_stats(scale,transformed,target),output_rms=rms(od,target)))
            groups['S0_S1_SPATIAL_OPERATOR_AUDIT'].append(dict(image_id=m['id'],s0_export_rms=rms(m['delta0'],target),s1_export_rms=rms(m['delta1'],target),s0_nominal_budget_image_mean_byte=mean(m['a0']*trainer.eps*255,target),s1_nominal_budget_image_mean_byte=mean(m['budget']*255,target),export_base_rms=rms(m['export_base'],target),scope='actual exports are observational; causal operator rows use same S0 training-state base per EOT'))
            for mode,group in [('A','CENTER_SCALING_FEASIBILITY'),('B','TSFL_SCALING_FEASIBILITY'),('C','FUSION_SCALING_FEASIBILITY'),('UniformB','UNIFORM_B_CONTROL')]:
                v=mode_outputs[mode];baseline=mode_outputs['S0'];weight=m['weights'][mode];new=np.mean([x['delta'][b,:,:h,:w] for x in v],axis=0);old=np.mean([x['delta'][b,:,:h,:w] for x in baseline],axis=0)
                proxy={k:float(np.mean([x['proxy'][b][k] for x in v])) if all(x['proxy'][b][k] is not None for x in v) else None for k in ['background_misdirection_proxy','target_relation_proxy','nt_feature_drift','nt_detection_proxy','nt_detection_proxy_clean']}
                draws=[x['delta'][b,:,:h,:w] for x in v];old_draws=[x['delta'][b,:,:h,:w] for x in baseline]
                old_energy=np.mean([(x**2).sum(0) for x in old_draws],axis=0);new_energy=np.mean([(x**2).sum(0) for x in draws],axis=0)
                effective_draws=[x['effective'][b,:,:h,:w] for x in v];effective_old=[x['effective'][b,:,:h,:w] for x in baseline]
                effective_energy=np.mean([(x**2).sum(0) for x in effective_draws],axis=0);effective_old_energy=np.mean([(x**2).sum(0) for x in effective_old],axis=0)
                effective_stats=dict(effective_mean_abs_delta=float(np.mean([mean(np.abs(x).mean(0),target) for x in effective_draws])),effective_mean_abs_delta_s0=float(np.mean([mean(np.abs(x).mean(0),target) for x in effective_old])),effective_rms=float(np.mean([rms(x,target) for x in effective_draws])),effective_s0_rms=float(np.mean([rms(x,target) for x in effective_old])),effective_overlap_rms=float(np.mean([rms(x,over) for x in effective_draws])) if over.any() else None,effective_high_tsfl_energy_retention=float(effective_energy[highq].sum()/effective_old_energy[highq].sum()) if effective_old_energy[highq].sum()>0 else None,effective_high_protection_energy_retention=float(effective_energy[highp].sum()/effective_old_energy[highp].sum()) if effective_old_energy[highp].sum()>0 else None)
                groups[group].append(dict(image_id=m['id'],mode=mode,requested_pre_image_clamp=True,**effective_stats,weight_mean=mean(weight,target),weight_smoothness=smoothness(weight,target),requested_mean_abs_delta=float(np.mean([mean(np.abs(x).mean(0),target) for x in draws])),requested_mean_abs_delta_s0=float(np.mean([mean(np.abs(x).mean(0),target) for x in old_draws])),requested_rms=float(np.mean([rms(x,target) for x in draws])),requested_s0_rms=float(np.mean([rms(x,target) for x in old_draws])),requested_overlap_rms=float(np.mean([rms(x,over) for x in draws])) if over.any() else None,high_tsfl_energy_retention=float(new_energy[highq].sum()/old_energy[highq].sum()) if old_energy[highq].sum()>0 else None,high_protection_energy_retention=float(new_energy[highp].sum()/old_energy[highp].sum()) if old_energy[highp].sum()>0 else None,high_protection_weight=mean(weight,highp),fixed_gout_norm_retention=norm((gp*m['a0'][None]*weight[None])[:,target])/max(norm((gp*m['a0'][None])[:,target]),1e-30),fourier_gradient_cosine=cosine(np.mean([x['gf'] for x in v],axis=0),np.mean(param_s0,axis=0)),outside_max=float(np.abs(new[:,~target]).max()) if (~target).any() else 0.,budget_excess=max(float(np.maximum(np.abs(x)-m['a0'][None]*trainer.eps,0).max()) for x in draws),**proxy))
            all_maps[m['id']]={k:m[k] for k in ['q','g_tsfl','g_nt','g_nt_normalized','budget','a0','pattern']};all_maps[m['id']].update(g_protect=gpmag,g_attack=attackmag,delta0=np.linalg.norm(m['delta0'],axis=0),delta1=np.linalg.norm(m['delta1'],axis=0),high_tsfl=highq,high_protection=highp)
            for om in m['object_maps']:
                for key in ['raw','q','region','radius']:all_maps[m['id']]['object_'+str(om['object_index'])+'_'+key]=om[key]
        if not torch.equal(initial_coeff,trainer.fourier_coeff):raise ValueError('Diagnostic updated Fourier parameters')
        if [model_hash(model),model_hash(adaptive)]!=initial_model_hashes:raise ValueError('Reference model state changed')
        progress.update(completed_batches=batch_idx//16+1,elapsed_seconds=time.perf_counter()-start);dump(out/'status.json',progress)
    import csv
    for name,rs in groups.items():
        if not rs:continue
        fields=sorted({k for row in rs for k in row})
        with (out/(name+'.csv')).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rs)
    mapsdir=out/'derived_maps';mapsdir.mkdir()
    for image_id,m in all_maps.items():np.savez_compressed(mapsdir/(image_id+'.npz'),**m)
    dump(out/'MATH_VALIDATION.json',validate_math());dump(out/'CANONICAL_EQUIVALENCE.json',equivalence)
    dump(out/'RUN_RECEIPT.json',dict(state='complete',device='cpu',threads=1,torch_version=torch.__version__,source_commit=os.popen('git rev-parse HEAD').read().strip(),canonical_module=str(canonical.__file__),frozen_reference_sha256=sha(cfg['surrogate']['ckpt']),selected_channels=trainer.channels.tolist(),feature_layer=trainer.feature_layer,model_state_unchanged=True,fourier_parameters_unchanged=True,optimizer_updates=0,virtual_five_steps='NOT_RUN_optional_not_required',images=len(rows),person_gt=sum(sum(a['cls']==14 for a in x['annotations']) for x in rows),elapsed_seconds=time.perf_counter()-start,limitations=['CPU FP32 diagnostic, not historical GPU bitwise replay','S0 export has no original Adam/RNG; diagnostic uses recorded final active Fourier and newly fixed paired RNG','protection derivatives use original batch16/EOT2 (smoke limit may shorten final batch), posttraining diagnostic counters18288','feature attribution uses nonnegative mean squared P5 selected response, not measured UE causal utility','NT detection proxy is frozen GT task-aligned evidence, not AP or full detection loss','no victim, dataset generation, model update, validation use or GitHub push']))
    progress.update(state='complete',stage='complete',elapsed_seconds=time.perf_counter()-start,model_state_unchanged=True,fourier_parameters_unchanged=True);dump(out/'status.json',progress)


if __name__=='__main__':main()
