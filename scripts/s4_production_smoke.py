"""Bounded current-production admission; no official metrics or data ingestion."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import types
import numpy as np
import torch
from ultralytics import YOLO

from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations
from ue_framework.methods.b2_semantic import B2SemanticTrainer
from ue_framework.methods.legacy_kproto_ret import _snapshot_metrics,LegacyKProtoGenerator
from ue_framework.methods.s4_tfss import S4Trainer,CleanMapCache,model_hash,rng_state,restore_rng
from ue_framework.methods.s4_optimization import optimize_s4
from ue_framework.methods.s4_pixel_audit import png_project
from ue_framework.methods.tfss_clean_maps import canonical_map_canvas
from ue_framework.methods.tfss_spatial import spatial_weights

ROOT=Path(__file__).resolve().parents[1]


def array_digest(value):
    array=np.ascontiguousarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def smoke(cfg,setup_trainer,prototypes,lock,out):
    manifest=json.loads((ROOT/'docs/tfss/FIXED_TRAIN_SAMPLE_MANIFEST.json').read_text())
    records=manifest['records'];images=[r['image_path'] for r in records]
    identity=dict(smoke_fixed_manifest='original TRAIN32',seed=0,source='current production',cache_identity=setup_trainer.s4_cache.identity)
    observed={};integrity=[];training_spatial_observations=[]
    def create(cls):
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        frozen=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
        adaptive=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
        t=cls(copy.deepcopy(cfg),copy.deepcopy(cfg['methods']['legacy_kproto_ret']),torch.device('cuda:0'),frozen,adaptive)
        t.tfss_mode='S0';return t

    class ActualCanonical(B2SemanticTrainer):
        """Observations only; original loss and original train_universal execute."""
        def _compose_delta_batched(self,*args,**kwargs):
            result=super()._compose_delta_batched(*args,**kwargs)
            self.ref_delta=result[1];self.ref_support=args[1] if len(args)>1 else kwargs['inner_t']
            return result
        def _batch_loss(self,items,prototypes):
            result=super()._batch_loss(items,prototypes)
            capture(self,items,*result,len(observed['reference'])//2,len(observed['reference'])%2,'reference')
            return result

    def capture(t,items,loss,metrics,step,eot,role):
        gradient=torch.autograd.grad(loss,t.fourier_coeff,retain_graph=True)[0].detach().cpu()
        delta=(t.ref_delta if role=='reference' else t.tfss_delta).detach().cpu()
        support=(t.ref_support if role=='reference' else t.tfss_support).detach().cpu()
        values={k:_snapshot_metrics(metrics)[k] for k in ('L_BG','L_rel','L_ret','L_orth_objective','L_total')}
        if role=='reference':
            path=out/'reference'/f'eot_{len(observed[role]):02d}.pt'
            torch.save(dict(gradient=gradient,delta=delta,support=support,values=values),path)
            observed[role].append(str(path))
        else:
            index=step*2+eot;ref=torch.load(observed['reference'][index],map_location='cpu',weights_only=True)
            difference=max(float((gradient-ref['gradient']).abs().max()),float((delta-ref['delta']).abs().max()),
                float((support-ref['support']).abs().max()),max(abs(values[k]-ref['values'][k]) for k in values))
            observed[role].append(difference)
    observed['reference']=[];observed['noop']=[]
    reference=create(ActualCanonical);reference.max_steps=5
    # Only bank loading is supplied with the exact immutable Parent prototypes.
    # The actual unchanged canonical optimization loop executes below.
    reference._build_bank=lambda *args,**kwargs:prototypes
    refdir=out/'reference';refdir.mkdir()
    before=[model_hash(reference.surrogate),model_hash(reference.adaptive.model)]
    # Override only target-image enumeration to the explicit bounded TRAIN32.
    reference._collect_target_images=lambda *args:images
    reference.train_universal(str(Path(images[0]).parent),str(Path(records[0]['label_path']).parent),
        str(refdir/'global_params.pt'),str(refdir/'diagnostics.csv'),str(refdir/'diagnostics.json'),0)
    ref_coeff=reference.fourier_coeff.detach().cpu().clone()
    integrity.append(before==[model_hash(reference.surrogate),model_hash(reference.adaptive.model)])
    del reference
    noop=create(S4Trainer);noop.max_steps=5
    result=optimize_s4(noop,images,Path(records[0]['label_path']).parent,prototypes,out/'noop',identity,stop_after=5,
        observe=lambda t,items,loss,metrics,step,eot:capture(t,items,loss,metrics,step,eot,'noop'))
    coefficient_difference=float((noop.fourier_coeff.detach().cpu()-ref_coeff).abs().max())
    maximum=max(observed['noop']+[coefficient_difference])
    if maximum>1e-7:raise ValueError('Actual canonical/no-op loss/gradient/delta/update equivalence failed: '+str(maximum))
    del noop
    # Production U loop and exact3->5 resume, same explicit fixed input selection.
    def utility():
        t=create(S4Trainer);t.tfss_mode='TFSS-U'
        t.s4_cache=setup_trainer.s4_cache
        return t

    def observe_training_spatial(t,items,loss,metrics,step,eot):
        if t.tfss_mode!='TFSS-U' or len(t.s4_maps)!=len(items) or len(t.s4_item_paths)!=len(items):
            raise ValueError('Missing actual Utility training maps or image identity')
        if t.tfss_applied_weight.ndim!=4 or t.tfss_applied_weight.shape[0]!=len(items):
            raise ValueError('Missing actual Utility applied spatial weights')
        for index,(item,path,(q,r,valid)) in enumerate(zip(items,t.s4_item_paths,t.s4_maps)):
            image,annotations,support=item;h,w=image.shape[:2]
            actual_weight=t.tfss_applied_weight[index,0,:h,:w]
            support_t=torch.as_tensor(support,device=t.device)
            formula_weight=spatial_weights(q,r,support_t,mode='TFSS-U',eta=.85)
            if tuple(q.shape)!=(h,w) or tuple(r.shape)!=(h,w) or tuple(valid.shape)!=(h,w):
                raise ValueError('Actual training map is not in original image coordinates')
            weight_error=float((actual_weight-formula_weight).abs().max())
            if weight_error>1e-7:
                raise ValueError(f'Applied training Utility weight differs from locked formula: {weight_error}')
            training_spatial_observations.append(dict(
                image_id=Path(path).stem,image_path=str(Path(path).resolve()),step=int(step)+1,eot=int(eot)+1,
                canvas_hw=list(canonical_map_canvas(image)),
                Q_sha256=array_digest(q),R_sha256=array_digest(r),valid_sha256=array_digest(valid),
                applied_W_sha256=array_digest(actual_weight),formula_W_sha256=array_digest(formula_weight),
                applied_vs_formula_W_max_abs=weight_error,
                actual_weight_min=float(actual_weight.min()),actual_weight_max=float(actual_weight.max())))
    u=utility()
    status=optimize_s4(u,images,Path(records[0]['label_path']).parent,prototypes,out/'utility5',identity,
        stop_after=5,observe=observe_training_spatial)
    with (out/'utility5'/'diagnostics.csv').open(newline='',encoding='utf-8') as stream:
        gradient_rows=list(csv.DictReader(stream))
    if len(gradient_rows)!=5:
        raise ValueError('Expected five completed Utility optimizer diagnostic rows')
    gradient_audit={}
    for key in ('grad_norm_fourier','grad_delta_adaptive','s4_protection_Fourier_norm'):
        values=[float(row[key]) for row in gradient_rows if row.get(key) not in ('',None)]
        if not values or not all(math.isfinite(value) and value>0 for value in values):
            raise ValueError(f'Utility smoke has missing/zero/nonfinite effective {key}')
        gradient_audit[key]=dict(count=len(values),minimum=min(values),maximum=max(values))
    expected=u.fourier_coeff.detach().cpu().clone()
    items=[]
    for rec in records:
        im=load_image_rgb_float(rec['image_path']);ann=read_yolo_annotations(rec['label_path'])
        sup=u._build_support(im.shape,ann,image_path=rec['image_path'])[0]
        items.append((im,ann,sup))
    # Independently exercise the legacy validated builder on singleton inputs,
    # training's batched cache call, and export's singleton cache calls. All
    # three must use the same image-local canvas and produce the same maps.
    module=types.ModuleType('validated_map_reference');module.__package__='ue_framework.methods'
    source=(ROOT/'docs/S4_TFSS_UTILITY/VALIDATED_MAP_BUILDER.py').read_text()
    exec(compile(source,'validated_tfss_clean_maps.py','exec'),module.__dict__)
    export_cache=CleanMapCache(out/'export_clean_maps',setup_trainer.s4_cache.identity)
    map_difference=0.
    map_rows=[];training_export_rows=[]
    for start in (0,16):
        batch=items[start:start+16];paths=images[start:start+16]
        cached,_=u.s4_cache.get(u,batch,paths)
        for offset,(item,path,train_map) in enumerate(zip(batch,paths,cached)):
            exported,_=export_cache.get(u,[item],[path]);exported=exported[0]
            saved=rng_state()
            try:reference,_=module.build_maps(u,[item])
            finally:restore_rng(saved)
            reference=reference[0]
            support_t=torch.as_tensor(item[2],device=u.device)
            pair_differences=[]
            for left,right in ((train_map,exported),(train_map,reference),(exported,reference)):
                left_weight=spatial_weights(left[0],left[1],support_t,mode='TFSS-U',eta=.85)
                right_weight=spatial_weights(right[0],right[1],support_t,mode='TFSS-U',eta=.85)
                values=[float((left[k].to(torch.float32)-right[k].to(torch.float32)).abs().max())
                        for k in (0,1)]
                values.append(float((left_weight-right_weight).abs().max()))
                values.append(float((left[2]!=right[2]).float().max()))
                pair_differences.append(values)
            differences=[max(row[column] for row in pair_differences) for column in range(4)]
            map_difference=max(map_difference,*differences)
            actual_training=[row for row in training_spatial_observations if row['image_id']==Path(path).stem]
            if not actual_training:
                raise ValueError('Fixed TRAIN image lacks actual optimizer-step spatial-weight evidence: '+Path(path).stem)
            export_q,export_r,export_valid=exported
            export_weight=spatial_weights(export_q,export_r,support_t,mode='TFSS-U',eta=.85)
            export_hashes=dict(Q_sha256=array_digest(export_q),R_sha256=array_digest(export_r),
                valid_sha256=array_digest(export_valid),W_sha256=array_digest(export_weight))
            used_hashes_ok=all(row['Q_sha256']==export_hashes['Q_sha256']
                and row['R_sha256']==export_hashes['R_sha256']
                and row['valid_sha256']==export_hashes['valid_sha256']
                and row['applied_W_sha256']==export_hashes['W_sha256']
                for row in actual_training)
            if not used_hashes_ok:
                raise ValueError('Actual training Q/R/W hashes differ from independent export maps: '+Path(path).stem)
            training_export_rows.append(dict(image_id=Path(path).stem,
                training_observation_count=len(actual_training),
                training_steps=sorted({row['step'] for row in actual_training}),
                training_eots=sorted({row['eot'] for row in actual_training}),
                actual_training_hashes=actual_training,export_hashes=export_hashes,
                exact_hash_match=True))
            map_rows.append(dict(image_id=Path(path).stem,
                training_canvas=list(canonical_map_canvas(item[0])),
                export_canvas=list(canonical_map_canvas(item[0])),
                Q_max_abs_difference=differences[0],R_max_abs_difference=differences[1],
                Utility_weight_max_abs_difference=differences[2],
                valid_domain_mismatches=int(differences[3]!=0),
                training_export_reference_max_difference=max(differences)))
    if map_difference>1e-7:raise ValueError('Image-canonical train/export map equivalence failed')
    # Save a bounded generation pack, then exercise the actual materialization
    # recipe on fixed32; original dataset stays read-only and no victim starts.
    idx=u._active_idx
    # Match the metadata schema written by LegacyKProtoTrainer.train_universal.
    # The smoke pack uses the current U coefficients, but the production
    # generator still validates the method identity and fixed configuration.
    torch.save(dict(coords=[list(u.freq_candidate_coords[int(i)]) for i in idx.cpu().tolist()],
        fourier_coeff=u.fourier_coeff.detach()[idx].cpu(),suppress_small=u.suppress_small.detach().cpu(),
        method='legacy_kproto_ret',support_mode=u.support_mode,target_class_id=u.target_class_id,
        eps=u.eps,feature_layer=u.feature_layer,prototype_cache=u.prototype_cache,
        num_prototypes=u.num_prototypes,margin=u.margin,
        loss_weights={'lambda_bg':u.lambda_bg,'lambda_rel':u.lambda_rel,'lambda_ret':u.lambda_ret}),
        out/'smoke_global.pt')
    generator=LegacyKProtoGenerator(cfg,cfg['methods']['legacy_kproto_ret'],torch.device('cuda:0'),u.surrogate,str(out/'smoke_global.pt'))
    from PIL import Image
    exports=out/'PNG_smoke';exports.mkdir()
    png_checks=[];exported_perturbation_rows=[]
    for index,(item,path) in enumerate(zip(items,images)):
        image,ann,support=item;maps,_=u.s4_cache.get(u,[item],[path]);q,r,_=maps[0]
        weight_tensor=spatial_weights(q,r,torch.as_tensor(support,device=u.device),mode='TFSS-U',eta=.85)
        exported_weight_hash=array_digest(weight_tensor)
        training_row=next(row for row in training_export_rows if row['image_id']==Path(path).stem)
        if exported_weight_hash!=training_row['export_hashes']['W_sha256']:
            raise ValueError('PNG export did not use the audited train/export Utility weight: '+Path(path).stem)
        weight=weight_tensor.cpu().numpy()
        result=generator.generate(image,ann,0,24,u.eps,'mask',image_path=path)
        weighted_perturbation=weight[:,:,None]*result.perturbation
        candidate=np.clip(image+weighted_perturbation,0,1)
        clean=np.rint(image*255).astype(np.uint8);poison=png_project(clean,candidate,support*u.eps)
        file=exports/(str(index)+'.png');Image.fromarray(poison).save(file)
        decoded=np.asarray(Image.open(file));diff=np.abs(decoded.astype(np.int16)-clean.astype(np.int16)).max(2)
        if np.any(diff[support==0]) or np.any(diff/255>support*u.eps+1e-7):raise ValueError('Exported smoke PNG budget failed')
        png_checks.append(dict(image_id=Path(path).stem,passed=True,linf_byte=int(diff.max()),overlap_linf_byte=int(diff[support==.5].max()) if (support==.5).any() else 0))
        exported_perturbation_rows.append(dict(image_id=Path(path).stem,
            map_weight_sha256=exported_weight_hash,
            actual_training_map_weight_sha256=training_row['export_hashes']['W_sha256'],
            training_to_export_weight_exact=exported_weight_hash==training_row['export_hashes']['W_sha256'],
            base_perturbation_sha256=array_digest(result.perturbation),
            weighted_preclamp_perturbation_sha256=array_digest(weighted_perturbation),
            actual_postclamp_delta_sha256=array_digest(candidate-image),
            decoded_png_delta_sha256=array_digest(decoded.astype(np.int16)-clean.astype(np.int16))))
    del generator,u
    prefix=utility();prefix_status=optimize_s4(prefix,images,Path(records[0]['label_path']).parent,prototypes,out/'utility3',identity,stop_after=3)
    del prefix
    resumed=utility();resume_status=optimize_s4(resumed,images,Path(records[0]['label_path']).parent,prototypes,out/'utility_resume5',identity,stop_after=5,resume=out/'utility3/latest_continuation.pt')
    replay=float((resumed.fourier_coeff.detach().cpu()-expected).abs().max())
    uninterrupted=torch.load(status['checkpoint'],map_location='cpu',weights_only=False)
    restored=torch.load(resume_status['checkpoint'],map_location='cpu',weights_only=False)
    if replay!=0 or (out/'utility5/diagnostics.csv').read_bytes()!=(out/'utility_resume5/diagnostics.csv').read_bytes():
        raise ValueError('Production3to5 checkpoint replay mismatch')
    for k in ('exp_avg','exp_avg_sq','step'):
        a=next(iter(uninterrupted['optimizer']['state'].values()))[k]
        b=next(iter(restored['optimizer']['state'].values()))[k]
        if not torch.equal(a,b):raise ValueError('Adam replay mismatch')
    if uninterrupted['scheduler']!=restored['scheduler']:raise ValueError('Scheduler replay mismatch')
    checkpoints=[str(p.relative_to(out)) for p in out.rglob('*.pt')]
    return dict(schema='prerun.pre-review-smoke.v1',production_entrypoint_reached=True,finite_loss=True,
        source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
        fixed_input='original TRAIN32/65PersonGT/46NTGT, two native batches, seed0; explicit bounded enumeration',
        optimizer_updates=20,optimizer_update_breakdown='canonical5+noop5+U5+prefix3+resume2',
        S0_noop_max_difference=maximum,parameter_update_difference=coefficient_difference,
        train_export_Q_R_W_max_difference=map_difference,train_export_map_rows=map_rows,
        actual_training_to_export_map_weight_correspondence=dict(
            passed=(len(training_export_rows)==len(records)
                and all(row['exact_hash_match'] for row in training_export_rows)
                and len(exported_perturbation_rows)==len(records)
                and all(row['training_to_export_weight_exact'] for row in exported_perturbation_rows)),
            fixed_train_images=len(records),observed_training_map_weight_rows=len(training_spatial_observations),
            formula_weight_max_abs_difference=max(row['applied_vs_formula_W_max_abs'] for row in training_spatial_observations),
            per_image=training_export_rows),
        exported_weighted_perturbations=exported_perturbation_rows,
        effective_gradient_audit=gradient_audit,
        resume3to5_max_difference=replay,resume_CSV_byte_identical=True,
        Adam_scheduler_replay_exact=True,PNG_checks=png_checks,frozen_adaptive_unchanged=all(integrity),
        baseline_equivalence_required=True,baseline_equivalence_probe=dict(reference_id='actual unchanged B2SemanticTrainer.train_universal, exact45 S0',
        reference_weights_path=cfg['surrogate']['ckpt'],candidate_weights_path=cfg['surrogate']['ckpt'],metric_name='max loss/Fourier gradient/delta/support/update difference',
        reference_metric=0.,candidate_metric=maximum,absolute_difference=maximum,tolerance=1e-7,
        evaluation_setting=dict(dataset='fixed TRAIN32',parameters=dict(EOT=2,batch=16),seed=0,sample_count=32,post_processing='same original image clamp'),
        evidence_paths=['reference/diagnostics.csv','noop/diagnostics.csv']),
        official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all',checkpoint_cleanup_completed=False,
        deleted_paths=[],checkpoint_paths_remaining=checkpoints,finite_outputs=True,
        method_change='image-local canonical map canvas for train/export consistency; TFSS formula, eta, losses, and optimizer unchanged')
