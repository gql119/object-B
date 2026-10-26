"""Train-only incidence/conditional-weight audit of the isolated risk extension.

No optimizer, poison dataset, victim, validation images or official metrics.
The population pass requires its separate validation-review closure. Undefined
original risks remain null; conditional W statistics are not global W statistics.
"""
import argparse,copy,json,math
from pathlib import Path


def run(release,output,device_name,limit=8):
    import numpy as np,torch
    from .runtime import ROOT,DATA_ROOT,load,sha,verify_release,atomic,write_once
    from .phase_fourier import accepted_calibration_dependency,copy_calibration_consumer_artifacts,runtime_config
    from .phase_calibration import train_manifest,load_teacher
    from .canonical_s4_training import configure,cpu_rng_isolation
    from .risk_fallback import gradients,POLICY
    from ue_framework.methods.s4_tfss import S4Trainer,model_hash
    from ue_framework.methods.tfss_spatial import spatial_weights
    from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations
    torch.set_num_threads(1);m=verify_release(release)
    if m.get('risk_estimator_policy')!=POLICY or m.get('validation_only') is not True:
        raise ValueError('Explicit validation-only risk protocol required')
    if limit not in (0,8):raise ValueError('Only fixed eight-image probe or complete25K audit allowed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    common=dict(source_commit=m['source_commit'],release_sha256=sha(release),risk_estimator_policy=POLICY,
        validation_only=True,official_metrics_disabled=True,artifact_ingest_disabled=True,
        parameter_updates=0,formal_READY_written=False,population=limit==0,device=device_name)
    atomic(output/'status.json',dict(common,state='RUNNING'))
    restore=None;handle=None
    try:
        if limit==0:
            dependency=m.get('risk_population_validation_closure',{})
            path=Path(dependency['path'])
            if sha(path)!=dependency['sha256']:raise ValueError('Population review closure drift')
            review=load(path)
            expected=dict(risk_estimator_policy=POLICY,source_commit=m['source_commit'],reviewer_count=1,
                scientific_blockers_closed=True,validation_population_allowed=True,formal_optimization_allowed=False)
            if any(review.get(k)!=v for k,v in expected.items()):raise ValueError('Population audit lacks its scientific closure')
        _,receipt,_=accepted_calibration_dependency(m,validation=True)
        imported,_=copy_calibration_consumer_artifacts(output,dict(manifest=m,receipt=receipt,
            phase_spec=load(m['dimension_reconstruction_dependency']['phase_spec_path'])))
        cfg,checkpoint=runtime_config(load(ROOT/m['phase_config_path']),m,imported,smoke=True,validation_batch_size=1)
        paths,hashes,digest=train_manifest()
        if digest!=receipt['accepted_clean_train_manifest_sha256']:raise ValueError('Train identity changed')
        if limit:
            by_id={p.stem:p for p in paths}
            ids=['000000247692']+[p.stem for p in paths if p.stem!='000000247692'][:7]
            paths=[by_id[i] for i in ids]
        device=torch.device(device_name)
        if device.type=='cpu' and torch.cuda.is_initialized():raise ValueError('CPU audit initialized CUDA')
        model,before=load_teacher(checkpoint,m['teacher_dependency']['checkpoint_sha256'],device)
        trainer=S4Trainer(cfg,cfg['methods']['legacy_kproto_ret'],device,model,copy.deepcopy(model).eval())
        if trainer.channels.numel()!=receipt['selected_channel_count'] or len(trainer.safe_bases)!=79:
            raise ValueError('Actual semantic coordinate/class schema drift')
        restore=configure(trainer,m,output,ROOT)
        counts=dict(images=0,GT=0,NT_GT=0,Person_images=0,eligible_Person_images=0,
            affected_images=0,fallback_GT=0,affected_Person_images=0,affected_eligible_Person_images=0,
            gradient_validated_fallback_GT_in_eligible_images=0)
        per_class={str(c):dict(NT_GT=0,fallback_GT=0,affected_images=0) for c in range(1,80)}
        histogram=np.zeros(150,dtype=np.int64);weight_count=0;weight_sum=0.;weight_square_sum=0.
        weight_min=1.;weight_max=.85;forward_count=0;raw_evidence=[]
        def raw_hook(module,args,result):
            nonlocal forward_count
            from .coco_eval import raw_tensor_device_details,raw_signature
            forward_count+=1
            if forward_count>(100 if limit else 100000):raise ValueError('Audit forward budget exceeded')
            devices,indices=raw_tensor_device_details(result)
            if devices!=[str(device)] or (device.type=='cuda' and indices!=[0]):raise ValueError('Actual raw device mismatch')
            def check(value):
                if torch.is_tensor(value):
                    if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite actual raw output')
                elif isinstance(value,dict):
                    for v in value.values():check(v)
                elif isinstance(value,(tuple,list)):
                    for v in value:check(v)
            check(result)
            if len(raw_evidence)<4:raw_evidence.append(dict(devices=devices,cuda_indices=indices,signature=raw_signature(result)))
        handle=trainer.surrogate.register_forward_hook(raw_hook)
        with (output/'per_image_risk_incidence.jsonl').open('x',encoding='utf-8') as stream:
            for index,path in enumerate(paths):
                label=DATA_ROOT/'labels/train'/(path.stem+'.txt')
                if (sha(path),sha(label))!=tuple(hashes[path.stem]):raise ValueError('Consumed train bytes changed: '+path.stem)
                image=load_image_rgb_float(str(path));anns=read_yolo_annotations(str(label))
                _,rows=gradients(trainer.surrogate,image,anns,imgsz=trainer.imgsz,coverage_only=True,
                    coverage_policy='independent_same_TAL_for_uncovered_valid_GT',target_class_id=0,num_classes=80)
                expected={j for j,a in enumerate(anns) if int(a['cls'])!=0}
                if len(rows)!=len(expected) or {r['gt_index'] for r in rows}!=expected:
                    raise ValueError('NT GT missing or duplicated')
                fallback=[r for r in rows if r.get('risk_estimator_policy')==POLICY]
                support,_,_=trainer._build_support(image.shape,anns,image_path=str(path))
                has_person=any(int(a['cls'])==0 for a in anns);eligible=float(support.sum())>10
                affected=bool(fallback);active=affected and eligible
                counts['images']+=1;counts['GT']+=len(anns);counts['NT_GT']+=len(expected)
                counts['Person_images']+=int(has_person);counts['eligible_Person_images']+=int(eligible)
                counts['affected_images']+=int(affected);counts['fallback_GT']+=len(fallback)
                counts['affected_Person_images']+=int(affected and has_person)
                counts['affected_eligible_Person_images']+=int(active)
                for row in rows:per_class[str(row['cls'])]['NT_GT']+=1
                for row in fallback:per_class[str(row['cls'])]['fallback_GT']+=1
                for cls in {r['cls'] for r in fallback}:per_class[str(cls)]['affected_images']+=1
                weight_record=None
                if active:
                    with cpu_rng_isolation(device):maps,evidence=trainer.s4_cache.get(trainer,[(image,anns,support)],[str(path)])
                    q,risk,_=maps[0];weight=spatial_weights(q,risk,torch.as_tensor(support,device=device),mode='TFSS-U',eta=.85)
                    matched=evidence[0]['nt_matching']
                    if len(matched)!=len(expected) or {r['gt_index'] for r in matched}!=expected:
                        raise ValueError('Actual gradient path lost NT GT')
                    actual_fallback={r['gt_index'] for r in matched if r.get('risk_estimator_policy')==POLICY}
                    if actual_fallback!={r['gt_index'] for r in fallback}:raise ValueError('Coverage/gradient trigger mismatch')
                    counts['gradient_validated_fallback_GT_in_eligible_images']+=len(actual_fallback)
                    values=weight[torch.as_tensor(support>0,device=device)].detach().double().cpu().numpy()
                    if not np.isfinite(values).all() or values.min()<.85 or values.max()>1:raise ValueError('Invalid W range')
                    histogram+=np.histogram(values,bins=150,range=(.85,1.))[0]
                    weight_count+=len(values);weight_sum+=float(values.sum());weight_square_sum+=float(np.square(values).sum())
                    weight_min=min(weight_min,float(values.min()));weight_max=max(weight_max,float(values.max()))
                    weight_record=dict(pixel_count=len(values),min=float(values.min()),max=float(values.max()),mean=float(values.mean()),
                        quantiles=np.quantile(values,[0,.1,.5,.9,1]).tolist(),original_Rnt=None,original_W=None,
                        original_undefined_reason='at least one legitimate NT GT has no native effective supervision')
                stream.write(json.dumps(dict(image_id=path.stem,GT=len(anns),NT_GT=len(expected),eligible_Person=eligible,
                    fallback_records=fallback,conditional_W=weight_record),allow_nan=False)+'\n')
                if (index+1)%100==0 or index+1==len(paths):
                    stream.flush();atomic(output/'progress.json',dict(common,state='RUNNING',counts=counts,actual_model_forwards=forward_count))
        if counts['images']!=len(paths) or sum(v['NT_GT'] for v in per_class.values())!=counts['NT_GT']:
            raise ValueError('Population accounting mismatch')
        if not limit and (counts['images']!=25000 or counts['GT']!=181475 or counts['eligible_Person_images']!=13630):
            raise ValueError('Complete original25K accounting mismatch')
        if model_hash(trainer.surrogate)!=before or sha(checkpoint)!=m['teacher_dependency']['checkpoint_sha256']:
            raise ValueError('Frozen Teacher changed')
        if any(p.grad is not None for p in trainer.surrogate.parameters()):raise ValueError('Teacher accumulated gradients')
        if device.type=='cpu' and torch.cuda.is_initialized():raise ValueError('CPU audit initialized CUDA')
        summary=dict(common,state='passed',counts=counts,per_NT_class=per_class,
            affected_train_image_fraction=counts['affected_images']/counts['images'],
            affected_NT_GT_fraction=counts['fallback_GT']/counts['NT_GT'] if counts['NT_GT'] else None,
            affected_TFSS_active_image_fraction=counts['affected_eligible_Person_images']/counts['eligible_Person_images'] if counts['eligible_Person_images'] else None,
            conditional_affected_eligible_W=dict(pixel_count=weight_count,min=weight_min if weight_count else None,
                max=weight_max if weight_count else None,mean=weight_sum/weight_count if weight_count else None,
                variance=max(0.,weight_square_sum/weight_count-(weight_sum/weight_count)**2) if weight_count else None,
                histogram=histogram.tolist(),histogram_edges=np.linspace(.85,1.,151).tolist()),
            original_complete_Rnt_on_affected_images=None,original_W_on_affected_images=None,
            full_training_W_distribution=None,distribution_scope='affected eligible Person images only',
            unaffected_comparison='original estimator delegation; bounded exact equivalence evidence, not a new whole-dataset comparison',
            actual_model_forwards=forward_count,actual_raw_evidence=raw_evidence,Teacher_unchanged=True,
            input_manifest_sha256=digest,per_image_report_sha256=sha(output/'per_image_risk_incidence.jsonl'),
            no_optimizer_instantiated=True,no_validation_images_opened=True,
            population_state='COMPLETE_25K' if not limit else 'BOUNDED_FIXTURE_ONLY')
        write_once(output/'risk_population_summary.json',summary);atomic(output/'status.json',summary)
        return summary
    except BaseException as exc:
        atomic(output/'status.json',dict(common,state='FAILED',error=repr(exc)));raise
    finally:
        if handle is not None:handle.remove()
        if restore is not None:restore()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda:0'),required=True);p.add_argument('--limit',type=int,default=8)
    a=p.parse_args();run(a.release,a.output,a.device,a.limit)
