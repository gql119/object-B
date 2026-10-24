"""COCO clean-train coordinate producer using the original S4 mathematics.

New component, not yet a released or validated formal calibration entrypoint.
No validation data, optimization, layer search or threshold adjustment occurs.
"""
import copy
import hashlib
from pathlib import Path
import shutil

import torch

from .runtime import atomic, now, sha, write_once
from ue_framework.data_utils import load_image_rgb_float, label_path_for_image, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import _rectangle, _feature_mask
from ue_framework.methods.semantic_coordinates import gt_evidence, select_channels, class_subspace
from ue_framework.methods.safebg import select_strict_tsfl
from ue_framework.methods.s4_tfss import model_hash


class ProtocolSpaceBlocked(ValueError):
    pass


def verify_expected_hashes(image_path, label_path, image_sha, label_sha, expected_hashes):
    if expected_hashes is None:
        return
    expected = expected_hashes.get(Path(image_path).stem)
    if expected is None or expected != (image_sha, label_sha):
        raise ValueError('Clean COCO train image/label differs from accepted manifest: '+Path(image_path).stem)


def collect(trainer, model, images, label_dir, output, *, expected_hashes=None):
    """Same per-GT gradients/pooling as the canonical P3/P5 builders.

    Both retained feature maps come from the same frozen forward; gradients are
    computed separately for each feature, retaining canonical summation order.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    images=sorted(map(str,images))
    if trainer.target_class_id!=0 or trainer.num_classes!=80 or trainer.feature_layer!='model.21':
        raise ValueError('COCO80 Person0 fixed P5 collector required')
    classes=list(range(1,80));layers=('model.15','model.21')
    stats={layer:dict(target_sum=None,nt_sum=None,nt_sums={c:None for c in classes},
                     nt_counts={c:0 for c in classes},pools={c:[] for c in classes}) for layer in layers}
    target_count=0;manifest=hashlib.sha256();eligible=[]
    frozen_before=model_hash(model)
    for index,path in enumerate(images):
        image=load_image_rgb_float(path)
        label=label_path_for_image(path,str(label_dir));anns=read_yolo_annotations(label)
        image_sha,label_sha=sha(path),sha(label)
        verify_expected_hashes(path,label,image_sha,label_sha,expected_hashes)
        manifest.update((Path(path).stem+':'+image_sha+':'+label_sha+'\n').encode())
        support,_,_=trainer._build_support(image.shape,anns,image_path=path)
        if support.sum()>10:
            eligible.append(path)
        if anns:
            tensor=trainer._to_tensor(image).requires_grad_(True)
            trainer.multi_features.clear();decoded=trainer._forward_raw(tensor)
            features={layer:trainer.multi_features[layer] for layer in layers}
            target_count+=sum(int(a['cls'])==0 for a in anns)
            for layer,feature in features.items():
                for ann in anns:
                    cls=int(ann['cls'])
                    if cls not in classes:continue
                    roi=_feature_mask(_rectangle(image.shape[:2],ann['bbox']),feature,image.shape[:2],trainer.imgsz)
                    pooled=trainer._pooled(feature.detach(),[roi])
                    if pooled:stats[layer]['pools'][cls].append(pooled[0].cpu())
            for cls,evidence in gt_evidence(decoded,anns,image.shape[:2],trainer.imgsz):
                for layer,feature in features.items():
                    gradient=torch.autograd.grad(evidence,feature,retain_graph=True)[0]
                    value=gradient.detach().abs().mean(dim=(0,2,3)).double().cpu()
                    if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite clean GT sensitivity')
                    item=stats[layer]
                    if cls==0:
                        item['target_sum']=value.clone() if item['target_sum'] is None else item['target_sum']+value
                    else:
                        item['nt_sum']=value.clone() if item['nt_sum'] is None else item['nt_sum']+value
                        item['nt_sums'][cls]=value.clone() if item['nt_sums'][cls] is None else item['nt_sums'][cls]+value
                        item['nt_counts'][cls]+=1
            del tensor,decoded,features,feature,gradient
        if (index+1)%100==0:
            atomic(output/'progress.json',dict(state='COLLECTING',images_done=index+1,images_total=len(images),UTC=now()))
    if model_hash(model)!=frozen_before:raise ValueError('Frozen COCO Teacher changed during clean calibration')
    result=dict(layers=stats,target_count=target_count,images=len(images),
                clean_train_manifest_sha256=manifest.hexdigest(),frozen_state_sha256=frozen_before,
                eligible_person_images=eligible,validation_used=False)
    # Keep raw arrays and pools even when downstream strict-space checks block.
    with (output/'raw_statistics.pt').open('xb') as f:torch.save(result,f)
    write_once(output/'eligible_support_manifest.json',dict(paths=eligible,count=len(eligible),
               predicate='original bbox_half_overlap support.sum()>10, Person0',
               optimizer_steps=24*((len(eligible)+15)//16),epochs=24,batch=16))
    return result


def build_coordinates(raw,cfg,output,*,builder_path,source_root):
    """Write native coordinate schemas; actual incompatible24D space blocks."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    reference=output/'reference';reference.mkdir()
    strict_dir=output/'coordinates';strict_dir.mkdir()
    method=cfg['methods']['legacy_kproto_ret'];s=method['semantic'];strict=s['strict_tsfl']
    if (Path(s['artifact_dir']).resolve()!=strict_dir.resolve()
            or Path(s['safe_bg']['reference_artifact_dir']).resolve()!=reference.resolve()):
        raise ValueError('Bind the actual new COCO coordinate/reference routes before calibration')
    if (strict['target_candidate_fraction'],strict['nt_safety_quantile'],strict['sensitivity_worst_k'],
        strict['target_channel_fraction'],strict['min_selected_channels'])!=(.4,.6,3,.15,4):
        raise ValueError('Frozen StrictP5 protocol changed')
    if (s['target_channel_fraction'],s['lambda_specificity'],s['subspace_variance'],s['subspace_min_rank'],s['subspace_max_rank_fraction'])!=(.75,1.,.9,4,.5):
        raise ValueError('Frozen P3 coordinate protocol changed')
    if raw['validation_used'] or raw['target_count']<=0 or not raw['eligible_person_images']:
        raise ValueError('Invalid train-only statistics')
    classes=list(range(1,80))
    for layer in ('model.15','model.21'):
        item=raw['layers'][layer]
        if any(item['nt_counts'][c]<=0 or item['nt_sums'][c] is None or len(item['pools'][c])<2 for c in classes):
            raise ProtocolSpaceBlocked('Actual COCO class statistics/subspace pools insufficient')
    base=dict(seed=0,checkpoint_sha256=sha(cfg['surrogate']['ckpt']),frozen_state_sha256=raw['frozen_state_sha256'],
              clean_train_manifest_sha256=raw['clean_train_manifest_sha256'],images=raw['images'],
              data_root=cfg['data']['dataset_root'],train_images=cfg['data']['train_images'],train_labels=cfg['data']['train_labels'],
              validation_used=False,builder_sha256=sha(builder_path),
              coordinate_code_sha256=sha(Path(source_root)/'ue_framework/methods/semantic_coordinates.py'),
              settings={k:s[k] for k in ('target_channel_fraction','lambda_specificity','subspace_variance','subspace_min_rank','subspace_max_rank_fraction')})
    p3=raw['layers']['model.15'];p5=raw['layers']['model.21']
    p3t=p3['target_sum']/raw['target_count']
    p3n=p3['nt_sum']/sum(p3['nt_counts'].values())
    indices,st,sn,score=select_channels(p3t,p3n,s['target_channel_fraction'],s['lambda_specificity'])
    identity3=dict(base,feature_layer='model.15',channels=len(p3t))
    channels3=dict(indices=indices,S_target=p3t,S_nt=p3n,S_target_normalized=st,S_nt_normalized=sn,specificity=score,identity=identity3)
    p3spaces={c:class_subspace(p3['pools'][c],s['subspace_variance'],s['subspace_min_rank'],s['subspace_max_rank_fraction']) for c in classes}
    torch.save(channels3,reference/'target_specific_channels.pt')
    torch.save(dict(classes=p3spaces,identity=identity3),reference/'non_target_subspaces.pt')
    write_once(reference/'coordinates_identity.json',identity3)
    p5t=p5['target_sum']/raw['target_count']
    p5by={c:p5['nt_sums'][c]/p5['nt_counts'][c] for c in classes}
    selected=select_strict_tsfl(p5t,p5by,candidate_fraction=.4,safety_quantile=.6,worst_k=3,
                              selected_fraction=.15,min_selected=4,eps=s['orth_eps'])
    idx=selected['indices'].cpu()
    if idx.numel()!=24:
        write_once(output/'BLOCKED_PROTOCOL_COMPATIBILITY.json',dict(state='BLOCKED_PROTOCOL_COMPATIBILITY',
                   actual_selected_count=idx.numel(),required_original_space=24,actual_summary=selected['summary'],
                   thresholds_unchanged=True,raw_statistics_retained=True))
        raise ProtocolSpaceBlocked('Actual StrictP5 selected space cannot support original24D SafeBG; do not relax gates')
    identity5=dict(base,feature_layer='model.21',channels=len(p5t),
                   safe_bg_code_sha256=sha(Path(source_root)/'ue_framework/methods/safebg.py'),
                   reference_target_channels_sha256=sha(reference/'target_specific_channels.pt'),
                   reference_non_target_subspaces_sha256=sha(reference/'non_target_subspaces.pt'),
                   strict_tsfl_settings=copy.deepcopy(strict),safe_bg_settings=copy.deepcopy(s['safe_bg']))
    mean_nt=torch.stack(list(p5by.values())).mean(0)
    channels5=dict(indices=idx,S_target=p5t,S_nt=mean_nt,S_target_normalized=selected['S_target_normalized'],
                   S_nt_normalized=selected['S_nt_worst_normalized'],S_nt_by_class=p5by,
                   S_nt_class_normalized=selected['S_nt_class_normalized'],S_nt_worst_normalized=selected['S_nt_worst_normalized'],
                   specificity=selected['ratio'],identity=identity5)
    torch.save(channels5,strict_dir/'target_specific_channels.pt')
    torch.save(dict(indices=idx,summary=selected['summary'],identity=identity5),strict_dir/'strict_tsfl_channels.pt')
    full={c:class_subspace(p5['pools'][c],s['subspace_variance'],s['subspace_min_rank'],s['subspace_max_rank_fraction']) for c in classes}
    safe={c:class_subspace([v.index_select(0,idx) for v in p5['pools'][c]],s['subspace_variance'],s['subspace_min_rank'],s['subspace_max_rank_fraction']) for c in classes}
    torch.save(dict(classes=full,identity=identity5),strict_dir/'non_target_subspaces.pt')
    torch.save(dict(classes=safe,identity=identity5),strict_dir/'safe_bg_selected_subspaces.pt')
    shutil.copyfile(reference/'target_specific_channels.pt',strict_dir/'orth_target_specific_channels.pt')
    shutil.copyfile(reference/'non_target_subspaces.pt',strict_dir/'orth_non_target_subspaces.pt')
    summary=dict(selected['summary'],images=raw['images'],target_objects=raw['target_count'],
                 non_target_objects_by_class=p5['nt_counts'],identity=identity5,
                 calibration_split='all clean training only; validation never opened')
    write_once(strict_dir/'strict_tsfl_summary.json',summary)
    chosen=torch.zeros(len(p5t),dtype=torch.bool);chosen[idx]=True
    write_once(strict_dir/'tsfl_summary.json',dict(num_channels_total=len(p5t),num_channels_selected=24,indices=idx.tolist(),
               target_channel_fraction=24/len(p5t),images=raw['images'],target_objects=raw['target_count'],
               non_target_objects=sum(p5['nt_counts'].values()),
               mean_S_target_selected=float(selected['S_target_normalized'][chosen].mean()),
               mean_S_nt_selected=float(mean_nt[chosen].mean()),
               mean_S_target_unselected=float(selected['S_target_normalized'][~chosen].mean()),
               mean_S_nt_unselected=float(mean_nt[~chosen].mean()),specificity_selected_mean=float(selected['ratio'][chosen].mean()),
               channel_selection_overlap_with_previous_run=len(set(idx.tolist())&set(indices.tolist()))/len(idx),
               legacy_overlap_field_definition='index-ID intersection with same-Teacher P3 reference; not cross-layer stability or an independent recalibration'))
    write_once(strict_dir/'coordinates_identity.json',identity5)
    selection_path=Path(strict['layer_selection_path'])
    write_once(selection_path,dict(state='selected',selection_policy='fixed_original_P5; no COCO layer search or AP tuning',
               selected_layer='P5',selected_feature_layer='model.21',selected_channel_indices=idx.tolist(),
               clean_train_manifest_sha256=raw['clean_train_manifest_sha256'],gate=dict(target_top_fraction=.4,worst_nt_quantile=.6)))
    torch.save(dict(selected_layer='P5',selected_channel_indices=idx,layer_selection_sha256=sha(selection_path)),strict_dir/'strict_tsfl_selected_layer.pt')
    write_once(strict_dir/'strict_tsfl_selected_layer_summary.json',dict(selected_layer='P5',feature_layer='model.21',
               total_channels=len(p5t),candidate_count=summary['safe_candidates'],selected_count=24,
               selected_fraction=24/len(p5t),LS=None,pearson_rho=None,spearman_rho=None,
               diagnostic_policy='fixed P5; layer-search diagnostics not measured',layer_selection_sha256=sha(selection_path)))
    write_once(strict_dir/'tsfl_sensitivity_stats.json',dict(target_mean_sensitivity=p5t.tolist(),
               classes={str(c):dict(object_count=p5['nt_counts'][c],mean_sensitivity=p5by[c].tolist()) for c in classes},identity=identity5))
    write_once(strict_dir/'subspace_summary.json',{cfg['experiment']['class_names'][c]:
               {k:v for k,v in item.items() if k not in ('basis','mean')} for c,item in full.items()})
    return dict(reference=str(reference),coordinates=str(strict_dir),selected_channels=idx.tolist(),
                eligible_count=len(raw['eligible_person_images']),validation_used=False)
