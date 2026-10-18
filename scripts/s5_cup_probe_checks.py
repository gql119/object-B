"""Production export/resume/native AMP checks for the bounded S5 smoke."""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def export_fixed(t,cfg,records,out,batch):
    import numpy as np
    import torch
    from PIL import Image
    from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations
    from ue_framework.methods.legacy_kproto_ret import LegacyKProtoGenerator,_rectangle
    from ue_framework.methods.s4_pixel_audit import png_project
    from ue_framework.methods.tfss_spatial import spatial_weights
    root=out/'PNG_smoke_dataset'
    for split in ('train','val'):
        for kind in ('images','labels'):(root/kind/split).mkdir(parents=True,exist_ok=True)
    params=out/f'batch{batch}_smoke_global.pt';idx=t._active_idx
    torch.save(dict(coords=[list(t.freq_candidate_coords[int(i)]) for i in idx.cpu().tolist()],
        fourier_coeff=t.fourier_coeff.detach()[idx].cpu(),suppress_small=t.suppress_small.detach().cpu(),
        method='legacy_kproto_ret',support_mode=t.support_mode,target_class_id=t.target_class_id,
        eps=t.eps,feature_layer=t.feature_layer,prototype_cache=t.prototype_cache,
        num_prototypes=t.num_prototypes,margin=t.margin,
        loss_weights={'lambda_bg':t.lambda_bg,'lambda_rel':t.lambda_rel,'lambda_ret':t.lambda_ret}),params)
    generator=LegacyKProtoGenerator(cfg,cfg['methods']['legacy_kproto_ret'],t.device,t.surrogate,str(params))
    rows=[]
    for record in records:
        path=record['image_path'];image=load_image_rgb_float(path);ann=read_yolo_annotations(record['label_path'])
        support=t._build_support(image.shape,ann,image_path=path)[0]
        maps,_=t.s4_cache.get(t,[(image,ann,support)],[path]);q,r,_=maps[0]
        weight=spatial_weights(q,r,torch.as_tensor(support,device=t.device),mode='TFSS-U',eta=.85)
        weight_hash=hashlib.sha256(weight.detach().cpu().numpy().tobytes()).hexdigest()
        if weight_hash!=t.cup_training_weight_hashes[Path(path).stem]:
            raise ValueError('Actual training and PNG export spatial-weight hashes differ')
        generated=generator.generate(image,ann,0,24,t.eps,'mask',image_path=path)
        candidate=np.clip(image+generated.perturbation*weight.cpu().numpy()[:,:,None],0,1)
        clean=np.rint(image*255).astype(np.uint8)
        destination=root/'images/train'/(Path(path).stem+'.png')
        if destination.exists():raise FileExistsError(destination)
        Image.fromarray(png_project(clean,candidate,support*t.eps)).save(destination)
        with Image.open(destination) as handle:decoded=np.asarray(handle)
        difference=np.abs(decoded.astype(np.int16)-clean.astype(np.int16)).max(2)
        if np.any(difference[support==0]) or np.any(difference/255>support*t.eps+1e-7):
            raise ValueError('Decoded S5 smoke PNG outside/budget violation')
        changed=[]
        for obj in ann:
            if int(obj['cls'])==14:
                mask=_rectangle(image.shape[:2],obj['bbox'])[0,0].numpy().astype(bool)
                changed.append(bool(np.any(difference[mask]>0)))
        if not all(changed):raise ValueError('Unchanged Person GT in S5 smoke PNG')
        label=root/'labels/train'/Path(record['label_path']).name
        if label.exists():raise FileExistsError(label)
        shutil.copyfile(record['label_path'],label)
        if hashlib.sha256(label.read_bytes()).hexdigest()!=record['label_sha256']:raise ValueError('Smoke label bytes changed')
        # Validation is disabled in the AMP smoke; keep the declared view clean.
        (root/'images/val'/Path(path).name).symlink_to(Path(path))
        (root/'labels/val'/label.name).symlink_to(Path(record['label_path']))
        rows.append(dict(image_id=Path(path).stem,batch=batch,passed=True,Person_GT_changed=sum(changed),
            global_Linf_byte=int(difference.max()),
            overlap_Linf_byte=int(difference[support==.5].max()) if (support==.5).any() else 0,
            outside_nonzero_pixels=int(np.count_nonzero(difference[support==0])),
            labels_byte_identical=True,training_export_W_hash_exact=True,W_sha256=weight_hash))
    del generator;gc.collect();torch.cuda.empty_cache()
    return rows


def resume_check(create,images,records,prototypes,out,batch,identity,expected,compare_tree):
    import torch
    from ue_framework.methods.s5_optimization import optimize_s5
    first=create('P3_CUP');prefix=out/f'batch{batch}_CUP_split3'
    optimize_s5(first,images,Path(records[0]['label_path']).parent,prototypes,prefix,identity,stop_after=3)
    first.s5_close_observer();del first;gc.collect();torch.cuda.empty_cache()
    resumed=create('P3_CUP');destination=out/f'batch{batch}_CUP_resume5'
    result=optimize_s5(resumed,images,Path(records[0]['label_path']).parent,prototypes,destination,identity,
                       stop_after=5,resume=prefix/'latest_continuation.pt')
    with Path(expected).open('rb') as stream:a=torch.load(stream,map_location='cpu',weights_only=False)
    with Path(result['checkpoint']).open('rb') as stream:b=torch.load(stream,map_location='cpu',weights_only=False)
    errors={key:compare_tree(a[key],b[key]) for key in ('trainer','optimizer','scheduler','rng','cup','cursor_state')}
    if any(v!=0 for v in errors.values()):raise ValueError('Exact S5 3-to-5 replay failed '+str(errors))
    for name in ('diagnostics.csv','cup_classwise_weights.jsonl'):
        if (Path(expected).parents[1]/name).read_bytes()!=(destination/name).read_bytes():
            raise ValueError('S5 replay journal differs '+name)
    resumed.s5_close_observer();del resumed;gc.collect();torch.cuda.empty_cache()
    return dict(batch=batch,passed=True,additional_optimizer_updates=5,errors=errors,
        cup_history_steps=b['cup']['history']['steps'],journals_exact=True,
        prefix_checkpoint=str(prefix/'latest_continuation.pt'),resume_checkpoint=result['checkpoint'])


def run_amp_checks(out,root,protocol):
    from scripts.s4_victim_recovery import compare_smoke
    from scripts.run_s4_tfss_pipeline import _resource_snapshot,_resource_blockers
    resource=_resource_snapshot(5.)
    blockers=_resource_blockers(resource,allow_own_gpu=True)
    if blockers:raise RuntimeError('AMP smoke resource conflict '+repr(blockers))
    parent=json.loads((root/'docs/S4_TFSS_UTILITY/PARENT_IDENTITY.json').read_text())
    args=dict(parent['victim_spec']['train_args'])
    # These controls disable non-training work only within bounded smoke.
    args.update(val=False,save=False,plots=False)
    config=dict(remote_root=str(out.parents[1]),poisoned_dataset_path=str(out/'PNG_smoke_dataset'),
        victim_protocol=dict(train_args=args,init_yaml_sha256=parent['victim_spec']['init_yaml_sha256']))
    path=out/'AMP_SMOKE_RUNTIME_CONFIG.json';path.write_text(json.dumps(config,indent=2)+'\n')
    trialroot=out/'AMP_trials';trialroot.mkdir(exist_ok=False)
    for name in ('native_S0','repaired_S0','real_S4','synthetic_S4'):
        subprocess.run([sys.executable,'-B',str(root/'scripts/s5_cup_probe.py'),'--output',str(trialroot/name),
            '--amp-trial',name,'--amp-config',str(path)],check=True,timeout=900,cwd=root)
    result=compare_smoke(trialroot)
    if result['total_actual_batches']>protocol['native_AMP_max_batches']:raise ValueError('AMP smoke batch cap exceeded')
    result.update(real_S4_legacy_trial_name_means='actual S5 TRAIN32 paired PNG development dataset',
        formal_S5_dataset=False,noise_states='two independent fixed16-image five-step trajectories',
        validation_enabled=False,smoke_only_val_save_plots_disabled=True,
        formal_victim_hyperparameters_unchanged=True,pre_AMP_resources=resource)
    (out/'native_AMP_smoke_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
