"""Independent full-target support ablation. No original S4 entrypoint is modified.

GPU smoke and formal execution require predecessor acceptance AND fresh queue audit.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_s4_tfss_pipeline import sha,dump,summarize,_resource_snapshot
from ue_framework.s4_full_contract import load_full_config,verify_queue
from ue_framework.s4_robustness_resources import capacity_blockers,sustained_iowait

def setup(lock,out):
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.data_utils import list_images,read_yolo_annotations
    from ue_framework.methods.s4_tfss import CleanMapCache
    from ue_framework.methods.s4_tfss_full import S4FullTrainer
    from ue_framework.methods.semantic_coordinates import semantic_identity
    from scripts.audit_safe_bg_ablation_fingerprint_20261004 import split_fingerprint
    cfg=load_full_config(ROOT,lock);mc=cfg['methods']['legacy_kproto_ret']
    meta=json.loads((ROOT/'docs/tfss/SOURCE_METADATA.json').read_text())
    provenance=json.loads((ROOT/'docs/S4_TFSS_UTILITY/PARENT_IDENTITY.json').read_text())
    if lock['tfss']['eta_U']!=.85 or lock['seed']!=0 or lock['target_class_id']!=14:
        raise ValueError('Frozen Utility/seed/target configuration changed')
    baseline=json.loads((ROOT/lock['baseline']['comparison_path']).read_text())
    if lock['victim_protocol']['train_args']!=baseline['training']['args'] or sha(ROOT/cfg['victim']['init'])!=provenance['victim_spec']['init_yaml_sha256']:
        raise ValueError('Matched victim initialization/protocol changed before long generation')
    for rel,expected in json.loads((ROOT/'docs/tfss/S0_CANONICAL_SOURCE_SHA256.json').read_text()).items():
        if sha(ROOT/rel)!=expected or sha(Path(meta['parent_root'])/rel)!=expected:
            raise ValueError('Canonical S0 byte identity changed: '+rel)
    if sha(ROOT/lock['parent_config_path'])!=lock['parent_config_sha256'] or sha(cfg['surrogate']['ckpt'])!=lock['reference_checkpoint_sha256']:
        raise ValueError('S0 config/reference identity changed')
    if semantic_identity(cfg,Path(meta['parent_root']))!=provenance['formal_status']['semantic_identity']:
        raise ValueError('TSFL/SafeBG semantic artifacts changed')
    if torch.__version__!=provenance['environment']['torch'] or ultralytics.__version__!=provenance['environment']['ultralytics']:
        raise ValueError('Matched environment changed; do not install replacement')
    source=Path(cfg['data']['dataset_root']);splits={};counts={};person_count=person_gt=0
    for split,expected in [('train',16551),('val',4952)]:
        images=[Path(p) for p in list_images(str(source/'images'/split))];labels=list((source/'labels'/split).glob('*.txt'))
        ids=[p.stem for p in images]
        if len(images)!=expected or len(labels)!=expected or len(set(ids))!=expected or set(ids)!={p.stem for p in labels}:
            raise ValueError('Source split count/unique image/label identity mismatch')
        splits[split]=split_fingerprint(images+labels);counts[split]=len(images)
        if split=='train':
            for p in labels:
                ann=read_yolo_annotations(str(p));n=sum(int(x['cls'])==14 for x in ann)
                person_count+=int(n>0);person_gt+=n
    combined=hashlib.sha256((splits['train']+'\n'+splits['val']).encode()).hexdigest()
    if combined!=lock['dataset_fingerprint'] or (person_count,person_gt)!=(6095,13256):
        raise ValueError('Matched source bytes/Person counts differ')
    dump(out/'source_audit.json',dict(state='passed',splits=splits,counts=counts,target_images=person_count,
        person_gt=person_gt,dataset_fingerprint=combined,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        parent_runtime_git_sha=None,parent_runtime_git_sha_reason='historical S0 contemporaneous commit unknown; exact45 source bytes bound',
        full_budget_mode='bbox_full_target8/0',historical_S0_initialized_from='original reference, not Parent victim or final noise'))
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    frozen=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0');adaptive=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
    trainer=S4FullTrainer(cfg,mc,torch.device('cuda:0'),frozen,adaptive)
    if (trainer.channels.tolist()!=meta['selected_channels'] or
        (trainer.universal_epochs,trainer.universal_batch_size,trainer.eot_samples,trainer.safe_bg_rank,trainer.safe_bg_removal_alpha,trainer.safe_bg_active_k)!=(24,16,2,8,.5,3)):
        raise ValueError('Parent channel order/schedule/SafeBG identity changed')
    proto=Path(meta['parent_root'])/'noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt'
    if sha(proto)!=lock['prototypes_sha256']:raise ValueError('Original prototypes changed')
    prototypes=torch.load(proto,map_location='cuda:0',weights_only=True)['prototypes']
    trainer.tfss_mode='TFSS-U'
    trainer.s4_cache=CleanMapCache(out/'clean_maps',dict(reference_sha256=lock['reference_checkpoint_sha256'],
        maps_code_sha256=sha(ROOT/'ue_framework/methods/tfss_clean_maps.py'),
        risk_code_sha256=sha(ROOT/'ue_framework/methods/tfss_nt_sensitivity.py'),map_version=lock['tfss']['map_version'],support_mode=lock['support_mode'],experiment_id=lock['method']))
    return cfg,trainer,prototypes,provenance,splits

def materialize_and_audit(cfg,trainer,noise,out,source_splits,lock):
    import numpy as np
    import torch
    from PIL import Image
    from ue_framework.data_utils import list_images,load_image_rgb_float,read_yolo_annotations
    from ue_framework.methods.legacy_kproto_ret import _rectangle
    from ue_framework.methods.s4_tfss_full import S4FullGenerator,export_project,spatial_diagnostics
    from ue_framework.methods.s4_pixel_audit import png_project,domains
    from ue_framework.methods.tfss_spatial import spatial_weights
    from scripts.audit_safe_bg_ablation_fingerprint_20261004 import split_fingerprint
    source=Path(cfg['data']['dataset_root']);data=out/'dataset';rows=[];audits=[]
    checkpoint_identity=torch.load(noise/'global_params.pt',map_location='cpu',weights_only=True)
    identity=checkpoint_identity.get('s4_identity',{})
    if (identity.get('method')!=lock['method'] or identity.get('support_mode')!='bbox_full_target'
        or identity.get('config_lock_sha256')!=sha(out/'config_lock_snapshot.json')
        or identity.get('source_commit')!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()):
        raise ValueError('Noise/export config/source/support identity mismatch')
    baseline_data=Path(lock['baseline']['dataset'])
    baseline_manifest=json.loads((Path(lock['baseline']['output'])/'materialization_manifest.json').read_text())
    baseline_hashes={row['id']:row['output_sha256'] for row in baseline_manifest}
    baseline_images={Path(p).stem:Path(p) for p in list_images(str(baseline_data/'images/train'))}
    if len(baseline_images)!=16551:raise ValueError('Original S4 decoded-RMS comparison dataset incomplete')
    gen=S4FullGenerator(cfg,cfg['methods']['legacy_kproto_ret'],torch.device('cuda:0'),trainer.surrogate,str(noise/'global_params.pt'))
    for split in ('train','val'):
        for folder in ('images','labels'):(data/folder/split).mkdir(parents=True,exist_ok=False)
    # SSIM is optional and uses existing installed implementation only. LPIPS
    # has no validated bound local weight/config here; never download weights.
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        structural_similarity=None
    for path in list_images(str(source/'images/train')):
        path=Path(path);label=source/'labels/train'/(path.stem+'.txt')
        ann=read_yolo_annotations(str(label));has=any(int(a['cls'])==14 for a in ann)
        dest=data/'images/train'/(path.stem+'.png' if has else path.name)
        if has:
            image=load_image_rgb_float(str(path));support=gen._build_support(image.shape,ann,image_path=str(path))[0]
            maps,records=trainer.s4_cache.get(trainer,[(image,ann,support)],[str(path)])
            q,r,_=maps[0];weight=spatial_weights(q,r,torch.as_tensor(support,device=trainer.device),mode='TFSS-U',eta=.85).cpu().numpy()
            result=gen.generate(image,ann,0,24,trainer.eps,'mask',image_path=str(path))
            # Scale the canonical pre-image-clamp delta, not its clamped image.
            candidate=np.clip(image+result.perturbation*weight[:,:,None],0,1)
            clean=np.rint(image*255).astype(np.uint8)
            Image.fromarray(export_project(clean,candidate,ann,trainer.eps)).save(dest)
            if len(audits)<6:
                sample=out/'visualization_samples';sample.mkdir(exist_ok=True)
                strength=np.abs(candidate-image).max(2);vis=np.rint(strength/max(trainer.eps,1e-12)*255).astype(np.uint8)
                Image.fromarray(vis).save(sample/(path.stem+'_derived_noise.png'))
        else:shutil.copyfile(path,dest)
        shutil.copyfile(label,data/'labels/train'/label.name)
        if sha(label)!=sha(data/'labels/train'/label.name):raise ValueError('Label bytes changed')
        row=dict(id=path.stem,has_target=has,source=str(path),output=str(dest),output_sha256=sha(dest),label_sha256=sha(label))
        rows.append(row)
        if not has:
            if sha(path)!=sha(dest):raise ValueError('Non-Person bytes changed')
        else:
            decoded=np.rint(load_image_rgb_float(str(dest))*255).astype(np.uint8)
            difference=decoded.astype(np.int16)-clean.astype(np.int16);absolute=np.abs(difference).max(2)
            target,other=domains(clean.shape,ann)
            if np.any(absolute[~target]!=0) or np.any(absolute/255>support*trainer.eps+1e-7):
                raise ValueError('Decoded PNG violates full-target8/0 budget: '+path.stem)
            changed=[]
            for a in ann:
                if int(a['cls'])==14:
                    mask=_rectangle(clean.shape[:2],a['bbox'])[0,0].numpy().astype(bool)
                    changed.append(bool(np.any(absolute[mask]>0)))
            if not all(changed):raise ValueError('Unchanged Person GT: '+path.stem)
            mse=float(np.mean(difference.astype(np.float64)**2))
            if sha(baseline_images[path.stem])!=baseline_hashes[path.stem]:raise ValueError('Historical S4 output hash changed')
            if sha(baseline_data/'labels/train'/label.name)!=sha(label):raise ValueError('Historical S4 diagnostic label identity changed')
            decoded0=np.rint(load_image_rgb_float(str(baseline_images[path.stem]))*255).astype(np.uint8)
            if decoded0.shape!=clean.shape:raise ValueError('Historical S4 comparison image geometry changed')
            diff0=decoded0.astype(np.int16)-clean.astype(np.int16)
            audits.append(dict(id=path.stem,changed_person_gt=sum(changed),linf_byte=int(absolute.max()),
                overlap_linf_byte=int(absolute[target&other].max()) if (target&other).any() else 0,
                rms_pixel=float(np.sqrt(mse)/255),S4_baseline_rms_pixel=float(np.sqrt(np.mean(diff0.astype(np.float64)**2))/255),
                rms_Person=float(np.sqrt(np.mean(difference[target].astype(np.float64)**2))/255),
                S4_baseline_rms_Person=float(np.sqrt(np.mean(diff0[target].astype(np.float64)**2))/255),
                psnr_db=float(10*np.log10(255**2/mse)) if mse else None,
                ssim=float(structural_similarity(clean,decoded,data_range=255,channel_axis=2)) if structural_similarity else None))
            audits[-1].update(spatial_diagnostics(clean/255.,difference/255.,ann,q=q.cpu().numpy(),risk=r.cpu().numpy(),weight=weight))
            audits[-1]['baseline_S4_spatial']=spatial_diagnostics(clean/255.,diff0/255.,ann)
            if len(audits)<=6:
                Image.fromarray(clean).save(sample/(path.stem+'_clean.png'))
                Image.fromarray(decoded).save(sample/(path.stem+'_full.png'))
                Image.fromarray(decoded0).save(sample/(path.stem+'_original_S4.png'))
                from PIL import ImageDraw
                overlay=Image.fromarray(clean);draw=ImageDraw.Draw(overlay)
                for a in ann:
                    cx,cy,bw,bh=a['bbox'];h,w=clean.shape[:2]
                    box=(max(0,int((cx-bw/2)*w)),max(0,int((cy-bh/2)*h)),
                         min(w-1,int(np.ceil((cx+bw/2)*w))),min(h-1,int(np.ceil((cy+bh/2)*h))))
                    draw.rectangle(box,outline='red' if int(a['cls'])==14 else 'green',width=2)
                overlay.save(sample/(path.stem+'_GT.png'))
                Image.fromarray(np.rint(absolute/8*255).astype(np.uint8)).save(sample/(path.stem+'_actual_delta_byte_heatmap.png'))
        if len(rows)%250==0:dump(out/'materialization_progress.json',dict(written=len(rows),remaining=16551-len(rows),audited_target=len(audits)))
    for path in list_images(str(source/'images/val')):
        path=Path(path);label=source/'labels/val'/(path.stem+'.txt')
        for original,dest in [(path,data/'images/val'/path.name),(label,data/'labels/val'/label.name)]:
            shutil.copyfile(original,dest)
            if sha(original)!=sha(dest):raise ValueError('Clean validation bytes changed')
    for split in ('train','val'):
        paths=[Path(p) for p in list_images(str(source/'images'/split))]+list((source/'labels'/split).glob('*.txt'))
        if split_fingerprint(paths)!=source_splits[split]:raise ValueError('Original VOC mutated')
        exported=list_images(str(data/'images'/split));labels=list((data/'labels'/split).glob('*.txt'))
        if len(exported)!={'train':16551,'val':4952}[split] or len(labels)!=len(exported):raise ValueError('Export count mismatch')
        if len({Path(p).stem for p in exported})!=len(exported):raise ValueError('Duplicate exported ID')
    if len(rows)!=16551 or len(audits)!=6095 or sum(x['changed_person_gt'] for x in audits)!=13256:
        raise ValueError('Incomplete full Person coverage')
    dump(out/'materialization_manifest.json',rows)
    integrity=dict(state='passed',support_mode='bbox_full_target',experiment_id=lock['method'],noise_checkpoint_sha256=sha(noise/'global_params.pt'),cache_identity=trainer.s4_cache.identity,train=16551,val=4952,target_images=6095,non_target_images_unchanged=10456,
        person_gt=13256,changed_person_gt=13256,labels_byte_identical=True,clean_val_byte_identical=True,
        non_target_images_byte_identical=True,outside_target_zero=True,per_pixel_frozen_budget_passed=True,
        decoded_PNG_audited=True,materialization_manifest_sha256=sha(out/'materialization_manifest.json'),PSNR_scope='full-image per target image',SSIM='measured' if structural_similarity else 'NOT_RUN_missing_installed_tool',
        LPIPS='NOT_RUN_no_validated_local_weight_binding',images=audits)
    dump(out/'dataset_integrity.json',integrity)
    from scripts.s4_full_report import spatial_summary
    dump(out/'SPATIAL_DIAGNOSTICS_SUMMARY.json',spatial_summary(audits))
    return data


def baseline_identity(lock):
    baseline=lock["baseline"]; old=Path(baseline["output"])
    checks=[(old/"noise/global_params.pt",baseline["noise_sha256"]),
            (old/"materialization_manifest.json",baseline["manifest_sha256"]),
            (old/"dataset_integrity.json",baseline["integrity_sha256"]),
            (ROOT/baseline["comparison_path"],baseline["comparison_sha256"]),
            (ROOT/"docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json","33d3d7af6c76c2c881db0dfc7d6fc66dafa9b5655026c07332701143ed52d6ec")]
    for path,expected in checks:
        if sha(path)!=expected:raise ValueError("Historical S4 identity changed: "+str(path))
    comparison=json.loads((ROOT/baseline["comparison_path"]).read_text())
    if comparison["source_provenance"]["last_sha256"]!=baseline["last_sha256"]:
        raise ValueError("Historical S4 last identity mismatch")
    args=comparison["training"]["args"]
    if sha(Path(args["project"])/args["name"]/"weights/last.pt")!=baseline["last_sha256"]:
        raise ValueError("Actual historical S4 checkpoint changed")
    return comparison


def resource_admission():
    samples=[]
    for index in range(3):
        samples.append(_resource_snapshot(10.))
        if index<2:time.sleep(30)
    blockers=[capacity_blockers(x) for x in samples]
    if any(blockers) or sustained_iowait(samples):
        raise RuntimeError("Capacity/I-O admission pending; no launch: "+repr(blockers))
    return samples


def evaluate_completed(cfg,out,lock):
    import torch
    import numpy as np
    from ultralytics import YOLO
    from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES
    victim=out/"victim_run/victim"
    if (out/'evaluation').exists() or (out/'victim_completion.json').exists():
        raise FileExistsError('Preserve existing evaluation; a recovery needs a new bound evaluation output')
    rows=list(csv.DictReader((victim/"results.csv").open()))
    if len(rows)!=200 or [int(float(list(row.values())[0])) for row in rows]!=list(range(1,201)):
        raise ValueError("Official evaluation requires contiguous 200 epochs")
    if any(not np.isfinite(float(v)) for row in rows for key,v in row.items() if "loss" in key):
        raise ValueError("Nonfinite recorded victim loss")
    dump(out/"victim_completion.json",dict(epochs=200,rows=200,
        last_sha256=sha(victim/"weights/last.pt"),best_sha256=sha(victim/"weights/best.pt")))
    for role,name in [("S4_TFSS_FULL_OVERLAP","last.pt"),("S4_TFSS_FULL_OVERLAP_best_secondary","best.pt")]:
        checkpoint=victim/"weights"/name
        result=YOLO(str(checkpoint)).val(data=str(out/"train_data.yaml"),imgsz=640,batch=36,
            workers=16,device="0",verbose=False,plots=False,seed=0,
            project=str(out/"evaluation"),name=role,exist_ok=False)
        record=dict(summarize(result.box),checkpoint_sha256=sha(checkpoint),
            checkpoint_role="last.pt@200" if name=="last.pt" else "best.pt",class_names=list(NAMES))
        dump(out/(role+"_evaluation.json"),record)


def train_evaluate(cfg,out,data,lock):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES
    from ue_framework.s4_victim import S4VictimTrainer,attach_safety_callbacks
    yaml=out/"train_data.yaml"
    yaml.write_text(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n"+
                    "\n".join(f"  {i}: {n}" for i,n in enumerate(NAMES))+"\n")
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    args=dict(lock["victim_protocol"]["train_args"])
    args.update(data=str(yaml),project=str(out/"victim_run"),name="victim")
    if (args["epochs"],args["batch"],args["workers"],args["seed"],args["amp"],args["optimizer"],
        args["patience"],args["pretrained"])!=(200,36,16,0,True,"SGD",0,False):
        raise ValueError("Frozen actual S4 victim protocol changed")
    init=ROOT/cfg["victim"]["init"]
    if sha(init)!=lock["victim_protocol"]["init_yaml_sha256"]:
        raise ValueError("Random-init model YAML changed")
    dump(out/"victim_run_spec.json",dict(initialization="random_yaml",init_yaml_sha256=sha(init),train_args=args))
    if (out/"victim_run").exists():raise FileExistsError("Fresh victim output required")
    model=YOLO(str(init));attach_safety_callbacks(model)
    model.train(trainer=S4VictimTrainer,**args)
    evaluate_completed(cfg,out,lock)


def gpu_smoke(cfg,trainer,prototypes,lock,out,identity):
    import numpy as np
    import torch
    from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations
    from ue_framework.methods.s4_tfss import S4Trainer,CleanMapCache,rng_state,restore_rng
    from ue_framework.methods.s4_full_optimization import optimize_s4_full
    from ue_framework.methods.s4_tfss_full import S4FullGenerator,export_project,original_training_eligible
    from ue_framework.methods.s4_pixel_audit import domains
    source=Path(cfg["data"]["dataset_root"])
    paths=trainer._collect_target_images(str(source/"images/train"),str(source/"labels/train"))
    selected=[];plain=None;overlap=None
    for path in paths:
        image=load_image_rgb_float(path);ann=read_yolo_annotations(str(source/"labels/train"/(Path(path).stem+".txt")))
        if not original_training_eligible(image.shape,ann):continue
        target,other=domains(image.shape,ann)
        if plain is None and not (target&other).any():plain=(path,image,ann)
        if overlap is None and (target&other).any():overlap=(path,image,ann)
        if len(selected)<16:selected.append(path)
        if plain is not None and overlap is not None and len(selected)==16:break
    if plain is None or overlap is None:raise ValueError("Smoke needs real overlap and nonoverlap")
    selected=list(dict.fromkeys([plain[0],overlap[0]]+selected))[:16]
    state=rng_state()
    parent_cfg=__import__("ue_framework.config",fromlist=["load_config"]).load_config(str(ROOT/lock["parent_config_path"]))
    from ultralytics import YOLO
    from ue_framework.methods.s4_tfss import model_hash
    ref=S4Trainer(parent_cfg,parent_cfg["methods"]["legacy_kproto_ret"],trainer.device,
                  YOLO(cfg["surrogate"]["ckpt"]).model.to(trainer.device),
                  YOLO(cfg["surrogate"]["ckpt"]).model.to(trainer.device))
    if model_hash(ref.surrogate)!=model_hash(trainer.surrogate) or model_hash(ref.adaptive.model)!=model_hash(trainer.adaptive.model):
        raise ValueError("Baseline/candidate teacher weights differ")
    counters={key:getattr(trainer,key) for key in ("_adaptive_calls","_semantic_calls")}
    for key in ("fourier_coeff","suppress_small","freq_score","freq_usage","freq_active_idx"):
        with torch.no_grad():getattr(ref,key).copy_(getattr(trainer,key))
    ref.tfss_mode="TFSS-U";ref.s4_cache=CleanMapCache(out/"reference_clean_maps",
        dict(trainer.s4_cache.identity,support_mode="bbox_half_overlap"))
    restore_rng(state)
    path,image,ann=plain
    evidence=[]
    for obj in (ref,trainer):
        restore_rng(state);obj._epoch=0;obj._active_idx=obj._get_active_freq_indices(0,0)
        obj._active_coords=[tuple(obj.freq_candidate_coords[int(i)]) for i in obj._active_idx.cpu().tolist()]
        support=obj._build_support(image.shape,ann,image_path=path)[0]
        obj.s4_item_paths=[path];obj.s4_measure_protection=False
        loss,_=obj._batch_loss([(image,ann,support)],prototypes)
        grad=torch.autograd.grad(loss,obj.fourier_coeff)[0]
        evidence.append((float(loss.detach()),obj.tfss_delta.detach().clone(),grad.detach()))
    loss_error=abs(evidence[0][0]-evidence[1][0])
    delta_error=float((evidence[0][1]-evidence[1][1]).abs().max())
    grad_error=float((evidence[0][2]-evidence[1][2]).abs().max())
    if max(loss_error,delta_error,grad_error)!=0:raise ValueError("Actual no-overlap production equivalence failed")
    del ref,evidence;restore_rng(state)
    for key,value in counters.items():setattr(trainer,key,value)
    observed=[]
    def observe(t,items,loss,metrics,step,eot):
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite real smoke loss")
        observed.append(dict(step=step,eot=eot,loss=float(loss.detach())))
    status=optimize_s4_full(trainer,selected,source/"labels/train",prototypes,out/"noise",
        identity,stop_after=2,observe=observe,formal_lock=lock)
    if status["steps"]!=2:raise ValueError("Smoke did not reach two actual Adam updates")
    idx=trainer._active_idx
    pack=dict(method="legacy_kproto_ret",support_mode="bbox_full_target",eps=trainer.eps,
        feature_layer=trainer.feature_layer,coords=[list(trainer.freq_candidate_coords[int(i)]) for i in idx.cpu().tolist()],
        fourier_coeff=trainer.fourier_coeff.detach()[idx].cpu(),suppress_small=trainer.suppress_small.detach().cpu(),
        s4_identity=identity)
    checkpoint=out/"smoke_global_params.pt";torch.save(pack,checkpoint)
    gen=S4FullGenerator(cfg,cfg["methods"]["legacy_kproto_ret"],trainer.device,trainer.surrogate,str(checkpoint))
    path,image,ann=overlap;support=gen._build_support(image.shape,ann,image_path=path)[0]
    maps,_=trainer.s4_cache.get(trainer,[(image,ann,support)],[path])
    from ue_framework.methods.tfss_spatial import spatial_weights
    q,r,_=maps[0];weight=spatial_weights(q,r,torch.as_tensor(support,device=trainer.device),mode="TFSS-U",eta=.85).cpu().numpy()
    generated=gen.generate(image,ann,0,24,trainer.eps,"mask",image_path=path)
    clean=np.rint(image*255).astype(np.uint8)
    png=export_project(clean,np.clip(image+generated.perturbation*weight[:,:,None],0,1),ann,trainer.eps)
    from PIL import Image
    Image.fromarray(png).save(out/"smoke_full_export.png")
    return dict(state="passed",actual_Adam_steps=2,observed=observed,
        production_entrypoint_reached=True,finite_loss=True,official_metrics_disabled=True,
        artifact_ingest_disabled=True,cleanup_policy="retain_all",deleted_paths=[],
        baseline_equivalence_probe=dict(reference_id="canonical S4Trainer",sample_id=Path(plain[0]).stem,
            loss_absolute_difference=loss_error,delta_max_absolute_difference=delta_error,
            Fourier_gradient_max_absolute_difference=grad_error,tolerance=0),
        selected_image_ids=[Path(p).stem for p in selected],decoded_PNG_8_0_passed=True,
        overlap_export_image_id=Path(path).stem,checkpoint_paths_remaining=[str(p) for p in out.rglob("*.pt")])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["smoke","formal"],required=True)
    parser.add_argument("--output",required=True);parser.add_argument("--lock",required=True)
    parser.add_argument("--admission",required=True);parser.add_argument("--review")
    args=parser.parse_args();lock=json.loads(Path(args.lock).read_text());out=Path(args.output).resolve()
    boundary=Path(lock["remote_root"]).resolve()
    if not ROOT.is_relative_to(boundary/"repos") or not out.is_relative_to(boundary/"outputs"):
        raise ValueError("Independent full-target repository/output boundary required")
    if out.exists():raise FileExistsError("Unique output only; no reset/resume")
    if args.phase=="formal" and out.name!=lock["run_id"]:raise ValueError("Single authorized formal RunID")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip():
        raise ValueError("Production source dirty")
    receipt=json.loads(Path(args.admission).read_text())
    verify_queue(lock,receipt,source_commit=commit,config_sha256=sha(args.lock))
    if args.phase=="formal":
        review=json.loads(Path(args.review).read_text()) if args.review else {}
        if (review.get("source_commit")!=commit or review.get("config_sha256")!=sha(args.lock)
            or review.get("pre_run_result")!="pass" or review.get("current_source_GPU_smoke_passed") is not True):
            raise ValueError("Current-source GPU smoke and one scientific review/closure required")
    samples=resource_admission()
    verify_queue(lock,receipt,source_commit=commit,config_sha256=sha(args.lock))
    out.mkdir(parents=True)
    for key,name in [("YOLO_CONFIG_DIR","yolo"),("MPLCONFIGDIR","mpl"),("XDG_CACHE_HOME","cache"),("TORCH_HOME","torch")]:
        folder=out/"runtime"/name;folder.mkdir(parents=True);os.environ[key]=str(folder)
    os.environ.update(CUDA_VISIBLE_DEVICES="0",CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",WANDB_MODE="disabled",PYTHONDONTWRITEBYTECODE="1")
    state=dict(state="running",run_id=out.name,source_commit=commit,config_sha256=sha(args.lock),support_mode="bbox_full_target",stage="source_preflight")
    def mark(stage,**values):
        state.update(stage=stage,**values);dump(out/"pipeline_status.json",state);print(json.dumps(state),flush=True)
    try:
        dump(out/"admission_snapshot.json",receipt);dump(out/"resource_admission.json",samples)
        shutil.copyfile(args.lock,out/"config_lock_snapshot.json")
        if args.phase=="formal":dump(out/"review_snapshot.json",review)
        baseline_identity(lock)
        cfg,trainer,prototypes,provenance,splits=setup(lock,out)
        identity=dict(config_lock_sha256=sha(args.lock),source_commit=commit,parent="S4-TFSS-Utility",
            method=lock["method"],support_mode=lock["support_mode"],eta=.85,
            dataset_fingerprint=lock["dataset_fingerprint"],cache_identity=trainer.s4_cache.identity)
        if args.phase=="smoke":
            mark("GPU_smoke");summary=gpu_smoke(cfg,trainer,prototypes,lock,out,identity)
            dump(out/"smoke_summary.json",dict(summary,source_commit=commit,config_sha256=sha(args.lock)))
            mark("complete",state="complete");return
        from ue_framework.methods.s4_full_optimization import optimize_s4_full
        source=Path(cfg["data"]["dataset_root"])
        images=trainer._collect_target_images(str(source/"images/train"),str(source/"labels/train"))
        if len(images)!=6095:raise ValueError("Missing full target dataset")
        mark("noise_generation")
        status=optimize_s4_full(trainer,images,source/"labels/train",prototypes,out/"noise",identity,formal_lock=lock)
        if status.get("schedule_complete") is not True:raise ValueError("Incomplete noise")
        mark("materialization_and_pixel_audit")
        data=materialize_and_audit(cfg,trainer,out/"noise",out,splits,lock)
        integrity=json.loads((out/"dataset_integrity.json").read_text())
        if integrity.get("state")!="passed" or integrity.get("support_mode")!="bbox_full_target":
            raise ValueError("Missing full-target audit")
        del trainer,prototypes
        import gc,torch
        gc.collect();torch.cuda.empty_cache()
        mark("waiting_victim_capacity");resource_admission()
        mark("victim_training_and_evaluation");train_evaluate(cfg,out,data,lock)
        from scripts.s4_full_report import finalize
        finalize(out,ROOT,commit,lock)
        mark("complete",state="complete",completed_at_unix=time.time())
    except BaseException as error:
        mark(state["stage"],state="failed",error=repr(error),failed_at_unix=time.time());raise


if __name__=="__main__":main()
