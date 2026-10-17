"""S4: exact S0+validated Utility -> full PNG audit -> matched random victim200.

All stages fail closed; no reset, cleanup, extra seed, method or AP tuning.
The smoke phase reaches the same trainer, map cache and optimizer production code.
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
BOUNDARY=Path('/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s4_tfss_utility_20261017')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def ram_available():
    fields={line.split(':')[0]:line.split(':')[1].split()[0] for line in Path('/proc/meminfo').read_text().splitlines()}
    return int(fields['MemAvailable'])*1024


def resource_gate():
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied; wait, never compete/interrupt')
    if os.getloadavg()[0]>24 or shutil.disk_usage(BOUNDARY.parent).free<150*1024**3:
        raise RuntimeError('CPU/disk resource gate failed; no cleanup')
    if float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<24000:
        raise RuntimeError('Need dedicated24GiB free GPU')
    if ram_available()<32*1024**3:raise RuntimeError('Need32GiB available RAM for unchanged cache/workers protocol')
    return dict(loadavg=os.getloadavg(),RAM_available_bytes=ram_available(),disk_free_bytes=shutil.disk_usage(BOUNDARY.parent).free,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip())


def setup(lock,out):
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.data_utils import list_images,read_yolo_annotations
    from ue_framework.methods.s4_tfss import S4Trainer,CleanMapCache
    from ue_framework.methods.semantic_coordinates import semantic_identity
    from scripts.audit_safe_bg_ablation_fingerprint_20261004 import split_fingerprint
    cfg=load_config(str(ROOT/lock['parent_config_path']));mc=cfg['methods']['legacy_kproto_ret']
    meta=json.loads((ROOT/'docs/tfss/SOURCE_METADATA.json').read_text())
    provenance=json.loads((ROOT/'docs/S4_TFSS_UTILITY/PARENT_IDENTITY.json').read_text())
    if lock['tfss']['eta_U']!=.85 or lock['seed']!=0 or lock['target_class_id']!=14:
        raise ValueError('Frozen Utility/seed/target configuration changed')
    if lock['victim_protocol']!=provenance['victim_spec'] or sha(ROOT/cfg['victim']['init'])!=provenance['victim_spec']['init_yaml_sha256']:
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
        full_budget_mode='original8/4/0',historical_S0_initialized_from='original reference, not Parent victim or final noise'))
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    frozen=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0');adaptive=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
    trainer=S4Trainer(cfg,mc,torch.device('cuda:0'),frozen,adaptive)
    if (trainer.channels.tolist()!=meta['selected_channels'] or
        (trainer.universal_epochs,trainer.universal_batch_size,trainer.eot_samples,trainer.safe_bg_rank,trainer.safe_bg_removal_alpha,trainer.safe_bg_active_k)!=(24,16,2,8,.5,3)):
        raise ValueError('Parent channel order/schedule/SafeBG identity changed')
    proto=Path(meta['parent_root'])/'noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt'
    if sha(proto)!=lock['prototypes_sha256']:raise ValueError('Original prototypes changed')
    prototypes=torch.load(proto,map_location='cuda:0',weights_only=True)['prototypes']
    trainer.tfss_mode='TFSS-U'
    trainer.s4_cache=CleanMapCache(out/'clean_maps',dict(reference_sha256=lock['reference_checkpoint_sha256'],
        maps_code_sha256=sha(ROOT/'ue_framework/methods/tfss_clean_maps.py'),
        risk_code_sha256=sha(ROOT/'ue_framework/methods/tfss_nt_sensitivity.py'),map_version=lock['tfss']['map_version']))
    return cfg,trainer,prototypes,provenance,splits


def materialize_and_audit(cfg,trainer,noise,out,source_splits):
    import numpy as np
    import torch
    from PIL import Image
    from ue_framework.data_utils import list_images,load_image_rgb_float,read_yolo_annotations
    from ue_framework.methods.legacy_kproto_ret import LegacyKProtoGenerator,_rectangle
    from ue_framework.methods.s4_pixel_audit import png_project,domains
    from ue_framework.methods.tfss_spatial import spatial_weights
    from scripts.audit_safe_bg_ablation_fingerprint_20261004 import split_fingerprint
    source=Path(cfg['data']['dataset_root']);data=out/'dataset';rows=[];audits=[]
    metadata=json.loads((ROOT/'docs/tfss/SOURCE_METADATA.json').read_text())
    s0_data=Path(metadata['s0_poison_data']['path'])
    s0_images={Path(p).stem:Path(p) for p in list_images(str(s0_data/'images/train'))}
    if len(s0_images)!=16551:raise ValueError('S0 decoded-RMS comparison dataset incomplete')
    gen=LegacyKProtoGenerator(cfg,cfg['methods']['legacy_kproto_ret'],torch.device('cuda:0'),trainer.surrogate,str(noise/'global_params.pt'))
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
            Image.fromarray(png_project(clean,candidate,support*trainer.eps)).save(dest)
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
                raise ValueError('Decoded PNG violates original8/4/0 budget: '+path.stem)
            changed=[]
            for a in ann:
                if int(a['cls'])==14:
                    mask=_rectangle(clean.shape[:2],a['bbox'])[0,0].numpy().astype(bool)
                    changed.append(bool(np.any(absolute[mask]>0)))
            if not all(changed):raise ValueError('Unchanged Person GT: '+path.stem)
            mse=float(np.mean(difference.astype(np.float64)**2))
            if sha(s0_data/'labels/train'/label.name)!=sha(label):raise ValueError('S0 diagnostic label identity changed')
            decoded0=np.rint(load_image_rgb_float(str(s0_images[path.stem]))*255).astype(np.uint8)
            if decoded0.shape!=clean.shape:raise ValueError('S0 comparison image geometry changed')
            diff0=decoded0.astype(np.int16)-clean.astype(np.int16)
            audits.append(dict(id=path.stem,changed_person_gt=sum(changed),linf_byte=int(absolute.max()),
                overlap_linf_byte=int(absolute[target&other].max()) if (target&other).any() else 0,
                rms_pixel=float(np.sqrt(mse)/255),S0_rms_pixel=float(np.sqrt(np.mean(diff0.astype(np.float64)**2))/255),
                rms_Person=float(np.sqrt(np.mean(difference[target].astype(np.float64)**2))/255),
                S0_rms_Person=float(np.sqrt(np.mean(diff0[target].astype(np.float64)**2))/255),
                psnr_db=float(10*np.log10(255**2/mse)) if mse else None,
                ssim=float(structural_similarity(clean,decoded,data_range=255,channel_axis=2)) if structural_similarity else None))
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
    integrity=dict(state='passed',train=16551,val=4952,target_images=6095,non_target_images_unchanged=10456,
        person_gt=13256,changed_person_gt=13256,labels_byte_identical=True,clean_val_byte_identical=True,
        non_target_images_byte_identical=True,outside_target_zero=True,per_pixel_frozen_budget_passed=True,
        decoded_PNG_audited=True,materialization_manifest_sha256=sha(out/'materialization_manifest.json'),PSNR_scope='full-image per target image',SSIM='measured' if structural_similarity else 'NOT_RUN_missing_installed_tool',
        LPIPS='NOT_RUN_no_validated_local_weight_binding',images=audits)
    dump(out/'dataset_integrity.json',integrity)
    return data


def summarize(box):
    import numpy as np
    ids=np.asarray(box.ap_class_index).astype(int).tolist();ap=np.asarray(box.all_ap,dtype=float)
    recall=np.asarray(box.r,dtype=float);precision=np.asarray(box.p,dtype=float)
    if ids!=list(range(20)) or ap.shape!=(20,10) or recall.shape!=(20,) or precision.shape!=(20,) or not all(np.isfinite(v).all() for v in (ap,recall,precision)):
        raise ValueError('Incomplete/nonfinite VOC20 validator metrics')
    keep=[i for i in range(20) if i!=14]
    result=dict(AP50_per_class=ap[:,0].tolist(),AP75_per_class=ap[:,5].tolist(),AP50_95_per_class=ap.mean(1).tolist(),
        Recall_per_class=recall.tolist(),Precision_per_class=precision.tolist(),precision_recall_definition='native validator global best-F1 point; AP full curve')
    for key,values in [('mAP50',ap[:,0]),('mAP75',ap[:,5]),('mAP50_95',ap.mean(1)),('Recall',recall),('Precision',precision)]:
        result.update({key+'_target':float(values[14]),key+'_non_target':float(values[keep].mean()),key+'_all':float(values.mean())})
    return result


def train_evaluate(cfg,out,data,provenance,lock):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES,pus,collateral
    from ue_framework.s4_victim import S4VictimTrainer,attach_safety_callbacks
    yaml=out/'train_data.yaml';yaml.write_text(f'path: {data}\ntrain: images/train\nval: images/val\nnames:\n'+'\n'.join(f'  {i}: {n}' for i,n in enumerate(NAMES))+'\n')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    args=dict(provenance['victim_spec']['train_args']);args.update(data=str(yaml),project=str(out/'victim_run'),name='victim',patience=0)
    init=ROOT/cfg['victim']['init']
    if sha(init)!=provenance['victim_spec']['init_yaml_sha256'] or args['epochs']!=200 or args['seed']!=0 or args.get('pretrained') is not False:
        raise ValueError('Random-init matched victim protocol changed')
    dump(out/'victim_run_spec.json',dict(initialization='random_yaml',init_yaml_sha256=sha(init),train_args=args))
    model=YOLO(str(init));attach_safety_callbacks(model)
    model.train(trainer=S4VictimTrainer,**args)
    victim=out/'victim_run/victim'
    rows=list(csv.DictReader((victim/'results.csv').open()))
    if len(rows)!=200 or [int(float(list(r.values())[0])) for r in rows]!=list(range(1,201)):
        raise ValueError('Victim not complete200 contiguous epochs')
    if not all(np.isfinite(float(v)) for row in rows for key,v in row.items() if 'loss' in key):
        raise FloatingPointError('Nonfinite recorded victim loss')
    dump(out/'victim_completion.json',dict(epochs=200,rows=200,last_sha256=sha(victim/'weights/last.pt'),best_sha256=sha(victim/'weights/best.pt')))
    metrics=json.loads((ROOT/'docs/S4_TFSS_UTILITY/HISTORICAL_METRICS.json').read_text())
    evaluation=out/'evaluation';evaluation.mkdir()
    for role,name in [('S4','last.pt'),('S4_best_secondary','best.pt')]:
        checkpoint=victim/'weights'/name
        result=YOLO(str(checkpoint)).val(data=str(yaml),imgsz=640,batch=36,workers=16,device='0',verbose=False,plots=False,seed=0,project=str(evaluation),name=role,exist_ok=False)
        metrics[role]=dict(summarize(result.box),checkpoint_sha256=sha(checkpoint),checkpoint_role='last.pt@200' if role=='S4' else 'best.pt',class_names=list(NAMES))
        dump(out/(role+'_evaluation.json'),metrics[role])
    for role,value in metrics.items():value.update(selectivity=pus(metrics['Clean'],value),collateral=collateral(metrics['Clean'],value))
    from scripts.s4_detection_diagnostics import evaluate_person_dog
    parent=Path(json.loads((ROOT/'docs/tfss/SOURCE_METADATA.json').read_text())['parent_root'])
    s0_checkpoint=parent/'victim_b2_to_safe_soft05_eps8_person_seed0_20261004/victim/weights/last.pt'
    if sha(s0_checkpoint)!=provenance['victim_last_sha256']:raise ValueError('S0 diagnostic checkpoint changed')
    direct={role:evaluate_person_dog(checkpoint,data,out/(role+'_person_dog.json'),lock['direct_Person_Dog_diagnostic'])
        for role,checkpoint in [('S0',s0_checkpoint),('S4',victim/'weights/last.pt')]}
    result=metrics['S4'];target_ok=result['mAP50_target']<=lock['success_gate']['target_max_literal']
    utility_ok=result['mAP50_non_target']>metrics['S0']['mAP50_non_target']
    category='SUCCESS_CANDIDATE' if target_ok and utility_ok else ('TARGET_REGRESSION' if not target_ok else 'UTILITY_NOT_IMPROVED')
    delta={name:{k:result[k]-metrics[name][k] for k in result if k.startswith(('mAP','Recall_','Precision_')) and isinstance(result[k],(float,int))} for name in ('S0','S1','S2','S3')}
    dump(out/'comparison.json',dict(state='complete',method='S4-TFSS-Utility',parent='S0',primary='last.pt@200',secondary='best.pt',
        metrics=metrics,delta=delta,outcome=category,single_seed_not_significance=True,causal_mechanism_established=False,
        person_to_dog=direct,person_to_dog_definition='new predeclared fixed-threshold GT-overlap diagnostic, not native AP/causal migration'))
    with (out/'voc20_comparison.csv').open('w',newline='') as f:
        rows=[]
        for i,name in enumerate(NAMES):
            row={'class':name}
            for role,value in metrics.items():
                for key in ('AP50','AP75','AP50_95','Recall','Precision'):row[role+'_'+key]=value[key+'_per_class'][i]
            for role in ('S0','S1','S2','S3'):
                for key in ('AP50','AP75','AP50_95','Recall','Precision'):row['S4_minus_'+role+'_'+key]=result[key+'_per_class'][i]-metrics[role][key+'_per_class'][i]
            rows.append(row)
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--lock',required=True)
    p.add_argument('--phase',choices=['smoke','formal'],required=True);p.add_argument('--review');p.add_argument('--resume-noise')
    a=p.parse_args();out=Path(a.output).resolve();lock=json.loads(Path(a.lock).read_text())
    if not out.is_relative_to(BOUNDARY/'outputs') or not ROOT.is_relative_to(BOUNDARY):raise ValueError('S4 write boundary')
    if out.exists():raise FileExistsError('Unique output only; never relaunch')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Production source dirty')
    if a.phase=='formal':
        review=json.loads(Path(a.review).read_text()) if a.review else {}
        if review.get('source_commit')!=commit or review.get('pre_run_result')!='pass' or review.get('config_sha256')!=sha(a.lock):
            raise ValueError('Current scientific review/closure and frozen lock required')
        if out.name!=lock['run_id'] and not a.resume_noise:raise ValueError('Formal RunID differs')
    resource=resource_gate();out.mkdir(parents=True)
    for key,name in [('YOLO_CONFIG_DIR','yolo'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','cache'),('TORCH_HOME','torch')]:
        d=out/'runtime'/name;d.mkdir(parents=True);os.environ[key]=str(d)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')
    state=dict(state='running',stage='source_preflight',run_id=out.name,source_commit=commit,config_lock_sha256=sha(a.lock),started_at_unix=time.time(),resource=resource)
    def mark(stage,**kw):state.update(stage=stage,**kw);dump(out/'pipeline_status.json',state);print(json.dumps(state),flush=True)
    mark('source_preflight')
    try:
        cfg,trainer,prototypes,prov,splits=setup(lock,out);dump(out/'config_lock_snapshot.json',lock)
        if a.phase=='smoke':
            from scripts.s4_production_smoke import smoke
            receipt=smoke(cfg,trainer,prototypes,lock,out)
            dump(out/'smoke_summary.json',receipt);mark('complete',state='complete',completed_at_unix=time.time());return
        dump(out/'review_snapshot.json',review)
        from ue_framework.methods.s4_optimization import optimize_s4
        source=Path(cfg['data']['dataset_root']);images=trainer._collect_target_images(str(source/'images/train'),str(source/'labels/train'))
        if len(images)!=6095:raise ValueError('Missing full target support')
        identity=dict(config_lock_sha256=sha(a.lock),source_commit=commit,parent='S0',method='TFSS-U',eta=.85,
            dataset_fingerprint=lock['dataset_fingerprint'],cache_identity=trainer.s4_cache.identity)
        mark('noise_generation');noise=out/'noise'
        result=optimize_s4(trainer,images,source/'labels/train',prototypes,noise,identity,resume=a.resume_noise,formal_lock=lock)
        if result.get('schedule_complete') is not True:raise ValueError('Incomplete noise')
        with (out/'S4_TFSS_SPATIAL_STATS.csv').open('w',newline='') as stream:
            epoch_rows=[json.loads(line) for line in (noise/'epoch_summary.jsonl').read_text().splitlines()]
            writer=csv.DictWriter(stream,fieldnames=list(epoch_rows[0]));writer.writeheader();writer.writerows(epoch_rows)
        mark('materialization_and_pixel_audit');data=materialize_and_audit(cfg,trainer,noise,out,splits)
        if json.loads((out/'dataset_integrity.json').read_text()).get('state')!='passed':raise ValueError('No dataset audit admission')
        # Completed data is reused in place while waiting; no stage is rerun.
        del trainer,prototypes
        import gc,torch
        gc.collect();torch.cuda.empty_cache()
        while True:
            pids={int(x) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()}
            if (not pids-{os.getpid()} and os.getloadavg()[0]<=24 and ram_available()>=32*1024**3
                    and shutil.disk_usage(out).free>=64*1024**3):break
            mark('waiting_victim_resources',other_GPU_pids=sorted(pids-{os.getpid()}),CPU_load=os.getloadavg(),RAM_available_bytes=ram_available())
            time.sleep(60)
        mark('victim_training_and_evaluation');train_evaluate(cfg,out,data,prov,lock)
        from scripts.s4_report import finalize
        finalize(out,ROOT,commit)
        mark('complete',state='complete',completed_at_unix=time.time())
    except BaseException as error:
        mark(state['stage'],state='failed',error=repr(error),failed_at_unix=time.time());raise


if __name__=='__main__':main()
