"""Bounded train-only CNLP unit-gradient calibration. No outer optimizer/victim."""
import argparse
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
sys.dont_write_bytecode=True
ALLOWED=Path('/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--parent-selection',required=True)
    parser.add_argument('--parent-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--inner-steps',type=int,choices=[1,3,5],required=True)
    args=parser.parse_args()
    output=Path(args.output).resolve()
    if not output.is_relative_to(ALLOWED) or not ROOT.is_relative_to(ALLOWED):
        raise ValueError('Admission source/output must stay in authorized guoqinglong project')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied; never compete or interrupt')
    if shutil.disk_usage(ALLOWED).free<4*1024**3:
        raise RuntimeError('Insufficient admission disk; no automatic deletion')
    selection=json.loads(Path(args.parent_selection).read_text())
    if selection.get('status')!='LOCKED' or selection.get('parent')!='S0':
        raise ValueError('Current admission requires verified S0 Parent selection')
    output.mkdir(parents=True,exist_ok=False)
    for name in ['yolo','mpl','cache','torch']:(output/'runtime'/name).mkdir(parents=True)
    os.environ.update(YOLO_CONFIG_DIR=str(output/'runtime/yolo'),MPLCONFIGDIR=str(output/'runtime/mpl'),
        XDG_CACHE_HOME=str(output/'runtime/cache'),TORCH_HOME=str(output/'runtime/torch'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')
    status=dict(state='running',stage='source_audit',official_metrics_disabled=True,artifact_ingest_disabled=True,
                formal_generation_enabled=False,victim_training_enabled=False,outer_optimizer_updates=0,
                inner_steps=args.inner_steps,started_at_unix=time.time())
    def write(name,value):
        p=output/name
        p.write_text(json.dumps(value,indent=2,allow_nan=False))
    def mark(stage,**fields):
        status.update(stage=stage,**fields);write('status.json',status);print(json.dumps(status),flush=True)
    mark('source_audit')
    try:
        import numpy as np
        import torch
        from ultralytics import YOLO
        import ultralytics
        from ue_framework.config import load_config
        from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations,label_path_for_image
        from ue_framework.methods.semantic_coordinates import semantic_identity
        from ue_framework.methods.counterfactual_utility import CNLPConfig
        from ue_framework.methods.counterfactual_trainer import S2S3FixedParentTrainer
        from ue_framework.methods.counterfactual_query import TrainQueryProvider
        from ue_framework.methods.s1_optimization import model_hash
        from scripts.prepare_counterfactual_inputs import train_manifest
        cfg=load_config(args.config);mc=cfg['methods']['legacy_kproto_ret'];parent=Path(args.parent_root).resolve()
        provenance=json.loads((ROOT/'docs/s1_evidence/parent_remote_identity.json').read_text())
        if str(parent)!=provenance['parent_root']:raise ValueError('Parent root mismatch')
        for rel,expected in provenance['source_sha256'].items():
            if sha(ROOT/rel)!=expected:raise ValueError('Parent source bytes changed: '+rel)
        if sha(args.config)!=selection['selected_identity']['config_sha256']:raise ValueError('Parent config changed')
        if sha(cfg['surrogate']['ckpt'])!=provenance['formal_status']['surrogate_sha256']:raise ValueError('Reference changed')
        if semantic_identity(cfg,parent)!=provenance['formal_status']['semantic_identity']:raise ValueError('Parent semantic artifacts changed')
        if torch.__version__!=provenance['environment']['torch'] or ultralytics.__version__!=provenance['environment']['ultralytics']:
            raise ValueError('Parent scientific environment changed')
        source=Path(cfg['data']['dataset_root'])
        mark('train_manifest')
        manifest=train_manifest(source/'images/train',source/'labels/train',source/'images/val')
        write('train_query_manifest.json',manifest)
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
        reference=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
        adaptive_model=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
        holder={}
        trainer=S2S3FixedParentTrainer(cfg,mc,torch.device('cuda:0'),reference,adaptive_model,
            cnlp_config=CNLPConfig(args.inner_steps,float(mc['adaptive']['inner_lr']),1.),
            query_provider=lambda ids,step:holder['provider'](ids,holder['case']),support_batch_size=1)
        if trainer.eot_samples!=2 or trainer.universal_batch_size!=16:
            raise ValueError('Parent batch16/EOT2 changed')
        if trainer.channels.tolist()!=[0,6,10,29,41,68,77,86,89,112,116,125,128,133,140,167,173,184,220,229,237,238,239,240]:
            raise ValueError('Parent selected channels changed')
        if (trainer.safe_bg_rank,trainer.safe_bg_removal_alpha,trainer.safe_bg_active_k)!=(8,.5,3):
            raise ValueError('Parent Safe-BG changed')
        provider=TrainQueryProvider(output/'train_query_manifest.json',sha(output/'train_query_manifest.json'),trainer.adaptive,1,0)
        holder['provider']=provider
        all_records=list(provider.records)
        order=lambda row:hashlib.sha256(('CNLP seed0 '+Path(row['image_id']).stem).encode()).hexdigest()
        persons=trainer._collect_target_images(str(source/'images/train'),str(source/'labels/train'))
        if len(persons)!=6095:raise ValueError('Full Person image identity missing')
        persons=sorted(persons,key=lambda p:hashlib.sha256(('CNLP seed0 '+Path(p).stem).encode()).hexdigest())
        classes={row['image_id']:{int(a['cls']) for a in read_yolo_annotations(row['label_path'])} for row in all_records}
        plan=[]
        for cls in range(20):
            if cls==14:continue
            for repeat in range(2):
                index=len(plan);support=[persons[(index*16+j)%len(persons)] for j in range(16)]
                eligible=sorted([r for r in all_records if cls in classes[r['image_id']] and r['image_id'] not in set(support)],key=order)
                if len(eligible)<2:raise ValueError('Insufficient fixed train-query coverage')
                plan.append(dict(case=index,class_stratum=cls,repeat=repeat,support_ids=support,query=eligible[repeat]))
        write('fixed_train_plan.json',dict(selection_policy='sha256 CNLP seed0,19NTclassesx2,disjoint train',cases=plan))
        proto=parent/'noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt'
        if sha(proto)!=provenance['noise_artifacts']['background_prototypes.pt']['sha256']:raise ValueError('Parent prototypes changed')
        prototypes=torch.load(proto,map_location='cuda:0',weights_only=True)['prototypes']
        frozen_hash=model_hash(trainer.surrogate);adaptive_hash=model_hash(trainer.adaptive.model)
        initial_coeff=trainer.fourier_coeff.detach().clone()
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        rows=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        mark('unit_gradient_calibration',cases_total=38,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        for item in plan:
            holder['case']=item['case'];provider.records=[item['query']]
            items=[]
            for path in item['support_ids']:
                image=load_image_rgb_float(path);anns=read_yolo_annotations(label_path_for_image(path,str(source/'labels/train')))
                support,_,_=trainer._build_support(image.shape,anns,image_path=path)
                if support.sum()<=10:raise ValueError('Selected support has insufficient target area')
                items.append((image,anns,support))
            trainer._epoch=0;trainer._active_idx=trainer._get_active_freq_indices(0,item['case'])
            trainer._active_coords=[tuple(trainer.freq_candidate_coords[int(j)]) for j in trainer._active_idx.cpu().tolist()]
            trainer.cnlp_begin_step(0);trainer.cnlp_bind_support(item['support_ids'])
            for eot in range(trainer.eot_samples):
                trainer.fourier_coeff.grad=None
                torch.cuda.synchronize();begin=time.perf_counter()
                total,metrics=trainer._batch_loss(items,prototypes)
                total.backward();torch.cuda.synchronize()
                if not torch.isfinite(trainer.fourier_coeff.grad).all():raise FloatingPointError('Nonfinite total Fourier gradient')
                vectors=trainer._cnlp_gradient_vectors
                valid=[c for c in range(20) if c!=14 and float(metrics[f'cnlp_class_{c}_valid'])>0]
                row=dict(case=item['case'],eot=eot,query_id=item['query']['image_id'],class_stratum=item['class_stratum'],seconds=time.perf_counter()-begin,
                         damage={str(c):float(metrics[f'cnlp_class_{c}_damage'].detach()) for c in valid},
                         clean_learning_gain={str(c):float((metrics[f'cnlp_class_{c}_initial']-metrics[f'cnlp_class_{c}_clean']).detach()) for c in valid},
                         norm={k:float(v.norm()) for k,v in vectors.items()},
                         cosine={k:float((vectors[k]*vectors['cnlp']).sum()/(vectors[k].norm()*vectors['cnlp'].norm()).clamp_min(1e-12)) for k in ['parent','attack','ret']},
                         parent_loss=float(metrics['L_parent'].detach()),cnlp_unit_loss=float(metrics['L_CNLP'].detach()),
                         target_proxy=float(metrics['target_score_clean_after'].detach()),max_abs_delta=float(metrics['max_abs_delta'].detach()),
                         outside_max=float(metrics['outside_support_max'].detach()))
                if row['max_abs_delta']>float(cfg['experiment']['eps'])+1e-6 or row['outside_max']!=0:raise ValueError('Parent spatial budget violated')
                rows.append(row);write('calibration_rows.json',rows)
            trainer.cnlp.abort_step()
            mark('unit_gradient_calibration',cases_completed=item['case']+1,eot_batches_completed=len(rows))
        if not torch.equal(initial_coeff,trainer.fourier_coeff):raise ValueError('Diagnostic changed Fourier coefficients')
        if model_hash(trainer.surrogate)!=frozen_hash or model_hash(trainer.adaptive.model)!=adaptive_hash:raise ValueError('Parent model changed')
        summary=dict(status,state='complete',passed=True,computation_kind='training',finite_loss=True,
                     production_entrypoint_reached=True,production_entrypoint='S2S3FixedParentTrainer._batch_loss',
                     cases=38,eot_batches=len(rows),virtual_sgd_updates=2*args.inner_steps*len(rows),outer_optimizer_updates=0,
                     source_commit=status['source_commit'],parent_selection_sha256=sha(args.parent_selection),
                     fixed_plan_sha256=sha(output/'fixed_train_plan.json'),manifest_sha256=sha(output/'train_query_manifest.json'),
                     duration_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     frozen_model_unchanged=True,adaptive_model_unchanged=True,fourier_unchanged=True,
                     cleanup_policy='retain_all',checkpoint_cleanup_completed=False,deleted_paths=[],checkpoint_paths_remaining=[])
        write('smoke_summary.json',summary);mark('complete',state='complete',passed=True)
    except BaseException as error:
        mark(status['stage'],state='failed',error=repr(error));raise


if __name__=='__main__':main()
