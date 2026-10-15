"""Bounded real-production Parent equivalence and CNLP/CUP continuation audit."""
import argparse,hashlib,json,os,random,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ALLOWED=Path('/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014')
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))
def worker(a):
    # The reference really imports the canonical historical source in its own process.
    sys.path.insert(0,str(Path(a.parent_root) if a.mode=='reference' else ROOT))
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied before worker allocation')
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations,label_path_for_image
    from ue_framework.methods.b2_semantic import B2SemanticTrainer
    import ue_framework.methods.b2_semantic as parent_module
    cfg=load_config(a.config);mc=cfg['methods']['legacy_kproto_ret'];parent=Path(a.parent_root)
    provenance=json.loads((ROOT/'docs/s1_evidence/parent_remote_identity.json').read_text())
    if str(parent)!=provenance['parent_root']:raise ValueError('Unexpected canonical Parent root')
    imported_root=Path(a.parent_root) if a.mode=='reference' else ROOT
    for rel,expected in provenance['source_sha256'].items():
        if sha(imported_root/rel)!=expected:raise ValueError('Parent source changed: '+rel)
    if sha(a.config)!=provenance['formal_status']['config_sha256']:raise ValueError('Parent config changed')
    if sha(cfg['surrogate']['ckpt'])!=provenance['formal_status']['surrogate_sha256']:raise ValueError('Reference changed')
    from ue_framework.methods.semantic_coordinates import semantic_identity
    if semantic_identity(cfg,parent)!=provenance['formal_status']['semantic_identity']:raise ValueError('Semantic artifacts changed')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    model=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0');learner=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
    if a.mode=='reference':
        trainer=B2SemanticTrainer(cfg,mc,torch.device('cuda:0'),model,learner)
    else:
        from ue_framework.methods.counterfactual_utility import CNLPConfig,CUPConfig
        from ue_framework.methods.counterfactual_trainer import S2S3FixedParentTrainer
        from ue_framework.methods.counterfactual_query import TrainQueryProvider
        cal=json.loads(Path(a.calibration).read_text())
        if cal['status']!='TRAIN_ONLY_ADMISSION_ANALYZED' or not cal['attack_conflict_gate_passed']:raise ValueError('Unpassed calibration')
        if sha(a.manifest)!=cal['results'][0]['manifest_sha256'] or sha(a.plan)!=cal['results'][0]['fixed_plan_sha256']:raise ValueError('Calibrated train plan/manifest changed')
        if sha(ROOT/'parent_selection.json')!=cal['parent_selection_sha256']:raise ValueError('Parent binding changed')
        proposal=json.loads(Path(a.proposal).read_text());policy=proposal['cnlp_config']['matching_policy']
        if proposal['calibration_sha256']!=sha(a.calibration):raise ValueError('Proposal/calibration mismatch')
        lam=0. if a.mode in ('disabled','parent_loop') else cal['lambda_candidate']
        holder={}
        trainer=S2S3FixedParentTrainer(cfg,mc,torch.device('cuda:0'),model,learner,
            cnlp_config=CNLPConfig(cal['selected_j'],float(mc['adaptive']['inner_lr']),lam,matching_policy=policy),
            cup_config=CUPConfig(**cal['cup_candidate']) if a.mode.startswith('s3_') or a.mode=='cup_sink' else None,
            query_provider=lambda ids,step:holder['provider'](ids,step),support_batch_size=1)
        holder['provider']=TrainQueryProvider(a.manifest,cal['results'][0]['manifest_sha256'],trainer.adaptive,1,0)
        trainer.cnlp_query_provider=holder['provider']
    if (trainer.universal_epochs,trainer.universal_batch_size,trainer.eot_samples)!=(24,16,2):raise ValueError('Parent schedule changed')
    plan=json.loads(Path(a.plan).read_text())['cases'];images=plan[0]['support_ids']+plan[1]['support_ids']
    if len(set(images))!=32:raise ValueError('Probe requires actual fixed32 unique train images')
    if a.mode=='cup_sink':
        candidates={r['query_id'] for r in cal['results'][0]['cases'] if len(r['damage'])>=2 and r['query_id'] not in set(images)}
        selected=sorted(candidates,key=lambda v:hashlib.sha256(('CUP sink seed0 '+Path(v).stem).encode()).hexdigest())[:4]
        if len(selected)!=4:raise ValueError('Insufficient preregistered multiclass CUP sink inputs')
        trainer.cnlp_query_provider.records=[r for r in trainer.cnlp_query_provider.records if r['image_id'] in selected]
    labels=Path(cfg['data']['dataset_root'])/'labels/train'
    proto=parent/'noise_full_b2_to_safe_soft05_eps8_person_seed0_20261004/background_prototypes.pt'
    if sha(proto)!=provenance['noise_artifacts']['background_prototypes.pt']['sha256']:raise ValueError('Prototypes changed')
    prototypes=torch.load(proto,map_location='cuda:0',weights_only=True)['prototypes']
    out=Path(a.output);arrays={};rows=[]
    def budget(items):
        poison=trainer._last_poison;h,w=poison.shape[-2:]
        import torch.nn.functional as F
        clean=torch.stack([F.pad(torch.from_numpy(im).permute(2,0,1).float(),(0,w-im.shape[1],0,h-im.shape[0]),value=.5) for im,_,_ in items]).to(poison)
        limit=torch.stack([F.pad(torch.from_numpy(support).float(),(0,w-im.shape[1],0,h-im.shape[0])) for im,_,support in items]).to(poison).unsqueeze(1)*trainer.eps
        delta=(poison-clean).abs();violation=float((delta-limit).clamp_min(0).max())
        outside=float(delta.expand_as(poison)[(limit==0).expand_as(poison)].max()) if bool((limit==0).any()) else 0.
        if violation>1e-7 or outside!=0:raise ValueError('8/4/0 production pixel budget failed')
        return dict(pixel_budget_violation=violation,outside_max=outside,max_delta=float(delta.max()))
    if a.mode=='matching_edges':
        out.mkdir(parents=True,exist_ok=False)
        evidence=json.loads(Path(a.edge_evidence).read_text())
        edge_ids=sorted({r['id'] for r in evidence['native_uncovered_train_objects']})
        all_records=list(trainer.cnlp_query_provider.records)
        edge_rows=[next(r for r in all_records if Path(r['image_id']).stem==ident) for ident in edge_ids]
        dump(out/'fixed_edge_queries.json',dict(source_evidence_sha256=sha(a.edge_evidence),queries=edge_rows))
        for i,row in enumerate(edge_rows):
            selected=images[:16] if row['image_id'] not in images[:16] else images[16:]
            items=[]
            for image_path in selected:
                im=load_image_rgb_float(image_path);anns=read_yolo_annotations(label_path_for_image(image_path,str(labels)))
                support,_,_=trainer._build_support(im.shape,anns,image_path=image_path);items.append((im,anns,support))
            trainer.cnlp_query_provider.records=[row]
            trainer._epoch=0;trainer._active_idx=trainer._get_active_freq_indices(0,i)
            trainer._active_coords=[tuple(trainer.freq_candidate_coords[int(k)]) for k in trainer._active_idx.cpu().tolist()]
            trainer.fourier_coeff.grad=None;trainer.cnlp_begin_step(0);trainer.cnlp_bind_support(selected)
            dump(out/'current_case.json',dict(case=i,query=row,full_support_ids=selected))
            total,metrics=trainer._batch_loss(items,prototypes);total.backward()
            if not torch.isfinite(trainer.fourier_coeff.grad).all():raise FloatingPointError('Edge gradient nonfinite')
            rows.append(dict(objects=[{k:v for k,v in rec.items() if k not in ('clean_components','poison_components')} for rec in trainer._cnlp_last_evidence['objects']],case=i,query_id=row['image_id'],nt_objects=float(metrics['cnlp_valid_objects']),loss=float(total.detach()),**budget(items)))
            dump(out/'edge_rows.json',rows);trainer.cnlp.abort_step()
        dump(out/'receipt.json',dict(mode=a.mode,passed=True,rows=rows,outer_optimizer_updates=0,query_count=len(edge_rows)))
    elif a.mode in ('reference','disabled'):
        out.mkdir(parents=True,exist_ok=False)
        opt=torch.optim.Adam([{'params':[trainer.fourier_coeff],'lr':trainer.universal_lr_fourier}])
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        for step in range(5):
            trainer._epoch=0;trainer._active_idx=trainer._get_active_freq_indices(0,step)
            trainer._active_coords=[tuple(trainer.freq_candidate_coords[int(i)]) for i in trainer._active_idx.cpu().tolist()]
            opt.zero_grad(set_to_none=True)
            selected=images[(step%2)*16:(step%2+1)*16];items=[]
            for p in selected:
                im=load_image_rgb_float(p);anns=read_yolo_annotations(label_path_for_image(p,str(labels)))
                support,_,_=trainer._build_support(im.shape,anns,image_path=p);items.append((im,anns,support))
            if a.mode=='disabled':trainer.cnlp_begin_step(step);trainer.cnlp_bind_support(selected)
            for eot in range(trainer.eot_samples):
                total,m=trainer._batch_loss(items,prototypes)
                if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite equivalence loss')
                rows.append(dict(step=step,eot=eot,parent_loss=float(m.get('L_parent',total).detach()),**budget(items)))
                arrays[f'{step}.{eot}.poison']=trainer._last_poison.detach().cpu()
                total.backward()
            trainer.fourier_coeff.grad.div_(trainer.eot_samples)
            arrays[f'{step}.gradient']=trainer.fourier_coeff.grad.detach().cpu().clone()
            opt.step()
            if a.mode=='disabled':trainer.cnlp_commit_step()
            arrays[f'{step}.coeff']=trainer.fourier_coeff.detach().cpu().clone()
        torch.save(arrays,out/'trajectory.pt')
        dump(out/'receipt.json',dict(mode=a.mode,rows=rows,production_steps=5,finite_loss=True,imported_parent_file=parent_module.__file__,
            parent_source_sha256={r:sha(imported_root/r) for r in provenance['source_sha256']},
            reference_checkpoint_path=cfg['surrogate']['ckpt'],reference_checkpoint_sha256=sha(cfg['surrogate']['ckpt']),config_sha256=sha(a.config),images=images))
    else:
        from ue_framework.methods.counterfactual_optimization import optimize_counterfactual,model_hash
        before=(model_hash(model),model_hash(learner))
        def observe(t,items,loss,values,step,eot):
            valid=[c for c in range(20) if c!=14 and float(values[f'cnlp_class_{c}_valid'])>0]
            uniform=torch.stack([values[f'cnlp_class_{c}_damage'].relu() for c in valid]).mean()
            g_uniform=torch.autograd.grad(uniform,t.fourier_coeff,retain_graph=True)[0]
            g_weighted=t._cnlp_gradient_vectors['cnlp']
            attack=t._semantic_terms['BG_sel']+t._semantic_terms['rel_sel']+t.lambda_adapt*values['L_adapt_target']
            g_attack=torch.autograd.grad(attack,t.fourier_coeff,retain_graph=True)[0]
            denom=g_attack.norm()*g_weighted.norm()
            full_attack_cosine=float((g_attack*g_weighted).sum()/denom) if float(denom)>0 else None
            ratio=float(g_weighted.norm()*t.cnlp.adapter.config.lambda_cnlp/t._cnlp_gradient_vectors['parent'].norm().clamp_min(1e-12))
            weights=torch.stack([values[f'cnlp_class_{c}_weight'] for c in valid])
            if abs(float(weights.sum())-1)>1e-6 or float(weights.min())<.5/len(valid)-1e-6 or float(weights.max())>2./len(valid)+1e-6:
                raise ValueError('Actual CUP sink weight normalization/bounds failed')
            rows.append(dict(step=step,eot=eot,support_ids=list(t.cnlp_support_ids),query_ids=sorted({r['image_id'] for r in t._cnlp_last_evidence['objects']}),target_proxy=float(values['target_score_clean_after'].detach()),uniform_cnlp_loss=float(uniform.detach()),weighted_uniform_gradient_difference=float((g_weighted-g_uniform).norm()),weighted_cnlp_to_parent_norm=ratio,cosine_full_attack_cnlp=full_attack_cosine,relative_weight_min=float(weights.min())*len(valid),relative_weight_max=float(weights.max())*len(valid),loss=float(loss.detach()),cnlp_loss=float(values['L_CNLP'].detach()),
                cup_min=float(values['cup_weight_min']),cup_max=float(values['cup_weight_max']),
                valid_classes=float(values['cnlp_valid_classes']),**budget(items)))
        target_steps=3 if a.mode.endswith('_pause') else (10 if a.mode=='cup_sink' else 5)
        state=optimize_counterfactual(trainer,images,labels,prototypes,out,
            dict(parent_selection_sha256=sha(ROOT/'parent_selection.json'),calibration_sha256=sha(a.calibration),source_commit=a.commit,diagnostic_query_subset=selected if a.mode=='cup_sink' else None),
            stop_after=target_steps,resume=a.resume,observe=observe)
        if (model_hash(model),model_hash(learner))!=before:raise ValueError('Frozen/Adaptive model changed')
        if state['steps']!=target_steps:raise ValueError('Incorrect bounded production step count')
        if trainer.cnlp.completed_steps!=state['steps']:raise ValueError('CUP/CNLP commit counter mismatch')
        if trainer.cnlp.cup and trainer.cnlp.cup.steps!=state['steps']:raise ValueError('CUP EMA updated other than once per actual optimizer step')
        png_checks=[]
        if a.mode.endswith('_full'):
            from ue_framework.methods.legacy_kproto_ret import LegacyKProtoGenerator
            from ue_framework.methods.csb_budget import png_project
            from PIL import Image
            idx=trainer.freq_active_idx if trainer._epoch+1>=trainer.adaptive_freq_warmup_epochs else trainer._active_idx
            pack=dict(coords=[list(trainer.freq_candidate_coords[int(i)]) for i in idx.cpu().tolist()],
                fourier_coeff=trainer.fourier_coeff.detach()[idx].cpu(),suppress_small=trainer.suppress_small.detach().cpu(),
                method='legacy_kproto_ret',support_mode=trainer.support_mode,eps=trainer.eps,feature_layer=trainer.feature_layer)
            torch.save(pack,out/'bounded_export_params.pt')
            gen=LegacyKProtoGenerator(cfg,mc,torch.device('cuda:0'),model,str(out/'bounded_export_params.pt'))
            for number,image_path in enumerate(images[:2]):
                im=load_image_rgb_float(image_path);lp=label_path_for_image(image_path,str(labels));before_label=sha(lp)
                anns=read_yolo_annotations(lp);support=gen._build_support(im.shape,anns,image_path=image_path)[0]
                b=support*trainer.eps;clean=np.rint(im*255).astype(np.uint8)
                result=gen.generate(im,anns,0,24,trainer.eps,'mask',image_path=image_path)
                dest=out/f'bounded_train_export_{number}.png';Image.fromarray(png_project(clean,result.poisoned_image,b)).save(dest)
                actual=np.asarray(Image.open(dest),dtype=np.int16);delta=np.abs(actual-clean.astype(np.int16)).max(2)
                violation=float(np.maximum(delta/255-b,0).max())
                outside=int(delta[support==0].max()) if np.any(support==0) else 0
                if violation>1e-7 or outside or sha(lp)!=before_label:raise ValueError('Decoded PNG budget/label failed')
                png_checks.append(dict(image_id=image_path,max_delta_byte=int(delta.max()),budget_violation=violation,outside_max=outside,label_unchanged=True))
        dump(out/'receipt.json',dict(mode=a.mode,status=state,rows=rows,finite_loss=True,base_models_unchanged=True,
            added_optimizer_steps=len(rows)//trainer.eot_samples,cnlp_commits=trainer.cnlp.completed_steps,
            cup_steps=trainer.cnlp.cup.steps if trainer.cnlp.cup else None,diagnostic_query_subset=selected if a.mode=='cup_sink' else None,png_checks=png_checks))

def main():
    p=argparse.ArgumentParser()
    for k in ['config','parent-root','output','calibration','plan','manifest','edge-evidence','proposal']:p.add_argument('--'+k,required=True)
    p.add_argument('--proxy-only',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--mode');p.add_argument('--resume');p.add_argument('--commit')
    a=p.parse_args()
    if a.worker:return worker(a)
    out=Path(a.output).resolve()
    if not out.is_relative_to(ALLOWED) or not ROOT.resolve().is_relative_to(ALLOWED):raise ValueError('Unauthorized write root')
    for k in ['config','parent_root','calibration','plan','manifest','edge_evidence','proposal']:setattr(a,k,str(Path(getattr(a,k)).resolve()))
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    if shutil.disk_usage(ALLOWED).free<4*1024**3:raise RuntimeError('Insufficient isolated admission disk; no deletion')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['yolo','mpl','cache','torch']:(out/'runtime'/name).mkdir(parents=True)
    env=dict(os.environ,YOLO_CONFIG_DIR=str(out/'runtime/yolo'),MPLCONFIGDIR=str(out/'runtime/mpl'),
        XDG_CACHE_HOME=str(out/'runtime/cache'),TORCH_HOME=str(out/'runtime/torch'),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=dict(state='running',stage='production_admission',source_commit=commit,formal_generation_enabled=False,victim_training_enabled=False,
        official_metrics_disabled=True,artifact_ingest_disabled=True)
    def mark(stage,**kw):status.update(stage=stage,**kw);dump(out/'status.json',status);print(json.dumps(status),flush=True)
    try:
        modes=['parent_loop','s2_full','s3_full'] if a.proxy_only else ['matching_edges','reference','disabled','s2_full','s2_pause','s2_resume','s3_full','s3_pause','s3_resume','cup_sink']
        for mode in modes:
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied between probes')
            mark(mode)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--mode',mode,'--commit',commit,'--output',str(out/mode)]
            for k in ['config','parent_root','calibration','plan','manifest','edge_evidence','proposal']:cmd+=['--'+k.replace('_','-'),getattr(a,k)]
            if mode.endswith('_resume'):
                saved=json.loads((out/(mode[:2]+'_pause')/'receipt.json').read_text())['status']['checkpoint'];cmd+=['--resume',saved]
            subprocess.run(cmd,check=True,env=env,cwd=out)
        if a.proxy_only:
            import statistics
            receipts={m:json.loads((out/m/'receipt.json').read_text()) for m in modes}
            ref=receipts['parent_loop']['rows'];checks={}
            for m in modes:
                rows=receipts[m]['rows']
                if len(rows)!=10 or any(a['support_ids']!=b['support_ids'] or a['query_ids']!=b['query_ids'] for a,b in zip(ref,rows)):
                    raise ValueError('Person proxy comparison requires IDENTICAL support/query/order/EOT')
                avg=statistics.mean(r['target_proxy'] for r in rows);baseline=statistics.mean(r['target_proxy'] for r in ref)
                checks[m]=dict(target_proxy_mean=avg,target_proxy_median=statistics.median(r['target_proxy'] for r in rows),relative_mean_change=avg/baseline-1 if baseline else None,
                    checkpoint_steps=receipts[m]['status']['steps'],actual_lambda=0. if m=='parent_loop' else json.loads(Path(a.calibration).read_text())['lambda_candidate'],
                    base_models_unchanged=receipts[m]['base_models_unchanged'],png_checks=receipts[m]['png_checks'])
            summary=dict(status,state='complete',passed=True,finite_loss=True,computation_kind='training',production_entrypoint_reached=True,
                production_entrypoint='S2S3FixedParentTrainer._batch_loss + optimize_counterfactual',production_optimizer_steps=15,eot_batches=30,
                matched_parent_proxy_checks=checks,proposal_sha256=sha(a.proposal),calibration_sha256=sha(a.calibration),
                cleanup_policy='retain_all',checkpoint_cleanup_completed=False,deleted_paths=[],checkpoint_paths_remaining=[str(v) for v in out.rglob('*.pt')])
            dump(out/'smoke_summary.json',summary);mark('complete',state='complete',passed=True);return
        import torch
        from scripts.s1_mechanism_probe import compare_tree
        ref=torch.load(out/'reference/trajectory.pt',map_location='cpu',weights_only=True)
        new=torch.load(out/'disabled/trajectory.pt',map_location='cpu',weights_only=True)
        diff=max(float((ref[k]-new[k]).abs().max()) for k in ref)
        rr=json.loads((out/'reference/receipt.json').read_text());nr=json.loads((out/'disabled/receipt.json').read_text())
        ld=max(abs(r['parent_loss']-n['parent_loss']) for r,n in zip(rr['rows'],nr['rows']))
        if diff>1e-6 or ld>1e-6:raise ValueError('Actual original Parent equivalence failed')
        checks={}
        for method in ['s2','s3']:
            f=json.loads((out/(method+'_full')/'receipt.json').read_text());r=json.loads((out/(method+'_resume')/'receipt.json').read_text())
            a1=torch.load(f['status']['checkpoint'],map_location='cpu',weights_only=False);a2=torch.load(r['status']['checkpoint'],map_location='cpu',weights_only=False)
            difference=compare_tree(a1,a2);same=(out/(method+'_full')/'diagnostics.csv').read_bytes()==(out/(method+'_resume')/'diagnostics.csv').read_bytes()
            checks[method]=dict(checkpoint_max_difference=difference,diagnostics_byte_identical=same,full_steps=f['status']['steps'],resumed_steps=r['status']['steps'],
                base_models_unchanged=f['base_models_unchanged'] and r['base_models_unchanged'],decoded_png_checks=f['png_checks'],cnlp_commits=r['cnlp_commits'],cup_steps=r['cup_steps'])
            if difference!=0 or not same:raise ValueError('Production continuation mismatch: '+method)
        cup=json.loads((out/'cup_sink/receipt.json').read_text())
        active=sum(r['weighted_uniform_gradient_difference']>1e-12 for r in cup['rows'])
        if cup['cup_steps']!=10 or active==0:raise ValueError('Real CUP aggregation did not affect Fourier gradient on fixed multiclass probe')
        summary=dict(status,state='complete',passed=True,finite_loss=True,computation_kind='training',production_entrypoint_reached=True,
            production_entrypoint='S2S3FixedParentTrainer._batch_loss + optimize_counterfactual',production_optimizer_steps=40,eot_batches=80,matching_edge_batches=12,
            baseline_equivalence_probe=dict(passed=True,reference_id=str(Path(a.parent_root)),reference_computation=rr['imported_parent_file'],candidate_computation='S2S3FixedParentTrainer lambda_CNLP=0',
                reference_weights_path=rr['reference_checkpoint_path'],candidate_weights_path=nr['reference_checkpoint_path'],reference_value=rr['rows'][0]['parent_loss'],candidate_value=nr['rows'][0]['parent_loss'],absolute_difference=ld,tolerance=1e-6,all_tensor_max_difference=diff),
            checks=checks,cup_sink=dict(optimizer_steps=10,eot_batches=20,weighted_gradient_difference_nonzero_batches=active,rows=cup['rows']),proposal_sha256=sha(a.proposal),matching_policy=json.loads(Path(a.proposal).read_text())['cnlp_config']['matching_policy'],calibration_sha256=sha(a.calibration),cleanup_policy='retain_all',checkpoint_cleanup_completed=False,deleted_paths=[],
            checkpoint_paths_remaining=[str(x) for x in out.rglob('*.pt')])
        dump(out/'smoke_summary.json',summary);mark('complete',state='complete',passed=True)
    except BaseException as e:mark(status['stage'],state='failed',error=repr(e));raise
if __name__=='__main__':main()
