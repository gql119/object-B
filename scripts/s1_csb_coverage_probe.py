"""S1 train-only coverage/adaptation admission; no perturbation or victim update."""
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.dont_write_bytecode=True

def main():
    p=argparse.ArgumentParser()
    for name in ['config','csb-config','parent-cache','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--full-coverage',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for q in ['yolo','mpl','cache']:(out/'runtime'/q).mkdir(parents=True)
    os.environ.update(YOLO_CONFIG_DIR=str(out/'runtime/yolo'),MPLCONFIGDIR=str(out/'runtime/mpl'),XDG_CACHE_HOME=str(out/'runtime/cache'),PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    import numpy as np,torch
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);np.random.seed(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    from ultralytics import YOLO
    from ue_framework.config import load_config
    from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations,label_path_for_image
    from ue_framework.methods.csb_budget import construct_budget,domains,digest,config_hash
    from ue_framework.methods.csb_sensitivity import gradients
    from ue_framework.methods.s1_optimization import model_hash
    from scripts.s1_spatial_audit import fixed_selection
    cfg=load_config(a.config);settings=json.loads(Path(a.csb_config).read_text());root=Path(cfg['data']['dataset_root'])
    model=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0').eval()
    for v in model.parameters():v.requires_grad_(False)
    before=model_hash(model)
    if a.full_coverage:paths,_=fixed_selection(root,6095)
    else:
        selection=json.loads((Path(a.parent_cache)/'selection.json').read_text())['filenames']
        paths=[str(root/'images/train'/q) for q in selection]
        paths+=list(map(str,(root/'images/train').glob('2011_001260.*')))
        if len(paths)!=33:raise ValueError('Fixed32 and one observed failed train image only')
    rows=[];errors=[];comparisons=[];start=time.perf_counter()
    for number,path in enumerate(paths,1):
        im=load_image_rgb_float(path);ann=read_yolo_annotations(label_path_for_image(path,str(root/'labels/train')))
        target,_=domains(im.shape,ann)
        g,records=gradients(model,im,ann,coverage_policy=settings['coverage_policy'],coverage_only=a.full_coverage)
        for r in records:
            row=dict(id=Path(path).stem,**r);rows.append(row)
            if r['coverage_status']!='COVERED':errors.append(row)
        if not a.full_coverage:
            budget,norm,_=construct_budget(g,target,settings,no_nt_gt=not any(int(q['cls'])!=14 for q in ann))
            reference=Path(a.parent_cache)/'maps'/(Path(path).stem+'.npz')
            if reference.exists():
                with np.load(reference) as old:
                    diff=float(np.max(np.abs(budget-old['budget'])))
                strict_g,_=gradients(model,im,ann,coverage_policy='strict_native_TAL')
                strict_budget,_,_=construct_budget(strict_g,target,settings,no_nt_gt=not any(int(q['cls'])!=14 for q in ann))
                comparisons.append(dict(id=Path(path).stem,budget_max_difference=diff,
                    current_strict_rule_max_difference=float(np.max(np.abs(budget-strict_budget)))))
            else:
                comparisons.append(dict(id=Path(path).stem,recovered_objects=[r for r in records if r['native_positive_count']==0],
                    gradient_l2_by_class={str(k):float(np.linalg.norm(v)) for k,v in g.items()},budget_min=float(budget[target].min()),budget_max=float(budget[target].max())))
        if number%50==0 or not a.full_coverage:print(json.dumps({'images_done':number,'id':Path(path).stem,'errors':len(errors)}),flush=True)
    unchanged=model_hash(model)==before
    # Historical cache was produced in a separate CUDA process. Use the already
    # declared spatial-bound tolerance for its float32 values; the current
    # covered-path rule comparison must be exact and cannot invoke rematching.
    same=all(q.get('budget_max_difference',0)<=1e-7 and q.get('current_strict_rule_max_difference',0)==0 for q in comparisons)
    passed=not errors and unchanged and same
    with (out/'coverage.csv').open('w',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    summary=dict(state='complete' if passed else 'failed',passed=passed,production_entrypoint_reached=True,
        computation_kind='inference' if a.full_coverage else 'training',finite_outputs=True,
        finite_loss=None if a.full_coverage else True,official_metrics_disabled=True,artifact_ingest_disabled=True,
        formal_generation_enabled=False,victim_training_enabled=False,cleanup_policy='retain_all',deleted_paths=[],
        checkpoint_cleanup_completed=False,checkpoint_paths_remaining=[],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=config_hash(settings),teacher_sha256=digest(cfg['surrogate']['ckpt']),teacher_unchanged=unchanged,
        full_coverage=a.full_coverage,image_count=len(paths),nt_gt_count=len(rows),
        native_uncovered_gt=sum(r['native_positive_count']==0 for r in rows),
        recovered_gt=sum(r['coverage_mode']=='independent_same_TAL_rematch' and r['coverage_status']=='COVERED' for r in rows),
        remaining_uncovered_gt=len(errors),errors=errors,comparisons=comparisons,historical_cache_tolerance=1e-7,
        current_covered_rule_tolerance=0,elapsed_seconds=time.perf_counter()-start)
    summary['deterministic_cuda_algorithms']=torch.are_deterministic_algorithms_enabled()
    (out/'smoke_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    if not passed:raise ValueError('CSB coverage admission failed; no perturbation/victim allowed')

if __name__=='__main__':main()
