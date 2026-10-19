"""Serial, source-bound S4 train-data robustness. AT and noise generation absent."""
import argparse
import csv
import hashlib
import importlib.metadata
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
DOC=ROOT/'docs/S4_ROBUSTNESS_20261019'
LOCK=DOC/'S4_ROBUSTNESS_LOCK.json'
NAMES=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_name(p.name+'.writing')
    temporary.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    temporary.replace(p)

def images(folder):
    rows=[p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')]
    result={p.stem:p for p in rows}
    if len(rows)!=len(result):raise ValueError('Duplicate image ID')
    return result

def child_run(command,out,label,timeout=None):
    process=subprocess.Popen(command,cwd=ROOT)
    try:
        stat=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
        ticks=int(stat[19])
    except FileNotFoundError:ticks=None
    dump(out/'active_child.json',dict(pid=process.pid,start_ticks=ticks,label=label,parent_pid=os.getpid(),started_at_unix=time.time(),command=command))
    try:code=process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Never orphan a still-running training child or free its RunID for reuse.
        dump(out/'child_timeout.json',dict(pid=process.pid,start_ticks=ticks,label=label,timeout_seconds=timeout,action='retain binding and await existing child; no relaunch or signal'))
        code=process.wait()
    dump(out/'active_child.json',dict(pid=process.pid,start_ticks=ticks,label=label,parent_pid=os.getpid(),exit_code=code,completed_at_unix=time.time(),active=False))
    if code:raise subprocess.CalledProcessError(code,command)

def output_setup(out,c):
    boundary=Path(c['remote_root'])/'outputs'
    if not out.is_relative_to(boundary):raise ValueError('Output outside authorized robustness root')
    out.mkdir(parents=True,exist_ok=False)
    for key,name in [('YOLO_CONFIG_DIR','yolo'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','cache'),('TORCH_HOME','torch')]:
        p=out/'runtime'/name;p.mkdir(parents=True);os.environ[key]=str(p)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')

def parent_audit(c,out,full):
    parent=Path(c['parent_output']);dataset=Path(c['poisoned_dataset_path']);clean=Path(c['clean_root'])
    proof={'parent_run':c['parent_noise_run_id'],'parent_dataset':str(dataset),'full_file_audit':full,'no_noise_optimization':True}
    for key,rel in [('parent_noise_checkpoint_sha256','noise/checkpoints/step_009144_epoch_24_cursor_00000.pt'),('parent_dataset_integrity_sha256','dataset_integrity.json'),('parent_materialization_manifest_sha256','materialization_manifest.json')]:
        assert sha(parent/rel)==c[key],key
        proof[key]=c[key]
    for package,version in c['runtime_versions'].items():assert importlib.metadata.version(package)==version,(package,version)
    native=Path(sys.prefix)/'lib/python3.11/site-packages/ultralytics/engine/trainer.py';assert sha(native)==c['native_trainer_sha256']
    assert sha(ROOT/'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json')==c['frozen_s4_config_sha256']
    manifest=json.loads((parent/'materialization_manifest.json').read_text());assert len(manifest)==16551
    if full:
        for row in manifest:
            original=Path(row['source']);dest=Path(row['output'])
            assert dest.is_relative_to(dataset/'images/train')
            assert original.is_relative_to(clean/'images/train')
            assert sha(dest)==row['output_sha256']
            for root in [dataset,clean]:assert sha(root/'labels/train'/(row['id']+'.txt'))==row['label_sha256']
        for split,n in [('train',16551),('val',4952)]:
            a=images(clean/'images'/split);b=images(dataset/'images'/split)
            assert len(a)==len(b)==n and set(a)==set(b)
            for image_id in a:
                assert sha(clean/'labels'/split/(image_id+'.txt'))==sha(dataset/'labels'/split/(image_id+'.txt'))
                if split=='val':assert sha(a[image_id])==sha(b[image_id])
        proof.update(train=16551,validation=4952,labels_exact=True,clean_validation_exact=True,all_S4_output_hashes_exact=True)
    dump(out/'parent_audit.json',proof)
    return proof

def wait_admission(out):
    from scripts.run_s4_tfss_pipeline import _resource_snapshot,_resource_blockers,_sustained_cpu_or_io_contention
    while True:
        samples=[]
        for i in range(3):
            samples.append(_resource_snapshot(10.0))
            if i<2:time.sleep(30)
        passed=not any(_resource_blockers(s) for s in samples) and not _sustained_cpu_or_io_contention(samples)
        dump(out/'resource_admission.json',dict(samples=samples,passed=passed,loadavg_record_only=True))
        if passed:return
        dump(out/'training_progress.json',dict(state='WAIT_FOR_RESOURCE',no_training_started=True))
        time.sleep(900)

def data_view(c,out,defense,arm,ids=None):
    from ue_framework.s4_robustness import read_rgb,write_png,transform_rgb
    import numpy as np
    source=Path(c['clean_root'] if arm=='clean' else c['poisoned_dataset_path'])
    clean=Path(c['clean_root']);view=out/'dataset';view.mkdir(exist_ok=False)
    for kind in ['images','labels']:
        for split in ['train','val']:(view/kind/split).mkdir(parents=True)
    source_images=images(source/'images/train');selected=sorted(source_images) if ids is None else list(ids)
    rows=[];total=0
    for image_id in selected:
        p=source_images[image_id];label=source/'labels/train'/(image_id+'.txt')
        rgb=read_rgb(p);result=transform_rgb(rgb,defense,c['defenses'].get(defense))
        assert rgb.shape==result.shape and rgb.dtype==result.dtype==np.uint8
        dest=view/'images/train'/(image_id+'.png')
        if defense in ('none','data_aug'):
            dest=dest.with_suffix(p.suffix);dest.symlink_to(p)
        else:
            write_png(dest,result);decoded=read_rgb(dest);assert np.array_equal(decoded,result)
        (view/'labels/train'/label.name).symlink_to(label)
        assert sha(label)==sha(clean/'labels/train'/label.name)
        total+=dest.stat().st_size
        rows.append(dict(id=image_id,arm=arm,defense=defense,source=str(p),source_sha256=sha(p),output=str(dest),output_sha256=sha(dest),label_sha256=sha(label),height=rgb.shape[0],width=rgb.shape[1],decoded_pixels_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
    for p in images(clean/'images/val').values():(view/'images/val'/p.name).symlink_to(p)
    for p in (clean/'labels/val').glob('*.txt'):(view/'labels/val'/p.name).symlink_to(p)
    assert len(images(view/'images/val'))==4952
    dump(out/'data_manifest.json',rows)
    audit=dict(state='passed',defense=defense,arm=arm,count=len(rows),source_dataset=str(source),parent_S4_manifest_sha256=c['parent_materialization_manifest_sha256'],geometry_unchanged=True,labels_byte_identical=True,decoded_PNG_exact=True,validation='original4952clean; transform never applied',output_bytes=total,manifest_sha256=sha(out/'data_manifest.json'))
    dump(out/'data_audit.json',audit)
    yaml=out/'data.yaml';yaml.write_text(f'path: {view}\ntrain: images/train\nval: images/val\nnames:\n'+'\n'.join(f'  {i}: {n}' for i,n in enumerate(NAMES))+'\n')
    return yaml,audit

class Boundary(Exception):pass

def local_amp_check(model):
    """Historical defense AMP selfcheck; no external weights or optimizer calls."""
    import torch
    was_training=model.training
    try:
        model.eval()
        sample=torch.zeros((1,3,256,256),dtype=torch.float32,device=next(model.parameters()).device)
        with torch.inference_mode():
            full=model(sample)
            with torch.autocast(device_type='cuda',enabled=True):mixed=model(sample)
        full=full[0] if isinstance(full,(tuple,list)) else full
        mixed=mixed[0] if isinstance(mixed,(tuple,list)) else mixed
        return bool(full.shape==mixed.shape and torch.isfinite(full).all() and torch.isfinite(mixed).all() and torch.allclose(full.float(),mixed.float(),atol=.5))
    finally:model.train(was_training)

def train(c,out,data,*,smoke=False,reference=False,synthetic=False,limit=10):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.engine import trainer as native_trainer
    from ue_framework.s4_victim import S4VictimTrainer,attach_safety_callbacks
    from ue_framework.s4_amp_observer import attach_amp_observer
    assert S4VictimTrainer.optimizer_step is DetectionTrainer.optimizer_step
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    args=dict(c['victim_protocol']['train_args']);args.update(data=str(data),project=str(out/'victim_run'),name='victim',patience=0)
    if smoke:args.update(val=False,save=False)
    defense=json.loads((out/'data_audit.json').read_text())['defense']
    if defense=='data_aug':
        overrides=c['defenses']['data_aug']['overrides'];assert overrides==dict(degrees=10.,shear=2.)
        args.update(**overrides)
    init=ROOT/'configs/voc_yolov8n_20cls.yaml';assert sha(init)==c['victim_protocol']['init_yaml_sha256']
    assert args['batch']==36 and args['workers']==16 and args['amp'] is True and args['epochs']==200 and args['seed']==0 and args['pretrained'] is False
    dump(out/'victim_run_spec.json',dict(run_id=out.parent.name+'__'+out.name,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),init_yaml_sha256=sha(init),train_args=args,defense=defense,smoke=smoke))
    model=YOLO(str(init))
    # Same selfcheck used by historical Clean defenses. Native optimizer_step,
    # scaler/clip/skip/update sequencing is never replaced.
    native_trainer.check_amp=local_amp_check
    holder=attach_amp_observer(model,out/'amp_audit',enforce=True,debug_hashes=True) if reference else attach_safety_callbacks(model,amp_log_dir=out/'amp_audit',debug_hashes=smoke)
    forced={'batch':None,'updates_before':None,'recovered':False,'actual_skip_batch':None}
    def verify_amp(t):
        assert bool(t.amp),'AMP disabled by hardware selfcheck; do not change formal protocol'
        dump(out/'AMP_SELF_CHECK.json',dict(passed=True,external_weights=False,source='historical run_robustness_victim.local_amp_check',native_optimizer_inherited=True))
    model.add_callback('on_train_start',verify_amp)
    if synthetic:
        def inject(t):
            original=t.model.loss
            def loss(batch,preds=None):
                result=original(batch,preds);o=holder['observer']
                if forced['batch'] is None and o.updates>=1:
                    forced['batch']=o.total_batches;forced['updates_before']=o.updates
                    return result[0]*1e20,result[1]
                return result
            t.model.loss=loss
        model.add_callback('on_train_start',inject)
    if smoke:
        def forbid_smoke_validation(t):raise AssertionError('Smoke validation/inference metrics are disabled')
        model.add_callback('on_val_start',forbid_smoke_validation)
        def boundary(t):
            o=holder['observer']
            if synthetic and forced['batch'] is not None:
                attempts=[row for row in o.rows[forced['batch']-1:] if row['optimizer_attempt_observed']]
                if attempts:
                    row=attempts[0]
                    assert row['native_scaler_nonfinite_skip'] and row['backoff'] and row['parameters_unchanged_on_skip']
                    forced['actual_skip_batch']=row['total_batches']
                    forced['recovered']=o.updates>forced['updates_before']
            # Limits count actual batches, not optimizer attempts. Accumulation
            # batches do not constitute a failed AMP attempt or a skipped update.
            ready=synthetic and forced['recovered'] or not synthetic and defense!='none' and o.updates>=1 and o.total_batches>=5
            if o.total_batches>=limit or ready:
                if synthetic:assert forced['recovered'],'Synthetic overflow did not recover within bound'
                if not reference and defense!='none':assert o.updates>=1,'No successful native AMP update within smoke bound'
                assert bool(t.amp),'AMP unexpectedly disabled'
                assert all(r['raw_loss_finite'] and r['model_finite'] and r['EMA_finite'] and r['optimizer_finite'] for r in o.rows)
                torch.save(dict(model=t.model.state_dict(),optimizer=t.optimizer.state_dict(),scheduler=t.scheduler.state_dict(),scaler=t.scaler.state_dict()),out/'retained_smoke_state.pt')
                dump(out/'trial_summary.json',dict(state='passed',rows=o.rows,initial=o.initial,batches=o.total_batches,actual_updates=o.updates,skips=o.skips,forced=forced,AMP=True,finite_loss=True,validation_forward_batches=0,official_metrics_disabled=True,artifact_ingest_disabled=True))
                raise Boundary()
        model.add_callback('on_train_batch_end',boundary)
    else:
        def retain(t):
            p=Path(t.save_dir)/'retained_robustness_epochs'/f'epoch_{t.epoch+1:03d}.pt';p.parent.mkdir(exist_ok=True)
            if p.exists():raise FileExistsError(p)
            shutil.copyfile(t.last,p)
            dump(out/'training_progress.json',dict(epoch=t.epoch+1,AMP=bool(t.amp),data_loader_workers=t.args.workers,data_loader_worker_rng='not independently serialized; no exact automatic resume',last_sha256=sha(t.last)))
        model.add_callback('on_model_save',retain)
    try:model.train(trainer=DetectionTrainer if reference else S4VictimTrainer,**args)
    except Boundary:
        if smoke:return
        raise
    if smoke:raise AssertionError('Smoke exited without bounded receipt')
    rows=list(csv.DictReader((out/'victim_run/victim/results.csv').open()))
    assert len(rows)==200 and [int(float(next(iter(r.values())))) for r in rows]==list(range(1,201))
    assert all(np.isfinite(float(v)) for row in rows for k,v in row.items() if 'loss' in k)
    victim=out/'victim_run/victim/weights'
    dump(out/'victim_completion.json',dict(epochs=200,last_sha256=sha(victim/'last.pt'),best_sha256=sha(victim/'best.pt')))

def evaluate(weights,data,out,role):
    from ultralytics import YOLO
    from scripts.run_s4_tfss_pipeline import summarize
    result=YOLO(str(weights)).val(data=str(data),imgsz=640,batch=36,workers=16,device='0',verbose=False,plots=False,seed=0,project=str(out/'evaluation'),name=role,exist_ok=False)
    m=dict(summarize(result.box),class_names=NAMES,checkpoint_sha256=sha(weights),checkpoint_role=role)
    dump(out/(role+'_evaluation.json'),m)
    return m

def trial(c,out,args):
    output_setup(out,c)
    records=json.loads((ROOT/'docs/tfss/FIXED_TRAIN_SAMPLE_MANIFEST.json').read_text())['records']
    ids=[r['image_id'] for r in records]
    assert len(ids)==32 and len(set(ids))==32
    for r in records:
        assert sha(r['image_path'])==r['image_sha256'] and sha(r['label_path'])==r['label_sha256']
    data,audit=data_view(c,out,args.defense,args.arm,ids)
    train(c,out,data,smoke=True,reference=args.reference,synthetic=args.synthetic,limit=args.limit)

def smoke(c,out,group):
    trials=[]
    plans={
      'first':[('native_S4','none','S4',15,['--reference']),('candidate_S4','none','S4',15,[])]+[(f'jpeg50_{arm}','jpeg50',arm,20,[]) for arm in ['S4','clean']]+[('overflow_S4','none','S4',20,['--synthetic'])],
      'second':[(f'{d}_{arm}',d,arm,15,[]) for d in ['jpeg30','gray','bdr'] for arm in ['S4','clean']],
      'phaseB':[(f'{d}_S4',d,'S4',20,[]) for d in c['phase_B']],
    }
    for name,defense,arm,limit,extra in plans[group]:
        cmd=[sys.executable,'-B',__file__,'--phase','trial','--output',str(out/name),'--defense',defense,'--arm',arm,'--limit',str(limit),*extra]
        child_run(cmd,out,name,timeout=1800)
        r=json.loads((out/name/'trial_summary.json').read_text());trials.append(dict(name=name,batches=r['batches'],actual_updates=r['actual_updates'],skips=r['skips']))
        dump(out/'smoke_progress.json',trials)
    if group=='first':
        a=json.loads((out/'native_S4/trial_summary.json').read_text());b=json.loads((out/'candidate_S4/trial_summary.json').read_text())
        assert a['actual_updates']>=1 and b['actual_updates']>=1
        assert a['initial']==b['initial']
        for x,y in zip(a['rows'],b['rows']):
            for k in ['input_tensor_sha256','input_ids','raw_loss','state','optimizer_update_executed','amp_scale_before','amp_scale_after']:assert x[k]==y[k],k
    assert sum(r['batches'] for r in trials)<=100
    dump(out/'smoke_summary.json',dict(state='passed',group=group,trials=trials,total_actual_batches=sum(r['batches'] for r in trials),baseline_equivalence_max_difference=0. if group=='first' else None,input_tensor_hashes_exact=True if group=='first' else None,finite_loss=True,production_entrypoint_reached=True,AMP=True,validation_forward_batches=0,official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all',deleted_paths=[]))

def formal(c,out,defense):
    assert defense in c['phase_A']+c['phase_B']
    receipt=json.loads((Path(c['remote_root'])/'receipts/PRERUN_CLOSURE.json').read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert receipt['pre_run_result']=='pass' and receipt['pre_run_code_commit']==commit
    assert receipt['config_sha256']==sha(LOCK)
    parent_audit(c,out,True)
    result={};audits={}
    for arm in ['S4','clean']:
        # Stage isolation: only the bound child owns CUDA; each child exits before next.
        armout=out/arm
        historical=c['historical_clean'].get(defense) if arm=='clean' else None
        if historical:
            armout.mkdir(exist_ok=False)
            data,audit=data_view(c,armout,'none','clean',ids=[])
            weights=Path(historical['root'])/'victim/weights/last.pt';assert sha(weights)==historical['last_sha256']
            m=evaluate(weights,data,armout,'last_primary')
            old=json.loads((Path(historical['root'])/'last_primary_clean_val_metrics.json').read_text())
            for i,name in enumerate(NAMES):
                assert abs(m['AP50_per_class'][i]-old['AP50_per_class'][name])<=1e-7,'Historical clean AP50 evaluation mismatch'
                assert abs(m['Recall_per_class'][i]-old['Recall_per_class'][name])<=1e-7,'Historical clean Recall evaluation mismatch'
            best=Path(historical['root'])/'victim/weights/best.pt';assert sha(best)==historical['best_sha256']
            evaluate(best,data,armout,'best_secondary')
            result[arm]=m;audits[arm]=dict(reused_clean=True,**historical)
        else:
            cmd=[sys.executable,'-B',__file__,'--phase','arm','--output',str(armout),'--defense',defense,'--arm',arm]
            child_run(cmd,out,arm)
            result[arm]=json.loads((armout/'last_primary_evaluation.json').read_text())
            audits[arm]=json.loads((armout/'data_audit.json').read_text())
    from ue_framework.s4_robustness import paired_scores
    pair=paired_scores(result['clean'],result['S4'])
    dump(out/'paired_results.json',dict(defense=defense,state='complete',source_commit=commit,config_sha256=sha(LOCK),parent_S4_manifest_sha256=c['parent_materialization_manifest_sha256'],metrics=result,scores=pair,audits=audits,primary='last.pt@200',secondary='best.pt',single_seed_descriptive=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','trial','formal','arm'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--defense',default='none');p.add_argument('--arm',choices=['S4','clean'],default='S4');p.add_argument('--reference',action='store_true');p.add_argument('--synthetic',action='store_true');p.add_argument('--limit',type=int,default=10);p.add_argument('--smoke-group',choices=['first','second','phaseB'],default='first');a=p.parse_args()
    c=json.loads(LOCK.read_text())
    if a.phase=='trial':trial(c,a.output,a);return
    output_setup(a.output,c)
    dump(a.output/'config_snapshot.json',c)
    status=dict(state='running',stage=a.phase,run_id=a.output.name,defense=a.defense,started_at_unix=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config_sha256=sha(LOCK))
    dump(a.output/'pipeline_status.json',status)
    try:
        if a.phase=='smoke':
            parent_audit(c,a.output,False);smoke(c,a.output,a.smoke_group)
        elif a.phase=='formal':formal(c,a.output,a.defense)
        else:
            assert a.defense in c['phase_A']+c['phase_B']
            parent_audit(c,a.output,False)
            data,audit=data_view(c,a.output,a.defense,a.arm)
            assert audit['count']==16551
            wait_admission(a.output)
            train(c,a.output,data)
            weights=a.output/'victim_run/victim/weights'
            for role,name in [('last_primary','last.pt'),('best_secondary','best.pt')]:evaluate(weights/name,data,a.output,role)
        status.update(state='complete',stage='complete',completed_at_unix=time.time());dump(a.output/'pipeline_status.json',status)
    except BaseException as e:
        status.update(state='failed',error=repr(e),failed_at_unix=time.time());dump(a.output/'pipeline_status.json',status);raise

if __name__=='__main__':main()
