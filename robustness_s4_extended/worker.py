"""Deferred scientific child. All GPU imports occur after predecessor acceptance."""
import argparse,csv,copy,json,math,os,random,sys,time
from pathlib import Path
from .common import ROOT,CONFIG,load,sha,atomic,commit,verify_source
from .metrics import paired

def check_environment(c):
    import importlib.metadata
    for k,v in c['baseline']['runtime_versions'].items():
        if importlib.metadata.version(k)!=v:raise ValueError('Pinned environment differs: '+k)
    u=Path(sys.prefix)/'lib/python3.11/site-packages/ultralytics'
    for rel,rec in c['native_sources'].items():
        if sha(u/rel)!=rec['sha256']:raise ValueError('Native implementation drift: '+rel)

def accepted_predecessor(c):
    receipt=load(Path(c['output_root'])/'PREDECESSOR_ACCEPTANCE.json')
    p=c['predecessor']
    if receipt['state']!='COMPLETED_AND_ACCEPTED' or receipt['run_id']!=p['run_id']:raise ValueError('Predecessor gate missing')
    s=load(Path(p['output'])/'pipeline_status.json')
    if s['state']!='complete' or sha(Path(p['output'])/'pipeline_status.json')!=receipt['native_state_sha256']:raise ValueError('Predecessor acceptance stale')
    # Acceptance includes actual model/CSV/manifest hashes, not process disappearance or idle GPU.
    return receipt

def review_gate(c):
    r=load(c['review_receipt'])
    if (r.get('source_commit'),r.get('config_sha256'),r.get('pre_run_result'),r.get('reviewer_count'))!=(commit(),sha(CONFIG),'pass',1):raise ValueError('Current-source scientific review required')
    for name in ['cutout','cutmix','mixup']:
        path=Path(c['output_root'])/'experiments'/('smoke_'+name)/'smoke_summary.json'
        s=load(path)
        if not s['finite_loss'] or s['source_commit']!=commit() or r['smoke_sha256'][name]!=sha(path):raise ValueError('Current source GPU evidence required')
    return r

def runtime_config(c):
    x=copy.deepcopy(c['baseline']);x['remote_root']=c['root'];return x

def no_op_probe(c,data,out):
    """Actual native vs disabled candidate dataset + CUDA loss on identical random weights."""
    import numpy as np, torch, yaml
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data import build
    from .integration import configure, ExtendedDataset
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    cfg=get_cfg(overrides=dict(c['baseline']['victim_protocol']['train_args'],data=str(data),mixup=0.,cutmix=0.))
    meta=yaml.safe_load(Path(data).read_text());cfg.task='detect'
    path=str(Path(meta['path'])/'images/train')
    ref=build_yolo_dataset(cfg,path,36,meta,mode='train',stride=32)
    zero=copy.deepcopy(c['defenses'][out.parent.name.replace('smoke_','')]);zero['probability']=0.
    configure(zero,out/'noop_audit')
    original=build.YOLODataset
    try:
        build.YOLODataset=ExtendedDataset;candidate=build_yolo_dataset(cfg,path,36,meta,mode='train',stride=32)
    finally:build.YOLODataset=original
    random.seed(0);np.random.seed(0);a=ref[0]
    random.seed(0);np.random.seed(0);b=candidate[0]
    for key in ['img','bboxes','cls','batch_idx']:
        if not torch.equal(a[key],b[key]):raise ValueError('Disabled native input/label equivalence failed: '+key)
    batches=[]
    for dataset,item in [(ref,a),(candidate,b)]:
        batch=dataset.collate_fn([item]);batch['img']=batch['img'].cuda().float()/255
        batches.append({k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()})
    model=YOLO(str(ROOT/'configs/voc_yolov8n_20cls.yaml')).model.cuda();model.train()
    model.args=vars(cfg)
    # Explicit dict args are converted by Ultralytics criteria via attribute access.
    from ultralytics.utils import IterableSimpleNamespace
    model.args=IterableSimpleNamespace(**vars(cfg))
    state={k:v.clone() for k,v in model.state_dict().items()}
    initial_weights=out/'baseline_preupdate_state.pt';torch.save(state,initial_weights)
    with torch.no_grad():left=float(model.loss(batches[0])[0].sum())
    model.load_state_dict(state)
    with torch.no_grad():right=float(model.loss(batches[1])[0].sum())
    if not math.isfinite(left) or left!=right:raise ValueError('Disabled native loss equivalence failed')
    atomic(out/'baseline_equivalence.json',dict(reference_id='pinned native YOLODataset and YOLOv8n loss',candidate_id='ExtendedDataset defense probability0, same YOLOv8n loss',reference_metric=left,candidate_metric=right,absolute_difference=abs(left-right),tolerance=0,metric_name='pre-update detection loss sum',actual_native_source=c['native_sources'],sample_count=1,image_and_labels_exact=True,reference_weights_path=str(initial_weights),candidate_weights_path=str(initial_weights),weights_sha256=sha(initial_weights),seed=0,weights='identical pre-update randomYAML state',official_metrics_disabled=True))
    del model,ref,candidate,state,batch,batches
    torch.cuda.empty_cache()

def train(c,defense,arm,out,smoke=False):
    from scripts import run_s4_robustness as old
    x=runtime_config(c);old.output_setup(out,x)
    ids=None
    if smoke:
        records=load(ROOT/'docs/tfss/FIXED_TRAIN_SAMPLE_MANIFEST.json')['records'];ids=[r['image_id'] for r in records]
    data,audit=old.data_view(x,out,'none',arm,ids=ids)
    audit.update(defense=defense,online=True,transform=c['defenses'][defense],input_identity='ORIGINAL_S4' if arm=='S4' else 'ORIGINAL_CLEAN')
    atomic(out/'data_audit.json',audit)
    if smoke and arm=='clean':no_op_probe(c,data,out)
    import numpy as np,torch
    from ultralytics import YOLO
    from ultralytics.engine import trainer as native
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ue_framework.s4_victim import attach_safety_callbacks
    from .integration import ExtendedTrainer,configure,epoch_callback
    assert ExtendedTrainer.optimizer_step is DetectionTrainer.optimizer_step
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    args=copy.deepcopy(x['victim_protocol']['train_args'])
    args.update(data=str(data),project=str(out/'victim_run'),name='victim',mixup=c['defenses'][defense]['probability'] if defense=='mixup' else 0.,cutmix=c['defenses'][defense]['probability'] if defense=='cutmix' else 0.)
    if smoke:args.update(val=False,save=False)
    if (args['epochs'],args['batch'],args['workers'],args['seed'],args['amp'],args['optimizer'],args['pretrained'])!=(200,36,16,0,True,'SGD',False):raise ValueError('Frozen training protocol changed')
    init=ROOT/'configs/voc_yolov8n_20cls.yaml'
    if sha(init)!=x['victim_protocol']['init_yaml_sha256']:raise ValueError('Initialization changed')
    atomic(out/'victim_run_spec.json',dict(source_commit=commit(),config_sha256=sha(CONFIG),arm=arm,defense=defense,train_args=args,init_yaml_sha256=sha(init),transform=c['defenses'][defense],validation='unchanged4952clean',smoke=smoke))
    configure(c['defenses'][defense],out/'transform_audit')
    model=YOLO(str(init));native.check_amp=old.local_amp_check
    holder=attach_safety_callbacks(model,amp_log_dir=out/'amp_audit',debug_hashes=smoke)
    model.add_callback('on_train_epoch_start',epoch_callback)
    def started(t):
        if not bool(t.amp):raise ValueError('AMP disabled; do not change formal protocol')
    model.add_callback('on_train_start',started)
    class EndSmoke(Exception):pass
    if smoke:
        def no_validation(t):raise AssertionError('Smoke validation forbidden')
        def boundary(t):
            obs=holder['observer']
            if obs.total_batches>=c['smoke_batches_per_arm'] or obs.total_batches>=5 and obs.updates>=1:
                if obs.updates<1 or not all(r['raw_loss_finite'] and r['model_finite'] and r['EMA_finite'] and r['optimizer_finite'] for r in obs.rows):raise ValueError('Smoke missing finite native update')
                torch.save(dict(model=t.model.state_dict(),optimizer=t.optimizer.state_dict(),scheduler=t.scheduler.state_dict(),scaler=t.scaler.state_dict()),out/'retained_smoke_state.pt')
                atomic(out/'trial_summary.json',dict(state='passed',finite_loss=True,batches=obs.total_batches,actual_updates=obs.updates,skips=obs.skips,validation_forward_batches=0,source_commit=commit(),config_sha256=sha(CONFIG),official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all'))
                raise EndSmoke()
        model.add_callback('on_val_start',no_validation);model.add_callback('on_train_batch_end',boundary)
    try:model.train(trainer=ExtendedTrainer,**args)
    except EndSmoke:
        if smoke:return
        raise
    if smoke:raise ValueError('Bounded smoke boundary not reached')
    accept_arm(c,out,defense,arm,evaluate=True)

def accept_arm(c,out,defense,arm,evaluate=False):
    import torch
    from scripts import run_s4_robustness as old
    torch.set_num_threads(1)
    history=list(csv.DictReader((out/'victim_run/victim/results.csv').open()))
    if len(history)!=200 or [int(float(next(iter(r.values())))) for r in history]!=list(range(1,201)):raise ValueError('Incomplete epoch200')
    weights=out/'victim_run/victim/weights';proof={}
    spec=load(out/'victim_run_spec.json')
    if (spec['source_commit'],spec['config_sha256'],spec['arm'],spec['defense'])!=(commit(),sha(CONFIG),arm,defense):raise ValueError('Training source/defense attribution differs')
    manifest=load(out/'data_manifest.json')
    if len(manifest)!=16551 or len({r['id'] for r in manifest})!=16551:raise ValueError('Incomplete source dataset')
    source=Path(c['baseline']['clean_root'] if arm=='clean' else c['baseline']['poisoned_dataset_path'])
    for row in manifest:
        if not Path(row['source']).is_relative_to(source/'images/train'):raise ValueError('Wrong poison source')
        if sha(row['output'])!=row['output_sha256'] or sha(row['source'])!=row['source_sha256']:raise ValueError('Source image changed during training')
        if sha(out/'dataset/labels/train'/(row['id']+'.txt'))!=row['label_sha256']:raise ValueError('Training GT changed')
    for name,role in [('last.pt','last_primary'),('best.pt','best_secondary')]:
        path=weights/name;ck=torch.load(path,map_location='cpu',weights_only=False)
        if not all(torch.isfinite(v).all() for v in ck['model'].state_dict().values() if torch.is_tensor(v)):raise ValueError('Nonfinite checkpoint')
        if name=='last.pt':
            ep=next(v for k,v in ck['train_results'].items() if k.strip()=='epoch')
            if len(ep)!=200 or int(float(ep[-1]))!=200:raise ValueError('Missing embedded200')
        del ck
        if evaluate:old.evaluate(path,out/'data.yaml',out,role)
        m=load(out/(role+'_evaluation.json'));paired(m,m)
        if sha(path)!=m['checkpoint_sha256']:raise ValueError('Metric checkpoint mismatch')
        proof[name]=sha(path)
    snap=out/'victim_run/victim/state_snapshots/epoch_200_full_state.pt'
    if not snap.is_file():raise ValueError('Full optimizer/RNG state missing')
    snapshot=torch.load(snap,map_location='cpu',weights_only=False)
    if snapshot.get('epoch')!=200 or not all(k in snapshot for k in ('optimizer','scheduler','rng','scaler')):raise ValueError('Final native training state missing')
    def finite(v):
        if torch.is_tensor(v):return bool(torch.isfinite(v).all())
        if isinstance(v,dict):return all(finite(q) for q in v.values())
        if isinstance(v,(list,tuple)):return all(finite(q) for q in v)
        if isinstance(v,float):return math.isfinite(v)
        return True
    if not finite({k:snapshot[k] for k in ['model','optimizer','scheduler','scaler']}):raise ValueError('Nonfinite final training state')
    del snapshot
    rec=dict(state='ACCEPTED',arm=arm,defense=defense,epochs=200,checkpoints=proof,source_commit=commit(),config_sha256=sha(CONFIG),input_manifest_sha256=sha(out/'data_manifest.json'),parent_S4_manifest_sha256=c['baseline']['parent_materialization_manifest_sha256'],validation='original4952clean',metrics_sha256=sha(out/'last_primary_evaluation.json'))
    atomic(out/'ACCEPTANCE.json',rec)
    return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['validate','smoke','arm','accept'],required=True);p.add_argument('--defense',choices=['cutout','cutmix','mixup']);p.add_argument('--arm',choices=['clean','S4']);p.add_argument('--output',type=Path);a=p.parse_args()
    c=load(CONFIG);verify_source(c);accepted_predecessor(c);check_environment(c)
    from scripts import run_s4_robustness as old
    if a.output and not a.output.resolve().is_relative_to(Path(c['root'])/'outputs'):raise ValueError('Output outside isolated root')
    if a.action=='validate':
        out=Path(c['output_root'])/'validation';out.mkdir(exist_ok=False)
        old.parent_audit(runtime_config(c),out,True)
        atomic(out/'CPU_IDENTITY_PASS.json',dict(source_commit=commit(),config_sha256=sha(CONFIG),passed=True))
    elif a.action=='smoke':
        a.output.mkdir(parents=True,exist_ok=False)
        for arm in ['clean','S4']:
            old.wait_admission(a.output);train(c,a.defense,arm,a.output/arm,True)
        trials=[load(a.output/arm/'trial_summary.json') for arm in ['clean','S4']]
        atomic(a.output/'smoke_summary.json',dict(source_commit=commit(),config_sha256=sha(CONFIG),finite_loss=True,total_actual_batches=sum(t['batches'] for t in trials),additional_preupdate_loss_forwards=2,total_production_units=sum(t['batches'] for t in trials)+2,trials=trials,baseline_equivalence=load(a.output/'clean/baseline_equivalence.json'),official_metrics_disabled=True,artifact_ingest_disabled=True,cleanup_policy='retain_all'))
    elif a.action=='arm':
        review_gate(c);old.wait_admission(Path(c['output_root']));train(c,a.defense,a.arm,a.output)
    else:
        review_gate(c)
        records={arm:accept_arm(c,a.output/arm,a.defense,arm) for arm in ['clean','S4']}
        clean=load(a.output/'clean/last_primary_evaluation.json');s4=load(a.output/'S4/last_primary_evaluation.json')
        scores=paired(clean,s4)
        atomic(a.output/'paired_results.json',dict(defense=a.defense,source_commit=commit(),config_sha256=sha(CONFIG),metrics=dict(clean=clean,S4=s4),scores=scores,primary='last.pt@200',secondary='best.pt',single_seed_descriptive=True))
        atomic(a.output/'ACCEPTANCE.json',dict(state='ACCEPTED',records=records,paired_sha256=sha(a.output/'paired_results.json')))
if __name__=='__main__':main()
