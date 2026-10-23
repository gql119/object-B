
import os,sys,json,hashlib,csv,math
from pathlib import Path
p=json.loads(sys.argv[1]);out=Path(p['output']);cfg=p['config'];defense=p['defense']
os.environ['YOLO_CONFIG_DIR']=str(out/'runtime/yolo')
import torch
torch.set_num_threads(1)
def load(f):return json.loads(Path(f).read_text())
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as fp:
  for b in iter(lambda:fp.read(1048576),b''):h.update(b)
 return h.hexdigest()
def finite(x):
 if torch.is_tensor(x):return bool(torch.isfinite(x).all())
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
 if isinstance(x,float):return math.isfinite(x)
 return True
status=load(out/'pipeline_status.json');pair=load(out/'paired_results.json');parent=load(out/'parent_audit.json')
assert status['state']==pair['state']=='complete' and parent['full_file_audit']
assert parent['all_S4_output_hashes_exact'] and parent['clean_validation_exact']
result={'remote_read_only':True,'GPU_forwards':0,'status':'passed','arms':{},'source_commit':pair['source_commit'],'paired_results_sha256':sha(out/'paired_results.json')}
for arm in ['S4','clean']:
 a=out/arm;ev=load(a/'last_primary_evaluation.json')
 hist=cfg['historical_clean'].get(defense) if arm=='clean' else None
 weights=Path(hist['root'])/'victim/weights' if hist else a/'victim_run/victim/weights'
 csvpath=Path(hist['root'])/'victim/results.csv' if hist else a/'victim_run/victim/results.csv'
 rows=list(csv.DictReader(csvpath.open()));assert len(rows)==200
 assert [int(float(next(iter(r.values())))) for r in rows]==list(range(1,201))
 d={'csv_rows':200,'last_sha256':sha(weights/'last.pt'),'best_sha256':sha(weights/'best.pt'),'clean_validation':'unchanged4952'}
 assert d['last_sha256']==ev['checkpoint_sha256']
 for role,name in [('last_primary','last.pt'),('best_secondary','best.pt')]:
  e=load(a/(role+'_evaluation.json'));assert e['checkpoint_sha256']==sha(weights/name)
  ck=torch.load(weights/name,map_location='cpu',weights_only=False)
  assert ck['train_args']['epochs']==200 and ck['train_args']['batch']==36 and ck['train_args']['workers']==16 and ck['train_args']['amp'] and ck['train_args']['seed']==0
  assert ck['train_args']['optimizer']=='SGD' and ck['train_args']['pretrained'] is False
  model=ck.get('ema') if ck.get('ema') is not None else ck.get('model');assert model is not None and finite(model.state_dict())
  embedded=ck.get('train_results');assert isinstance(embedded,dict)
  epoch_key=next(k for k in embedded if k.strip()=='epoch');embedded_rows=embedded[epoch_key]
  if role=='last_primary':assert len(embedded_rows)==200 and [int(float(x)) for x in embedded_rows]==list(range(1,201))
  d[role+'_embedded_csv_rows']=len(embedded_rows)
  d[role+'_native_epoch_field']=ck.get('epoch')
  del ck,model
 if hist:
  assert d['last_sha256']==hist['last_sha256'] and d['best_sha256']==hist['best_sha256']
  d['historical_clean_reuse']=True
 else:
  completion=load(a/'victim_completion.json');assert completion['last_sha256']==d['last_sha256'] and completion['best_sha256']==d['best_sha256']
  audit=load(a/'data_audit.json');manifest=load(a/'data_manifest.json')
  assert len(manifest)==16551 and len({r['id'] for r in manifest})==16551
  assert sha(a/'data_manifest.json')==audit['manifest_sha256']
  source=cfg['poisoned_dataset_path' if arm=='S4' else 'clean_root']
  assert audit['source_dataset']==source
  for r in manifest:
   assert r['arm']==arm and r['defense']==defense and Path(r['source']).is_relative_to(Path(source)/'images/train')
   assert sha(r['output'])==r['output_sha256']
   assert sha(a/'dataset/labels/train'/(r['id']+'.txt'))==r['label_sha256']
  snapshot=a/'victim_run/victim/state_snapshots/epoch_200_full_state.pt'
  state=torch.load(snapshot,map_location='cpu',weights_only=False)
  assert state['epoch']==200 and state['automatic_resume_allowed'] is False
  assert all(k in state for k in ['model','optimizer','scheduler','scaler','rng'])
  assert all(k in state['rng'] for k in ['python','numpy','torch','cuda']) and state['optimizer']['state']
  assert all(finite(state[k]) for k in ['model','optimizer','scheduler','scaler'])
  assert state['train_args']['workers']==16 and state['train_args']['batch']==36
  d.update(dataset_output_hashes_exact=16551,dataset_manifest_sha256=audit['manifest_sha256'],epoch200_state_sha256=sha(snapshot),optimizer_scheduler_scaler_rng_complete=True,exact_prefetch_resume_claim=False)
  del state
 result['arms'][arm]=d
print(json.dumps(result,allow_nan=False))
