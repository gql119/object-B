"""Real native CPU few-batch validation, synthetic images, never a COCO result."""
from pathlib import Path
import datetime, hashlib, json, os, sys
R=Path(__file__).resolve().parents[1]
EX=R/'execution'
EX.mkdir(exist_ok=True)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
O=EX/('teacher_cpu_probe_'+stamp)
O.mkdir(exist_ok=False)
os.environ['YOLO_CONFIG_DIR']=str(O/'yolo_settings')
os.environ['WANDB_MODE']='disabled'
sys.path.insert(0,str(R))
import numpy as np
from PIL import Image
import torch
import yaml
torch.set_num_threads(1)
from coco_s4.teacher import COCOTrainOnlyTeacher,teacher_overrides
from ue_framework.s4_victim import S4VictimTrainer,attach_safety_callbacks
from ultralytics.models.yolo.detect.train import DetectionTrainer

assert COCOTrainOnlyTeacher.optimizer_step is S4VictimTrainer.optimizer_step
assert COCOTrainOnlyTeacher.preprocess_batch is DetectionTrainer.preprocess_batch
source_sha=hashlib.sha256((R/'coco_s4/teacher.py').read_bytes()).hexdigest()
dataset=O/'synthetic_dataset'
(dataset/'images/train').mkdir(parents=True)
(dataset/'labels/train').mkdir(parents=True)
rng=np.random.default_rng(29)
for i in range(72):
    Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(dataset/f'images/train/{i:012d}.jpg')
    (dataset/f'labels/train/{i:012d}.txt').write_text('0 0.4 0.4 0.3 0.3\n79 0.7 0.7 0.2 0.2\n',encoding='utf-8')
mapping=json.loads((R/'docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json').read_text())['mapping']
dy=O/'teacher_train_only.yaml'
dy.write_text(yaml.safe_dump(dict(path=str(dataset),train='images/train',val=None,nc=80,names={x['yolo_class_id']:x['name'] for x in mapping}),sort_keys=False),encoding='utf-8')
cfg=yaml.safe_load((R/'docs/S4_COCO_MINITRAIN_20261023/S4_COCO_EXPERIMENT_CONFIG.yaml').read_text())
cfg['victim']['init']=str(R/cfg['victim']['init'])
overrides=teacher_overrides(cfg,dy,O/'train')
overrides.update(epochs=1,imgsz=64,workers=0,amp=False,cache=False,close_mosaic=0,save_period=-1,device='cpu')
trainer=COCOTrainOnlyTeacher(overrides=overrides,cpu_probe=True)
holder=attach_safety_callbacks(trainer,amp_log_dir=O/'amp_audit',debug_hashes=False)
trainer.train()
audit=json.loads((O/'train/TEACHER_TRAIN_ONLY_AUDIT.json').read_text())
assert audit['train_image_count']==72 and audit['validation_image_count']==0
assert audit['validation_loader_requests_blocked']>=1 and audit['native_validate_dispatches_blocked']==1 and audit['final_eval_dispatches_blocked']==1
assert audit['validation_forwards']==audit['EMA_forwards']==0 and trainer.validator.calls==0
assert trainer.model.nc==80 and trainer.model.names[0]=='person'
assert isinstance(trainer.optimizer,torch.optim.SGD)
assert all(torch.isfinite(x).all() for x in trainer.model.state_dict().values() if torch.is_tensor(x))
csv=(O/'train/results.csv').read_text();assert len(csv.strip().splitlines())==2
assert 'val/' not in csv and 'metrics/' not in csv
last=O/'train/weights/last.pt'
ckpt=torch.load(last,map_location='cpu',weights_only=False)
assert ckpt['epoch']==0 and ckpt['optimizer'] is not None and ckpt['scaler'] is not None
assert not ckpt['ema']._forward_pre_hooks
obs=trainer.amp_audit
assert obs.updates>0 and obs.total_batches==2
collision=False
try: COCOTrainOnlyTeacher(overrides=overrides,cpu_probe=True)
except FileExistsError: collision=True
assert collision
assert not torch.cuda.is_initialized()
proof={'state':'TEACHER_CPU_PRODUCTION_COMPONENT_VALIDATED','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'source_sha256':source_sha,'output':str(O),'data':'72 synthetic CPU images, not COCO scientific data',
       'native_training_batches':obs.total_batches,'actual_native_SGD_updates':obs.updates,
       'finite_native_loss_and_model':True,'class_count':80,'Person_ID':0,
       'validation_loader_images':0,'validation_forwards':0,'EMA_forwards':0,
       'native_final_epoch_validation_blocked':True,'native_final_eval_validation_blocked':True,
       'last_checkpoint_optimizer_and_scaler_retained':True,'collision_refused':True,
       'CSV_contains_only_actual_training_losses_and_LR':True,'native_optimizer_step_and_preprocess_inherited':True,
       'native_S4_save_invoked_without_serialized_audit_hooks':True,
       'environment':{'Python':sys.version.split()[0],'torch':torch.__version__,'Ultralytics':__import__('ultralytics').__version__},
       'GPU_initialized':False,'GPU_smoke_passed':False,'scientific_review_passed':False,'formal_run_allowed':False,
       'not_full_pipeline_readiness':True,'deleted_paths':[],'retained_last':str(last)}
(EX/'TEACHER_CPU_VALIDATION_20261025.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof),flush=True)
