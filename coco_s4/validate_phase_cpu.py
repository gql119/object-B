"""Focused CPU probes of new release guards and native pre-update comparison.

Does not repeat the accepted 45 compatibility or Teacher train-only probes.
"""
from pathlib import Path
from types import SimpleNamespace
import datetime
import json
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'execution'/('phase_cpu_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
OUT.mkdir(parents=True, exist_ok=False)
os.environ['YOLO_CONFIG_DIR'] = str(OUT/'yolo_settings')
os.environ['MPLCONFIGDIR'] = str(OUT/'mpl')
(OUT/'yolo_settings').mkdir()
(OUT/'mpl').mkdir()
import torch
torch.set_num_threads(1)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg
from coco_s4.phase_teacher import baseline_probe
from coco_s4.runtime import safe_relative, write_once, sha, load

for name in ('../outside', '/absolute', 'C:/outside', 'bad\\path'):
    try:
        safe_relative(name)
        raise AssertionError('Bad binding accepted')
    except ValueError:
        pass
p=OUT/'immutable.json';write_once(p, {'phase':'teacher'})
try:
    write_once(p, {'phase':'other'})
    raise AssertionError('Immutable receipt overwritten')
except FileExistsError:
    pass
torch.manual_seed(0)
model = DetectionModel(str(ROOT/'configs/coco_yolov8n_80cls.yaml'), nc=80, verbose=False).train()
args = get_cfg(overrides=dict(model=str(ROOT/'configs/coco_yolov8n_80cls.yaml'), imgsz=64, seed=0))
model.args=args
model.names={i: str(i) for i in range(80)}
batch=dict(img=torch.rand(2,3,64,64),cls=torch.tensor([[0.],[79.]]),
           bboxes=torch.tensor([[.4,.5,.4,.4],[.6,.5,.3,.3]]),batch_idx=torch.tensor([0,1]))
before={k:v.clone() for k,v in model.state_dict().items()}
t=SimpleNamespace(model=model,args=args,train_loader=[batch],device=torch.device('cpu'),preprocess_batch=lambda b:b,
                  baseline_probe_dataset='two synthetic CPU images, not COCO scientific data')
baseline_probe(t, OUT)
assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
proof=load(OUT/'baseline_equivalence.json')
assert proof['absolute_difference']==0 and proof['native_loss_inherited']
assert not torch.cuda.is_initialized()
report=dict(state='passed',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),output=str(OUT),
            immutable_receipt_collision_refused=True, unsafe_release_paths_refused=4,
            actual_CPU_native_loss_equivalence=proof, candidate_model_state_restored=True,
            GPU_initialized=False, new_GPU_smoke=False,
            source={n:sha(ROOT/n) for n in ('coco_s4/runtime.py','coco_s4/phase_teacher.py','coco_s4/serial.py')})
(ROOT/'execution/PHASE_TEACHER_CPU_VALIDATION_20261025.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
