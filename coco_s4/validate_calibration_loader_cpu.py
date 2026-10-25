"""Actual native EMA-container/dtype/gradient parity; synthetic CPU only."""
import copy,json,os
from pathlib import Path
from .runtime import ROOT,now,sha

def main():
    out=ROOT/'execution'/('calibration_loader_cpu_'+now().replace(':','').replace('+','_'))
    out.mkdir(exist_ok=False);os.environ['YOLO_CONFIG_DIR']=str(out/'settings')
    import torch
    from ultralytics import YOLO
    from .phase_calibration import load_teacher
    from .coco_eval import raw_signature
    from ue_framework.methods.s4_tfss import model_hash
    torch.set_num_threads(1);torch.manual_seed(0)
    mapping=json.loads((ROOT/'docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json').read_text())['mapping']
    model=YOLO(str(ROOT/'configs/coco_yolov8n_80cls.yaml')).model
    model.names={x['yolo_class_id']:x['name'] for x in mapping}
    checkpoint=out/'random_native_ema_last.pt'
    torch.save(dict(epoch=199,ema=copy.deepcopy(model).half(),model=None,
                    train_args=dict(task='detect',model=str(ROOT/'configs/coco_yolov8n_80cls.yaml'))),checkpoint)
    actual,before=load_teacher(checkpoint,sha(checkpoint),torch.device('cpu'))
    reference=YOLO(str(checkpoint)).model.float().eval()
    for p in reference.parameters():p.requires_grad_(False)
    assert type(actual).__name__=='DetectionModel' and actual.model[-1].nc==80
    assert all(p.dtype==torch.float32 and not p.requires_grad for p in actual.parameters())
    assert model_hash(actual)==model_hash(reference)
    x=torch.rand(1,3,640,640,requires_grad=True);y=x.detach().clone().requires_grad_(True)
    a=actual(x);b=reference(y)
    assert raw_signature(a)==raw_signature(b)
    def tensors(value):
        if torch.is_tensor(value):yield value
        elif isinstance(value,dict):
            for v in value.values():yield from tensors(v)
        elif isinstance(value,(tuple,list)):
            for v in value:yield from tensors(v)
    ga=torch.autograd.grad(next(tensors(a)).square().mean(),x)[0]
    gb=torch.autograd.grad(next(tensors(b)).square().mean(),y)[0]
    assert torch.equal(ga,gb) and torch.isfinite(ga).all() and ga.abs().sum()>0
    assert model_hash(actual)==before and not torch.cuda.is_initialized()
    proof=dict(state='passed',UTC=now(),checkpoint_kind='synthetic random native DetectionModel EMA/FP16/epoch199',
        actual_container='DetectionModel',actual_dtype='torch.float32',native_YOLO_state_exact=True,
        native_raw_outputs_exact=True,native_input_gradient_exact=True,finite_nonzero_input_gradient=True,
        frozen_model_unchanged=True,GPU_initialized=False,official_metrics_generated=False,
        actual_formal_Teacher_checkpoint_used=False,production_loader_source_sha256=sha(ROOT/'coco_s4/phase_calibration.py'),
        retained_output=str(out),deleted_paths=[])
    with (ROOT/'execution/CALIBRATION_NATIVE_LOADER_CPU_VALIDATION_20261025.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2)
    print(json.dumps(proof))

if __name__=='__main__':main()
