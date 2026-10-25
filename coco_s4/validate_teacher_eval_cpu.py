"""Real native prediction CPU probe, synthetic images/random model, not COCO AP."""
from pathlib import Path
import json
import os

from .runtime import ROOT, now, sha


def main():
    out=ROOT/'execution'/('teacher_eval_cpu_'+now().replace(':','').replace('+','_'))
    out.mkdir(parents=True,exist_ok=False)
    os.environ['YOLO_CONFIG_DIR']=str(out/'settings')
    os.environ['MPLCONFIGDIR']=str(out/'mpl')
    import torch
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    from . import coco_eval
    from .phase_teacher_eval import native_reference
    torch.set_num_threads(1);torch.manual_seed(0)
    mapping=json.loads((ROOT/'docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json').read_text())['mapping']
    model=YOLO(str(ROOT/'configs/coco_yolov8n_80cls.yaml'))
    model.model.names={x['yolo_class_id']:x['name'] for x in mapping}
    checkpoint=out/'random_native80.pt';model.save(checkpoint)
    images=out/'images';images.mkdir()
    rng=np.random.default_rng(42)
    for i in range(2):Image.fromarray(rng.integers(0,256,(64,80,3),dtype=np.uint8)).save(images/f'{i:012d}.jpg')
    meta=out/'synthetic_5000_id_schema.json'
    meta.write_text(json.dumps(dict(images=[dict(id=i,file_name=f'{i:012d}.jpg',width=80,height=64) for i in range(5000)])))
    predictions=out/'prediction';predictions.mkdir()
    original=dict(coco_eval.PREDICT_ARGS)
    try:
        # Explicit CPU-only test overrides; production config remains device0/b36.
        coco_eval.PREDICT_ARGS.update(device='cpu',batch=2)
        path,audit=coco_eval.predict(checkpoint,images,meta,mapping,predictions,smoke_batches=1)
        baseline=native_reference(checkpoint,images,meta,mapping,1,path)
    finally:
        coco_eval.PREDICT_ARGS.clear();coco_eval.PREDICT_ARGS.update(original)
    assert audit['image_count']==2 and audit['raw_tensors_checked']>0 and audit['raw_output_signatures']
    assert baseline['raw_outputs_exact'] and baseline['decoded_predictions_exact']
    assert baseline['evaluation_setting']['parameters']['device']=='cpu'
    assert baseline['evaluation_setting']['parameters']['batch']==2
    assert audit['actual_device']=='cpu' and audit['actual_forward_tensor_devices']==['cpu']
    assert audit['actual_cuda_device_index'] is None and audit['actual_cuda_device_indices']==[]
    assert baseline['reference_actual_device']=='cpu' and baseline['reference_cuda_device_index'] is None
    assert baseline['reference_forward_tensor_devices']==['cpu']
    assert baseline['reference_cuda_device_indices']==[] and audit['official_metrics_disabled']
    rejected_nan=False
    try:coco_eval.raw_signature(torch.tensor([float('nan')]))
    except FloatingPointError:rejected_nan=True
    assert rejected_nan and not torch.cuda.is_initialized()
    proof=dict(state='passed',UTC=now(),data='two synthetic CPU images; random native COCO80 head',
               native_raw_and_decoded_predictions_exact=True,pre_NMS_nonfinite_rejected=True,
               image_count=2,production_parameters_except_explicit_CPU_test=dict(device='cpu',batch=2,imgsz=640),
               GPU_initialized=False,official_COCO_metrics_computed=False,scientific_review_passed=False,
               formal_phase_allowed=False,baseline_equivalence_probe=baseline,
               source_sha256={n:sha(ROOT/n) for n in ['coco_s4/coco_eval.py','coco_s4/phase_teacher_eval.py']},
               output=str(out),deleted_paths=[])
    (ROOT/'execution/TEACHER_EVAL_DEVICE_INDEX_CPU_VALIDATION_20261025.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof))


if __name__=='__main__':main()
