"""Explicit COCO80 mapping and read-only fixed-checkpoint official evaluation.

This component is not a released or scientifically accepted evaluation phase.
Its caller must supply the accepted checkpoint/data/release and phase closure.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

from .runtime import atomic, load, now, sha, write_once

PREDICT_ARGS = dict(imgsz=640, batch=36, device='0', conf=.001, iou=.7,
                    max_det=300, half=False, augment=False, agnostic_nms=False,
                    rect=False, save=False, save_txt=False, save_conf=False,
                    verbose=False, stream=True)


def raw_signature(value):
    """Numerical identity before threshold/NMS, used only by bounded probes."""
    import torch
    rows=[]
    def visit(v):
        if torch.is_tensor(v):
            if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite reference raw output')
            t=v.detach().cpu().contiguous()
            rows.append(dict(shape=list(t.shape),dtype=str(t.dtype),
                             sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest()))
        elif isinstance(v,dict):
            for k in sorted(v):visit(v[k])
        elif isinstance(v,(tuple,list)):
            for item in v:visit(item)
    visit(value)
    return rows


def raw_tensor_devices(value):
    """Return device names observed on the actual pre-NMS forward outputs."""
    import torch
    devices=set()
    def visit(v):
        if torch.is_tensor(v):
            devices.add(str(v.device))
        elif isinstance(v,dict):
            for k in sorted(v):visit(v[k])
        elif isinstance(v,(tuple,list)):
            for item in v:visit(item)
    visit(value)
    return sorted(devices)


def load_evaluator(directory, expected):
    """Use only the separately pinned addon; never modify core site-packages."""
    root=Path(directory).resolve()
    for rel, expected_sha in expected.items():
        p=root/rel
        if not p.resolve().is_relative_to(root) or sha(p)!=expected_sha:
            raise ValueError('Pinned evaluator source/binary changed: '+rel)
    sys.path.insert(0,str(root))
    import pycocotools
    if not Path(pycocotools.__file__).resolve().is_relative_to(root):
        raise ValueError('Wrong official evaluator import route')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    return COCO,COCOeval


def official_metrics(annotation_json, predictions, mapping, output, evaluator_dir, evaluator_files):
    """Official crowd/ignore handling and standard all-area maxDets100 arrays."""
    COCO,COCOeval=load_evaluator(evaluator_dir,evaluator_files)
    from .evaluation import per_class
    gt=COCO(str(annotation_json))
    categories={x['id']:x['name'] for x in gt.dataset['categories']}
    if (len(mapping)!=80 or {x['yolo_class_id'] for x in mapping}!=set(range(80))
            or any(categories.get(x['category_id'])!=x['name'] for x in mapping)):
        raise ValueError('Official sparse category mapping mismatch')
    data=load(predictions)
    if data:
        result=gt.loadRes(data)
    else:
        result=COCO();result.dataset=dict(images=list(gt.dataset['images']),annotations=[],categories=gt.dataset['categories'])
        result.createIndex()
    ev=COCOeval(gt,result,'bbox')
    ev.params.imgIds=sorted(gt.getImgIds())
    ev.params.catIds=[x['category_id'] for x in mapping]
    ev.params.maxDets=[1,10,100]
    ev.evaluate();ev.accumulate();ev.summarize()
    rows=per_class(ev,mapping)
    import numpy as np
    i50=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.5))[0])
    for row in rows:
        tp=fp=0
        for image_eval in ev.evalImgs:
            if (image_eval is None or image_eval['category_id']!=row['category_id']
                    or image_eval['aRng']!=ev.params.areaRng[0]):
                continue
            valid=~image_eval['dtIgnore'][i50]
            matched=image_eval['dtMatches'][i50]>0
            tp+=int((valid&matched).sum());fp+=int((valid&~matched).sum())
        row['Precision50']=tp/(tp+fp) if tp+fp else None
    for row in rows:
        for field in ('AP50','AP75','AP50_95','Recall50','Precision50'):
            v=row[field]
            if v is not None and (not math.isfinite(v) or not 0<=v<=1):
                raise ValueError('Nonfinite/invalid official metric')
    with (Path(output)/'COCO_PER_CLASS_AP.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_once(Path(output)/'official_metrics.json',dict(state='complete',UTC=now(),classes=rows,
               image_count=len(ev.params.imgIds),annotation_sha256=sha(annotation_json),predictions_sha256=sha(predictions),
               official_COCOeval=True,IoU_thresholds=ev.params.iouThrs.tolist(),maxDets=ev.params.maxDets,
               Precision50_definition='TP/(TP+FP), IoU0.5, all area, maxDets100, conf>=0.001; official dtIgnore applied',
               crowd_ignore_policy='unchanged official COCO annotations and evaluator',
               checkpoint_selection_allowed=False,parameter_updates=False))
    return rows


def predict(checkpoint, images_dir, annotation_json, mapping, output, *, smoke_batches=0):
    """Raw-output finiteness is checked before NMS; boxes use original dimensions."""
    import torch
    from ultralytics import YOLO
    if not isinstance(smoke_batches,int) or not 0<=smoke_batches<=100:
        raise ValueError('Invalid inference smoke budget')
    meta=load(annotation_json)
    entries=sorted(meta['images'],key=lambda im:im['id'])
    if len(entries)!=5000 or len({x['id'] for x in entries})!=5000:
        raise ValueError('Full official val2017 must contain 5000 unique IDs')
    by_id={x['id']:x for x in entries}
    if smoke_batches:
        entries=entries[:PREDICT_ARGS['batch']*smoke_batches]
    paths=[str(Path(images_dir)/x['file_name']) for x in entries]
    if not all(Path(p).is_file() for p in paths):
        raise ValueError('Missing accepted val image')
    initial=sha(checkpoint)
    model=YOLO(str(checkpoint))
    if model.model.model[-1].nc!=80 or len(model.model.names)!=80 or model.model.names[0]!='person':
        raise ValueError('Not a COCO80/Person0 checkpoint')
    counter=dict(raw_forward_calls=0,raw_tensors_checked=0)
    signatures=[];forward_devices=set()
    def checked(module,args,result):
        counter['raw_forward_calls']+=1
        forward_devices.update(raw_tensor_devices(result))
        def check(v):
            if torch.is_tensor(v):
                counter['raw_tensors_checked']+=1
                if not torch.isfinite(v).all():
                    raise FloatingPointError('Nonfinite raw detection output before NMS')
            elif isinstance(v,dict):
                for item in v.values():check(item)
            elif isinstance(v,(tuple,list)):
                for item in v:check(item)
        check(result)
        if smoke_batches:signatures.append(raw_signature(result))
    hook=model.model.register_forward_hook(checked)
    table={r['yolo_class_id']:r['category_id'] for r in mapping}
    seen=[];detections=[]
    try:
        for result in model.predict(source=paths,**PREDICT_ARGS):
            image_id=int(Path(result.path).stem)
            im=by_id[image_id]
            if tuple(result.orig_shape)!=(im['height'],im['width']):
                raise ValueError('Prediction geometry differs from accepted official dimensions')
            seen.append(image_id)
            for box in result.boxes.data.detach().cpu().tolist():
                x1,y1,x2,y2,score,cls=box
                if not all(math.isfinite(v) for v in box) or cls!=int(cls) or int(cls) not in table:
                    raise ValueError('Invalid decoded prediction')
                if not 0<=x1<=x2<=im['width'] or not 0<=y1<=y2<=im['height'] or not 0<=score<=1:
                    raise ValueError('Prediction boxes escaped original image')
                detections.append(dict(image_id=image_id,category_id=table[int(cls)],bbox=[x1,y1,x2-x1,y2-y1],score=score))
    finally:
        hook.remove()
    if seen!=[im['id'] for im in entries] or len(set(seen))!=len(entries):
        raise ValueError('Val prediction IDs missing, repeated or out of bound')
    if not counter['raw_tensors_checked'] or not counter['raw_forward_calls']:
        raise ValueError('No pre-NMS finite raw forward evidence')
    if not forward_devices:
        raise ValueError('No tensor-device evidence from the production pre-NMS forward')
    if sha(checkpoint)!=initial:
        raise ValueError('Read-only evaluation modified checkpoint bytes')
    target=Path(output)/'detections.json'
    write_once(target,detections)
    receipt=dict(state='passed' if smoke_batches else 'complete',UTC=now(),checkpoint_sha256=initial,
                 predict_args=dict(PREDICT_ARGS),image_count=len(seen),detections=len(detections),
                 finite_outputs=True,checked_outputs='all raw tensor outputs of DetectionModel, before NMS',
                 sparse_category_id_mapping_explicit=True,original_pixel_xywh=True,**counter,
                 raw_output_signatures=signatures,actual_forward_tensor_devices=sorted(forward_devices),
                 actual_device=(next(iter(forward_devices)) if len(forward_devices)==1 else 'mixed'),
                 official_metrics_disabled=bool(smoke_batches),artifact_ingest_disabled=bool(smoke_batches),
                 cleanup_policy='retain_all',deleted_paths=[])
    write_once(Path(output)/'prediction_audit.json',receipt)
    return target,receipt
