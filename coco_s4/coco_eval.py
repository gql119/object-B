"""Explicit COCO80 mapping and read-only fixed-checkpoint official evaluation.

This component is not a released or scientifically accepted evaluation phase.
Its caller must supply the accepted checkpoint/data/release and phase closure.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
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


def raw_tensor_device_details(value):
    """Return device names and CUDA indices from actual pre-NMS tensors."""
    import torch
    devices=set();cuda_indices=set()
    def visit(v):
        if torch.is_tensor(v):
            device=v.device
            devices.add(str(device))
            if device.type=='cuda' and device.index is not None:
                cuda_indices.add(int(device.index))
        elif isinstance(v,dict):
            for k in sorted(v):visit(v[k])
        elif isinstance(v,(tuple,list)):
            for item in v:visit(item)
    visit(value)
    return sorted(devices),sorted(cuda_indices)


def raw_tensor_devices(value):
    """Compatibility helper for callers that need the observed device names."""
    return raw_tensor_device_details(value)[0]


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


def validate_val2017_image_identity(annotation_json, images_dir, image_manifest,
                                    expected_manifest_sha256, *, expected_count=5000):
    """Bind every evaluated val image to the accepted ID/name/geometry/byte manifest."""
    manifest_path = Path(image_manifest).resolve()
    if sha(manifest_path) != expected_manifest_sha256:
        raise ValueError('Pinned val2017 image manifest bytes changed')
    annotation_path = Path(annotation_json).resolve()
    metadata = load(annotation_path)
    annotation_rows = metadata.get('images')
    if not isinstance(annotation_rows, list) or len(annotation_rows) != expected_count:
        raise ValueError('Official val2017 annotation image count differs from the frozen contract')
    annotations = {}
    for row in annotation_rows:
        image_id = row.get('id')
        if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id in annotations:
            raise ValueError('Official val2017 annotation IDs are invalid or duplicated')
        annotations[image_id] = row
    if len(annotations) != expected_count:
        raise ValueError('Official val2017 annotation IDs are incomplete')

    root = Path(images_dir).resolve()
    manifest_rows = {}
    with manifest_path.open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'Blank val2017 image-manifest row at line {line_number}')
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid val2017 image-manifest JSON at line {line_number}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Invalid val2017 image-manifest record at line {line_number}')
            raw_id = row.get('id', row.get('image_id'))
            if 'id' in row and 'image_id' in row and row['id'] != row['image_id']:
                raise ValueError(f'Conflicting val2017 image IDs at line {line_number}')
            if (isinstance(raw_id, bool)
                    or not (isinstance(raw_id, int)
                            or isinstance(raw_id, str) and raw_id.isdecimal())):
                raise ValueError(f'Invalid val2017 image ID at line {line_number}')
            image_id = int(raw_id)
            if image_id in manifest_rows:
                raise ValueError('Duplicate val2017 image ID in pinned manifest')
            annotation = annotations.get(image_id)
            if annotation is None:
                raise ValueError('Pinned image manifest contains an ID absent from official val2017')
            relative_image = row.get('image')
            if not isinstance(relative_image, str) or not relative_image.startswith('images/val/'):
                raise ValueError('Pinned val2017 image path is not in the accepted images/val view')
            filename = relative_image[len('images/val/'):]
            if (not isinstance(filename, str) or not filename or Path(filename).name != filename
                    or '/' in filename or '\\' in filename or ':' in filename):
                raise ValueError('Unsafe or missing val2017 manifest filename')
            dimensions = (row.get('width'), row.get('height'))
            if (any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions)
                    or dimensions != (annotation.get('width'), annotation.get('height'))
                    or filename != annotation.get('file_name')):
                raise ValueError('Pinned val2017 ID/name/dimensions differ from official annotations')
            byte_count = row.get('image_bytes')
            image_digest = row.get('image_sha256')
            if (isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0
                    or not isinstance(image_digest, str) or not re.fullmatch(r'[0-9a-f]{64}', image_digest)):
                raise ValueError('Pinned val2017 image byte-count/SHA fields are invalid')
            image_path = (root / filename).resolve()
            if not image_path.is_relative_to(root) or not image_path.is_file():
                raise ValueError('Pinned val2017 image is missing or escapes its accepted root')
            actual_bytes = image_path.stat().st_size
            actual_sha256 = sha(image_path)
            if actual_bytes != byte_count or actual_sha256 != image_digest:
                raise ValueError('Actual val2017 image bytes differ from the pinned per-image identity')
            manifest_rows[image_id] = dict(id=image_id, file_name=filename,
                width=int(dimensions[0]), height=int(dimensions[1]),
                image_bytes=byte_count, image_sha256=image_digest)
    if len(manifest_rows) != expected_count or set(manifest_rows) != set(annotations):
        raise ValueError('Pinned val2017 manifest does not cover the exact official image-ID set')
    rows = [manifest_rows[image_id] for image_id in sorted(manifest_rows)]
    canonical = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return dict(state='passed', expected_image_count=expected_count, image_count=len(rows),
        annotation_sha256=sha(annotation_path), manifest_path=str(manifest_path),
        manifest_sha256=sha(manifest_path), expected_manifest_sha256=expected_manifest_sha256,
        images_root=str(root), total_image_bytes=sum(row['image_bytes'] for row in rows),
        verified_image_records_sha256=hashlib.sha256(canonical).hexdigest(),
        identity_check='annotation IDs/names/dimensions + actual per-image byte count/SHA256',
        verified_images=rows)


def official_metrics(annotation_json, predictions, mapping, output, evaluator_dir, evaluator_files,
                     *, image_identity_receipt_sha256=None, protocol_identity=None):
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
               image_identity_receipt_sha256=image_identity_receipt_sha256,
               protocol_identity=protocol_identity,
               official_COCOeval=True,IoU_thresholds=ev.params.iouThrs.tolist(),maxDets=ev.params.maxDets,
               Precision50_definition='TP/(TP+FP), IoU0.5, all area, maxDets100, conf>=0.001; official dtIgnore applied',
               crowd_ignore_policy='unchanged official COCO annotations and evaluator',
               checkpoint_selection_allowed=False,parameter_updates=False))
    return rows


def predict(checkpoint, images_dir, annotation_json, mapping, output, *, smoke_batches=0,
            image_identity_receipt_sha256=None):
    """Raw-output finiteness is checked before NMS; boxes use original dimensions."""
    import torch
    from ultralytics import YOLO
    if not isinstance(smoke_batches,int) or not 0<=smoke_batches<=100:
        raise ValueError('Invalid inference smoke budget')
    if (image_identity_receipt_sha256 is not None
            and not re.fullmatch(r'[0-9a-f]{64}', image_identity_receipt_sha256)):
        raise ValueError('Invalid pinned val2017 image-identity receipt hash')
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
    signatures=[];forward_devices=set();forward_cuda_indices=set();input_batch_sizes=[]
    def checked(module,args,result):
        counter['raw_forward_calls']+=1
        batch=int(args[0].shape[0])
        if not 1<=batch<=PREDICT_ARGS['batch']:
            raise ValueError('Actual native inference batch exceeded the frozen batch setting')
        input_batch_sizes.append(batch)
        devices,indices=raw_tensor_device_details(result)
        forward_devices.update(devices);forward_cuda_indices.update(indices)
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
    # A Python path list is eagerly loaded as one native in-memory batch. A
    # text source selects LoadImagesAndVideos, which consumes batch=36.
    source=Path(output)/'inference_images.txt'
    with source.open('x',encoding='utf-8') as f:
        f.write(''.join(p+'\n' for p in paths))
    try:
        for result in model.predict(source=str(source),**PREDICT_ARGS):
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
                 forward_input_batch_sizes=input_batch_sizes,observed_max_input_batch=max(input_batch_sizes),
                 planned_input_batch_sizes=[min(PREDICT_ARGS['batch'],len(paths)-i) for i in range(0,len(paths),PREDICT_ARGS['batch'])],
                 native_source_kind='LoadImagesAndVideos text manifest',input_manifest_sha256=sha(source),
                 image_identity_receipt_sha256=image_identity_receipt_sha256,
                 actual_cuda_device_index=(next(iter(forward_cuda_indices)) if len(forward_cuda_indices)==1 else None),
                 actual_cuda_device_indices=sorted(forward_cuda_indices),
                 actual_device=(next(iter(forward_devices)) if len(forward_devices)==1 else 'mixed'),
                 official_metrics_disabled=bool(smoke_batches),artifact_ingest_disabled=bool(smoke_batches),
                 cleanup_policy='retain_all',deleted_paths=[])
    write_once(Path(output)/'prediction_audit.json',receipt)
    return target,receipt
