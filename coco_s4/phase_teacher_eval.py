"""Fixed last@200 read-only Teacher assessment, after its separate phase gate.

No method decision consumes these validation AP values. Bounded smoke executes
the production prediction sink and a native reference, with official metrics off.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .runtime import ROOT, DATA_ROOT, atomic, load, now, sha, write_once
from .phase_contract import OWNED, WORKFLOW_OUTPUT, phase_release, formal_phase_gate
from .coco_eval import PREDICT_ARGS, predict, official_metrics


def validate_config(cfg):
    expected=dict(schema='s4.coco.teacher-eval-config.v1',phase='teacher_eval',
                  checkpoint_policy='fixed native EMA last@200',image_count=5000,nc=80,target_class_id=0,
                  predict_args=PREDICT_ARGS,parameter_updates=False,checkpoint_selection_allowed=False,
                  method_statistics_allowed=False,metric_driven_tuning_allowed=False,
                  evaluator_version='2.0.10',
                  evaluator_wheel_sha256='1760c10459dfb4229e7436ae380228428efb0115bbe332a51b72d07fa085d8c0')
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Frozen post-training Teacher evaluation protocol changed')


def native_reference(checkpoint, images, annotation, mapping, smoke_batches, candidate):
    """Independent native YOLO.predict invocation, same weights/inputs/protocol."""
    from ultralytics import YOLO
    from .coco_eval import raw_signature, raw_tensor_device_details
    entries=sorted(load(annotation)['images'],key=lambda im:im['id'])[:PREDICT_ARGS['batch']*smoke_batches]
    paths=[str(Path(images)/x['file_name']) for x in entries]
    model=YOLO(str(checkpoint));signatures=[];reference_devices=set();reference_cuda_indices=set();input_batch_sizes=[]
    def reference_hook(module,args,result):
        batch=int(args[0].shape[0])
        if not 1<=batch<=PREDICT_ARGS['batch']:
            raise ValueError('Independent native reference batch exceeded frozen batch setting')
        input_batch_sizes.append(batch)
        devices,indices=raw_tensor_device_details(result)
        reference_devices.update(devices);reference_cuda_indices.update(indices)
        signatures.append(raw_signature(result))
    hook=model.model.register_forward_hook(reference_hook)
    table={r['yolo_class_id']:r['category_id'] for r in mapping};detections=[]
    source=Path(candidate).parent/'reference_images.txt'
    with source.open('x',encoding='utf-8') as f:
        f.write(''.join(p+'\n' for p in paths))
    try:
        for r in model.predict(source=str(source),**PREDICT_ARGS):
            for box in r.boxes.data.detach().cpu().tolist():
                x1,y1,x2,y2,score,cls=box
                detections.append(dict(image_id=int(Path(r.path).stem),category_id=table[int(cls)],
                                       bbox=[x1,y1,x2-x1,y2-y1],score=score))
    finally:
        hook.remove()
    audit=load(Path(candidate).parent/'prediction_audit.json')
    raw_equal=signatures==audit['raw_output_signatures']
    decoded_equal=detections==load(candidate)
    if not raw_equal or not decoded_equal:
        raise ValueError('Independent canonical native inference differs from production sink')
    if len(reference_devices)!=1:
        raise ValueError('Independent reference did not expose one unambiguous forward device')
    return dict(reference_id='pinned ultralytics.YOLO.predict, independent model instance',
                candidate_id='coco_s4.coco_eval.predict production sink',
                reference_weights_path=str(checkpoint),candidate_weights_path=str(checkpoint),
                weights_sha256=sha(checkpoint),metric_name='raw and decoded native inference exact identity',
                reference_metric=0,candidate_metric=0,absolute_difference=0,tolerance=0,
                raw_outputs_exact=raw_equal,decoded_predictions_exact=decoded_equal,
                reference_actual_device=next(iter(reference_devices)),
                reference_cuda_device_index=(next(iter(reference_cuda_indices)) if len(reference_cuda_indices)==1 else None),
                reference_forward_tensor_devices=sorted(reference_devices),
                reference_cuda_device_indices=sorted(reference_cuda_indices),
                reference_raw_forward_calls=len(signatures),
                reference_forward_input_batch_sizes=input_batch_sizes,
                reference_observed_max_input_batch=max(input_batch_sizes),
                raw_reference_sha256=hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest(),
                evaluation_setting=dict(dataset='first sorted annotation image IDs',sample_count=len(paths),seed=0,
                                        parameters=dict(PREDICT_ARGS),post_processing='identical native NMS'))


def run(release_path,output,*,registration=None,smoke_batches=0):
    manifest,cfg,checkpoint,teacher=phase_release(release_path,'teacher_eval')
    validate_config(cfg)
    config_sha=sha(ROOT/manifest['phase_config_path'])
    output=Path(output).resolve()
    if smoke_batches:
        # Reference repeats these batches, so total production work stays <=100.
        if not 1<=smoke_batches<=49 or not output.is_relative_to((OWNED/'validation').resolve()):
            raise ValueError('Inference smoke requires1..49batches in the isolated owned validation root')
    else:
        if output!=(WORKFLOW_OUTPUT/'teacher_eval').resolve():
            raise ValueError('Wrong existing workflow phase output')
        formal_phase_gate(registration,'teacher_eval',release_path,manifest,config_sha)
    output.mkdir(parents=True,exist_ok=False)
    os.environ['YOLO_CONFIG_DIR']=str(output/'yolo_settings')
    os.environ['MPLCONFIGDIR']=str(output/'mpl')
    os.environ['WANDB_MODE']='disabled'
    import torch
    import numpy as np
    import random
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise ValueError('Production Teacher assessment requires admitted CUDA device')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['epoch']!=199 or ck['ema'] is None or ck['ema'].model[-1].nc!=80 or ck['ema'].names[0]!='person':
        raise ValueError('Not the native fixed EMA COCO80 last@200 checkpoint')
    if not all(torch.isfinite(p).all() for p in ck['ema'].parameters()):
        raise FloatingPointError('Nonfinite Teacher checkpoint')
    del ck
    mapping=load(ROOT/'docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json')['mapping']
    addon=manifest['evaluator']
    if (Path(addon['directory']).resolve()!=(OWNED/'evaluator/pycocotools_2_0_10/addon').resolve()
            or addon['wheel_sha256']!=cfg['evaluator_wheel_sha256']):
        raise ValueError('Unexpected isolated evaluator binding')
    common=dict(phase='teacher_eval',source_commit=manifest['source_commit'],release_sha256=sha(release_path),
                phase_config_sha256=config_sha,teacher_checkpoint_sha256=sha(checkpoint),
                previous_stage_receipt_sha256=manifest['teacher_dependency']['accepted_sha256'],
                computation_kind='inference',production_entrypoint_reached=True,
                official_metrics_disabled=bool(smoke_batches),artifact_ingest_disabled=bool(smoke_batches),
                parameter_updates=False,checkpoint_selection_allowed=False,method_statistics_allowed=False,
                cleanup_policy='retain_all',deleted_paths=[])
    status=dict(common,state='RUNNING',started_UTC=now())
    atomic(output/'status.json',status)
    write_once(output/'phase_spec.json',dict(common,config=cfg,teacher_acceptance=teacher))
    try:
        predictions,audit=predict(checkpoint,DATA_ROOT/'images/val',DATA_ROOT/'metadata/instances_val2017.json',
                                  mapping,output,smoke_batches=smoke_batches)
        if (audit['actual_device']!='cuda:0' or audit.get('actual_forward_tensor_devices')!=['cuda:0']
                or audit.get('actual_cuda_device_indices')!=[0]):
            raise ValueError('Production pre-NMS forward tensors did not prove admitted CUDA0')
        if smoke_batches:
            baseline=native_reference(checkpoint,DATA_ROOT/'images/val',DATA_ROOT/'metadata/instances_val2017.json',
                                      mapping,smoke_batches,predictions)
            if (baseline['reference_actual_device']!=audit['actual_device']
                    or baseline['reference_cuda_device_index']!=0
                    or baseline['reference_forward_tensor_devices']!=['cuda:0']
                    or baseline['reference_cuda_device_indices']!=[0]):
                raise ValueError('Independent reference and production tensors did not prove the same CUDA0 device')
            if sha(checkpoint)!=manifest['teacher_dependency']['checkpoint_sha256']:
                raise ValueError('Read-only inference probe changed checkpoint bytes')
            total_forwards=audit['raw_forward_calls']+baseline['reference_raw_forward_calls']
            if not 1<=total_forwards<=100:raise ValueError('Total raw inference work exceeded bounded smoke')
            status.update(state='passed',finite_outputs=audit['finite_outputs'],
                          actual_inference_batches=smoke_batches*2,total_raw_forwards_including_warmup=total_forwards,
                          production_raw_forward_calls=audit['raw_forward_calls'],
                          raw_tensors_checked=audit['raw_tensors_checked'],image_count=audit['image_count'],
                          actual_device=audit['actual_device'],
                          actual_cuda_device_index=audit['actual_cuda_device_indices'][0],
                          actual_forward_tensor_devices=audit['actual_forward_tensor_devices'],
                          actual_cuda_device_indices=audit['actual_cuda_device_indices'],
                          reference_actual_device=baseline['reference_actual_device'],
                          reference_cuda_device_index=baseline['reference_cuda_device_indices'][0],
                          reference_forward_tensor_devices=baseline['reference_forward_tensor_devices'],
                          reference_cuda_device_indices=baseline['reference_cuda_device_indices'],
                          forward_input_batch_sizes=audit['forward_input_batch_sizes'],
                          observed_max_input_batch=audit['observed_max_input_batch'],
                          planned_input_batch_sizes=audit['planned_input_batch_sizes'],
                          reference_forward_input_batch_sizes=baseline['reference_forward_input_batch_sizes'],
                          reference_observed_max_input_batch=baseline['reference_observed_max_input_batch'],
                          baseline_equivalence_probe=baseline,completed_UTC=now(),checkpoint_paths_remaining=[])
            write_once(output/'smoke_summary.json',status)
        else:
            rows=official_metrics(DATA_ROOT/'metadata/instances_val2017.json',predictions,mapping,output,
                                  addon['directory'],addon['files'])
            if audit['image_count']!=5000 or len(rows)!=80:raise ValueError('Incomplete full COCO80 assessment')
            if sha(checkpoint)!=manifest['teacher_dependency']['checkpoint_sha256']:
                raise ValueError('Read-only assessment changed checkpoint bytes')
            status.update(state='ACCEPTED',finite_outputs=True,image_count=5000,classes=80,
                          predictions_sha256=sha(predictions),metrics_sha256=sha(output/'official_metrics.json'),
                          per_class_csv_sha256=sha(output/'COCO_PER_CLASS_AP.csv'),completed_UTC=now())
            write_once(output/'accepted.json',status)
    except BaseException as exc:
        status.update(state='FAILED',error=repr(exc),ended_UTC=now())
        raise
    finally:
        atomic(output/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--registration',type=Path)
    p.add_argument('--smoke-batches',type=int,default=0);a=p.parse_args()
    run(a.release,a.output,registration=a.registration,smoke_batches=a.smoke_batches)
