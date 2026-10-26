"""Bounded real-Teacher risk-estimator validation; no optimizer or formal metrics."""
import argparse,hashlib,json
from pathlib import Path


def run(release,output,device_name):
    import numpy as np,torch
    from .runtime import ROOT,DATA_ROOT,verify_release,load,sha,atomic,write_once
    from .phase_contract import teacher_dependency
    from .phase_calibration import load_teacher
    from .risk_fallback import gradients as candidate,POLICY
    from ue_framework.methods.tfss_nt_sensitivity import gradients as reference
    from ue_framework.methods.s4_tfss import model_hash
    from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.cfg import get_cfg
    torch.set_num_threads(1);m=verify_release(release)
    if m.get('risk_estimator_policy')!=POLICY or m.get('validation_only') is not True:raise ValueError('Explicit isolated risk policy required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    common=dict(source_commit=m['source_commit'],release_sha256=sha(release),risk_estimator_policy=POLICY,
        official_metrics_disabled=True,artifact_ingest_disabled=True,parameter_updates=0,formal_READY_written=False,
        validation_only=True,device=device_name)
    atomic(output/'status.json',dict(common,state='RUNNING'))
    device=torch.device(device_name)
    if device.type=='cpu' and torch.cuda.is_initialized():raise ValueError('CPU validation initialized CUDA')
    counter=[];forward0=DetectionModel.forward
    def observed_forward(model,x,*args,**kwargs):
        result=forward0(model,x,*args,**kwargs)
        counter.append(dict(input_shape=list(x.shape),device=str(x.device)))
        if len(counter)>100:raise ValueError('Exceeded bounded native-forward budget')
        return result
    DetectionModel.forward=observed_forward
    try:
        checkpoint,_=teacher_dependency(m['teacher_dependency'])
        model,before=load_teacher(checkpoint,m['teacher_dependency']['checkpoint_sha256'],device)
        raw_records=[]
        def raw_hook(module,args,result):
            from .coco_eval import raw_signature,raw_tensor_device_details
            signature=raw_signature(result);devices,indices=raw_tensor_device_details(result)
            if not signature:raise FloatingPointError('Nonfinite or missing native raw output')
            if device.type=='cuda' and (devices!=['cuda:0'] or indices!=[0]):raise ValueError('No actual CUDA0 raw-output evidence')
            raw=result[1] if isinstance(result,tuple) else result
            raw_records.append(dict(input_shape=list(args[0].shape),devices=devices,cuda_indices=indices,
                actual_feature_shapes=[list(v.shape) for v in raw['feats']]))
        handle=model.register_forward_hook(raw_hook)
        def data(stem):
            image=DATA_ROOT/'images/train'/(stem+'.jpg');label=DATA_ROOT/'labels/train'/(stem+'.txt')
            return load_image_rgb_float(str(image)),read_yolo_annotations(str(label)),dict(id=stem,image_sha256=sha(image),label_sha256=sha(label))
        kwargs=dict(imgsz=640,coverage_policy='independent_same_TAL_for_uncovered_valid_GT',target_class_id=0,num_classes=80)
        def captured(function,image,anns,**options):
            captures=[];grad0=torch.autograd.grad
            def record_grad(*args,**kw):
                result=grad0(*args,**kw)
                captures.append(result[0].detach().cpu().clone())
                return result
            torch.autograd.grad=record_grad
            try:result=function(model,image,anns,**options)
            finally:torch.autograd.grad=grad0
            return result,captures
        for stem in m['covered_fixture_image_ids']:
            image,anns,normal_meta=data(stem)
            if any(int(a['cls'])!=0 for a in anns):break
        else:raise ValueError('Normal equivalence fixture has no NT GT')
        # CUDA equality uses the same explicitly deterministic comparison domain
        # for both sides; restore production backend before the new branch test.
        backend=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark)
        if device.type=='cuda':torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        try:
            (old,old_rows),old_grad=captured(reference,image,anns,**kwargs)
            (new,new_rows),new_grad=captured(candidate,image,anns,**kwargs)
        finally:torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark=backend
        assert old and old_rows==new_rows and set(old)==set(new) and all(np.array_equal(old[k],new[k]) for k in old)
        assert len(old_grad)==len(new_grad) and all(torch.equal(a,b) for a,b in zip(old_grad,new_grad))
        normal=dict(**normal_meta,state='passed',maps_exact=True,records_exact=True,full_input_gradients_exact=True,
            input_gradient_hashes=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in new_grad],classes=sorted(new),
            comparison_backend='same deterministic cuDNN comparison domain; restored' if device.type=='cuda' else 'same actual CPU')
        image,anns,failed_meta=data('000000247692')
        (maps,rows),grads=captured(candidate,image,anns,**kwargs)
        expected={i for i,a in enumerate(anns) if a['cls']!=0}
        assert {x['gt_index'] for x in rows}==expected and len(rows)==len(expected)
        assert set(maps)=={int(a['cls']) for a in anns if a['cls']!=0}
        fallback=[x for x in rows if x.get('risk_estimator_policy')==POLICY]
        assert any(x['gt_index']==13 and x['cls']==24 and x['positive_count']==0 and x['risk_candidate_count']>0 for x in fallback)
        assert all(np.isfinite(x).all() and np.linalg.norm(x)>0 for x in maps.values())
        assert len(grads)==len(maps) and all(torch.isfinite(v).all() and v.norm()>0 for v in grads)
        # Actual co-occurring GT overlap; no synthetic labels enter the dataset.
        def xyxy(a):
            x,y,w,h=a['bbox'];return x-w/2,y-h/2,x+w/2,y+h/2
        overlap_pairs=0
        for i,a in enumerate(anns):
            for b in anns[i+1:]:
                l,t,r,bottom=xyxy(a);ll,tt,rr,bb=xyxy(b)
                overlap_pairs+=int(min(r,rr)>max(l,ll) and min(bottom,bb)>max(t,tt))
        assert overlap_pairs>0
        failed=dict(**failed_meta,state='passed',GT_count=len(anns),NT_GT_count=len(expected),risk_class_count=len(maps),
            GT_indices_each_once=True,class_keys_exact=True,overlap_pairs=overlap_pairs,fallback_GT_records=fallback,
            class_input_gradient_norms={str(c):float(v.norm()) for c,v in zip(sorted(maps),grads)},
            original_complete_risk=None,original_undefined_reason='NO_VALID_POSITIVE; no zero imputation')
        # Empty-NT and schema contracts are fixtures only, never altered files.
        calls_before=len(counter)
        assert candidate(model,image,[],**kwargs)==({},[])
        assert len(counter)==calls_before
        noNT=dict(fixture='empty annotation contract only; original dataset unchanged',state='passed',model_forwards=0)
        handle.remove()
        assert model_hash(model)==before and sha(checkpoint)==m['teacher_dependency']['checkpoint_sha256']
        # Real random-YAML VOC regression on the exact original input fixture.
        torch.manual_seed(21)
        voc=DetectionModel(str(ROOT/'configs/voc_yolov8n_20cls.yaml'),nc=20,verbose=False).to(device).eval();voc.args=get_cfg()
        for p in voc.parameters():p.requires_grad_(False)
        im=np.random.default_rng(1).random((96,160,3),dtype=np.float32)
        ann=[{'cls':14,'bbox':[.3,.5,.35,.5]},{'cls':19,'bbox':[.7,.5,.4,.5]}]
        vk=dict(imgsz=64,coverage_policy='independent_same_TAL_for_uncovered_valid_GT',target_class_id=14,num_classes=20)
        a,ar=reference(voc,im,ann,**vk);b,br=candidate(voc,im,ann,**vk)
        assert ar==br and all(np.array_equal(a[k],b[k]) for k in a)
        assert not any(p.grad is not None for p in model.parameters())
        if device.type=='cpu':assert not torch.cuda.is_initialized()
        result=dict(common,state='passed',actual_native_forwards=len(counter),actual_forward_evidence=counter,
            real_Teacher_raw_evidence=raw_records,covered_COCO=normal,failed_GT=failed,empty_NT=noNT,
            VOC20_real_random_YAML_gradient_maps_exact=True,Teacher_unchanged=True,checkpoint_unchanged=True,
            full25K_trigger_statistics=None,full25K_state='NOT_RUN',Fourier_gradient_gate='OPEN_SEPARATE_FULL_CHAIN_VALIDATION',
            current_scope='component risk estimator only; not full phase readiness')
        write_once(output/'risk_validation_summary.json',result);atomic(output/'status.json',result)
        return result
    except BaseException as exc:
        atomic(output/'status.json',dict(common,state='FAILED',error=repr(exc)));raise
    finally:DetectionModel.forward=forward0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',choices=('cpu','cuda:0'),required=True);a=p.parse_args();run(a.release,a.output,a.device)
