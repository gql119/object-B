"""Locked direct Person/Dog overlap diagnostic, separate from full-curve AP.

Not a causal class migration test or the native one-to-one confusion matrix.
Count each real Person GT once: Dog IoU>=.5; optionally no Person IoU>=.5.
"""
import numpy as np


def person_dog_counts(person_boxes,boxes,classes):
    gt=np.asarray(person_boxes,dtype=float).reshape(-1,4)
    pred=np.asarray(boxes,dtype=float).reshape(-1,4);classes=np.asarray(classes,dtype=int)
    if len(pred)!=len(classes) or not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError('Invalid diagnostic boxes')
    def covered(cls):
        p=pred[classes==cls]
        if not len(p):return np.zeros(len(gt),dtype=bool)
        lo=np.maximum(gt[:,None,:2],p[None,:,:2]);hi=np.minimum(gt[:,None,2:],p[None,:,2:])
        inter=np.maximum(hi-lo,0).prod(2)
        union=((gt[:,2:]-gt[:,:2]).prod(1)[:,None]+(p[:,2:]-p[:,:2]).prod(1)[None,:]-inter)
        iou=inter/np.maximum(union,1e-12)
        return (iou>=.5).any(1)
    dog=covered(11);person=covered(14)
    return dict(Person_GT=len(gt),Person_GT_with_Dog_overlap=int(dog.sum()),
        Person_GT_with_Dog_without_Person_overlap=int((dog&~person).sum()),
        Person_GT_with_Person_overlap=int(person.sum()))


def evaluate_person_dog(checkpoint,data,output,protocol):
    from pathlib import Path
    import json
    from ultralytics import YOLO
    from ue_framework.data_utils import read_yolo_annotations
    if protocol!={'conf':.25,'iou':.7,'match_iou':.5,'imgsz':640,'batch':36,'max_det':300,'agnostic_nms':False}:
        raise ValueError('Predeclared direct detection protocol changed')
    root=Path(data);counts=dict(Person_GT=0,Person_GT_with_Dog_overlap=0,Person_GT_with_Dog_without_Person_overlap=0,Person_GT_with_Person_overlap=0)
    n=0
    options={k:v for k,v in protocol.items() if k!='match_iou'}
    for result in YOLO(str(checkpoint)).predict(source=str(root/'images/val'),stream=True,
            device='0',save=False,verbose=False,**options):
        if len(result.names)!=20 or result.names[14]!='person':raise ValueError('Diagnostic class identity mismatch')
        h,w=result.orig_shape;ann=read_yolo_annotations(str(root/'labels/val'/(Path(result.path).stem+'.txt')))
        gt=[]
        for a in ann:
            if int(a['cls'])==14:
                x,y,bw,bh=a['bbox'];gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        values=person_dog_counts(gt,result.boxes.xyxy.cpu().numpy(),result.boxes.cls.cpu().numpy())
        for k,v in values.items():counts[k]+=v
        n+=1
    if n!=4952 or not counts['Person_GT']:raise ValueError('Incomplete matched direct-detection diagnostic')
    report=dict(images=n,counts=counts,protocol=dict(conf=.25,NMS_iou=.7,match_iou=.5,imgsz=640,batch=36,max_det=300,agnostic_nms=False),
        scope='GT-wise Dog overlap and Dog-overlap-without-Person-overlap; no one-to-one assignment; separate from AP',
        rate_Dog_without_Person=counts['Person_GT_with_Dog_without_Person_overlap']/counts['Person_GT'])
    Path(output).write_text(json.dumps(report,indent=2));return report
