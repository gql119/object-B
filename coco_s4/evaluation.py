"""Official COCOeval array reduction and the project's signed paired metrics."""
import math
import numpy as np

def mean_valid(values):
    v=np.asarray(values);v=v[v>=0]
    return float(v.mean()) if v.size else None

def per_class(coco_eval,mapping):
    p=coco_eval.params
    if list(p.catIds)!=[x['category_id'] for x in mapping] or p.maxDets[-1]!=100 or p.areaRngLbl[0]!='all':raise ValueError('Expected standard COCO80 all-area maxDets100 evaluation')
    precisions=coco_eval.eval['precision'];recalls=coco_eval.eval['recall'];rows=[]
    i50=int(np.flatnonzero(np.isclose(p.iouThrs,.5))[0]);i75=int(np.flatnonzero(np.isclose(p.iouThrs,.75))[0])
    for k,c in enumerate(mapping):
        rows.append(dict(c,AP50=mean_valid(precisions[i50,:,k,0,-1]),AP75=mean_valid(precisions[i75,:,k,0,-1]),AP50_95=mean_valid(precisions[:,:,k,0,-1]),Recall50=mean_valid(recalls[i50,k,0,-1:])))
    return rows

def paired_metrics(clean,poison,target=0):
    if len(clean)!=80 or len(poison)!=80 or {x['yolo_class_id'] for x in clean}!=set(range(80)):raise ValueError('Need complete matched COCO80 rows')
    a={x['yolo_class_id']:x for x in clean};b={x['yolo_class_id']:x for x in poison}
    if set(a)!=set(b):raise ValueError('Class identities differ')
    for cls in a:
        for f in ['AP50','AP75','AP50_95']:
            if any(v is None or not math.isfinite(v) or not 0<=v<=1 for v in (a[cls][f],b[cls][f])):raise ValueError('Missing/invalid official per-class AP')
    nt=[c for c in a if c!=target];ac=sum(a[c]['AP50'] for c in nt)/len(nt);bc=sum(b[c]['AP50'] for c in nt)/len(nt)
    P=1-b[target]['AP50']/a[target]['AP50'] if a[target]['AP50']>0 else None
    U=bc/ac if ac>0 else None
    S=2*P*U/(P+U) if P is not None and U is not None and P+U!=0 else None
    damage={c:1-b[c]['AP50']/a[c]['AP50'] if a[c]['AP50']>0 else None for c in nt}
    ranked=sorted((v,c) for c,v in damage.items() if v is not None)[::-1]
    complete=len(ranked)==len(nt)
    return {'P':P,'U':U,'S':S,'WCD':ranked[0][0] if complete else None,'TCD3':sum(v for v,c in ranked[:3])/3 if complete else None,'damage_by_class':damage,'raw_signed_no_clip':True,'zero_denominator':'null'}
