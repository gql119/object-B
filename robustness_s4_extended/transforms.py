"""Class-agnostic Cutout on uint8 images and absolute xyxy detection boxes."""
import hashlib
import numpy as np

def sample_seed(seed, epoch, sample_id, operation):
    key=f'{seed}|{epoch}|{sample_id}|{operation}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')

def validate(image, boxes, classes):
    boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    classes=np.asarray(classes).reshape(-1)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected uint8 HWC three-channel image')
    h,w=image.shape[:2]
    if len(boxes)!=len(classes) or not np.isfinite(boxes).all():raise ValueError('Invalid detection labels')
    if not np.isfinite(classes).all() or np.any(classes!=classes.astype(int)) or np.any((classes<0)|(classes>=20)):
        raise ValueError('VOC20 integer class IDs required')
    if len(boxes) and (np.any(boxes[:,:2]<0) or np.any(boxes[:,2]>w) or np.any(boxes[:,3]>h) or np.any(boxes[:,2:]<=boxes[:,:2])):
        raise ValueError('Invalid absolute xyxy boxes')
    return boxes,classes

def occlusion_fractions(boxes, rectangles):
    """Continuous box/union-of-rectangles area, including overlapping holes exactly."""
    result=[]
    for box in np.asarray(boxes).reshape(-1,4):
        clips=[]
        for r in rectangles:
            a,b=max(box[0],r[0]),max(box[1],r[1]);c,d=min(box[2],r[2]),min(box[3],r[3])
            if c>a and d>b:clips.append((a,b,c,d))
        xs=sorted({v for r in clips for v in (r[0],r[2])});area=0.
        for l,r in zip(xs,xs[1:]):
            intervals=sorted((q[1],q[3]) for q in clips if q[0]<r and q[2]>l)
            height=0.;end=-float('inf')
            for y,z in intervals:
                height+=max(0,z-max(y,end));end=max(end,z)
            area+=(r-l)*height
        result.append(area/((box[2]-box[0])*(box[3]-box[1])))
    return np.array(result,dtype=np.float64)

def cutout(image, boxes, classes, config, rng, *, rectangles=None):
    boxes,classes=validate(image,boxes,classes)
    output=image.copy()
    info=dict(applied=False,reason='probability',rectangles=[],occlusion_per_gt=[0.]*len(boxes),class_ids=classes.astype(int).tolist())
    if rng.random()>=config['probability']:return output,info
    h,w=image.shape[:2]
    for attempt in range(config['max_attempts']):
        holes=[]
        for _ in range(config['n_holes']):
            a=rng.uniform(*config['area_fraction']);ratio=np.sqrt(a)
            cw=max(1,round(w*ratio));ch=max(1,round(h*ratio));cx=int(rng.integers(w));cy=int(rng.integers(h))
            holes.append((max(0,cx-cw//2),max(0,cy-ch//2),min(w,cx+(cw+1)//2),min(h,cy+(ch+1)//2)))
        if rectangles is not None:
            holes=[tuple(int(v) for v in r) for r in rectangles]
            if any(not (0<=a<c<=w and 0<=b<d<=h) for a,b,c,d in holes):raise ValueError('Invalid hole')
        fractions=occlusion_fractions(boxes,holes)
        info.update(candidate_occlusion_per_gt=fractions.tolist(),attempts=attempt+1)
        if len(fractions) and fractions.max()>config['max_gt_occlusion']:
            info['reason']='reject_whole_candidate_any_class_severe_occlusion'
            if rectangles is not None:break
            continue
        for a,b,c,d in holes:output[b:d,a:c]=config['fill_value']
        info.update(applied=True,reason='accepted',rectangles=holes,occlusion_per_gt=fractions.tolist())
        break
    return output,info
