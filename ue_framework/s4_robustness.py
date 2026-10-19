"""S4-only train-data defenses. No perturbation generator or CUP is used."""
from io import BytesIO
import math
import cv2
import numpy as np
from PIL import Image, JpegImagePlugin

OFFLINE = ('jpeg50','jpeg30','gray','bdr','gauss','meanf','medianf')

def transform_rgb(x, defense, params=None):
    params=dict(params or {})
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError('Expected original-resolution HWC uint8 RGB')
    if defense in ('none','data_aug'):
        return x.copy()
    if defense in ('jpeg50','jpeg30'):
        quality=int(params.get('quality',50 if defense=='jpeg50' else 30))
        sampling=int(params.get('subsampling',2))
        assert quality==(50 if defense=='jpeg50' else 30) and sampling==2
        assert params.get('optimize',False) is False and params.get('progressive',False) is False
        buffer=BytesIO()
        Image.fromarray(x).save(buffer,format='JPEG',quality=quality,
            subsampling=sampling,optimize=False,progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            assert JpegImagePlugin.get_sampling(decoded)==2
            return np.asarray(decoded.convert('RGB')).copy()
    if defense=='gray':
        return cv2.cvtColor(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
    if defense=='bdr':
        bits=int(params.get('bits',5));assert bits==5
        levels=2**bits-1
        # Exact retained project implementation: float32, 31 levels, ties-to-even.
        z=x.astype(np.float32)/255.0
        return np.clip(np.round((np.round(z*levels)/levels)*255),0,255).astype(np.uint8)
    if defense in ('gauss','meanf'):
        kernel=tuple(params.get('kernel',[3,3]));assert kernel==(3,3)
        if defense=='gauss':
            sigma=float(params.get('sigma',0));assert sigma==0
            return cv2.GaussianBlur(x,kernel,sigma,borderType=cv2.BORDER_DEFAULT)
        return cv2.blur(x,kernel,borderType=cv2.BORDER_DEFAULT)
    if defense=='medianf':
        kernel=int(params.get('kernel',3));assert kernel==3
        return cv2.medianBlur(x,kernel)
    raise ValueError('Unapproved defense: '+defense)

def read_rgb(path):
    bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if bgr is None:raise ValueError('Unreadable image: '+str(path))
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

def write_png(path,x):
    if not cv2.imwrite(str(path),cv2.cvtColor(x,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3]):
        raise RuntimeError('PNG encoding failed: '+str(path))

def paired_scores(clean,poison):
    """Signed ratios, no clipping. Undefined denominators stay unavailable."""
    a=float(clean['mAP50_target']);b=float(clean['mAP50_non_target'])
    if not all(math.isfinite(v) and 0<=v<=1 for v in [a,b,float(poison['mAP50_target']),float(poison['mAP50_non_target'])]):
        raise ValueError('AP must be finite fractions in [0,1]')
    p=None if a==0 else 1-float(poison['mAP50_target'])/a
    u=None if b==0 else float(poison['mAP50_non_target'])/b
    s=None if p is None or u is None or p+u==0 else 2*p*u/(p+u)
    values=clean['AP50_per_class'];other=poison['AP50_per_class']
    if len(values)!=20 or len(other)!=20:raise ValueError('Expected all VOC20 classes')
    damages={i:None if values[i]==0 else (values[i]-other[i])/values[i] for i in range(20) if i!=14}
    valid=sorted([(i,v) for i,v in damages.items() if v is not None],key=lambda x:x[1],reverse=True)
    return dict(P=p,U=u,S=s,WCD=valid[0][1] if len(valid)==19 else None,
        WCD_class_id=valid[0][0] if len(valid)==19 else None,
        TCD3=sum(v for _,v in valid[:3])/3 if len(valid)==19 else None,
        signed_damage=damages,denominator_policy='undefined is null; never clamp negative P/damage or cap U at1')
