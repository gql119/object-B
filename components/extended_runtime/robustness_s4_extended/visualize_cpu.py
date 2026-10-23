"""Small synthetic train-image/GT visual audit using actual production transform classes."""
import copy,json,random
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .common import ROOT,load,CONFIG,atomic,sha
from .transforms import cutout

def main():
    from .tests.test_defenses import NativeTests,CFG
    from .integration import DetectionCutMix
    from ultralytics.data.augment import MixUp
    NativeTests.setUpClass();helper=NativeTests()
    records=[];pairs=[]
    cases=[('person',[[20,10,60,60]],[14],(0,0,20,20)),('person+dog',[[0,0,20,20],[30,30,60,60]],[14,11],(30,30,60,60)),('person+horse overlap',[[15,10,60,60],[5,30,70,60]],[14,12],(25,30,45,50)),('no person',[[20,10,60,60]],[6],(0,0,20,20)),('small target',[[10,10,12,12]],[14],(10,10,12,12)),('full occlusion rejected',[[0,0,80,64]],[14],(0,0,80,64)),('boundary',[[5,5,50,50]],[14],(0,0,6,6)),('empty GT',[],[],(0,0,30,30))]
    for name,boxes,cls,hole in cases:
        a=np.full((64,80,3),[130,190,170],np.uint8);b,info=cutout(a,boxes,cls,CFG,np.random.default_rng(0),rectangles=[hole]);pairs.append((name,a,b,boxes,boxes,cls,cls,info['reason']));records.append(dict(case=name,**info))
    a=helper.labels([[30,30,60,60]],[14]);b=helper.labels([[2,2,8,8],[5,5,15,15]],[11,12],220)
    t=DetectionCutMix(helper.dataset(b),p=1.);t._rand_bbox=lambda w,h:(0,0,10,10);r=t(copy.deepcopy(a));pairs.append(('CutMix native+safe donor',a['img'],r['img'],a['instances'].bboxes,r['instances'].bboxes,[14],r['cls'].reshape(-1),'donor horse <50% visible excluded'))
    t=MixUp(helper.dataset(b),p=1.);random.seed(0);np.random.seed(0);r=t(copy.deepcopy(a));pairs.append(('MixUp native',a['img'],r['img'],a['instances'].bboxes,r['instances'].bboxes,[14],r['cls'].reshape(-1),'3 boxes retained; no class loss mixing'))
    canvas=Image.new('RGB',(900,len(pairs)*195),'white');draw=ImageDraw.Draw(canvas)
    for row,(name,a,b,ba,bb,ca,cb,note) in enumerate(pairs):
        y=row*195;draw.text((10,y+6),name,fill='black');draw.text((10,y+25),note,fill='black')
        for x,pix,boxes,cls in [(10,a,ba,ca),(320,b,bb,cb)]:
            im=Image.fromarray(pix).resize((240,144),Image.Resampling.NEAREST);d=ImageDraw.Draw(im)
            for box,k in zip(boxes,cls):
                h,w=pix.shape[:2];x1,y1,x2,y2=box;d.rectangle((x1*240/w,y1*144/h,x2*240/w,y2*144/h),outline='red' if int(k)==14 else 'blue',width=2);d.text((x1*240/w,y1*144/h),str(int(k)),fill='black')
            canvas.paste(im,(x,y+46))
    p=ROOT/'robustness_s4_extended/reports/CPU_VISUAL_AUDIT.png';canvas.save(p)
    atomic(p.with_suffix('.json'),dict(scope='synthetic CPU inputs; no VOC or poison image transferred; actual native mix classes',cutout_cases=records,png_sha256=sha(p)))
    print(p)
if __name__=='__main__':main()
