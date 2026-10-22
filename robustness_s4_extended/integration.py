"""Version-pinned native detection augmentation integration, process-local only."""
from copy import deepcopy
import json, os, random
from pathlib import Path
import numpy as np
from ultralytics.data.augment import CutMix, MixUp
from ultralytics.data.dataset import YOLODataset
from ue_framework.s4_victim import S4VictimTrainer
from .transforms import cutout, sample_seed, validate, occlusion_fractions

CONTEXT={}

class CutoutTransform:
    def __init__(self,config):self.config=config
    def __call__(self,labels):
        instance=deepcopy(labels['instances']);h,w=labels['img'].shape[:2]
        instance.convert_bbox('xyxy');instance.denormalize(w,h)
        img,info=cutout(labels['img'],instance.bboxes,labels['cls'],self.config,np.random.default_rng(np.random.randint(0,2**32,dtype=np.uint64)))
        labels['img']=img
        # Every GT keeps its full original bbox; unacceptable occlusion rejects the image transform.
        labels['_extended_audit']=info
        return labels

class DetectionCutMix(CutMix):
    """Native CutMix with a stricter donor-visibility filter; no primary object may overlap."""
    def __init__(self,*a,donor_min_visible=.5,min_pixels=2.,**kw):
        super().__init__(*a,**kw);self.donor_min_visible=donor_min_visible;self.min_pixels=min_pixels
    def get_params(self,labels):
        p=super().get_params(labels)
        if p.get('skip'):return p
        donor=deepcopy(labels['mix_labels'][0]['instances']);donor.convert_bbox('xyxy');donor.denormalize(p['w'],p['h'])
        boxes=donor.bboxes;area=p['area'];fractions=occlusion_fractions(boxes,[area])
        lo=np.maximum(boxes[:,:2],area[:2]);hi=np.minimum(boxes[:,2:],area[2:]);wh=np.maximum(hi-lo,0)
        p['indexes2']=np.array([i for i in p['indexes2'] if fractions[i]>=self.donor_min_visible and np.min(wh[i])>=self.min_pixels],dtype=int)
        if not len(p['indexes2']):p['skip']=True
        return p

class SeededOperation:
    def __init__(self,operation,defense,dataset):self.operation=operation;self.defense=defense;self.dataset=dataset;self.calls=0
    def __call__(self,labels):
        # Only the new operation's RNG is locally seeded/restored. Base augmentations retain native worker RNG.
        epoch=int(self.dataset.ext_epoch.value);sid=Path(labels['im_file']).stem
        seed=sample_seed(0,epoch,sid,self.defense)
        a,b=random.getstate(),np.random.get_state()
        random.seed(seed);np.random.seed(seed)
        try:
            result=self.operation(labels)
            probe=deepcopy(result['instances']);h,w=result['img'].shape[:2];probe.convert_bbox('xyxy');probe.denormalize(w,h)
            validate(result['img'],probe.bboxes,result['cls'])
            audit=result.pop('_extended_audit',{})
            # Bounded evidence per worker, not one file or stdout event per training image.
            if self.calls<4:
                folder=self.dataset.ext_audit
                folder.mkdir(parents=True,exist_ok=True)
                with (folder/f'worker_{os.getpid()}.jsonl').open('a') as f:
                    f.write(json.dumps(dict(defense=self.defense,epoch=epoch,sample_id=sid,seed=seed,labels=len(probe),**audit))+'\n')
            self.calls+=1
            return result
        finally:random.setstate(a);np.random.set_state(b)

class ExtendedDataset(YOLODataset):
    def __init__(self,*a,**kw):
        self.ext_config=deepcopy(CONTEXT['config']);self.ext_epoch=CONTEXT['epoch'];self.ext_audit=Path(CONTEXT['audit'])
        super().__init__(*a,**kw)
    def build_transforms(self,hyp=None):
        result=super().build_transforms(hyp)
        if not self.augment:return result
        name=self.ext_config['name'];cfg=self.ext_config
        if name=='cutout':
            result.insert(-1,SeededOperation(CutoutTransform(cfg),name,self))
        else:
            found=False
            for i,t in enumerate(result.transforms):
                if name=='mixup' and type(t) is MixUp:
                    result[i]=SeededOperation(t,name,self);found=True
                if name=='cutmix' and type(t) is CutMix:
                    native=DetectionCutMix(t.dataset,pre_transform=t.pre_transform,p=t.p,beta=cfg['beta'],num_areas=cfg['num_areas'],donor_min_visible=cfg['donor_min_visible'],min_pixels=cfg['min_pixels'])
                    result[i]=SeededOperation(native,name,self);found=True
            if not found:raise RuntimeError('Pinned native augmentation sink absent')
        return result

class ExtendedTrainer(S4VictimTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        if mode!='train':return super().build_dataset(img_path,mode,batch)
        # Factory replacement is scoped to this construction, never edits installed files.
        from ultralytics.data import build
        original=build.YOLODataset
        try:
            build.YOLODataset=ExtendedDataset
            return super().build_dataset(img_path,mode,batch)
        finally:build.YOLODataset=original

def configure(config,audit_dir):
    import multiprocessing as mp
    CONTEXT.update(config=deepcopy(config),epoch=mp.get_context('spawn').Value('i',0),audit=str(audit_dir))

def epoch_callback(trainer):
    CONTEXT['epoch'].value=trainer.epoch
