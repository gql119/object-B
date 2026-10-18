"""Detached feature-drift priority history, without any CNLP dependency."""
from dataclasses import asdict,dataclass
import hashlib
import json
import math
from pathlib import Path
import torch


def load_cup_config(path):
    raw=Path(path).read_bytes();value=json.loads(raw)
    if value['schema']!='s5.cup-calibration.v1' or value['layer']!='P3':
        raise ValueError('Require the measured selected-P3 calibration')
    keys=('beta','temperature','damage_clip','min_weight_factor','max_weight_factor','normalization_floor')
    return S5CUPConfig(**{k:value[k] for k in keys},
        known_scales={int(k):v for k,v in value['known_scales'].items()},
        calibration_sha256=hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True)
class S5CUPConfig:
    beta: float
    temperature: float
    damage_clip: float
    min_weight_factor: float
    max_weight_factor: float
    normalization_floor: float
    known_scales: dict
    calibration_sha256: str

    def __post_init__(self):
        vals=(self.beta,self.temperature,self.damage_clip,self.min_weight_factor,
              self.max_weight_factor,self.normalization_floor)
        if not all(math.isfinite(x) for x in vals):raise ValueError('Nonfinite CUP settings')
        if not (0<=self.beta<1 and self.temperature>0 and self.damage_clip>=1
                and 0<self.min_weight_factor<=1<=self.max_weight_factor
                and self.normalization_floor>0):raise ValueError('Invalid CUP bounds/normalization')
        if len(self.calibration_sha256)!=64:raise ValueError('Missing calibration identity')
        for k,v in self.known_scales.items():
            if type(k) is not int or k==14 or not 0<=k<20 or not math.isfinite(v) or v<self.normalization_floor:
                raise ValueError('Invalid train-calibrated scale')


class S5CUPHistory:
    def __init__(self,config):
        self.config=config
        self.raw_mean=torch.zeros(20,dtype=torch.float64)
        self.mean=torch.zeros(20,dtype=torch.float64)
        self.scale=torch.zeros(20,dtype=torch.float64)
        self.observations=torch.zeros(20,dtype=torch.long)
        self.steps=0
        for k,v in config.known_scales.items():self.scale[k]=v

    @staticmethod
    def _classes(classes):
        if len(set(classes))!=len(classes) or any(type(k) is not int or k==14 or not 0<=k<20 for k in classes):
            raise ValueError('Invalid non-target class set')

    def weights(self,classes,pi,reference):
        classes=list(classes);self._classes(classes)
        pi=torch.as_tensor(pi,dtype=torch.float64,device='cpu').detach()
        if pi.shape!=(len(classes),) or not torch.isfinite(pi).all() or (pi<=0).any():
            raise ValueError('Invalid original instance proportions')
        if not classes:return reference.new_empty(0),dict(priority=[],weighted_sum=0.)
        if abs(float(pi.sum())-1)>1e-12:raise ValueError('Original pi must sum to1')
        priority=torch.where(self.observations[classes]>0,self.mean[classes],torch.ones(len(classes),dtype=torch.float64))
        if bool((priority==priority[0]).all()):
            w=torch.ones_like(pi)
        else:
            proposed=torch.softmax(pi.log()+priority/self.config.temperature,0)/pi
            lo,hi=self.config.min_weight_factor,self.config.max_weight_factor
            left,right=float(proposed.min())-hi,float(proposed.max())-lo
            for _ in range(80):
                mid=(left+right)/2
                if float((pi*(proposed-mid).clamp(lo,hi)).sum())>1:left=mid
                else:right=mid
            w=(proposed-(left+right)/2).clamp(lo,hi)
        residual=abs(float((pi*w).sum())-1)
        if residual>1e-12:raise ValueError('Weighted simplex residual')
        if (w<self.config.min_weight_factor-1e-12).any() or (w>self.config.max_weight_factor+1e-12).any():
            raise ValueError('Relative CUP bounds violated')
        allocation=pi*w
        return w.to(reference).detach(),dict(priority=priority.tolist(),
            relative_factors_float64=w.tolist(),absolute_allocation_float64=allocation.tolist(),
            weighted_sum=float(allocation.sum()),projection_residual=residual,
            entropy=float(-(allocation*allocation.log()).sum()))

    def commit(self,eot_damage):
        grouped={}
        for sample in eot_damage:
            self._classes(list(sample))
            for k,v in sample.items():
                v=float(v.detach()) if torch.is_tensor(v) else float(v)
                if not math.isfinite(v) or v<0:raise FloatingPointError('Invalid feature-drift damage')
                grouped.setdefault(k,[]).append(v)
        planned=[]
        for k,values in grouped.items():
            raw=sum(values)/len(values)
            scale=float(self.scale[k]) or max(raw,self.config.normalization_floor)
            normalized=raw/scale
            if not math.isfinite(normalized):raise FloatingPointError('Nonfinite normalized priority')
            clipped=min(normalized,self.config.damage_clip)
            planned.append((k,raw,scale,clipped,len(values),normalized))
        rows=[]
        # Everything is validated before modifying checkpointed state.
        for k,raw,scale,value,count,unclipped in planned:
            seen=bool(self.observations[k])
            self.scale[k]=scale
            self.raw_mean[k]=self.config.beta*self.raw_mean[k]+(1-self.config.beta)*raw if seen else raw
            self.mean[k]=self.config.beta*self.mean[k]+(1-self.config.beta)*value if seen else value
            self.observations[k]+=count
            rows.append(dict(cls=k,raw_damage=raw,scale=scale,normalized_damage=unclipped,
                clipped_damage=value,was_clipped=unclipped>self.config.damage_clip,
                priority_after=float(self.mean[k]),observations=int(self.observations[k])))
        self.steps+=1
        return rows

    def state_dict(self):
        return dict(config=asdict(self.config),steps=self.steps,raw_mean=self.raw_mean.clone(),
                    mean=self.mean.clone(),scale=self.scale.clone(),observations=self.observations.clone())

    def load_state_dict(self,state):
        if state['config']!=asdict(self.config) or type(state['steps']) is not int or state['steps']<0:
            raise ValueError('CUP checkpoint identity mismatch')
        for key in ('mean','raw_mean','scale','observations'):
            value=state[key];expected=getattr(self,key)
            if (not torch.is_tensor(value) or value.device.type!='cpu' or value.shape!=expected.shape
                    or value.dtype!=expected.dtype or not torch.isfinite(value).all()
                    or bool((value<0).any()) or value[14]!=0):
                raise ValueError('Invalid CUP checkpoint field '+key)
        if bool((state['mean']>self.config.damage_clip).any()):raise ValueError('Priority outside clip bound')
        for k,v in self.config.known_scales.items():
            if state['scale'][k]!=v:raise ValueError('Known calibrated class scale changed')
        if bool(((state['observations']>0)&(state['scale']<self.config.normalization_floor)).any()):
            raise ValueError('Observed class has no valid scale')
        for key in ('mean','raw_mean','scale','observations'):getattr(self,key).copy_(state[key])
        self.steps=state['steps']
