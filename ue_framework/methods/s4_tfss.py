"""S4 production wiring. Parent loss and the validated TFSS formulas are reused."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F

from .b2_semantic import B2SemanticTrainer
from .tfss_clean_maps import build_maps, canonical_map_canvas
from .tfss_spatial import SoftSpatialProjection, spatial_weights


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(rng):
    random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda'])


def model_hash(model):
    h = hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class CleanMapCache:
    """Clean-only maps keyed by image bytes and image-local padding, never poison.

    Each image uses its own ceil-to-32 canvas, independent of batch peers, so
    training and per-image export call the same attribution context. RNG is
    restored even on failure; map construction is not an outer update.
    """
    def __init__(self, root, identity):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.hits = self.misses = 0

    def get(self, trainer, items, paths):
        if len(items) != len(paths) or not items:
            raise ValueError('Clean-map input identity missing')
        result, records = [], []
        for item,path in zip(items,paths):
            image,ann,support = item
            canvas = canonical_map_canvas(image)
            identity = dict(self.identity, path=str(Path(path).resolve()), image_sha256=digest(path),
                annotations=ann, canvas=list(canvas), shape=list(image.shape))
            key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            npz = self.root/(key+'.npz'); meta = self.root/(key+'.json')
            if npz.exists() != meta.exists():
                raise ValueError('Partial map-cache artifact: preserve and diagnose '+key)
            if npz.exists():
                record = json.loads(meta.read_text())
                if record['identity'] != identity or digest(npz) != record['npz_sha256']:
                    raise ValueError('Clean cache identity/hash mismatch')
                with np.load(npz,allow_pickle=False) as z:
                    values = tuple(torch.from_numpy(z[k].copy()).to(trainer.device) for k in ('q','risk','valid'))
                self.hits += 1
            else:
                if shutil.disk_usage(self.root).free < 64*1024**3:
                    raise RuntimeError('Insufficient free disk for clean maps; retain all data')
                before = rng_state()
                try:
                    maps,evidence = build_maps(trainer,[item])
                finally:
                    restore_rng(before)
                values = maps[0]
                np.savez_compressed(npz, q=values[0].cpu().numpy(),risk=values[1].cpu().numpy(),valid=values[2].cpu().numpy())
                record = dict(identity=identity, map_evidence=evidence[0], npz_sha256=digest(npz))
                meta.write_text(json.dumps(record,indent=2,allow_nan=False))
                self.misses += 1
            q,risk,valid = values
            if tuple(q.shape) != support.shape or tuple(risk.shape) != support.shape or valid.dtype != torch.bool:
                raise ValueError('Invalid map geometry/dtype')
            result.append(values); records.append(record['map_evidence'])
        return result,records


class S4Trainer(SoftSpatialProjection,B2SemanticTrainer):
    """Only final original a0-composed delta is multiplied by detached W_U."""
    def _batch_loss(self,items,prototypes):
        if self.tfss_mode != 'S0':
            maps, records = self.s4_cache.get(self,items,self.s4_item_paths)
            self.s4_maps, self.s4_map_records = maps,records
            ph,pw = ((max(x[0].shape[d] for x in items)+31)//32*32 for d in (0,1))
            ws = [F.pad(spatial_weights(q,r,torch.as_tensor(s,device=self.device),mode='TFSS-U',eta=.85),
                        (0,pw-s.shape[1],0,ph-s.shape[0]),value=1)
                  for (_,_,s),(q,r,_) in zip(items,maps)]
            self.tfss_weight = torch.stack(ws)[:,None]
        result = super()._batch_loss(items,prototypes)
        if result is None:
            raise ValueError('Canonical S0 produced no valid real supervision')
        loss,metrics = result
        actual = ((self.tfss_input+self.tfss_delta).clamp(0,1)-self.tfss_input).detach()
        support = self.tfss_support
        if not torch.isfinite(actual).all() or bool((actual.abs()>support*self.eps+1e-7).any()):
            raise ValueError('Actual float delta violates original8/4/0 budget')
        if bool((actual.masked_select((support==0).expand_as(actual)) != 0).any()):
            raise ValueError('Outside Person delta nonzero')
        if getattr(self,'s4_measure_protection',False):
            term = metrics['weighted_L_BG']+metrics['weighted_L_rel']+self.lambda_adapt*metrics['L_adapt_target']
            gp = torch.autograd.grad(term,self.fourier_coeff,retain_graph=True)[0]
            metrics['s4_protection_Fourier_norm'] = float(gp.norm())
        else:
            metrics['s4_protection_Fourier_norm'] = None
        if self.tfss_mode != 'S0':
            rows=[]
            base=((self.tfss_input+self.tfss_s0_delta).clamp(0,1)-self.tfss_input).detach()
            for i,((im,ann,s),(q,r,valid),rec) in enumerate(zip(items,self.s4_maps,self.s4_map_records)):
                h,w=im.shape[:2]; dom=torch.as_tensor(s>0,device=self.device)
                d=actual[i,:,:h,:w];b=base[i,:,:h,:w];weight=self.tfss_applied_weight[i,0,:h,:w]
                def energy(mask):
                    denominator=b[:,mask].double().square().sum()
                    return float(d[:,mask].double().square().sum()/denominator) if denominator>0 else None
                hi=(q>=.9)&dom&valid; low=(r>=.9)&(q<=.1)&dom&valid
                row=dict(image_id=Path(self.s4_item_paths[i]).stem,person_rms=float(d[:,dom].double().square().mean().sqrt()),
                    base_person_rms=float(b[:,dom].double().square().mean().sqrt()),
                    mean_abs=float(d[:,dom].abs().double().mean()),weight_mean=float(weight[dom].mean()),
                    weight_min=float(weight[dom].min()),weight_max=float(weight[dom].max()),
                    high_Q_energy=energy(hi),high_R_energy=energy((r>=.9)&dom),high_R_low_Q_energy=energy(low),
                    actual_linf=float(d.abs().max()),overlap_linf=float(d[:,torch.as_tensor(s==.5,device=self.device)].abs().max()) if (s==.5).any() else 0.,
                    saturation_fraction=float((d[:,dom].abs()>=torch.as_tensor(s,device=self.device)[dom]*self.eps-1e-7).float().mean()),
                    no_nt_gt=int(rec['no_nt_gt']),small_gt_fallback=sum(x.get('reason')=='canonical_P5_pool_undercoverage' for x in rec['target_objects']),
                    invalid_Q_objects=sum(not x['valid'] for x in rec['target_objects']))
                for name,v in [('Q',q[dom]),('R',r[dom]),('W',weight[dom])]:
                    for p,x in zip(('min','q10','q50','q90','max'),torch.quantile(v.float(),v.new_tensor([0.,.1,.5,.9,1.]))):row[name+'_'+p]=float(x)
                rows.append(row)
            self.s4_spatial_rows=rows
            for k in rows[0]:
                if k=='image_id':continue
                values=[r[k] for r in rows if r[k] is not None]
                metrics['s4_'+k]=(max(values) if k in ('actual_linf','overlap_linf') else sum(values)/len(values)) if values else None
        return loss,metrics
