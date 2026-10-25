"""Bounded cache of exact ordered SVDs; unchanged Safe-BG projection formula."""
from collections import OrderedDict
import hashlib
import torch

class OnDemandSafeBGTripletCache:
    def __init__(self,class_bases,projector_rtol=1e-6,safe_rank=8,max_entries=8192):
        self.classes=tuple(sorted(class_bases))
        if len(self.classes)<3 or safe_rank!=8 or max_entries<1:
            raise ValueError('Need complete class bases, rank8 and a positive cache bound')
        first=class_bases[self.classes[0]]
        if first.ndim!=2 or first.shape[0]<safe_rank or not 0<projector_rtol<1:
            raise ValueError('Invalid selected dimension or projector tolerance')
        self.selected_dim=first.shape[0];self.device,self.dtype=first.device,first.dtype
        self.projector_rtol=projector_rtol;self.safe_rank=safe_rank;self.max_entries=max_entries
        self.bases={};self.entries=OrderedDict();h=hashlib.sha256()
        for cls in self.classes:
            b=class_bases[cls]
            if b.ndim!=2 or b.shape[0]!=self.selected_dim or not torch.isfinite(b).all() or b.device!=self.device or b.dtype!=self.dtype:
                raise ValueError(f'Invalid class basis {cls}')
            if b.shape[1]==0 or torch.linalg.matrix_rank(b,rtol=projector_rtol)!=b.shape[1]:
                raise ValueError(f'Invalid class rank {cls}; no silent omission')
            gram=(b.T@b-torch.eye(b.shape[1],device=b.device,dtype=b.dtype)).abs().max()
            if not torch.isfinite(gram) or gram>1e-5:
                raise ValueError(f'Nonorthonormal class basis {cls}')
            self.bases[cls]=b.detach().clone()
            h.update(str(cls).encode());h.update(b.detach().cpu().contiguous().numpy().tobytes())
        self.sha256=h.hexdigest();self.precompute_seconds=0.;self.misses=0

    def lookup(self,active):
        order=tuple(active)
        if len(order)!=3 or len(set(order))!=3 or any(x not in self.bases for x in order):
            raise ValueError('Exactly three distinct known active classes required')
        if order in self.entries:
            self.entries.move_to_end(order);return self.entries[order]
        # Same concatenation ORDER and same SVD as the original six variants.
        concatenated=torch.cat([self.bases[c] for c in order],dim=1)
        u,singular,_=torch.linalg.svd(concatenated,full_matrices=False)
        if u.shape[1]<self.safe_rank or not torch.isfinite(singular).all() or singular[self.safe_rank-1]<=singular.max()*self.projector_rtol:
            raise ValueError(f'Invalid rank8 triplet {order}')
        q=u[:,:self.safe_rank]
        gram=(q.T@q-torch.eye(self.safe_rank,device=self.device,dtype=self.dtype)).abs().max()
        rank=torch.linalg.matrix_rank(q@q.T,rtol=self.projector_rtol)
        if not torch.isfinite(gram) or gram>1e-5 or int(rank)!=self.safe_rank:
            raise ValueError('Invalid Safe-BG projector')
        energy=singular.square()
        # Compact owned tensors keep the bound proportional to D*rank, rather
        # than retaining the full U storage through a tensor view.
        value=(q.clone(),singular[:16].clone(),gram,rank,energy[:8].sum()/energy.sum().clamp_min(1e-12))
        self.entries[order]=value;self.misses+=1
        if len(self.entries)>self.max_entries:self.entries.popitem(last=False)
        return value

    def memory_report(self):
        element_bytes=torch.empty((),dtype=self.dtype).element_size()
        per_entry=(self.selected_dim*self.safe_rank+min(self.selected_dim,16)+2)*element_bytes+8
        return dict(selected_dim=self.selected_dim,classes=len(self.classes),entries=len(self.entries),
                    max_entries=self.max_entries,misses=self.misses,
                    basis_bytes=sum(b.numel()*b.element_size() for b in self.bases.values()),
                    entry_bytes=sum(v.numel()*v.element_size() for entry in self.entries.values() for v in entry),
                    entry_upper_bound_bytes=self.max_entries*per_entry)
