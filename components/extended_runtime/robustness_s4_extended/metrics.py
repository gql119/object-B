"""Paired signed fractional scores, without clipping or alternate Clean denominators."""
import math
def paired(clean,poison):
    for m in (clean,poison):
        for key in ('AP50','AP75','AP50_95','Recall','Precision'):
            a=m[key+'_per_class']
            if len(a)!=20 or any(not math.isfinite(float(v)) or not 0<=v<=1 for v in a):raise ValueError('Full finite fractional VOC20 metrics required')
    c,p=clean['AP50_per_class'],poison['AP50_per_class'];nt=[i for i in range(20) if i!=14]
    P=1-p[14]/c[14] if c[14] else None
    U=sum(p[i] for i in nt)/sum(c[i] for i in nt) if sum(c[i] for i in nt) else None
    S=2*P*U/(P+U) if P is not None and U is not None and P+U else None
    damage={i:1-p[i]/c[i] if c[i] else None for i in nt};rank=sorted((v for v in damage.values() if v is not None),reverse=True)
    return dict(P=P,U=U,S=S,WCD=rank[0] if rank else None,TCD3=sum(rank[:3])/3 if len(rank)>=3 else None,relative_collateral_by_class=damage,target_clean=c[14],target_S4=p[14],NT_clean=sum(c[i] for i in nt)/19,NT_S4=sum(p[i] for i in nt)/19)
