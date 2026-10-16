"""Diagnostic statistics only; no model or original method modification."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def corr(a, b, mask):
    a, b = np.asarray(a)[mask], np.asarray(b)[mask]
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    def rank(x):
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts-1)/2)[inv]
    return float(np.corrcoef(rank(a), rank(b))[0,1])


def top_mask(values, mask, fraction=.2):
    ids = np.flatnonzero(mask)
    out = np.zeros(mask.size, bool)
    if not len(ids): return out.reshape(mask.shape)
    vals = np.asarray(values).reshape(-1)[ids]
    order = np.lexsort((ids, -vals))  # stable exact count; report flat-map limits separately
    chosen=ids[order[:max(1,math.ceil(fraction*len(ids)))]]
    out[chosen] = True
    return out.reshape(mask.shape)


def share(values, selected, domain=None):
    values = np.asarray(values, dtype=np.float64)
    den = float(values[domain].sum()) if domain is not None else float(values.sum())
    return float(values[selected].sum()/den) if den > 0 else None


def mean(values, mask):
    return float(np.asarray(values)[mask].mean()) if mask.any() else None


def rms(delta, mask):
    return float(np.sqrt(np.square(delta[:,mask].astype(np.float64)).mean())) if mask.any() else None


def norm(v):
    return float(np.linalg.norm(np.asarray(v).reshape(-1).astype(np.float64)))


def cosine(a, b):
    aa,bb=np.asarray(a).ravel().astype(np.float64),np.asarray(b).ravel().astype(np.float64)
    den=np.linalg.norm(aa)*np.linalg.norm(bb)
    return float(np.dot(aa,bb)/den) if den>0 else None


def normalized_attribution(g, mask):
    t=torch.from_numpy(np.asarray(g,dtype=np.float32))[None,None]
    smooth=F.avg_pool2d(t,3,stride=1,padding=1,count_include_pad=False)[0,0].numpy()
    lo,hi=np.quantile(smooth[mask],[.1,.9])
    if hi-lo<=max(float(smooth.max())*1e-7,1e-20):
        return None,dict(valid=False,q10=float(lo),q90=float(hi),reason='constant/degenerate attribution; no invented spatial pattern')
    q=np.clip((smooth-lo)/(hi-lo),0,1).astype(np.float32)
    return q*mask,dict(valid=True,q10=float(lo),q90=float(hi))


def center_weight(shape, bbox, mask, eta=.5, sigma=.5):
    h,w=shape;cx,cy,bw,bh=bbox
    yy,xx=np.mgrid[:h,:w]
    rr=((xx+.5-cx*w)/(bw*w/2))**2+((yy+.5-cy*h)/(bh*h/2))**2
    return (eta+(1-eta)*np.exp(-rr/(2*sigma*sigma)))*mask, np.sqrt(rr)


def smoothness(weight, target):
    dx=target[:,1:]&target[:,:-1];dy=target[1:,:]&target[:-1,:]
    vals=[]
    if dx.any():vals.extend(np.abs(np.diff(weight,axis=1))[dx].tolist())
    if dy.any():vals.extend(np.abs(np.diff(weight,axis=0))[dy].tolist())
    return float(np.mean(vals)) if vals else None


def vector_stats(reference, value, domain):
    a,b=reference[:,domain],value[:,domain];an=norm(a);bn=norm(b)
    active=np.abs(a)>0
    return dict(reference_norm=an,norm=bn,norm_retention=bn/an if an else None,
                cosine=cosine(a,b),sign_agreement=float((np.sign(a[active])==np.sign(b[active])).mean()) if active.any() else None)


def operator_jacobian(base, support, budget, *, eps, literal_s1=False):
    """Execute min/max autograd exactly, including ties and literal legacy branch."""
    x=torch.from_numpy(base).requires_grad_(True)
    a=torch.from_numpy(support)[None];b=torch.from_numpy(budget)[None]
    clip=torch.maximum(torch.minimum(x,b),-b)*(a>0)
    legacy=bool(torch.equal(b,a*eps))
    out=x*a if literal_s1 and legacy else clip
    j=torch.autograd.grad(out.sum(),x)[0]
    return out.detach().numpy(),j.numpy(),legacy


def gradient_energy(gout, jacobian, domain):
    """Scalar-channel energy relative to g_out, including partial RGB blocking/ties."""
    g=np.asarray(gout,dtype=np.float64)[:,domain]
    j=np.asarray(jacobian,dtype=np.float64)[:,domain]
    den=float(np.square(g).sum())
    if den==0:return dict(blocked_scalar_energy_share=None,attenuated_energy_share=None,fully_blocked_pixel_energy_share=None)
    return dict(blocked_scalar_energy_share=float((g*g*(j==0)).sum()/den),
                attenuated_energy_share=float(1-(g*g*j*j).sum()/den),
                fully_blocked_pixel_energy_share=float((g*g)[:,np.all(j==0,axis=0)].sum()/den))


def validate_math():
    x=torch.tensor([-2.,-1.,0.,1.,2.],requires_grad=True)
    y=torch.maximum(torch.minimum(x,torch.tensor(1.)),torch.tensor(-1.))
    j=torch.autograd.grad(y.sum(),x)[0]
    assert torch.equal(j,torch.tensor([0.,.5,1.,.5,0.]))
    assert corr(np.ones((2,2)),np.ones((2,2)),np.ones((2,2),bool)) is None
    w,_=center_weight((9,9),[.5,.5,1,1],np.ones((9,9),bool))
    assert .5<=w.min()<=w.max()<=1 and w[4,4]==w.max()
    m=top_mask(np.ones((5,5)),np.ones((5,5),bool),.2)
    assert m.sum()==5
    e=gradient_energy(np.array([1.,2.,3.])[:,None,None],np.array([0.,1.,1.])[:,None,None],np.ones((1,1),bool))
    assert abs(e['blocked_scalar_energy_share']-1/14)<1e-12 and e['fully_blocked_pixel_energy_share']==0
    t=gradient_energy(np.ones((3,1,1)),np.full((3,1,1),.5),np.ones((1,1),bool))
    assert t['blocked_scalar_energy_share']==0 and t['attenuated_energy_share']==.75
    x=.99;requested=.03
    assert abs((min(1,x+requested*.5)-x)/(min(1,x+requested)-x)-1)<1e-12
    cap=.031372549
    support=np.ones((1,1),np.float32);budget=support*cap
    base=np.full((3,1,1),float(budget[0,0]),np.float32)
    _,jj,legacy=operator_jacobian(base,support,budget,eps=cap,literal_s1=True)
    assert legacy and np.all(jj==1)
    _,jj,legacy=operator_jacobian(base,support,budget,eps=cap,literal_s1=False)
    assert legacy and np.all(jj==.5)
    return dict(boundary_jacobian=j.tolist(),partial_rgb_energy=e,boundary_energy=t,
                realized_clamped_amplitude_retention=1.,actual_decimal_epsilon_legacy_verified=True,
                ties='min/max splits equally at +/-cap: 0.5; outside mask multiplication is zero; exact legacy branch uses actual configured epsilon',tests_passed=9)
