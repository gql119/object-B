"""CPU-only isolated OP mathematics. No dataset/model/runner imports or training CLI."""
import numpy as np

def orthonormal_basis(weight_rows, rtol=1e-10):
    w=np.asarray(weight_rows,dtype=np.float64)
    if w.ndim!=2 or not np.isfinite(w).all():raise ValueError('Finite 2D weights required')
    if not w.size:return np.empty((w.shape[1],0))
    _,s,v=np.linalg.svd(w,full_matrices=False)
    rank=int(np.sum(s>rtol*s[0])) if len(s) and s[0]>0 else 0
    return v[:rank].T

def remove_projection(x,q):
    x=np.asarray(x,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    if x.ndim!=2 or q.ndim!=2 or x.shape[1]!=q.shape[0] or not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Incompatible finite matrices')
    if not np.allclose(q.T@q,np.eye(q.shape[1]),atol=1e-9):raise ValueError('Basis must be orthonormal')
    return x-(x@q)@q.T

def bounded_reconstruction(x,q,shape):
    raw=remove_projection(x,q)
    clipped=np.clip(raw,0,1)
    return clipped.reshape(shape),dict(effective_rank=q.shape[1],raw_min=float(raw.min()),raw_max=float(raw.max()),clipped_fraction=float(np.mean(raw!=clipped)),projection_rms=float(np.sqrt(np.mean((np.asarray(x)-raw)**2))))
