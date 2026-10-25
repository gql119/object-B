"""Independent frozen24D projection probe, with all RNG state restored."""
from pathlib import Path
import torch

from .runtime import ROOT,sha
from .phase_contract import OWNED
from ue_framework.methods.safebg import safe_background_direction


def probe(binding,device,teacher_checkpoint):
    reference=Path(binding['path']).resolve()
    if (not reference.is_relative_to((OWNED/'releases').resolve())
            or sha(reference)!=binding['sha256']):
        raise ValueError('Frozen pre-adaptation SafeBG reference source drift')
    ns={};exec(compile(reference.read_bytes(),str(reference),'exec'),ns)
    before_cpu=torch.get_rng_state().clone()
    before_cuda=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
        generator=torch.Generator(device=device).manual_seed(157)
        bases={c:torch.linalg.qr(torch.randn(24,4,device=device,generator=generator)).Q for c in range(20) if c!=14}
        clean=torch.randn(2,24,device=device,generator=generator)
        background=clean+torch.randn(clean.shape,device=device,generator=generator)
        kwargs=dict(mode='low_rank',safe_rank=8,removal_alpha=.5)
        old=ns['safe_background_direction'](clean,background,bases,**kwargs)
        new=safe_background_direction(clean,background,bases,**kwargs)
        if not torch.equal(old[0],new[0]):raise ValueError('Frozen24D GPU/CPU projection output is numerically inconsistent')
        for key,value in old[1].items():
            if not (torch.equal(value,new[1][key]) if torch.is_tensor(value) else value==new[1][key]):
                raise ValueError('Frozen24D projection diagnostic is inconsistent: '+key)
        reference_metric=float(old[0].norm());candidate_metric=float(new[0].norm())
    if not torch.equal(before_cpu,torch.get_rng_state()):raise ValueError('Equivalence probe changed CPU RNG')
    if device.type=='cuda' and not torch.equal(before_cuda,torch.cuda.get_rng_state(device)):
        raise ValueError('Equivalence probe changed CUDA RNG')
    return dict(reference_id='frozen original SafeBG weightless24D kernel '+binding['sha256'],
        reference_weights_path=str(teacher_checkpoint),candidate_weights_path=str(teacher_checkpoint),
        weights_usage='checkpoint belongs to the production loss path; this isolated projection kernel has no learned weights',
        reference_source_path=str(reference),reference_source_sha256=binding['sha256'],
        candidate_source_path=str(ROOT/'ue_framework/methods/safebg.py'),
        candidate_source_sha256=sha(ROOT/'ue_framework/methods/safebg.py'),
        metric_name='24D projected output Frobenius norm',reference_metric=reference_metric,
        candidate_metric=candidate_metric,absolute_difference=abs(reference_metric-candidate_metric),tolerance=0,
        all_outputs_and_diagnostics_exact=True,RNG_restored=True,actual_tensor_device=str(device),
        evaluation_setting=dict(dataset='fixed synthetic24D tensor fixture; no VOC/COCO AP',
            seed=157,sample_count=2,parameters=dict(rank=8,removal_alpha=.5,active_k=3),post_processing='none'))
