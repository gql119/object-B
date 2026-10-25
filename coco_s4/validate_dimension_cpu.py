"""Measured VOC24 equivalence and dynamic selected-space CPU contracts.

Synthetic features exercise the real SafeBG/Fourier/loss primitives. A learned
Teacher full-loss CPU/CUDA test remains a separate required production gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch

from .cache import OnDemandSafeBGTripletCache
from .dimension_protocol import PROTOCOL, selected_dimension, validate_identity
from .runtime import ROOT, now, sha, write_once
from ue_framework.methods.fourier import build_fourier_pattern, sample_midfreq_coords
from ue_framework.methods.legacy_kproto_ret import kproto_losses
from ue_framework.methods.safebg import SafeBGTripletCache, safe_background_direction, select_strict_tsfl
from ue_framework.methods.semantic_coordinates import normalized_orth, project_bases_to_channels

REFERENCE_COMMIT='3a8db4c4277d2ffbcadef7864240063d84fff6b9'


def exact_outputs(first, second):
    assert torch.equal(first[0],second[0]), 'Safe-BG output mismatch'
    assert first[1].keys()==second[1].keys()
    for key,a in first[1].items():
        b=second[1][key]
        assert torch.equal(a,b) if torch.is_tensor(a) else a==b, key


def run(output):
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    reference_bytes=subprocess.check_output(['git','-C',str(ROOT),'show',REFERENCE_COMMIT+':ue_framework/methods/safebg.py'])
    reference={};exec(compile(reference_bytes,'committed_original_safebg.py','exec'),reference)
    original_lock=json.loads((ROOT/'docs/S4_COCO_MINITRAIN_20261023/ORIGINAL_S4_METHOD_SOURCE_LOCK.json').read_text())
    assert hashlib.sha256(reference_bytes).hexdigest()==original_lock['files']['ue_framework/methods/safebg.py']
    comparisons=[]
    for dtype in (torch.float32,torch.float64):
        torch.manual_seed(109)
        bases={c:torch.linalg.qr(torch.randn(24,4,dtype=dtype)).Q for c in range(20) if c!=14}
        x=torch.randn(4,24,dtype=dtype);bg=x+torch.randn_like(x)
        args=dict(mode='low_rank',safe_rank=8,removal_alpha=.5)
        old=reference['safe_background_direction'](x,bg,bases,**args)
        new=safe_background_direction(x,bg,bases,**args)
        exact_outputs(old,new)
        old_cache=reference['SafeBGTripletCache'](bases)
        candidate_cache=SafeBGTripletCache(bases)
        online_cache=OnDemandSafeBGTripletCache(bases,max_entries=4)
        for cache in (old_cache,candidate_cache,online_cache):
            # Original reference cache has no dimension metadata because its
            # code predates the new guard; execute it through the old function.
            fn=reference['safe_background_direction'] if cache is old_cache else safe_background_direction
            exact_outputs(old,fn(x,bg,bases,triplet_cache=cache,**args))
        assert old_cache.sha256==candidate_cache.sha256
        comparisons.append(dict(dtype=str(dtype),output_max_absolute_difference=0,
                                all_diagnostics_exact=True,eager_cache_projector_hash_exact=True))
    torch.manual_seed(193)
    dimension=39
    bases={c:torch.linalg.qr(torch.randn(dimension,12)).Q for c in range(1,80)}
    cache=OnDemandSafeBGTripletCache(bases,max_entries=3)
    clean=torch.randn(7,dimension);background=clean+torch.randn_like(clean)
    direct=safe_background_direction(clean,background,bases,mode='low_rank',safe_rank=8,removal_alpha=.5)
    cached=safe_background_direction(clean,background,bases,mode='low_rank',safe_rank=8,removal_alpha=.5,triplet_cache=cache)
    exact_outputs(direct,cached)
    assert direct[1]['projection_ratios'].shape==(7,79)
    assert (direct[1]['projector_rank']==8).all() and direct[1]['orthonormal_error'].max()<=1e-5
    assert torch.allclose(direct[1]['d_safe_scaled_norm'],direct[1]['d_bg_norm'],rtol=1e-6,atol=1e-6)
    for first in range(1,70):cache.lookup([first,first+1,first+2])
    memory=cache.memory_report()
    assert memory['entries']==3 and memory['entry_bytes']<=memory['entry_upper_bound_bytes']
    # Runtime sink ranking retains every79 class, including a deliberately
    # dominant class79; reversed mapping insertion cannot change the result.
    probe=clean[0:1];direction=bases[79][:,0][None,:]
    rank_a=safe_background_direction(probe,probe+direction,bases,mode='low_rank',safe_rank=8,removal_alpha=.5)
    rank_b=safe_background_direction(probe,probe+direction,dict(reversed(list(bases.items()))),mode='low_rank',safe_rank=8,removal_alpha=.5)
    exact_outputs(rank_a,rank_b);assert rank_a[1]['active_classes'][0][0]==79
    amplitudes=torch.linspace(.2,.9,16,requires_grad=True)
    pattern=build_fourier_pattern(16,16,sample_midfreq_coords(16,16,16,0),amplitudes,torch.device('cpu'))
    feature_map=torch.randn(dimension,256)
    poison=clean[:1]+.03*(feature_map@pattern.flatten())[None,:]
    safe=safe_background_direction(clean[:1],background[:1],bases,mode='low_rank',safe_rank=8,removal_alpha=.5)[0]
    losses=kproto_losses(poison,clean[:1],safe,torch.empty(0,dimension),torch.empty(0,dimension),.2)
    gradients={}
    for name,loss in zip(('L_BG','L_rel'),losses[:2]):
        grad=torch.autograd.grad(loss,amplitudes,retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.norm()>0,name
        gradients[name]=dict(loss=float(loss.detach()),Fourier_gradient_norm=float(grad.norm()))
    # P3 remains a separate64D layer with48 channels under its .75 rule.
    p3_bases={c:torch.linalg.qr(torch.randn(64,8)).Q for c in range(1,80)}
    p3_indices=torch.arange(48);orth=project_bases_to_channels(p3_bases,p3_indices)
    p3_delta=torch.randn(2,48,requires_grad=True)
    before=normalized_orth(p3_delta,orth,torch.ones(2))
    safe_background_direction(clean,background,bases,mode='low_rank',safe_rank=8,removal_alpha=.5,triplet_cache=cache)
    after=normalized_orth(p3_delta,orth,torch.ones(2))
    assert torch.equal(before[0],after[0]) and torch.equal(before[1],after[1]) and before[2]==after[2]
    idx=torch.arange(dimension,dtype=torch.long)
    identity=dict(dimension_protocol=PROTOCOL,selected_space_dimension=dimension,
                  selected_channel_indices=idx.tolist(),num_classes=80,target_class_id=0)
    assert validate_identity(identity,idx,256)==dimension
    rejected=0
    for invalid in (idx.flip(0),torch.zeros(dimension,dtype=torch.long),torch.arange(7),torch.arange(260)):
        try:selected_dimension(invalid,256)
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid coordinate accepted')
    for invalid in (dict(bases,**{}),):
        invalid[1]=torch.zeros(dimension,12)
        try:OnDemandSafeBGTripletCache(invalid)
        except ValueError:rejected+=1
        else:raise AssertionError('Degenerate class silently accepted')
    collapsed={c:torch.eye(dimension)[:,:2] for c in range(1,80)}
    try:OnDemandSafeBGTripletCache(collapsed).lookup([1,2,3])
    except ValueError:rejected+=1
    else:raise AssertionError('Rank8 was silently reduced')
    assert not torch.cuda.is_initialized()
    report=dict(state='passed',UTC=now(),protocol=PROTOCOL,reference_commit=REFERENCE_COMMIT,
                reference_source_sha256=hashlib.sha256(reference_bytes).hexdigest(),VOC24_equivalence=comparisons,
                COCO39_rank=8,COCO39_all_NT_classes=79,COCO39_direct_cache_exact=True,
                class79_ranks_first=True,cache_memory=memory,synthetic_Fourier_gradients=gradients,
                P3_coordinates=dict(full_dimension=64,selected_dimension=48,loss_and_conflicts_exact=True),
                invalid_cases_rejected=rejected,GPU_initialized=False,
                real_Teacher_full_loss_CPU='OPEN',current_candidate_GPU='OPEN',scientific_review='OPEN',
                sources={name:sha(ROOT/name) for name in ('ue_framework/methods/safebg.py','ue_framework/methods/b2_semantic.py',
                    'coco_s4/cache.py','coco_s4/dimension_protocol.py','coco_s4/calibration.py')})
    write_once(output/'report.json',report)
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
