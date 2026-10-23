"""Numerical contracts for actual CUP reducers/history, not GPU evidence."""
import copy
from pathlib import Path
import torch
import pytest
from ue_framework.methods.s5_cup_history import S5CUPConfig,S5CUPHistory,load_cup_config
from ue_framework.methods.s5_cup_trainer import S5CUPTrainer


def config():
    return S5CUPConfig(.95,1.,10.,.5,2.,1e-12,{1:1.,11:1.},'a'*64)


def reducer(x,enabled):
    t=object.__new__(S5CUPTrainer)
    t.orth_feature_layer='model.15';t.s5_ret_layer='P3';t.s5_decomposition_noop=True
    t.s5_retention_records={l:[dict(image_index=0,gt_non_target_ordinal=i,
        cls=1 if i<2 else 11,included=True,loss=z.square()) for i,z in enumerate(x)] for l in ('P3','P5')}
    t.cup=S5CUPHistory(config());t.cup_enabled=enabled;t.cup_active_step=False;t.cup_pending=[]
    return t


def test_disabled_and_neutral_recover_original_scalar_and_gradient_exactly():
    for enabled in (False,True):
        x=torch.tensor([.1,.2,.3],requires_grad=True);t=reducer(x,enabled);base=x.square().mean()
        actual=t.s5_reduce_retention(base,x)
        assert actual is base
        assert torch.equal(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(base,x)[0])
        assert all(r['w']==1. for r in t.cup_audit)


def test_weighted_sink_uses_instance_pi_and_retains_class_gradient():
    x=torch.tensor([.1,.2,.3],requires_grad=True);t=reducer(x,True)
    t.cup.commit([{1:3.,11:.25}]);base=x.square().mean();actual=t.s5_reduce_retention(base,x)
    expected=sum(row['pi']*row['w']*t.s5_class_losses['P3'][row['cls']] for row in t.cup_audit)
    assert torch.allclose(actual,expected,atol=1e-8,rtol=1e-6)
    assert torch.allclose(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0],atol=1e-8,rtol=1e-6)
    assert t.cup_audit[0]['w']>t.cup_audit[1]['w']
    assert abs(sum(r['allocation'] for r in t.cup_audit)-1)<1e-7


def test_projection_is_bounded_normalized_detached_and_priority_ordered():
    h=S5CUPHistory(config());classes=[0,1,3,5,11,19]
    h.commit([{k:float(i+1) for i,k in enumerate(classes)}])
    for counts in ([1,1,1,1,1,1],[1,50,3,200,1,20],[200,1,1,1,1,1]):
        pi=[n/sum(counts) for n in counts];w,e=h.weights(classes,pi,torch.tensor(0.,requires_grad=True))
        assert not w.requires_grad and .5<=float(w.min())<=float(w.max())<=2
        assert abs(sum(a*b for a,b in zip(pi,w.tolist()))-1)<1e-7
        assert e['projection_residual']<1e-12
        priority=e['priority']
        assert all((priority[i]-priority[j])*(float(w[i])-float(w[j]))>=-1e-7 for i in range(6) for j in range(6))


def test_eot_once_per_step_missing_hold_unknown_scale_and_state_replay():
    h=S5CUPHistory(config());h.commit([{1:2.,3:2.},{1:4.,3:4.}])
    assert h.steps==1 and h.observations[1]==2 and h.raw_mean[1]==3 and h.mean[1]==3
    assert h.scale[3]==3 and h.mean[3]==1
    before=h.state_dict();h.commit([{}]);assert h.steps==2
    assert torch.equal(h.mean,before['mean']) and torch.equal(h.observations,before['observations'])
    clone=S5CUPHistory(config());clone.load_state_dict(h.state_dict())
    for state in (h,clone):state.commit([{1:1.,3:6.},{1:3.,3:3.}])
    for name in ('mean','raw_mean','scale','observations'):assert torch.equal(getattr(h,name),getattr(clone,name))
    assert h.scale[3]==3 and h.steps==clone.steps==3


def test_invalid_commit_and_checkpoint_preserve_previous_history():
    h=S5CUPHistory(config());h.commit([{1:2.}]);before=h.state_dict()
    with pytest.raises(FloatingPointError):h.commit([{1:1.},{11:float('nan')}])
    assert torch.equal(h.mean,before['mean']) and h.steps==before['steps']
    broken=copy.deepcopy(before);broken['mean'][11]=float('nan')
    with pytest.raises(ValueError):h.load_state_dict(broken)
    assert torch.equal(h.mean,before['mean'])


def test_step_observers_never_change_history_until_commit_and_empty_nt_has_zero_gradient():
    x=torch.tensor([.1,.2,.3],requires_grad=True);t=reducer(x,True);t.cup_last_commit=[]
    t.cup_begin_step()
    for _ in range(2):t.s5_reduce_retention(x.square().mean(),x)
    assert t.cup.steps==0 and len(t.cup_pending)==2 and not t.cup.observations.any()
    t.cup_end_step();assert t.cup.steps==1 and len(t.cup_pending)==0 and t.cup.observations[1]==2
    z=reducer(x,True);z.s5_retention_records={'P3':[],'P5':[]}
    actual=z.s5_reduce_retention(x.sum()*0,x)
    assert torch.equal(torch.autograd.grad(actual,x)[0],torch.zeros_like(x))


def test_frozen_calibration_loads_only_measured_classes():
    cfg=load_cup_config(Path(__file__).resolve().parents[1]/'configs/s5_cup_calibration.json')
    assert cfg.beta==.95 and cfg.temperature==1 and len(cfg.known_scales)==12 and 14 not in cfg.known_scales
    assert len(cfg.calibration_sha256)==64
