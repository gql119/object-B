"""CPU correctness evidence, not VOC experiment efficacy or GPU admission."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ue_framework.methods.counterfactual_utility import (
    CNLPConfig, CUPConfig, CUPHistory, CounterfactualSGD,
    capture_rng, same_rng, validate_pair,
)
from ue_framework.methods.counterfactual_extension import CounterfactualExtension


def batch(image, name):
    return dict(img=image, image_ids=[name], split='train', preprocessing_id='fixed-test',
                cls=torch.tensor([[1.]]), bboxes=torch.tensor([[.5,.5,.5,.5]]),
                batch_idx=torch.tensor([0]))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.8, dtype=torch.float64))

    def forward(self, x):
        return self.weight * x


def native_loss(forward, b):
    return (forward(b['img']) - .25).square().mean()


def query_loss(c, p, q):
    return {1: (c-.2).square().mean()}, {1: (p-.2).square().mean()}, {}


def toy_run(value, steps=1):
    model = Toy()
    engine = CounterfactualSGD(model, CNLPConfig(steps,.1,.2), ['weight'])
    delta = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    clean = batch(torch.full((1,1,1,1),.5,dtype=torch.float64), 's')
    poison = {**clean, 'img': clean['img']+delta}
    query = batch(torch.full((1,1,1,1),.6,dtype=torch.float64), 'q')
    result = engine.evaluate(clean,poison,query,native_loss,query_loss)
    return delta, result


@pytest.mark.parametrize('steps',[1,3,5])
def test_real_higher_order_delta_chain_matches_finite_difference(steps):
    delta, out = toy_run(-.1, steps)
    value = out['damage'][1]
    grad = torch.autograd.grad(value,delta)[0]
    _, plus = toy_run(-.1+1e-6,steps)
    _, minus = toy_run(-.1-1e-6,steps)
    difference = (plus['damage'][1]-minus['damage'][1])/(2e-6)
    torch.testing.assert_close(grad, difference, rtol=1e-6,atol=1e-8)
    assert grad.abs() > 1e-6
    assert out['clean_loss'][1].requires_grad is False


@pytest.mark.parametrize('steps',[1,3,5])
def test_zero_delta_two_paths_equal(steps):
    _, out = toy_run(0., steps)
    assert out['loss'] == 0
    assert out['damage'][1] == 0


def test_clean_vs_poison_damage_not_direct_input_loss():
    _, out = toy_run(-.1,3)
    assert out['clean_support_loss'].numel() == 3
    assert out['clean_support_loss'][0] != out['clean_support_loss'][-1]
    assert out['damage'][1] > 0 and out['loss'] > 0


@pytest.mark.parametrize('steps',[1,3,5])
def test_learning_gain_is_same_query_before_and_after_virtual_update(steps):
    _,out=toy_run(-.1,steps)
    clean_gain=out['initial_loss'][1]-out['clean_loss'][1]
    poison_gain=out['initial_loss'][1]-out['poison_loss'][1]
    assert clean_gain>0
    torch.testing.assert_close(clean_gain-poison_gain,out['damage'][1],rtol=0,atol=1e-15)
    assert not out['initial_loss'][1].requires_grad


@pytest.mark.parametrize('violation',['overlap','split','GT','ids','preprocessing','detach'])
def test_data_contract_rejects_bad_pairs(violation):
    clean = batch(torch.ones(1,1,1,1)*.5,'s')
    poison = {**clean, 'img': clean['img'].clone().requires_grad_()}
    query = batch(torch.ones(1,1,1,1)*.6,'q')
    if violation == 'overlap': query['image_ids']=['s']
    if violation == 'split': query['split']='val'
    if violation == 'GT': poison['cls']=torch.tensor([[2.]])
    if violation == 'ids': poison['image_ids']=['other']
    if violation == 'preprocessing': poison['preprocessing_id']='other'
    if violation == 'detach': poison['img']=poison['img'].detach()
    with pytest.raises(ValueError): validate_pair(clean,poison,query)


def test_rng_and_bn_state_are_isolated():
    model = nn.Sequential(nn.BatchNorm2d(1),nn.Dropout(.4),nn.Conv2d(1,1,1)).double().train()
    engine = CounterfactualSGD(model,CNLPConfig(3,.01,.2),[k for k,_ in model.named_parameters()])
    clean = batch(torch.rand(1,1,4,4,dtype=torch.float64),'s')
    poison = {**clean, 'img': clean['img'].clone().requires_grad_()}
    query = batch(torch.rand(1,1,4,4,dtype=torch.float64),'q')
    before = {k:v.clone() for k,v in model.state_dict().items()}
    rng = capture_rng()
    out = engine.evaluate(clean,poison,query,native_loss,query_loss)
    assert same_rng(rng,capture_rng())
    assert model.training and model[0].training
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert out['damage'][1] == 0


def test_random_draw_mismatch_fails_and_restores_rng():
    model = Toy(); engine = CounterfactualSGD(model,CNLPConfig(1,.1,.2),['weight'])
    c=batch(torch.full((1,1,1,1),.5,dtype=torch.float64),'s')
    p={**c,'img':c['img'].clone().requires_grad_()}
    q=batch(torch.full((1,1,1,1),.6,dtype=torch.float64),'q')
    def bad(forward,b):
        if b['img'].requires_grad: torch.rand(1)
        return native_loss(forward,b)
    rng = capture_rng()
    with pytest.raises(ValueError,match='random draws'): engine.evaluate(c,p,q,bad,query_loss)
    assert same_rng(rng,capture_rng())


def test_legal_absent_nt_query_returns_graph_zero():
    model = Toy(); engine = CounterfactualSGD(model,CNLPConfig(1,.1,.2),['weight'])
    c=batch(torch.full((1,1,1,1),.5,dtype=torch.float64),'s')
    p={**c,'img':c['img'].clone().requires_grad_()}
    q=batch(torch.full((1,1,1,1),.6,dtype=torch.float64),'q')
    out=engine.evaluate(c,p,q,native_loss,lambda c,p,q: ({},{},{'legal_no_nt_gt':True}))
    assert out['loss'] == 0
    assert torch.equal(torch.autograd.grad(out['loss'],p['img'])[0],torch.zeros_like(p['img']))


def cup():
    # Unit-test settings, explicitly not calibrated research settings.
    return CUPHistory(CUPConfig(.8,.2,.5,2.,5.))


def test_cup_uniform_initialization_history_and_absence():
    h=cup(); ref=torch.tensor(0.)
    torch.testing.assert_close(h.weights([1,2,3],ref),torch.ones(3)/3)
    h.commit([{1:2.,2:-1.},{1:4.}])
    assert h.mean[1] == 3. and h.mean[2] == -1.
    before=h.mean[2].clone(); h.commit([{1:1.}])
    assert h.mean[2] == before and h.mean[1] == pytest.approx(2.6)
    w=h.weights([1,2,3],ref)
    assert w[0] > w[2] >= w[1] and not w.requires_grad


@pytest.mark.parametrize('classes',[[1],[1,2],[1,2,3],list(range(14))+list(range(15,20))])
def test_cup_noncollapse_bounded_simplex(classes):
    h=cup(); h.commit([{classes[0]:1e10}])
    w=h.weights(classes,torch.tensor(0.,dtype=torch.float64))
    assert w.sum() == pytest.approx(1.)
    assert bool((w>=.5/len(classes)-1e-12).all())
    assert bool((w<=2/len(classes)+1e-12).all())
    assert h.mean[classes[0]] == 5.


def test_cup_resume_identical_future_and_bad_state_atomic():
    h=cup(); h.commit([{1:.7,2:.4}]); restored=cup(); restored.load_state_dict(h.state_dict())
    for sample in [{1:.8},{3:-.2},{2:.1,3:1.}]:
        h.commit([sample]); restored.commit([sample])
    torch.testing.assert_close(h.mean,restored.mean,rtol=0,atol=0)
    torch.testing.assert_close(h.weights([1,2,3],torch.tensor(0.)),restored.weights([1,2,3],torch.tensor(0.)),rtol=0,atol=0)
    before=h.state_dict(); bad=copy.deepcopy(before); bad['mean'][14]=1.
    with pytest.raises(ValueError): h.load_state_dict(bad)
    assert torch.equal(h.mean,before['mean'])


def test_cup_invalid_damage_commit_is_atomic():
    h=cup(); before=h.state_dict()
    with pytest.raises(FloatingPointError): h.commit([{1:1.,2:float('nan')}])
    assert torch.equal(h.mean,before['mean']) and h.steps == 0


class ToyAdapter:
    config=CNLPConfig(1,.1,.2)
    def evaluate(self,*args):
        d={1:torch.tensor(1.,requires_grad=True),2:torch.tensor(.1,requires_grad=True)}
        return dict(loss=torch.stack(list(d.values())).mean(),damage=d)


def test_eot_history_commit_once_and_checkpoint_boundary():
    e=CounterfactualExtension(ToyAdapter(),CUPConfig(.8,.2,.5,2.,5.))
    e.begin_step(0)
    _,first=e.add(torch.tensor(2.),None,None,None)
    _,second=e.add(torch.tensor(2.),None,None,None)
    assert e.cup.steps == 0
    assert torch.equal(first['weights'],second['weights'])
    with pytest.raises(ValueError): e.state_dict()
    e.commit_step(); saved=e.state_dict()
    assert e.cup.steps == 1 and e.cup.observations[1] == 2
    other=CounterfactualExtension(ToyAdapter(),CUPConfig(.8,.2,.5,2.,5.)); other.load_state_dict(saved)
    e.begin_step(1); other.begin_step(1)
    _,a=e.add(torch.tensor(2.),None,None,None); _,b=other.add(torch.tensor(2.),None,None,None)
    assert torch.equal(a['weights'],b['weights'])
    e.abort_step(); other.abort_step()
    assert e.completed_steps == 1


def test_uniform_cup_exactly_equals_uniform_cnlp():
    a=CounterfactualExtension(ToyAdapter())
    b=CounterfactualExtension(ToyAdapter(),CUPConfig(.8,.2,1.,1.,5.))
    a.begin_step(0); b.begin_step(0)
    x,_=a.add(torch.tensor(2.),None,None,None)
    y,_=b.add(torch.tensor(2.),None,None,None)
    torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_zero_weight_equivalence_probe_preserves_parent_loss():
    adapter=ToyAdapter(); adapter.config=CNLPConfig(1,.1,0.)
    extension=CounterfactualExtension(adapter)
    parent=torch.tensor(2.,requires_grad=True)
    extension.begin_step(0)
    total,_=extension.add(parent,None,None,None)
    torch.testing.assert_close(total,parent,rtol=0,atol=0)
    assert torch.autograd.grad(total,parent)[0]==1


def test_cup_no_nt_query_preserves_connected_zero():
    class EmptyAdapter:
        config=CNLPConfig(1,.1,.2)
        def evaluate(self,c,p,q): return dict(loss=p.sum()*0,damage={})
    extension=CounterfactualExtension(EmptyAdapter(),CUPConfig(.8,.2,.5,2.,5.))
    variable=torch.tensor([.1],requires_grad=True)
    extension.begin_step(0)
    _,result=extension.add(torch.tensor(2.),None,variable,None)
    assert torch.autograd.grad(result['loss'],variable)[0]==0
