"""CPU contracts for the actual retention sink and source-derived layer probe."""
from types import SimpleNamespace
import ast
from pathlib import Path

import torch

from ue_framework.methods.s5_retention_probe import RetentionMeasurements, S5LayerTrainer


def reducer(records, layer='P3'):
    obj=SimpleNamespace(orth_feature_layer='model.15',s5_retention_records=records,
                        s5_ret_layer=layer,s5_decomposition_noop=True)
    return obj


def test_original_instance_reduction_and_gradient_are_exact_at_p3_baseline():
    x=torch.tensor([.1,.2,.3],requires_grad=True)
    losses=x.square()
    records={layer:[dict(image_index=0,gt_non_target_ordinal=i,cls=2 if i<2 else 11,
                        included=True,loss=v) for i,v in enumerate(losses)] for layer in ('P3','P5')}
    o=reducer(records);base=losses.mean()
    actual=RetentionMeasurements.s5_reduce_retention(o,base,x)
    assert torch.equal(actual,base)
    assert torch.equal(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(base,x)[0])
    assert o.s5_class_pi['P3']=={2:2/3,11:1/3}


def test_p5_uses_its_full_channel_retention_and_exposes_missing_small_gt():
    x=torch.tensor([.1,.2],requires_grad=True)
    records={'P3':[dict(image_index=0,gt_non_target_ordinal=i,cls=2,included=True,loss=v)
                   for i,v in enumerate(x.square())],
             'P5':[dict(image_index=0,gt_non_target_ordinal=0,cls=2,included=True,loss=3*x[0].square()),
                   dict(image_index=0,gt_non_target_ordinal=1,cls=2,included=False,loss=None)]}
    o=reducer(records,'P5');actual=RetentionMeasurements.s5_reduce_retention(o,x.square().mean(),x)
    assert torch.equal(actual,3*x[0].square())
    assert o.s5_missing_P5_rois==[(0,1)]
    assert torch.equal(torch.autograd.grad(actual,x)[0],torch.tensor([.6,0.]))


def test_empty_gt_keeps_connected_zero_and_mro_keeps_original_adaptive_wrapper():
    x=torch.ones(3,requires_grad=True);o=reducer({'P3':[],'P5':[]})
    actual=RetentionMeasurements.s5_reduce_retention(o,x.sum()*0,x)
    assert torch.equal(torch.autograd.grad(actual,x)[0],torch.zeros_like(x))
    names=[c.__name__ for c in S5LayerTrainer.__mro__]
    assert names.index('B2AdaptiveTrainer')<names.index('S5FrozenRetention')<names.index('SemanticFrozenTrainer')


def test_static_copy_diff_is_limited_to_two_observations_and_one_retention_sink():
    root=Path(__file__).resolve().parents[1]
    def method(path,cls):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_batch_loss')
    parent=method(root/'ue_framework/methods/b2_semantic.py','SemanticFrozenTrainer')
    candidate=method(root/'ue_framework/methods/s5_retention_core.py','S5FrozenRetention')
    class StripOnlyDeclaredChanges(ast.NodeTransformer):
        def visit_Expr(self,node):
            call=node.value
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='s5_record_retention':return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='s5_reduce_retention':return None
            return self.generic_visit(node)
    stripped=StripOnlyDeclaredChanges().visit(candidate)
    assert ast.dump(parent,include_attributes=False)==ast.dump(stripped,include_attributes=False)
