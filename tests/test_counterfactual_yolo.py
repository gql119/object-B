"""Random-init, synthetic CPU YOLO: native loss/second derivatives, not VOC."""
from types import SimpleNamespace

import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from ue_framework.methods.counterfactual_utility import CNLPConfig
from ue_framework.methods.counterfactual_yolo import YOLOCounterfactual


@pytest.fixture
def adapter():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model=DetectionModel('yolov8n.yaml',nc=20,verbose=False)
    model.args=get_cfg()
    model.criterion=model.init_criterion()
    fixed={id(p) for p in model.model[-1].dfl.parameters()}
    trainable=[k for k,p in model.named_parameters() if id(p) not in fixed]
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return YOLOCounterfactual(SimpleNamespace(model=model,trainable=trainable),CNLPConfig(1,1e-4,.1))


def sample(image,name,classes=(1,14)):
    return dict(img=image,split='train',image_ids=[name],preprocessing_id='identity64',
                cls=torch.tensor(classes,dtype=torch.float32).reshape(-1,1),
                bboxes=torch.tensor([[.4,.4,.4,.4],[.7,.7,.3,.3]])[:len(classes)],
                batch_idx=torch.zeros(len(classes),dtype=torch.long))


@pytest.mark.parametrize('steps',[1,3,5])
def test_native_yolo_shared_assignment_second_order_delta(adapter,steps):
    adapter.engine.config=CNLPConfig(steps,1e-4,.1)
    clean=sample(torch.rand(1,3,64,64),'support')
    delta=torch.zeros_like(clean['img'],requires_grad=True)
    poison={**clean,'img':clean['img']+delta}
    query=sample(torch.rand(1,3,64,64),'query')
    initial={k:v.clone() for k,v in adapter.model.state_dict().items()}
    result=adapter.evaluate(clean,poison,query)
    assert set(result['damage']) == {1}
    assert result['damage'][1] == 0
    assert len(result['evidence']['objects']) == 1
    assert result['evidence']['objects'][0]['positive_count'] > 0
    grad=torch.autograd.grad(result['poison_loss'][1],delta)[0]
    assert torch.isfinite(grad).all() and float(grad.norm()) > 0
    assert all(torch.equal(initial[k],v) for k,v in adapter.model.state_dict().items())
    assert all(p.grad is None for p in adapter.model.parameters())


def test_person_only_query_legal_zero(adapter):
    clean=sample(torch.rand(1,3,64,64),'support')
    poison={**clean,'img':clean['img'].clone().requires_grad_()}
    query=sample(torch.rand(1,3,64,64),'query',(14,))
    result=adapter.evaluate(clean,poison,query)
    assert result['damage'] == {} and result['loss'] == 0
    assert result['evidence']['legal_no_nt_gt']


def test_missing_nt_positive_is_not_zero_fallback(adapter):
    query=sample(torch.rand(1,3,64,64),'query',(1,))
    with torch.no_grad(): clean=adapter.model(query['img'])
    native=adapter.criterion.assigner
    def empty(*args,**kwargs):
        label,box,score,fg,indices=native(*args,**kwargs)
        return label,box,score,torch.zeros_like(fg),indices
    adapter.criterion.assigner=empty
    with pytest.raises(ValueError,match='CNLP_NO_VALID_POSITIVE'):
        adapter.query_loss(clean,clean,query)
