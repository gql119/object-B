import copy
import pytest

from ue_framework.methods.counterfactual_parent import select_parent,require_formal_lock,REQUIRED_GATES


def evidence(target,nt):
    return dict(run_id='test-provenance',gates={k:True for k in REQUIRED_GATES},
                protocol=dict(dataset_sha256='a'*64,clean_val_sha256='b'*64,architecture='YOLOv8n',
                              seed=0,epochs=200,initialization='random',validator_sha256='c'*64,
                              evaluation_role='last.pt'),
                metrics=dict(person_ap50=target,nt_map50=nt),
                identity=dict(git_sha='d'*40,config_sha256='e'*64,last_checkpoint_sha256='f'*64,
                              budget_rule='test-only'))


@pytest.mark.parametrize('target,nt,parent',[(.038993,.681698,'S1'),(.038994,.691698,'S0'),
                                           (.028993,.681697,'S0'),(.01,.75,'S1')])
def test_exact_one_pp_two_metric_selection(target,nt,parent):
    assert select_parent(evidence(.028993,.691698),evidence(target,nt))['parent']==parent


@pytest.mark.parametrize('issue',['incomplete','best','validator','nan','identity'])
def test_parent_gate_rejects_unsupported_selection(issue):
    a,b=evidence(.028993,.691698),evidence(.02,.72)
    if issue=='incomplete': b['gates']['victim_200']=False
    if issue=='best': b['protocol']['evaluation_role']='best.pt'
    if issue=='validator': b['protocol']['validator_sha256']='x'*64
    if issue=='nan': b['metrics']['nt_map50']=float('nan')
    if issue=='identity': b['identity']['git_sha']='UNKNOWN'
    with pytest.raises(ValueError): select_parent(a,b)


def test_draft_config_never_enables_formal_run():
    with pytest.raises(ValueError,match='disabled'):
        require_formal_lock(dict(status='DRAFT',formal_generation_enabled=False))


def test_helper_metrics_cannot_override_parent_rule():
    a,b=evidence(.028993,.691698),evidence(.1,.75)
    b['metrics'].update(S=1.,WCD=0.)
    assert select_parent(a,b)['parent']=='S0'
