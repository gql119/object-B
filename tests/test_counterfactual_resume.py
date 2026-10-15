"""Synthetic CPU outer-loop continuation; real VOC/GPU admission still pending."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ue_framework.methods.counterfactual_utility import CNLPConfig,CUPConfig
from ue_framework.methods.counterfactual_extension import CounterfactualExtension
from ue_framework.methods.counterfactual_optimization import optimize_counterfactual


class Adapter:
    config=CNLPConfig(1,.1,.2)
    def evaluate(self,clean,poison,query):
        d={0:poison.square().mean()}
        return dict(loss=d[0].relu(),damage=d)


class Provider:
    def state_dict(self): return {'test_manifest':'fixed'}
    def load_state_dict(self,state): assert state==self.state_dict()


class Trainer:
    def __init__(self):
        self.device=torch.device('cpu')
        self.fourier_coeff=torch.tensor([[.2],[.3]],requires_grad=True)
        self.suppress_small=torch.tensor([0.])
        self.freq_score=torch.zeros(2); self.freq_usage=torch.zeros(2)
        self.freq_active_idx=torch.arange(2); self.freq_candidate_coords=[(0,0),(0,1)]
        self.universal_lr_fourier=.01; self.universal_epochs=3; self.universal_batch_size=1
        self.eot_samples=2; self.enable_adaptive_freq_basis=True
        self.adaptive_freq_warmup_epochs=1; self.freq_active_num_bases=1
        self._adaptive_calls=0; self._semantic_calls=0
        self.surrogate=torch.nn.Linear(1,1); self.adaptive=SimpleNamespace(model=torch.nn.Linear(1,1))
        for model in (self.surrogate,self.adaptive.model):
            for p in model.parameters(): p.data.fill_(.1); p.requires_grad_(False)
        self.bases={0:None}; self.worst_k=1; self.support_mode='test'; self.eps=8/255
        self.feature_layer='test'; self.num_prototypes=1; self.margin=.1
        self.cnlp=CounterfactualExtension(Adapter(),CUPConfig(.8,.2,.5,2.,5.))
        self.cnlp_query_provider=Provider()
    def _build_support(self,shape,ann,image_path): return np.ones(shape[:2]),None,None
    def _get_active_freq_indices(self,epoch,step): return self.freq_active_idx
    def cnlp_begin_step(self,step): self.cnlp.begin_step(step)
    def cnlp_bind_support(self,ids): self.ids=ids
    def cnlp_commit_step(self): self.cnlp.commit_step()
    def cnlp_state_dict(self): return self.cnlp.state_dict()
    def cnlp_load_state_dict(self,s): self.cnlp.load_state_dict(s)
    def _batch_loss(self,items,prototypes):
        # Simulate EOT randomness and a differentiable perturbation loss.
        noise=torch.rand(())
        loss,out=self.cnlp.add((self.fourier_coeff-.1).square().mean()+noise*.001,
                              None,self.fourier_coeff,None)
        self._adaptive_calls+=1; self._semantic_calls+=1
        return loss,dict(L_orth_aeroplane=loss.detach(),L_CNLP=out['loss'])


def test_optimizer_scheduler_rng_frequency_cup_resume_matches_uninterrupted(tmp_path):
    images=tmp_path/'images'; labels=tmp_path/'labels'; images.mkdir(); labels.mkdir()
    paths=[]
    for j in range(2):
        path=images/f'{j}.png'; Image.fromarray(np.ones((8,8,3),dtype=np.uint8)*128).save(path)
        (labels/f'{j}.txt').write_text('14 0.5 0.5 0.5 0.5\n')
        paths.append(path)
    a=Trainer(); full=optimize_counterfactual(a,paths,labels,None,tmp_path/'full',{'test':'fixed'},stop_after=99)
    b=Trainer(); short=optimize_counterfactual(b,paths,labels,None,tmp_path/'interrupted',{'test':'fixed'},stop_after=2)
    c=Trainer(); resumed=optimize_counterfactual(c,paths,labels,None,tmp_path/'resumed',{'test':'fixed'},
                                                stop_after=99,resume=short['checkpoint'])
    assert full['state']=='complete' and resumed['state']=='complete'
    assert full['steps']==resumed['steps']==6
    torch.testing.assert_close(a.fourier_coeff,c.fourier_coeff,rtol=0,atol=0)
    torch.testing.assert_close(a.cnlp.cup.mean,c.cnlp.cup.mean,rtol=0,atol=0)
    assert a.cnlp.completed_steps==c.cnlp.completed_steps==6
    assert (tmp_path/'full/diagnostics.csv').read_bytes()==(tmp_path/'resumed/diagnostics.csv').read_bytes()
    af=torch.load(full['checkpoint'],weights_only=False)
    cf=torch.load(resumed['checkpoint'],weights_only=False)
    assert af['scheduler']==cf['scheduler']
    for key in ('exp_avg','exp_avg_sq','step'):
        torch.testing.assert_close(af['optimizer']['state'][0][key],cf['optimizer']['state'][0][key],rtol=0,atol=0)
    torch.testing.assert_close(af['rng']['torch'],cf['rng']['torch'],rtol=0,atol=0)
    # Already-completed boundary recovery performs no extra optimizer/CUP update.
    d=Trainer(); finished=optimize_counterfactual(d,paths,labels,None,tmp_path/'completed_resume',{'test':'fixed'},
                                                stop_after=99,resume=full['checkpoint'])
    assert finished['steps']==6 and d.cnlp.completed_steps==6
    torch.testing.assert_close(a.fourier_coeff,d.fourier_coeff,rtol=0,atol=0)
