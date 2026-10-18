"""S4 selected-P3 retention with detached, bounded class priorities only."""
import torch
from .s5_retention_probe import S5LayerTrainer
from .s5_cup_history import S5CUPHistory


class S5CUPTrainer(S5LayerTrainer):
    def __init__(self,*args,cup_config,cup_enabled=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.s5_ret_layer='P3'
        self.cup=S5CUPHistory(cup_config)
        self.cup_enabled=bool(cup_enabled)
        self.cup_active_step=False
        self.cup_pending=[]
        self.cup_last_commit=[]

    def cup_begin_step(self):
        if self.cup_active_step or self.cup_pending:raise ValueError('Uncommitted CUP boundary')
        self.cup_active_step=True

    def cup_end_step(self):
        if not self.cup_active_step:raise ValueError('CUP commit outside an Adam boundary')
        self.cup_last_commit=self.cup.commit(self.cup_pending if self.cup_enabled else [])
        self.cup_pending=[];self.cup_active_step=False

    def cup_cancel_empty_step(self):
        if self.cup_pending:raise ValueError('Cannot discard valid uncommitted damage')
        self.cup_active_step=False

    def _batch_loss(self,items,prototypes):
        loss,metrics=super()._batch_loss(items,prototypes)
        metrics.update(L_ret_base=self.cup_base_ret,L_ret_CUP=self.cup_weighted_ret,
            cup_entropy=self.cup_weight_summary.get('entropy',0.),
            cup_min_w=min([x['w'] for x in self.cup_audit] or [1.]),
            cup_max_w=max([x['w'] for x in self.cup_audit] or [1.]),
            cup_allocation_residual=abs(self.cup_weight_summary['effective_weighted_sum']-1.) if self.cup_audit else 0.)
        names=('cup_ret_Fourier_norm','cup_base_ret_Fourier_norm','cup_ret_to_protection_norm',
               'cup_ret_protection_cosine','cup_base_protection_cosine')
        metrics.update({key:None for key in names})
        if getattr(self,'s4_measure_protection',False):
            protection=metrics['weighted_L_BG']+metrics['weighted_L_rel']+self.lambda_adapt*metrics['L_adapt_target']
            gp=torch.autograd.grad(protection,self.fourier_coeff,retain_graph=True)[0]
            gb=torch.autograd.grad(self.cup_base_ret,self.fourier_coeff,retain_graph=True)[0]
            gw=torch.autograd.grad(self.cup_weighted_ret,self.fourier_coeff,retain_graph=True)[0]
            def cosine(a,b):
                scale=a.double().norm()*b.double().norm()
                return (a.double()*b.double()).sum()/scale if scale>0 else None
            metrics.update(cup_ret_Fourier_norm=gw.norm()*self.lambda_ret,
                cup_base_ret_Fourier_norm=gb.norm()*self.lambda_ret,
                cup_ret_to_protection_norm=gw.norm()*self.lambda_ret/gp.norm() if gp.norm()>0 else None,
                cup_ret_protection_cosine=cosine(gw,gp),cup_base_protection_cosine=cosine(gb,gp))
            for row in self.cup_audit:
                gk=torch.autograd.grad(self.s5_class_losses['P3'][row['cls']],self.fourier_coeff,retain_graph=True)[0]
                row.update(raw_class_Fourier_norm=float(gk.norm()),
                    actual_weighted_class_Fourier_norm=float(gk.norm())*row['allocation']*self.lambda_ret,
                    class_protection_cosine=float(cosine(gk,gp)) if cosine(gk,gp) is not None else None)
        return loss,metrics

    def s5_reduce_retention(self,original,poison):
        base=super().s5_reduce_retention(original,poison)
        classes=sorted(self.s5_class_losses['P3'])
        pi=[self.s5_class_pi['P3'][k] for k in classes]
        factors,allocation=self.cup.weights(classes,pi,original)
        if not self.cup_enabled:factors=torch.ones_like(factors)
        if factors.requires_grad:raise ValueError('Class priority must be detached')
        if self.cup_active_step and self.cup_enabled:
            self.cup_pending.append({k:float(self.s5_class_losses['P3'][k].detach()) for k in classes})
        self.cup_audit=[]
        for i,k in enumerate(classes):
            self.cup_audit.append(dict(cls=k,pi=pi[i],w=float(factors[i]),
                allocation=pi[i]*float(factors[i]),raw_damage=float(self.s5_class_losses['P3'][k].detach()),
                raw_history=float(self.cup.raw_mean[k]),priority_before=float(self.cup.mean[k]) if self.cup.observations[k] else 1.,
                scale=float(self.cup.scale[k]) if self.cup.scale[k]>0 else None,
                observations_before=int(self.cup.observations[k]),history_step=self.cup.steps,
                enabled=self.cup_enabled))
        self.cup_weight_summary=allocation
        self.cup_weight_summary['enabled']=self.cup_enabled
        self.cup_weight_summary['effective_weighted_sum']=sum(x['allocation'] for x in self.cup_audit)
        if classes and abs(self.cup_weight_summary['effective_weighted_sum']-1)>1e-7:
            raise ValueError('Applied FP32 allocation no longer normalized')
        # Explicitly preserve original scalar/gradient when disabled or neutral.
        if not self.cup_enabled or not classes or bool((factors==1).all()):
            weighted=original
        else:
            residual=torch.stack([original.new_tensor(pi[i])*(factors[i]-1)*self.s5_class_losses['P3'][k]
                                  for i,k in enumerate(classes)]).sum()
            weighted=base+residual
        self.cup_base_ret=original
        self.cup_weighted_ret=weighted
        return weighted

    def cup_state_dict(self):
        if self.cup_active_step or self.cup_pending:raise ValueError('Checkpoint before CUP commit')
        return dict(enabled=self.cup_enabled,layer=self.s5_ret_layer,history=self.cup.state_dict())

    def cup_load_state_dict(self,state):
        if state['enabled']!=self.cup_enabled or state['layer']!='P3':
            raise ValueError('CUP resume mode/layer mismatch')
        self.cup.load_state_dict(state['history'])
        self.cup_pending=[];self.cup_active_step=False;self.cup_last_commit=[]
