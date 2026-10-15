"""Add CNLP/CUP around unchanged Safe-Soft05 or S1 Parent computation.

Caller must provide an identity-checked train-only query provider and complete
outer-step lifecycle. This is preparation code, NOT a formal launch protocol.
"""
import torch
import torch.nn.functional as F

from .b2_semantic import B2SemanticTrainer
from .s1_csb import S1CSBTrainer
from .counterfactual_extension import CounterfactualExtension
from .counterfactual_yolo import YOLOCounterfactual


class CNLPTrainerMixin:
    def __init__(self, *args, cnlp_config, query_provider, support_batch_size,
                 cup_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        if cnlp_config.target_class_id != self.target_class_id:
            raise ValueError('CNLP target identity must inherit Parent')
        if cnlp_config.inner_lr != self.adaptive.inner_lr:
            raise ValueError('CNLP virtual LR must inherit Parent Adaptive LR')
        if not 1 <= support_batch_size <= self.universal_batch_size:
            raise ValueError('Explicit CNLP support batch size required')
        self.cnlp = CounterfactualExtension(YOLOCounterfactual(self.adaptive,cnlp_config),cup_config)
        self.cnlp_query_provider = query_provider
        self.cnlp_support_batch_size = support_batch_size
        self.cnlp_support_ids = None

    def cnlp_begin_step(self, step):
        self.cnlp.begin_step(step)

    def cnlp_bind_support(self, ids):
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Explicit unique support IDs required')
        self.cnlp_support_ids = tuple(ids)

    def cnlp_commit_step(self):
        self.cnlp.commit_step()
        self.cnlp_support_ids = None

    def cnlp_state_dict(self):
        return self.cnlp.state_dict()

    def cnlp_load_state_dict(self, state):
        self.cnlp.load_state_dict(state)
        self.cnlp_support_ids = None

    def _batch_loss(self, items, prototypes):
        if self.cnlp.pending is None or self.cnlp_support_ids is None or len(items) != len(self.cnlp_support_ids):
            raise ValueError('CNLP loop must bind actual filtered support IDs before Parent computation')
        parent_result = super()._batch_loss(items,prototypes)
        if parent_result is None:
            return None
        parent_loss, metrics = parent_result
        poisoned = self._last_poison
        h,w = poisoned.shape[-2:]
        count=min(self.cnlp_support_batch_size,len(items))
        start=self.cnlp.completed_steps % len(items)
        indices=[(start+j)%len(items) for j in range(count)]
        selected=[items[j] for j in indices]
        clean=torch.stack([F.pad(torch.from_numpy(im).permute(2,0,1).float(),
                              (0,w-im.shape[1],0,h-im.shape[0]),value=.5)
                           for im,_,_ in selected]).to(self.device)
        clean_batch=self.adaptive._batch(clean,selected)
        poison_batch=self.adaptive._batch(poisoned[indices],selected)
        ids=[self.cnlp_support_ids[j] for j in indices]
        identity=dict(split='train',image_ids=ids,
                      preprocessing_id='Parent_padded_letterbox_shared_no_new_augmentation')
        clean_batch.update(identity); poison_batch.update(identity)
        # Exclude ALL current support IDs, including images outside the CNLP subset.
        query=self.cnlp_query_provider(self.cnlp_support_ids,self.cnlp.completed_steps)
        if set(query['image_ids']) & set(self.cnlp_support_ids):
            raise ValueError('CNLP query overlaps full Parent support batch')
        total,result=self.cnlp.add(parent_loss,clean_batch,poison_batch,query)
        self._cnlp_last_evidence = result['evidence']
        raw=result['loss']
        # Measurements use the SAME original Fourier optimization variable.
        g_parent=torch.autograd.grad(parent_loss,self.fourier_coeff,retain_graph=True)[0]
        g_cnlp=torch.autograd.grad(raw,self.fourier_coeff,retain_graph=True)[0]
        attack=self._semantic_terms['BG_sel']+self._semantic_terms['rel_sel']
        g_attack=torch.autograd.grad(attack,self.fourier_coeff,retain_graph=True)[0]
        g_ret=torch.autograd.grad(self._semantic_terms['ret'],self.fourier_coeff,retain_graph=True)[0]
        self._cnlp_gradient_vectors = {
            'parent':g_parent.detach(), 'cnlp':g_cnlp.detach(),
            'attack':g_attack.detach(), 'ret':g_ret.detach(),
        }
        if not torch.isfinite(g_cnlp).all():
            raise FloatingPointError('Nonfinite CNLP perturbation gradient')
        cosine=(g_parent*g_cnlp).sum()/(g_parent.norm()*g_cnlp.norm()).clamp_min(1e-12)
        cos_attack=(g_attack*g_cnlp).sum()/(g_attack.norm()*g_cnlp.norm()).clamp_min(1e-12)
        cos_ret=(g_ret*g_cnlp).sum()/(g_ret.norm()*g_cnlp.norm()).clamp_min(1e-12)
        weights=result['weights']
        metrics.update(L_parent=parent_loss,L_CNLP=raw,L_CNLP_weighted=result['weighted_loss'],
                       L_total=total,grad_cnlp_raw_norm=g_cnlp.norm(),
                       grad_cnlp_weighted_norm=g_cnlp.norm()*self.cnlp.adapter.config.lambda_cnlp,
                       cos_grad_parent_cnlp=cosine,
                       cos_grad_parent_cnlp_valid=raw.new_tensor(float(g_parent.norm()>0 and g_cnlp.norm()>0)),
                       cos_grad_attack_cnlp=cos_attack,cos_grad_ret_cnlp=cos_ret,
                       cos_grad_attack_cnlp_valid=raw.new_tensor(float(g_attack.norm()>0 and g_cnlp.norm()>0)),
                       cos_grad_ret_cnlp_valid=raw.new_tensor(float(g_ret.norm()>0 and g_cnlp.norm()>0)),
                       cnlp_valid_classes=raw.new_tensor(len(result['damage'])),
                       cnlp_valid_objects=raw.new_tensor(len(result['evidence']['objects'])),
                       cnlp_inner_steps=raw.new_tensor(self.cnlp.adapter.config.inner_steps),
                       cnlp_clean_support_loss=result['clean_support_loss'].mean(),
                       cnlp_poison_support_loss=result['poison_support_loss'].mean(),
                       cup_weight_entropy=-(weights*weights.clamp_min(1e-12).log()).sum(),
                       cup_weight_max=weights.max() if weights.numel() else raw.new_zeros(()),
                       cup_weight_min=weights.min() if weights.numel() else raw.new_zeros(()))
        classes=sorted(result['damage'])
        for cls in range(self.cnlp.adapter.config.num_classes):
            if cls == self.target_class_id: continue
            valid=cls in result['damage']
            # Absent values are accompanied by explicit validity masks, not findings.
            metrics[f'cnlp_class_{cls}_valid']=raw.new_tensor(float(valid))
            for name,key in [('clean','clean_loss'),('poison','poison_loss'),('damage','damage')]:
                metrics[f'cnlp_class_{cls}_{name}']=result[key][cls] if valid else raw.new_zeros(())
            metrics[f'cnlp_class_{cls}_initial']=result['initial_loss'][cls] if valid else raw.new_zeros(())
            metrics[f'cnlp_class_{cls}_weight']=weights[classes.index(cls)] if valid else raw.new_zeros(())
        return total,metrics


class S2S3FixedParentTrainer(CNLPTrainerMixin,B2SemanticTrainer):
    """S0 fixed8/4/0 Parent: no CSB estimator/cache/projection mixed in."""


class S2S3SpatialParentTrainer(CNLPTrainerMixin,S1CSBTrainer):
    """S1 Parent: inherits actual locked CSB cache/projection unchanged."""
