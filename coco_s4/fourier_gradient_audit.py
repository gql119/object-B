"""Passive validation of the actual loss-to-coefficient autograd path."""
import math

MEASUREMENT = 'direct_autograd_fourier_coeff_v1'


def measure(terms, parameter):
    import torch
    if not parameter.is_leaf or not parameter.requires_grad or parameter.ndim != 2 or parameter.shape[1] != 3:
        raise ValueError('Expected the actual native leaf Fourier coefficient parameter')
    before = None if parameter.grad is None else parameter.grad.detach().clone()
    gradients = {}
    for name, key in (('L_BG', 'BG_sel'), ('L_rel', 'rel_sel')):
        gradient, = torch.autograd.grad(terms[key], parameter, retain_graph=True)
        if gradient.shape != parameter.shape or not torch.isfinite(gradient).all() or gradient.norm().item() <= 0:
            raise ValueError('Actual production '+name+' has no finite nonzero Fourier coefficient gradient')
        gradients[name] = gradient.detach().clone()
    if (before is None and parameter.grad is not None) or (before is not None and not torch.equal(before, parameter.grad)):
        raise ValueError('Passive coefficient differentiation changed optimizer gradient accumulation')
    return gradients


def validate_summary(summary):
    evidence = summary.get('per_loss_fourier_gradient_measurement', {})
    if (evidence.get('measurement') != MEASUREMENT or evidence.get('target') != 'trainer.fourier_coeff'
            or evidence.get('parameter_device') != summary.get('actual_device')
            or evidence.get('parameter_shape') != [64, 3] or evidence.get('EOT_draws') != 2
            or evidence.get('optimizer_gradient_accumulation_unchanged') is not True):
        raise ValueError('Missing actual production per-loss coefficient differentiation evidence')
    for name in ('L_BG', 'L_rel'):
        value = summary.get('SafeBG_loss_Fourier_gradient_norms', {}).get(name, 0)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid mean-EOT coefficient gradient: '+name)
