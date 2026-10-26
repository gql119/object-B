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
    from .canonical_s4_training import validate_proof
    validate_proof(summary.get('tfss_training_proof', {}), draws=2)
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


def validate_production_batch16(summary):
    """Fail closed on old single-image evidence and unverified numerical fixes."""
    validate_summary(summary)
    ids=summary.get('actual_training_image_ids',[])
    if (summary.get('actual_training_images')!=16 or summary.get('actual_native_batch_sizes')!=[16,16]
            or summary.get('sample_order_policy')!='original_full_eligible_numpy_seed0'
            or summary.get('full_eligible_image_count',0)<16 or len(ids)!=16 or len(set(ids))!=16
            or ids!=summary.get('expected_training_image_ids')):
        raise ValueError('Actual original seeded batch16 validation is missing')
    cache=summary.get('cache_memory',{})
    check=summary.get('projector_precision_validation',{})
    gram=check.get('maximum_native_cached_Gram',float('nan'))
    if (cache.get('svd_precision_policy')!='selected_dimension_float64_svd_v1'
            or check.get('thresholds_unchanged') is not True
            or check.get('all_native_cached_ranks_eight') is not True
            or not math.isfinite(gram) or gram>1e-5):
        raise ValueError('Native fixed-threshold rank8/orthogonality precision gate is open')
    if summary.get('actual_device')=='cuda:0':
        if (check.get('historical_failed_order')!=[38,24,26]
                or check.get('historical_failed_order_in_native_cache') is not True
                or not math.isfinite(check.get('historical_failed_order_Gram',float('nan')))
                or check['historical_failed_order_Gram']>1e-5):
            raise ValueError('The original failing CUDA triplet was not exercised successfully')
