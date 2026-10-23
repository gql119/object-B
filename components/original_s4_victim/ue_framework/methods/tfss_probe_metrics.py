"""Read-only gradient/strength diagnostics, outside the optimization objective."""
import torch


def cosine(a, b):
    denominator = a.double().norm()*b.double().norm()
    return float((a.double()*b.double()).sum()/denominator) if denominator > 0 else None


def save_tensor_artifact(path, value):
    # Python file IO also supports managed Windows paths where the C++ filename
    # writer cannot open a path. Serialization/computation itself is unchanged.
    with open(path, 'wb') as handle:
        torch.save(value, handle)


def stats(value, mask):
    selected = value[:, mask]
    if not selected.numel():
        return {'rms': None, 'mean_abs': None, 'linf': None, 'pixels': 0}
    return {'rms': float(selected.double().square().mean().sqrt()),
            'mean_abs': float(selected.abs().double().mean()),
            'linf': float(selected.abs().max()), 'pixels': int(mask.sum())}


def collect(trainer, metrics, items, maps, *, mode, reference_path, output_prefix):
    """Reference is same fixed batch/step/EOT along its OWN S0 trajectory.

    Pairing is descriptive after trajectories diverge, not a causal decomposition.
    Only derived gradients/deltas are retained, never source image RGB.
    """
    protection = metrics['weighted_L_BG'] + metrics['weighted_L_rel'] + trainer.lambda_adapt*metrics['L_adapt_target']
    gradient_node = trainer.tfss_s0_delta
    gp = torch.autograd.grad(protection, gradient_node, retain_graph=True)[0]
    # Actual train-dynamic meta-gradient includes original virtual Adaptive update.
    fourier = torch.autograd.grad(protection, trainer.fourier_coeff, retain_graph=True)[0]
    gr = torch.autograd.grad(metrics['weighted_L_ret'], gradient_node, retain_graph=True)[0]
    state = {'protection_gradient': gp.detach().cpu(), 'fourier_protection_gradient': fourier.detach().cpu(),
             's0_delta': trainer.tfss_delta.detach().cpu(), 'support': trainer.tfss_support.detach().cpu()}
    if mode == 'S0':
        save_tensor_artifact(reference_path, state)
        reference = state
    else:
        reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    if mode == 'S0-noop':
        if not torch.equal(state['support'], reference['support']):
            raise ValueError('Canonical/noop spatial mask mismatch')
        if float((state['s0_delta']-reference['s0_delta']).abs().max()) > 1e-7:
            raise ValueError('Canonical/noop actual delta mismatch')
    ref_fourier = reference['fourier_protection_gradient']
    save_tensor_artifact(str(output_prefix)+'_fourier_gradient.pt', {'protection': fourier.detach().cpu()})
    rows = []
    actual = ((trainer.tfss_input+trainer.tfss_delta).clamp(0, 1)-trainer.tfss_input).detach()
    for i, ((image, annotations, support), (q, risk, valid_q_domain)) in enumerate(zip(items, maps)):
        h, w = image.shape[:2]
        target = torch.as_tensor(support > 0, device=actual.device)
        overlap = torch.as_tensor(support == .5, device=actual.device)
        exclusive = torch.as_tensor(support == 1, device=actual.device)
        g = gp[i, :, :h, :w].detach().cpu()
        ref = reference['protection_gradient'][i, :, :h, :w]
        ref_energy = ref.square().sum(0)
        domain = target.cpu()
        top = torch.zeros_like(domain)
        indices = domain.flatten().nonzero().flatten()
        # Stable tie policy; zero energy has no invented attribution/retention.
        valid = bool(ref_energy[domain].sum() > 0)
        high_valid = valid and bool(ref_energy[domain].max()-ref_energy[domain].min() > ref_energy[domain].max()*1e-7)
        if high_valid:
            order = torch.argsort(ref_energy.flatten()[indices], descending=True, stable=True)
            top.flatten()[indices[order[:max(1, (len(indices)+4)//5)]]] = True
        weight = trainer.tfss_applied_weight[i, 0, :h, :w].detach()
        d = actual[i, :, :h, :w]
        q = q.to(d.device);risk = risk.to(d.device)
        high_q = (q >= .9) & target & valid_q_domain.to(target.device)
        high_r = (risk >= .9) & target
        dx = target[:, 1:] & target[:, :-1];dy = target[1:] & target[:-1]
        differences = torch.cat([(weight[:, 1:]-weight[:, :-1]).abs()[dx],
                                 (weight[1:]-weight[:-1]).abs()[dy]])
        high_retention = float((ref_energy[top]*weight.cpu()[top].square()).sum()/ref_energy[top].sum()) if high_valid else None
        rows.append({'image_index': i, 'person': stats(d, target), 'overlap': stats(d, overlap),
            'exclusive': stats(d, exclusive), 'high_tsfl': stats(d, high_q), 'high_nt_risk': stats(d, high_r),
            'weight_min': float(weight[target].min()), 'weight_mean': float(weight[target].mean()),
            'weight_max': float(weight[target].max()),
            'weight_quantiles': torch.quantile(weight[target], d.new_tensor([.1, .25, .5, .75, .9])).tolist(),
            'weight_smoothness': float(differences.mean()) if len(differences) else None,
            'effective_area': float((d.abs().amax(0)[target] > 0).float().mean()),
            'saturation_fraction': float((d.abs() >= torch.as_tensor(support, device=d.device)*trainer.eps-1e-7)[:, target].float().mean()),
            'protection_gradient_norm': float(g[:, domain].norm()),
            'protection_gradient_cosine_to_s0': cosine(g[:, domain], ref[:, domain]),
            'protection_gradient_norm_ratio_to_s0': float(g[:, domain].norm()/ref[:, domain].norm()) if valid else None,
            'high_protection_operator_energy_retention': high_retention,
            'protection_vs_retention_gradient_cosine': cosine(gp[i], gr[i]),
            'fourier_protection_norm': float(fourier.norm()),
            'fourier_protection_cosine_to_s0': cosine(fourier.detach().cpu(), ref_fourier),
            'target_confidence_proxy': float(metrics['target_score_poison_after'].detach()),
            'target_clean_query_proxy': float(metrics['target_score_clean_after'].detach()),
            'proxy_scope': 'original Adaptive selected support, repeated batch scalar; not per-object AP',
            'inner_model_update_norm': float(metrics['inner_update_norm'].detach()),
            'gradient_type': 'actual Parent background/relative + original Adaptive target training objective'})
    return rows
