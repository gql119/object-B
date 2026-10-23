"""Diagnostics for same-state comparisons. No optimizer or training changes."""
import torch
from .tfss_spatial import realized_rms


def feasibility(image, base_delta, support, target, absolute=1e-7, relative=1e-4):
    maximum = float(realized_rms(image, base_delta, support))
    gap = float(target)-maximum
    return {'target_RMS': float(target), 'maximum_RMS_alpha1': maximum,
            'target_minus_maximum': gap, 'relative_gap': gap/float(target),
            'unreachable_at_locked_tolerance': gap > 0 and
            (gap > absolute or gap/float(target) > relative)}


def spatial_selectivity(image, base, candidate, uniform, support, q, risk, valid_q):
    """All energies use actual post-clamp perturbations, SAME incoming base."""
    base, candidate, uniform = [((image+d).clamp(0, 1)-image).detach().double()
                                for d in (base, candidate, uniform)]
    domain = support > 0
    q = q.to(base.device);risk = risk.to(base.device);valid_q = valid_q.to(base.device)
    masks = {'person': domain, 'overlap': support == .5, 'exclusive': support == 1,
             'high_Q': domain & valid_q & (q >= .9),
             'high_R': domain & (risk >= .9),
             'high_R_low_Q': domain & valid_q & (risk >= .9) & (q <= .1)}
    rows = {}
    for key, mask in masks.items():
        count = int(mask.sum())
        row = {'pixels': count, 'base_RMS': None, 'Utility_RMS': None,
               'Uniform_RMS': None, 'Utility_energy_retention': None,
               'Uniform_energy_retention': None}
        if count:
            powers = [d[:, mask].square().mean() for d in (base, candidate, uniform)]
            row.update(zip(['base_RMS','Utility_RMS','Uniform_RMS'],
                           [float(p.sqrt()) for p in powers]))
            if powers[0] > 0:
                row['Utility_energy_retention'] = float(powers[1]/powers[0])
                row['Uniform_energy_retention'] = float(powers[2]/powers[0])
        rows[key] = row
    for key, weight in [('Q_weighted', q*valid_q*domain), ('R_weighted', risk*domain)]:
        energy = [(d.square().sum(0)*weight).sum() for d in (base,candidate,uniform)]
        rows[key] = {'Utility_energy_retention': float(energy[1]/energy[0]) if energy[0]>0 else None,
                     'Uniform_energy_retention': float(energy[2]/energy[0]) if energy[0]>0 else None}
    return rows
