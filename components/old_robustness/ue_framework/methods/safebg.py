"""Strict TSFL selection and Safe-BG projection helpers for B2-TO-Safe."""
from __future__ import annotations

import math
import hashlib
import itertools
import time

import torch


class SafeBGTripletCache:
    """Precompute frozen low-rank projectors for all 19 choose 3 class sets.

    Each canonical triplet retains its six column orders. The online SVD uses
    descending projection-ratio order; retaining that order avoids float32 SVD
    differences near a narrow eighth/ninth singular-value gap.
    """

    def __init__(self, class_bases, projector_rtol=1e-6, safe_rank=8):
        classes = tuple(sorted(class_bases))
        if len(classes) != 19 or safe_rank not in (4, 8):
            raise ValueError("Safe-BG triplet cache requires 19 classes and rank 4 or 8")
        first = class_bases[classes[0]]
        if first.ndim != 2 or first.shape[0] != 24:
            raise ValueError("Safe-LR triplet cache requires 24 selected channels")
        self.classes = classes
        self.device, self.dtype = first.device, first.dtype
        self.projector_rtol = projector_rtol
        self.safe_rank = safe_rank
        for cls in classes:
            basis = class_bases[cls]
            if (basis.ndim != 2 or basis.shape[0] != 24
                    or basis.device != self.device or basis.dtype != self.dtype
                    or not torch.isfinite(basis).all()):
                raise ValueError(f"Invalid selected-space basis for class {cls}")
        started = time.perf_counter()
        self.entries = {}
        all_q = []
        for triple in itertools.combinations(classes, 3):
            variants = {}
            for order in itertools.permutations(triple):
                concatenated = torch.cat([class_bases[c] for c in order], dim=1)
                u, singular, _ = torch.linalg.svd(concatenated, full_matrices=False)
                if (u.shape[1] < safe_rank or not torch.isfinite(singular).all()
                        or singular[safe_rank - 1] <= singular.max() * projector_rtol):
                    raise ValueError(f"Invalid rank-{safe_rank} Safe-BG triplet {order}")
                q = u[:, :safe_rank]
                gram_error = (q.T @ q - torch.eye(safe_rank, device=self.device,
                                                  dtype=self.dtype)).abs().max()
                if not torch.isfinite(gram_error) or gram_error > 1e-5:
                    raise ValueError(f"Nonorthonormal Safe-BG triplet {order}")
                projector_rank = torch.linalg.matrix_rank(q @ q.T, rtol=projector_rtol)
                if int(projector_rank) != safe_rank:
                    raise ValueError(f"Wrong Safe-BG projector rank for {order}")
                energy = singular.square()
                variants[order] = (q, singular[:16], gram_error, projector_rank,
                                   energy[:8].sum() / energy.sum().clamp_min(1e-12))
                all_q.append(q)
            self.entries[triple] = variants
        self.sha256 = hashlib.sha256(torch.stack(all_q).detach().cpu().contiguous()
                                     .numpy().tobytes()).hexdigest()
        self.precompute_seconds = time.perf_counter() - started

    def lookup(self, active):
        ordered = tuple(active)
        return self.entries[tuple(sorted(ordered))][ordered]


def select_strict_tsfl(target_sensitivity, class_sensitivity, candidate_fraction=0.30,
                       safety_quantile=0.50, worst_k=3, selected_fraction=0.15,
                       min_selected=4, eps=1e-12):
    """Apply target-gate, non-target-gate, then stable target/NT ratio ranking."""
    if target_sensitivity.ndim != 1 or not class_sensitivity:
        raise ValueError("Strict TSFL expects C target scores and per-class NT scores")
    if not 0 < candidate_fraction <= 1 or not 0 <= safety_quantile <= 1:
        raise ValueError("Invalid Strict TSFL gate settings")
    if not 0 < selected_fraction <= 1 or not 1 <= worst_k <= len(class_sensitivity):
        raise ValueError("Invalid Strict TSFL ranking settings")
    classes = sorted(class_sensitivity)
    values = torch.stack([class_sensitivity[c] for c in classes]).to(target_sensitivity)
    if values.shape[1:] != target_sensitivity.shape or not torch.isfinite(values).all():
        raise ValueError("Invalid or nonfinite class-wise sensitivity tensor")
    if not torch.isfinite(target_sensitivity).all() or target_sensitivity.mean() <= 0:
        raise ValueError("Invalid target sensitivity")
    if (values.mean(1) <= 0).any():
        raise ValueError("A non-target class has zero sensitivity")
    st_hat = target_sensitivity / (target_sensitivity.mean() + eps)
    nt_class_hat = values / (values.mean(1, keepdim=True) + eps)
    k = min(worst_k, len(classes))
    nt_worst_hat = nt_class_hat.topk(k, dim=0, largest=True, sorted=True).values.mean(0)
    target_threshold = torch.quantile(st_hat, 1.0 - candidate_fraction)
    nt_threshold = torch.quantile(nt_worst_hat, safety_quantile)
    target_mask = st_hat >= target_threshold
    safe_mask = nt_worst_hat <= nt_threshold
    eligible = target_mask & safe_mask
    eligible_idx = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_idx.numel() == 0:
        raise ValueError("Strict TSFL gates leave no eligible channels")
    ratio = st_hat / (nt_worst_hat + eps)
    requested = max(1, math.ceil(selected_fraction * target_sensitivity.numel()))
    count = min(requested, int(eligible_idx.numel()))
    order = torch.argsort(ratio[eligible_idx], descending=True, stable=True)
    indices = eligible_idx[order[:count]].sort().values
    if indices.numel() < min_selected:
        raise ValueError(f"Strict TSFL selected {indices.numel()} channels; minimum is {min_selected}")
    selected = torch.zeros_like(target_mask)
    selected[indices] = True
    unselected = ~selected
    conflicts = {
        int(channel): [classes[i] for i in torch.argsort(nt_class_hat[:, channel], descending=True, stable=True)[:k].tolist()]
        for channel in indices.tolist()
    }
    summary = {
        "total_channels": int(target_sensitivity.numel()),
        "target_candidates": int(target_mask.sum()),
        "safe_candidates": int(eligible.sum()),
        "final_selected_channels": int(indices.numel()),
        "final_selected_fraction": float(indices.numel() / target_sensitivity.numel()),
        "target_candidate_fraction": float(candidate_fraction),
        "target_gate_quantile": float(1.0 - candidate_fraction),
        "target_gate_threshold": float(target_threshold),
        "nt_safety_quantile": float(safety_quantile),
        "nt_safety_threshold": float(nt_threshold),
        "sensitivity_worst_k": int(k),
        "selected_channel_fraction": float(selected_fraction),
        "mean_S_t_selected": float(st_hat[selected].mean()),
        "mean_S_t_unselected": float(st_hat[unselected].mean()),
        "mean_S_nt_worst_selected": float(nt_worst_hat[selected].mean()),
        "mean_S_nt_worst_unselected": float(nt_worst_hat[unselected].mean()),
        "mean_ratio_selected": float(ratio[selected].mean()),
        "mean_ratio_unselected": float(ratio[unselected].mean()) if unselected.any() else None,
        "selected_channel_indices": indices.tolist(),
        "selected_channel_top_conflicting_classes": conflicts,
    }
    return {
        "indices": indices,
        "S_target_normalized": st_hat,
        "S_nt_class_normalized": nt_class_hat,
        "S_nt_worst_normalized": nt_worst_hat,
        "ratio": ratio,
        "target_candidate_mask": target_mask,
        "safe_candidate_mask": eligible,
        "summary": summary,
        "classes": classes,
    }


def safe_background_direction(target_clean, background, class_bases, active_k=3,
                              projector_rtol=1e-6, eps=1e-12,
                              mode="full_rank", safe_rank=None, triplet_cache=None,
                              removal_alpha=1.0):
    """Project each clean BG direction outside its top-k active NT subspaces.

    Inputs are frozen selected-space vectors [N,D]. The function intentionally
    does not fallback when projection removes most of a direction; it reports
    that collapse through the returned retention and collapse mask.
    """
    if target_clean.ndim != 2 or background.shape != target_clean.shape or not class_bases:
        raise ValueError("Safe-BG expects matching [N,D] inputs and class bases")
    classes = sorted(class_bases)
    if not 1 <= active_k <= len(classes):
        raise ValueError("Invalid Safe-BG active_k")
    if projector_rtol <= 0 or eps <= 0:
        raise ValueError("Invalid Safe-BG numerical tolerance")
    if mode not in ("full_rank", "low_rank"):
        raise ValueError("Unknown Safe-BG projection mode")
    if mode == "low_rank" and (safe_rank not in (4, 8) or target_clean.shape[1] != 24):
        raise ValueError("Safe-BG requires preregistered rank 4 or 8 in 24 dimensions")
    if removal_alpha not in (0.5, 1.0) or (safe_rank == 4 and removal_alpha != 1.0):
        raise ValueError("Only rank-4 hard or rank-8 hard/half removal is admitted")
    if triplet_cache is not None and (mode != "low_rank" or active_k != 3
            or triplet_cache.classes != tuple(classes)
            or triplet_cache.device != target_clean.device
            or triplet_cache.dtype != target_clean.dtype
            or triplet_cache.projector_rtol != projector_rtol
            or triplet_cache.safe_rank != safe_rank):
        raise ValueError("Safe-BG triplet cache does not match runtime inputs")
    d_bg = (background - target_clean).detach()
    if not torch.isfinite(d_bg).all():
        raise ValueError("Nonfinite raw background direction")
    directions, retentions, residual_norms, removed, active_rows, ratios_rows, active_ratio_rows, orth_cosines = [], [], [], [], [], [], [], []
    projector_ranks, projector_columns, orthonormal_errors, decomposition_errors, energy_ratios, spectra = [], [], [], [], [], []
    projected_norms, retained_projected_norms = [], []
    for d in d_bg:
        norm_bg = d.norm()
        if norm_bg <= eps:
            raise ValueError("Raw background direction has zero norm")
        ratios = []
        for cls in classes:
            basis = class_bases[cls].to(device=d.device, dtype=d.dtype)
            if triplet_cache is None and (basis.ndim != 2 or basis.shape[0] != d.numel()
                                          or not torch.isfinite(basis).all()):
                raise ValueError(f"Invalid selected-space basis for class {cls}")
            proj = basis @ (basis.transpose(0, 1) @ d)
            ratios.append(proj.square().sum() / (norm_bg.square() + eps))
        ratios_t = torch.stack(ratios)
        # Classes are already sorted, so stable descending argsort also gives
        # deterministic class-ID tie breaking without a device-to-host sync.
        order = torch.argsort(ratios_t, descending=True, stable=True)[:active_k]
        active = [classes[i] for i in order.tolist()]
        active_ratios = ratios_t.index_select(0, order)
        if triplet_cache is not None:
            u_active, singular_top16, gram_error, projector_rank, top8_energy = triplet_cache.lookup(active)
        else:
            concatenated = torch.cat([class_bases[c].to(device=d.device, dtype=d.dtype) for c in active], dim=1)
            u, singular, _ = torch.linalg.svd(concatenated, full_matrices=False)
            if singular.numel() == 0 or not torch.isfinite(singular).all():
                raise ValueError("Unstable active non-target projector SVD")
            if mode == "low_rank":
                if u.shape[1] < safe_rank or singular[safe_rank - 1] <= singular.max() * projector_rtol:
                    raise ValueError(f"Active non-target span cannot support preregistered rank {safe_rank}")
                u_active = u[:, :safe_rank]
            else:
                keep = singular > singular.max() * projector_rtol
                if not keep.any():
                    raise ValueError("Active non-target projector has zero rank")
                u_active = u[:, keep]
            gram_error = (u_active.transpose(0, 1) @ u_active -
                          torch.eye(u_active.shape[1], device=u.device, dtype=u.dtype)).abs().max()
            if not torch.isfinite(gram_error) or gram_error > 1e-5:
                raise ValueError("Active non-target projector basis is not orthonormal")
            projector_rank = torch.linalg.matrix_rank(u_active @ u_active.T, rtol=projector_rtol)
            spectrum_energy = singular.square()
            top8_energy = spectrum_energy[:8].sum() / spectrum_energy.sum().clamp_min(eps)
            singular_top16 = singular[:16]
        projected = u_active @ (u_active.transpose(0, 1) @ d)
        removed_component = removal_alpha * projected
        residual = d - removed_component
        norm_safe = residual.norm()
        retention = norm_safe / (norm_bg + eps)
        if not torch.isfinite(retention):
            raise ValueError("Nonfinite Safe-BG direction retention")
        residual_norms.append(norm_safe)
        if mode == "low_rank":
            if norm_safe <= eps:
                raise ValueError("Low-rank Safe-BG residual has zero norm")
            scaled = norm_bg * residual / (norm_safe + eps)
        else:
            scaled = torch.where(norm_safe > eps, residual * (norm_bg / norm_safe.clamp_min(eps)), torch.zeros_like(residual))
        orth_cos = (residual @ projected) / (residual.norm() * projected.norm()).clamp_min(eps)
        decomposition_error = (d - (removed_component + residual)).norm()
        directions.append(scaled)
        retentions.append(retention)
        projected_norms.append(projected.norm())
        retained_projected_norms.append((1.0 - removal_alpha) * projected.norm())
        removed.append(removed_component.norm())
        active_rows.append(active)
        ratios_rows.append(ratios_t)
        active_ratio_rows.append(active_ratios)
        orth_cosines.append(orth_cos)
        projector_ranks.append(projector_rank)
        projector_columns.append(u_active.shape[1])
        orthonormal_errors.append(gram_error)
        decomposition_errors.append(decomposition_error)
        energy_ratios.append(top8_energy)
        spectra.append(singular_top16)
    return target_clean.detach() + torch.stack(directions), {
        "d_bg_norm": torch.stack([d.norm() for d in d_bg]),
        "d_safe_norm": torch.stack(residual_norms),
        "d_safe_scaled_norm": torch.stack([d.norm() for d in directions]),
        "retention": torch.stack(retentions),
        "removed_norm": torch.stack(removed),
        "removed_ratio": torch.stack(removed) / (torch.stack([d.norm() for d in d_bg]) + eps),
        "projected_component_norm": torch.stack(projected_norms),
        "retained_projected_component_norm": torch.stack(retained_projected_norms),
        "active_classes": active_rows,
        "projection_ratios": torch.stack(ratios_rows),
        "active_projection_ratios": torch.stack(active_ratio_rows),
        "orth_cosine": torch.stack(orth_cosines),
        "collapsed": torch.stack(retentions) < 0.05,
        "projector_rank": torch.stack(projector_ranks),
        "projector_basis_columns": torch.tensor(projector_columns, device=d_bg.device),
        "orthonormal_error": torch.stack(orthonormal_errors),
        "decomposition_error": torch.stack(decomposition_errors),
        "top8_energy_ratio": torch.stack(energy_ratios),
        "singular_values_top16": torch.stack(spectra),
    }
