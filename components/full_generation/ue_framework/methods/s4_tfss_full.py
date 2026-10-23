"""Opt-in S4-TFSS-Full; canonical S4 files remain byte-identical.

Only the final geometric support changes. Fourier/JND/EOT/STE flooring retain
the original bbox-half computation, followed by the inherited TFSS projection.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from .base import PoisonResult
from .fourier import spectrum_to_numpy
from .legacy_kproto_ret import _StrictTargetSupport, LegacyKProtoGenerator
from .tausb_universal import _TAUSBCommon
from .s4_tfss import S4Trainer
from .s4_pixel_audit import domains, png_project

SUPPORT_MODE = "bbox_full_target"
EXPERIMENT_ID = "S4_TFSS_FULL_OVERLAP"


def full_support(shape, annotations):
    target, _ = domains(shape, annotations)
    return target.astype(np.float32)


def original_training_eligible(shape, annotations):
    """Keep original sample admission; no half coefficient reaches the delta."""
    target, other = domains(shape, annotations)
    return float(target.sum() - .5*(target & other).sum()) > 10


class FullTargetSupport(_StrictTargetSupport):
    """Inserted immediately before _StrictTargetSupport in the existing MRO."""
    def _build_support(self, image_shape, annotations, support_type="mask",
                       ring_width=4, image_path=None):
        if self.support_mode != SUPPORT_MODE:
            return super()._build_support(image_shape, annotations, support_type,
                                          ring_width, image_path)
        if support_type != "mask":
            raise ValueError("Full target support requires mask argument")
        support = full_support(image_shape, annotations)
        return support, np.zeros_like(support), SUPPORT_MODE if support.any() else "no_target"

    def _compose_delta_batched(self, img_t, inner_t, ring_t, coords,
                               fourier_coeff, suppress_small, current_epoch=0):
        if self.support_mode != SUPPORT_MODE:
            return super()._compose_delta_batched(img_t, inner_t, ring_t, coords,
                fourier_coeff, suppress_small, current_epoch)
        if not bool(((inner_t == 0) | (inner_t == 1)).all()):
            raise ValueError("bbox_full_target requires binary support")
        binary = (inner_t > 0).to(inner_t.dtype)
        # Same actual pre-support implementation reached by the original
        # _StrictTargetSupport super-call. No replacement Fourier/JND/EOT.
        raw, base, _, _, jnd = _TAUSBCommon._compose_delta_batched(
            self, img_t, binary, torch.zeros_like(ring_t), coords,
            fourier_coeff, suppress_small, current_epoch)
        # Preserve the original 2.01-byte STE floor, even though no halving is
        # needed here. Removing it would introduce another experimental variable.
        minimum = 2.01 / 255.0
        sign = torch.where(base < 0, -torch.ones_like(base), torch.ones_like(base))
        sign = torch.where((img_t <= 0) & (sign < 0), -sign, sign)
        sign = torch.where((img_t >= 1) & (sign > 0), -sign, sign)
        floored = sign * torch.maximum(base.abs(), base.new_tensor(minimum))
        base = base + (floored - base).detach()
        delta = base * inner_t
        return raw * inner_t, delta, (img_t + delta).clamp(0, 1), binary, jnd


def spatial_diagnostics(clean, delta, annotations, *, q=None, risk=None, weight=None):
    """Actual image-clamped RGB delta in fractional units; geometry from GT."""
    clean = np.asarray(clean, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    if clean.shape != d.shape or clean.ndim != 3 or clean.shape[-1] != 3:
        raise ValueError("Expected matching HWC RGB")
    if not np.isfinite(d).all():
        raise ValueError("Nonfinite spatial diagnostic")
    target, other = domains(clean.shape, annotations)
    overlap, exclusive = target & other, target & ~other

    def rms(mask):
        return float(np.sqrt(np.mean(d[mask] ** 2))) if mask.any() else None
    def mean_map(value, mask):
        if value is None or not mask.any():
            return None
        a = np.asarray(value)
        if a.shape != target.shape or not np.isfinite(a).all():
            raise ValueError("Spatial map/GT coordinates differ")
        return float(a[mask].mean())
    mse = float(np.mean(d ** 2))
    per_class = {}
    for cls in sorted({int(a["cls"]) for a in annotations if int(a["cls"]) != 14}):
        _, nt = domains(clean.shape, [a for a in annotations if int(a["cls"]) == cls])
        mask = target & nt
        per_class[str(cls)] = dict(area_pixels=int(mask.sum()), RMS_delta=rms(mask),
            mean_abs_delta=float(np.mean(np.abs(d[mask]))) if mask.any() else None)
    return dict(overlap_area_ratio=float(overlap.sum()/target.sum()) if target.any() else None,
        overlap_area_pixels=int(overlap.sum()), perturbation_region_area_pixels=int(target.sum()),
        changed_area_pixels=int(np.any(d != 0, axis=2).sum()),
        overlap_mean_abs_delta=float(np.abs(d[overlap]).mean()) if overlap.any() else None,
        overlap_RMS_delta=rms(overlap),
        overlap_Linf_delta=float(np.abs(d[overlap]).max()) if overlap.any() else None,
        non_overlap_RMS_delta=rms(exclusive), person_RMS_delta=rms(target),
        overlap_mean_W_U=mean_map(weight, overlap), overlap_mean_Q_t=mean_map(q, overlap),
        overlap_mean_R_nt=mean_map(risk, overlap),
        PSNR_db=10*math.log10(1/mse) if mse > 0 else None,
        PSNR_zero_MSE=bool(mse == 0), overlap_by_NT_class=per_class)


def export_project(clean_uint8, candidate_float, annotations, eps):
    """Shared decoded-PNG projection/audit: no residual 4-byte overlap cap."""
    if not np.isfinite(eps) or abs(eps-8/255) > 1e-9:
        raise ValueError("Frozen 8/255 epsilon changed")
    support = full_support(clean_uint8.shape, annotations)
    result = png_project(clean_uint8, candidate_float, support*eps)
    difference = result.astype(np.int16)-clean_uint8.astype(np.int16)
    target = support > 0
    if np.any(difference[~target] != 0) or np.any(np.abs(difference) > 8):
        raise ValueError("Full target decoded PNG violates 8/0 budget")
    return result


class S4FullTrainer(S4Trainer, FullTargetSupport):
    def __init__(self, cfg, method_cfg, device, surrogate, adaptive):
        if method_cfg.get("support_mode") != SUPPORT_MODE:
            raise ValueError("S4-TFSS-Full configuration identity missing")
        super().__init__(cfg, method_cfg, device, surrogate, adaptive)

    def _batch_loss(self, items, prototypes):
        if self.tfss_mode != "TFSS-U":
            raise ValueError("S4-TFSS-Full requires unchanged TFSS-U, eta=.85")
        loss, metrics = super()._batch_loss(items, prototypes)
        # Parent losses/update graphs remain intact. Replace only the old
        # support==0.5 overlap diagnostic, which is empty for full support.
        for i, ((image, ann, _), (q, risk, _)) in enumerate(zip(items, self.s4_maps)):
            h, w = image.shape[:2]
            delta = ((self.tfss_input+self.tfss_delta).clamp(0, 1)-self.tfss_input)
            d = delta[i, :, :h, :w].detach().permute(1, 2, 0).cpu().numpy()
            weight = self.tfss_applied_weight[i, 0, :h, :w].detach().cpu().numpy()
            row = spatial_diagnostics(image, d, ann, q=q.cpu().numpy(),
                                      risk=risk.cpu().numpy(), weight=weight)
            self.s4_spatial_rows[i].update(row)
            self.s4_spatial_rows[i]["overlap_linf"] = row["overlap_Linf_delta"] or 0.
        for key in self.s4_spatial_rows[0]:
            if key in ("image_id", "overlap_by_NT_class", "PSNR_zero_MSE"):
                continue
            values = [r.get(key) for r in self.s4_spatial_rows]
            numeric = [v for v in values if isinstance(v, (float, int)) and not isinstance(v, bool)]
            if numeric:
                metrics["s4_"+key] = max(numeric) if key in ("overlap_linf", "overlap_Linf_delta", "actual_linf") else sum(numeric)/len(numeric)
            else:
                metrics["s4_"+key] = None
        return loss, metrics


class S4FullGenerator(LegacyKProtoGenerator, FullTargetSupport):
    """Canonical generator copied below with only minimum_support 10 -> 0.

    This preserves export of tiny valid Person boxes, just as bbox_half_overlap.
    Checkpoint support_mode validation remains inherited and rejects old S4 noise.
    """

    def __init__(self, cfg, method_cfg, device, surrogate, global_params_path):
        if method_cfg.get("support_mode") != SUPPORT_MODE:
            raise ValueError("S4-TFSS-Full generator requires full support configuration")
        super().__init__(cfg, method_cfg, device, surrogate, global_params_path)
        pack = torch.load(global_params_path, map_location="cpu", weights_only=True)
        if (pack.get("s4_identity", {}).get("method") != EXPERIMENT_ID
                or pack.get("s4_identity", {}).get("support_mode") != SUPPORT_MODE):
            raise ValueError("Checkpoint is not the independent S4-TFSS-Full experiment")


    def generate(self, image: np.ndarray, annotations: List[dict], seed: int, steps: int, eps: float, support_type: str, image_path: str = None) -> PoisonResult:
        if support_type != "mask":
            raise ValueError(f"tausb_mask only supports support_type='mask', got {support_type}")

        inner_np, ring_np, support_source = self._build_support(
            image_shape=image.shape,
            annotations=annotations,
            support_type="mask",
            ring_width=self.ring_width,
            image_path=image_path,
        )

        minimum_support = 0.0  # Full GT union, same tiny-target export policy as original S4
        if float(inner_np.sum()) <= minimum_support:
            zero = image * 0.0
            return PoisonResult(
                poisoned_image=image.copy(),
                perturbation=zero,
                support_mask=inner_np,
                ring_mask=ring_np,
                losses={"L_total": 0.0},
                extras={
                    "note": "empty_support",
                    "is_poisoned": False,
                    "poisoned": 0,
                    "support_source": support_source,
                    "mask_path": self._resolve_instance_mask_path(image_path) or "",
                },
            )

        with torch.no_grad():
            img_t = self._to_tensor(image)
            raw_perturb, perturb, adv, _support, jnd = self._compose_delta(
                img_t=img_t,
                inner_np=inner_np,
                ring_np=ring_np,
                coords=self.coords,
                fourier_coeff=self.fourier_coeff,
                suppress_small=self.suppress_small,
            )
            budget_violation = F.relu(torch.max(torch.abs(raw_perturb)) - self.eps)

            h, w = image.shape[:2]
            spectrum = spectrum_to_numpy(
                self.imgsz,
                self.imgsz,
                self.coords,
                torch.tanh(self.fourier_coeff[:, 0] / self.tanh_temp).detach().cpu().numpy(),
            )
            pattern_vis = self._build_global_freq_pattern(h, w, self.coords, self.fourier_coeff).mean(dim=1)

        return PoisonResult(
            poisoned_image=self._to_numpy(adv),
            perturbation=self._to_numpy(perturb),
            support_mask=inner_np,
            ring_mask=ring_np,
            losses={
                "L_budget": float(budget_violation.item()),
                "L_total": float(budget_violation.item()),
            },
            extras={
                "note": "poisoned",
                "is_poisoned": True,
                "poisoned": 1,
                "coords": self.coords,
                "jnd_gain": jnd.squeeze(0).squeeze(0).detach().cpu().numpy(),
                "spectrum": spectrum,
                "pattern": pattern_vis.squeeze(0).detach().cpu().numpy(),
                "support_source": support_source,
                "mask_path": self._resolve_instance_mask_path(image_path) or "",
            },
        )
