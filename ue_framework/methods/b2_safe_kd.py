"""Safe-Soft05 with an independent teacher-aligned classification KD term.

The inherited frozen-reference feature retention term is deliberately untouched.
"""

import hashlib
import json
from pathlib import Path

import torch

from .b2_semantic import B2SemanticTrainer
from .kd_utility import KDAdaptiveLearner

SAFE_SOFT05_CHANNELS = [0, 6, 10, 29, 41, 68, 77, 86, 89, 112, 116, 125,
                        128, 133, 140, 167, 173, 184, 220, 229, 237, 238, 239, 240]


def _cosine(left, right):
    return (left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-12)


class SafeSoft05KDTrainer(B2SemanticTrainer):
    def __init__(self, cfg, method_cfg, device, frozen_reference, adaptive_model):
        super().__init__(cfg, method_cfg, device, frozen_reference, adaptive_model)
        kd_cfg = method_cfg["semantic"]["kd_cls"]
        if (not kd_cfg.get("enabled") or float(kd_cfg["tau"]) != 2.0
                or float(kd_cfg["beta"]) != 1.0
                or kd_cfg["scale_mode"] != "fixed_reference_scale"):
            raise ValueError("Safe-Soft05-KDcls scientific settings changed")
        if (self.target_class_id != 14 or self.safe_bg_rank != 8
                or self.safe_bg_active_k != 3 or self.safe_bg_removal_alpha != 0.5
                or self.lambda_orth != 0.0005
                or self.channels.tolist() != SAFE_SOFT05_CHANNELS
                or abs(self.eps - 8 / 255) > 1e-9
                or method_cfg["support_mode"] != "bbox_half_overlap"):
            raise ValueError("Safe-Soft05 parent scientific settings changed")
        self.kd_beta = 1.0
        calibration_path = Path(kd_cfg["calibration_artifact"])
        artifact_bytes = calibration_path.read_bytes()
        if hashlib.sha256(artifact_bytes).hexdigest() != kd_cfg["calibration_sha256"]:
            raise ValueError("KD calibration artifact hash changed")
        calibration = json.loads(artifact_bytes)
        if (calibration["seed"] != 0 or calibration["optimizer_step_count"] != 0
                or not all(calibration["checks"].values())):
            raise ValueError("KD calibration is not a passing seed-0 no-update probe")
        ret_ref = float(calibration["metrics"]["L_ret_raw"])
        kd_ref = float(calibration["metrics"]["L_KD_raw"])
        if not (ret_ref > 1e-12 and kd_ref > 1e-12):
            raise ValueError("KD calibration loss scales must be positive")
        self.kd_scale = torch.tensor(ret_ref / kd_ref, device=device).detach()
        # The existing pre-update adaptive parameters equal the frozen reference
        # checkpoint; only the student receives the differentiable virtual step.
        self.adaptive = KDAdaptiveLearner(
            adaptive_model, self.adaptive_feature_layer, self.target_class_id,
            self.imgsz, float(method_cfg["adaptive"]["inner_lr"]),
        )

    def _batch_loss(self, items, prototypes):
        result = super()._batch_loss(items, prototypes)
        if result is None:
            return None
        old_total, metrics = result
        ret = metrics["L_ret"]
        kd = metrics["L_KD_cls_nt_raw"]
        scale = self.kd_scale
        weighted_kd = self.lambda_ret * self.kd_beta * scale * kd
        total = old_total + weighted_kd
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Non-finite Safe-Soft05-KDcls total loss")

        # Parent L_ret, weighted_L_ret and its computation graph stay identical.
        # A hashed no-update train-only probe fixes the KD-only scale for this run.
        metrics.update({
            "L_ret_raw": ret,
            "L_KD_raw": kd,
            "kd_fixed_scale": scale,
            "L_KD_scaled": scale * kd,
            "L_KD_weighted": weighted_kd,
            "L_util_diagnostic": ret + self.kd_beta * scale * kd,
            "L_total": total,
            "kd_beta": kd.new_tensor(self.kd_beta),
            "kd_tau": kd.new_tensor(2.0),
            "kd_scale_calibrated": kd.new_ones(()),
        })
        poison = self._last_poison
        g_kd = torch.autograd.grad(weighted_kd, poison, retain_graph=True)[0]
        g_total = torch.autograd.grad(total, poison, retain_graph=True)[0]
        metrics["grad_delta_total"] = g_total.norm()
        if bool(metrics["gradient_diagnostics_active"]):
            weighted_ret = self._semantic_terms["ret"]
            g_ret = torch.autograd.grad(weighted_ret, poison, retain_graph=True)[0]
            attack = (self._semantic_terms["BG_sel"]
                      + self._semantic_terms["rel_sel"] + metrics["L_adaptive"])
            g_attack = torch.autograd.grad(attack, poison, retain_graph=True)[0]
            g_util = g_ret + g_kd
            ratio = g_util.norm() / g_attack.norm().clamp_min(1e-12)
            metrics.update({
                "grad_ret_norm": g_ret.norm(),
                "grad_KD_norm": g_kd.norm(),
                "grad_util_norm": g_util.norm(),
                "grad_attack_norm": g_attack.norm(),
                "utility_attack_grad_ratio": ratio,
                "cos_grad_KD_ret": _cosine(g_kd, g_ret),
                "cos_grad_KD_attack": _cosine(g_kd, g_attack),
                "cos_grad_util_attack": _cosine(g_util, g_attack),
                "grad_total_norm": g_total.norm(),
            })
        else:
            for key in ("grad_ret_norm", "grad_KD_norm", "grad_util_norm",
                        "grad_attack_norm", "utility_attack_grad_ratio",
                        "cos_grad_KD_ret", "cos_grad_KD_attack",
                        "cos_grad_util_attack"):
                metrics[key] = kd.new_zeros(())
        return total, metrics
