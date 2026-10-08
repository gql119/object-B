"""Safe-Soft05 with an independent teacher-aligned classification KD term.

The inherited frozen-reference feature retention term is deliberately untouched.
"""

import hashlib
import json
import math
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
                or kd_cfg["scale_mode"] != "fixed_gradient_lambda"
                or kd_cfg.get("reduction") != "object_balanced"):
            raise ValueError("Safe-Soft05-KDcls scientific settings changed")
        if (self.target_class_id != 14 or self.safe_bg_rank != 8
                or self.safe_bg_active_k != 3 or self.safe_bg_removal_alpha != 0.5
                or self.lambda_orth != 0.0005
                or self.channels.tolist() != SAFE_SOFT05_CHANNELS
                or abs(self.eps - 8 / 255) > 1e-9
                or method_cfg["support_mode"] != "bbox_half_overlap"):
            raise ValueError("Safe-Soft05 parent scientific settings changed")
        calibration_path = Path(kd_cfg["calibration_artifact"])
        artifact_bytes = calibration_path.read_bytes()
        if hashlib.sha256(artifact_bytes).hexdigest() != kd_cfg["calibration_sha256"]:
            raise ValueError("KD calibration artifact hash changed")
        calibration = json.loads(artifact_bytes)
        if (calibration.get("schema") != "safe-soft05-kdcls-object-balanced-lambda-v1"
                or calibration["seed"] != 0 or calibration["optimizer_step_count"] != 0
                or calibration["batch_size"] != 16
                or calibration["query_indices_per_batch"] != [0, 4, 8, 12]
                or len(calibration["selected_stems"]) != 64
                or calibration["query_count"] != 16
                or calibration["gate"] != "KD_OBJECT_BALANCED_CALIBRATION_PASS"
                or not all(calibration["checks"].values())):
            raise ValueError("KD calibration is not the passing preregistered 16-query probe")
        candidate_lambdas = [float(calibration[name]) for name in
                             ("lambda_from_ret", "lambda_from_q90", "lambda_from_max")]
        final_lambda = float(calibration["final_lambda_KD"])
        if (not all(math.isfinite(value) and value > 0 for value in candidate_lambdas)
                or not math.isfinite(final_lambda) or final_lambda <= 0
                or abs(final_lambda - min(candidate_lambdas)) > 1e-8):
            raise ValueError("KD lambda does not match the locked gradient rule")
        self.lambda_kd = final_lambda
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
        weighted_kd = self.lambda_kd * kd
        total = old_total + weighted_kd
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Non-finite Safe-Soft05-KDcls total loss")

        # Parent L_ret, weighted_L_ret and its computation graph stay identical.
        # A hashed no-update train-only gradient calibration fixes KD-only lambda.
        metrics.update({
            "L_ret_raw": ret,
            "L_KD_raw": kd,
            "lambda_KD": kd.new_tensor(self.lambda_kd),
            "L_KD_weighted": weighted_kd,
            "L_util_diagnostic": self.lambda_ret * ret + weighted_kd,
            "L_total": total,
            "kd_tau": kd.new_tensor(2.0),
            "kd_gradient_calibrated": kd.new_ones(()),
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
