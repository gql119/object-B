# Safe-Soft05-KDcls gradient calibration audit

Status: **GPU gradient distribution pending**. This document separates the completed loss-scale audit from the 16-query delta-gradient audit. It is not a formal poison or victim result.

## Fixed scale and sample identity

The original `3131.96337890625` was a **ratio of two raw loss values from one clean training query**, not a gradient norm or gradient ratio:

`L_ret_raw / L_KD_raw = 7.316342816920951e-5 / 2.3360243517345225e-8`.

The replacement was preregistered before the new smoke. From the first 64 sorted training images containing both person and non-target GT, four disjoint 16-image support batches were formed. Each batch used clean-query positions 0, 4, 8 and 12, for **4 batches and 16 query measurements** with seed 0, one EOT per measurement and zero perturbation optimizer updates. The recorded teacher-TAL non-target positive-unit count summed to **240**. No validation image or final AP entered the statistic. The existing formal Safe-Soft05 background prototype cache was reused with a verified cache key and SHA256 `12889ba0f80b29c194a59bd9eb2a6d8b137e9aa4977fdb94ad84444628b316a7`.

The new factor is the **ratio of arithmetic means across the 16 query measurements**:

`s_KD = mean_q(L_ret_raw[q]) / mean_q(L_KD_raw[q])`

`= 0.00010502697568881558 / 0.0012989567871263263`

`= 0.08085486501915594`.

No median, trimmed mean, RMS, gradient norm or gradient ratio entered this calibration. The sum of the 16 raw KD losses is `0.02078330859402122`; one query (`batch 1`, query stem `000133`) contributed `0.019322259351611137`, or **92.97%** of that sum. The loss distribution is therefore strongly long-tailed. The preregistered mean and fixed factor were retained without outlier exclusion. The calibration artifact is `experiments/safe_soft05_kdcls_20261007/calibration_multibatch.json`, SHA256 `f203bde588023fb932d2667fb2c02c5e8a5653ab97eb1b4135fa54e636f8959a`.

`lambda_ret = 0.25`, `beta = 1`, `tau = 2`. The added loss is `lambda_ret * beta * s_KD * L_KD_cls_nt`, so the **effective KD coefficient** is `0.020213716254788985`. The scale itself is `0.08085486501915594`; those two quantities must not be conflated. The inherited retention loss remains `0.25 * L_ret`.

## The earlier 0.03354 diagnostic

The five-step `0.033537253737449646` was the **maximum across five optimizer steps** of

`||grad_poison(0.25 L_ret + 0.25 s_KD L_KD)|| / max(||grad_poison(L_BG + L_rel + L_adaptive)||, 1e-12)`.

It was a ratio of **weighted gradients with respect to the composed poisoned-image tensor**, not a KD-only/attack ratio, not a batch mean and not a delta-gradient measurement. The present audit explicitly differentiates each weighted term against the **same composed delta tensor** on each of the 16 fixed query measurements. Ratios use denominator `norm + 1e-12`; cosine denominators use `max(product of norms, 1e-12)`. Distribution quantiles are linear-interpolated across the 16 queries; standard deviation is population std (`ddof=0`).

## Inherited retention and KD correspondence

`git diff 0d43da4..8c80efe` changes neither `ue_framework/methods/legacy_kproto_ret.py`, `ue_framework/methods/b2_semantic.py`, `ue_framework/methods/b2_adaptive.py` nor `ue_framework/methods/adaptive_learner.py`. The parent `L_ret` uses the unchanged `non_target_retention_loss`: squared channelwise feature difference, summed over channels and averaged over non-target pooled instances. With Safe-Soft05's P5/P3 split, its non-target features come from the `model.15` orthogonality feature path; the P5 target TSFL path remains `model.21`. The clean/poison EOT pair, masks, reduction, computational graph and `lambda_ret=0.25` are inherited. The parent configuration diff contains only the added KD stanza and isolated run/artifact paths.

The frozen Teacher is the pre-update model on the existing clean query; the Student is that model after the existing differentiable one-step Adaptive update, evaluated on the **same clean query**. Teacher-only, non-target-GT TAL positives determine a common `(batch, location)` index for both raw-logit tensors. Person GT is masked before TAL. Person logit is removed from both 20-class vectors, leaving 19 non-target logits normalized by softmax at temperature 2. The loss is classification KL(T||S) only. No box, DFL or objectness KD was added.

## Delta-gradient distribution and gate

Pending the locked 16-query GPU audit. The run will write `kd_gradient_calibration_batches.csv`, `kd_gradient_calibration_batch_summary.csv` and `kd_gradient_calibration_summary.json` in a fresh, collision-checked RunID. It will not update noise or train a victim. The final verdict will use distinct KD/ret, KD/attack and utility/attack ratios and the predeclared weak/strong thresholds; no coefficient changes will be applied automatically.
