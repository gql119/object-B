# Safe-Soft05-KDcls gradient calibration audit

Status: **KD_TOO_WEAK** after the locked 16-query delta-gradient audit. This is not a formal poison or victim result. Neither the fixed KD factor nor any other scientific setting was changed after observing these results.

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

## Locked delta-gradient audit

RunID `gradient_audit_mb16_a270038_seed0_20261007` exited 0 on an NVIDIA RTX A6000 (peak allocated memory 4.41 GiB). It used the four original 16-image support batches, the same 16 query stems in the same order, seed 0, the same Teacher/Student checkpoint, and the same parent prototype cache. It performed **zero optimizer updates**. Teacher, Adaptive base and noise state checks all passed before/after; person GT/logits were excluded and the Teacher's TAL location indices were shared with the Student. The audit reads the production KD term and differentiates three **already weighted** expressions against one identical composed `delta` tensor for each query:

- `g_ret = grad_delta(0.25 * L_ret)`;
- `g_KD = grad_delta(0.25 * 1 * 0.080854865 * L_KD_cls_nt)`;
- `g_attack = grad_delta(1 * L_BG + 1 * L_rel + 1 * L_adaptive)`;
- `g_util = g_ret + g_KD`.

The per-query raw `L_BG`, `L_rel`, `L_ret`, `L_KD`, weighted `L_adaptive`, and raw `L_orth_worst` are in `experiments/safe_soft05_kdcls_20261007/gradient_audit_mb16_a270038_seed0_20261007/kd_gradient_calibration_batches.csv`. Four batch-level means, each over its four predeclared queries, are in `kd_gradient_calibration_batch_summary.csv`. The primary distributions below use all **16 query observations**, not the four batch means. Full precision and all checks are in `kd_gradient_calibration_summary.json`.

| Metric | Min | Q10 | Q25 | Median | Q75 | Q90 | Max | Mean | Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `||g_ret||` | 0.0005694 | 0.0008619 | 0.0009839 | 0.001539 | 0.003245 | 0.006680 | 0.006802 | 0.002647 | 0.002365 |
| `||g_KD||` | 1.961e-8 | 2.403e-7 | 1.443e-6 | 6.971e-6 | 9.940e-5 | 0.0006064 | 0.004708 | 0.0003932 | 0.001139 |
| `||g_util||` | 0.0005733 | 0.0008623 | 0.0009990 | 0.001768 | 0.003252 | 0.006738 | 0.008176 | 0.002775 | 0.002525 |
| `||g_attack||` | 0.3071 | 0.3183 | 0.3226 | 0.3407 | 0.3626 | 0.4061 | 0.4106 | 0.3500 | 0.03445 |
| `r_KD_ret` | 9.806e-6 | 0.0001205 | 0.0003125 | **0.006025** | 0.09924 | 0.4519 | 0.9645 | 0.1360 | 0.2734 |
| `r_KD_attack` | 4.899e-8 | 6.946e-7 | 4.398e-6 | **2.142e-5** | 0.0002790 | 0.001524 | 0.01407 | 0.001132 | 0.003392 |
| `r_util_attack` | 0.001396 | 0.002644 | 0.002782 | 0.004739 | 0.009504 | 0.02078 | **0.02444** | 0.008210 | 0.007748 |
| `cos_KD_ret` | -0.01009 | -0.009138 | -0.003619 | 0.001380 | 0.02950 | 0.1013 | 0.1751 | 0.02677 | 0.05207 |
| `cos_KD_attack` | -0.2357 | -0.07272 | -0.02580 | 0.0008701 | 0.02707 | 0.09716 | 0.1368 | -0.001088 | 0.08349 |
| `cos_util_attack` | -0.03836 | -0.03013 | -0.02520 | -0.01752 | 0.002936 | 0.01351 | 0.02967 | -0.01129 | 0.01863 |

`r_KD_ret = ||g_KD||/(||g_ret||+1e-12)`, `r_KD_attack = ||g_KD||/(||g_attack||+1e-12)`, and `r_util_attack = ||g_ret+g_KD||/(||g_attack||+1e-12)`. These are distinct quantities. Median cosines are near zero, so this short fixed-set audit does not show a stable strong alignment or conflict. It cannot establish victim-level utility preservation.

The registered long-tail test is true for all three requested metrics. Their maximum/median and Q90/median ratios are respectively `675.3 / 87.0` for `||g_KD||`, `160.1 / 75.0` for `r_KD_ret`, and `656.8 / 71.1` for `r_KD_attack`, each above the predeclared limits of 10 or 5. Query `000133` in batch 1 is the largest KD-gradient outlier: `||g_KD||=0.004708`, `r_KD_ret=0.7043`, `r_KD_attack=0.01407`, and `r_util_attack=0.02444`; its KD-positive class is horse (VOC class 12). Query `000372` is another horse-dominated KD/ret outlier (`r_KD_ret=0.9645`). The batch-1 mean raw KD loss is `0.004839`; the other batch means are `2.728e-5`, `1.321e-5`, and `0.0003413`.

The fixed sample identities, ordering and raw `L_ret` values reproduced exactly. Raw KD is not bitwise identical on re-evaluation: maximum per-query absolute difference is `7.776e-5`, and its 16-query mean changed by `+0.48%` (`0.0012989568` to `0.0013051901`). The fixed calibration artifact and coefficient were **not** recomputed. Thus the protocol/sample selection is reproducible, while exact floating-point KD equality was not observed on this GPU; this limits any claim of bitwise replay.

## Gate and coefficient implication

The predeclared weak criterion is met: median `r_KD_ret=0.006025 < 0.1` **and** median `r_KD_attack=2.142e-5 < 0.02`. The strong criterion is not met: median `r_KD_attack` is below 0.5, Q90 `r_util_attack=0.02078 < 1`, and max `r_util_attack=0.02444 < 10`. This is **KD_TOO_WEAK**, despite a few nonzero outliers. It is not `KD_CALIBRATION_PASS`; do not start formal poison generation or victim training.

For a future **user-approved new candidate only**, linear rescaling of the measured fixed-set gradients to the lower target median `r_KD_ret=0.25` would multiply the effective KD coefficient by `0.25/0.006025113 = 41.493`, from `0.020213716` to approximately **`0.8387`** (equivalent KD-only scale `3.3549` while `lambda_ret=0.25`, `beta=1`). This is a suggested diagnostic starting value, **not applied or validated**; it would still project median `r_KD_attack` to only `0.000889`, below the desired 0.05. Conversely, a scalar large enough to project median KD/attack to 0.05 would be about 2,334 times the current coefficient and project the observed worst KD/attack ratio to about 32.8, violating the hard safety cap. The measured long tail therefore offers no single fixed scalar that meets both suggested median targets and the existing worst-case safety line on this fixed set. No coefficient or method change is authorized by this audit.
