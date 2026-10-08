# KD Mechanism Diagnostic Audit

**Decision: RECOMMEND_DETECTION_COMPONENT_AUDIT.** The fixed train-query audit finds one local Softmax confidence blind spot, but does not show that this mechanism caused the independent victim's non-target AP loss. The present evidence does not justify starting Sigmoid-KD experiment C.

## Scope and identity

This is a diagnostic of the completed Safe-Soft05-KD-ObjBal seed-0 run. The existing fixed calibration selection was reused: 16 clean training queries from four disjoint 16-image batches, covering 24 non-target GT objects and 240 Teacher-assigned positive locations. The final formal Fourier state was loaded and its delta evaluated in memory. There were **zero optimizer steps**, no new poison image or dataset write, and no victim training.

The exact formal config SHA256 was `8107f4a3f7859b3d775cfa3c37d461a774bc379b2782206ec7b0c39b363fc741`; fixed query set SHA256 `f203bde588023fb932d2667fb2c02c5e8a5653ab97eb1b4135fa54e636f8959a`; saved formal perturbation state SHA256 `43f8a8a9fb1e4fe589d4201a7f81ad20016b00f78ff5c18dc2483df4f5595d2a`. The parent prototype cache retained SHA256 `12889ba0f80b29c194a59bd9eb2a6d8b137e9aa4977fdb94ad84444628b316a7`. The script checks the locked config, calibration order and selected stems, cache identity, 24 P5 channels, Safe-BG rank 8/alpha 0.5, person class, epsilon 8/255, and the fixed `lambda_KD=0.7160846249749698`. The parent `L_ret`, orthogonality and attack losses are called through the unchanged formal trainer.

The detector's raw head scores are **logits**, not probabilities: for both Teacher and Student, `sigmoid(raw_scores)` equalled decoded class probabilities with maximum absolute error **0.0**. Observed raw logits ranged from **-69.9332 to 2.8132**. All Sigmoid values below were computed once from raw logits. Both KD diagnostics use `tau=2` and multiply KL by `tau²`. Softmax excludes the person logit and uses the formal object-balanced reduction; the diagnostic Bernoulli KL sums the 19 separate non-target class terms before the same reduction. Their unweighted loss values are on different scales.

The same delta tensor was used for all four gradients. `g_ret` includes the formal 0.25 weight; `g_attack` is the weighted Safe-BG + relative + adaptive objective, as in the fixed calibration; Softmax and diagnostic Sigmoid KD are shown both at unit coefficient and at the existing `lambda_KD`. Sigmoid KD was **never added to the total loss**. Directions were normalized to equal unit norm for cosine comparisons; the separate norm and ratio columns preserve scale.

## 1. Absolute confidence drift and the Softmax blind spot

The predeclared anomaly criterion was **Softmax KL ≤ 10⁻⁴** and **GT-class Sigmoid probability change ≤ -0.05** at a Teacher-assigned positive. It flagged **1/240 locations in 1/24 objects**. In horse query `000407`, location `8151`, Softmax KL was **3.3507×10⁻⁵**, while GT-class Sigmoid fell **0.58352 → 0.52220** (change **-0.06132**). Diagnostic Bernoulli KL was **0.008151**, about **243 times** that Softmax KL. Assignment to the same GT remained present and GT-aligned box IoU increased by **0.00511** at this location. This is a measured local example of normalized Softmax missing a substantial absolute confidence shift.

Across all locations, **204/240** Softmax KL values were ≤ 10⁻⁴, but only **8/240** GT-class Sigmoid changes were ≤ -0.05. All eight large declines were horse locations: seven from one object in query `000133` and one from `000407`. The Softmax KL median was **2.478×10⁻⁶**, Q90 **1.759×10⁻⁴**, maximum **0.04341**; Bernoulli KL median **1.093×10⁻⁵**, Q90 **3.891×10⁻⁴**, maximum **0.63314**. The median GT-class Sigmoid change was **+0.000686**; range **-0.13988 to +0.43141**.

| Unit and drift measure | Softmax Pearson | Softmax Spearman | Bernoulli Pearson | Bernoulli Spearman |
| --- | ---: | ---: | ---: | ---: |
| 240 positives, absolute change | 0.9225 | 0.6004 | 0.8937 | 0.7617 |
| 240 positives, signed change | 0.6847 | 0.2727 | 0.7906 | 0.3956 |
| 240 positives, downward magnitude | 0.2200 | -0.0836 | 0.0848 | -0.1403 |
| 24 object means, absolute change | 0.9640 | 0.6348 | 0.9442 | 0.7452 |

Bernoulli KL captures the flagged location and ranks absolute drift better by Spearman, while Softmax has the higher Pearson correlation with absolute drift. The downward-only rank correlations are poor for both; many locations have no decline, and the eight large drops cluster within two horse objects. Thus the observation supports a **local blind spot**, not a general claim that Bernoulli KD explains or fixes the victim AP change. Positive locations from one object are correlated, so 240 is not 240 independent objects.

## 2. Gradient effectiveness and direction

Every number below is across the **16 fixed queries** and on the same final-state delta. Norms are Euclidean norms over the delta tensor. Sigmoid weighted rows apply the current Softmax coefficient only for a scale diagnostic; that coefficient is not calibrated for Bernoulli KL.

| Norm or ratio | Q10 | Q25 | median | Q75 | Q90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `||g_ret||`, weighted | 0.0203 | 0.0314 | 0.0389 | 0.143 | 0.443 | 0.448 |
| `||g_KD_softmax||`, raw | 1.56e-5 | 8.89e-5 | 7.25e-4 | 0.00392 | 0.0283 | 0.535 |
| `||g_KD_softmax||`, weighted | 1.12e-5 | 6.36e-5 | 5.19e-4 | 0.00281 | 0.0203 | 0.383 |
| `||g_KD_sigmoid||`, raw | 9.49e-5 | 5.23e-4 | 0.00179 | 0.0116 | 1.647 | 5.285 |
| `||g_KD_sigmoid||`, same lambda | 6.79e-5 | 3.74e-4 | 0.00128 | 0.00831 | 1.179 | 3.785 |
| `||g_attack||`, weighted | 0.284 | 0.288 | 0.292 | 0.328 | 0.432 | 0.472 |
| Softmax/ret weighted norm | 8.24e-5 | 4.91e-4 | 0.00224 | 0.0744 | 0.803 | 1.204 |
| Sigmoid/ret at same lambda | 4.30e-4 | 0.00290 | 0.0153 | 0.413 | 4.848 | 112.54 |
| Softmax/attack weighted norm | 3.90e-5 | 2.19e-4 | 0.00171 | 0.00972 | 0.0460 | 1.307 |
| Sigmoid/attack at same lambda | 2.36e-4 | 0.00129 | 0.00445 | 0.0196 | 2.531 | 12.915 |

| Unit-normalized direction cosine | Q10 | Q25 | median | Q75 | Q90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Softmax KD vs attack | -0.0725 | -0.0285 | +0.0030 | +0.0105 | +0.0263 | +0.0490 |
| Sigmoid KD vs attack | -0.0662 | -0.00860 | +0.00611 | +0.0154 | +0.0375 | +0.0562 |
| Softmax KD vs retention | -0.00812 | -0.000756 | +0.00538 | +0.0635 | +0.1439 | +0.1764 |
| Sigmoid KD vs retention | -0.00519 | -0.000391 | +0.00376 | +0.0641 | +0.1366 | +0.1531 |
| Softmax KD vs Sigmoid KD | +0.692 | +0.835 | +0.960 | +0.996 | +0.999 | +0.99997 |

The present weighted Softmax KD is **weak on the typical final-state query**: median `||g_KD||/||g_attack|| = 0.00171`, and median `||g_KD||/||g_ret|| = 0.00224`. It is not uniformly weak: the maximum Softmax/attack ratio is **1.307**, so the zero-perturbation calibration's tail bound does not characterize every final-state query. This is a scale and state-dependence observation, not a result-based change to `lambda_KD`.

Sigmoid and Softmax directions are usually similar (median cosine **0.960**), although one query has an opposite direction (minimum **-0.876**, query `000308`). Each KD has negative cosine with attack on **6/16** queries, with near-zero median cosine; there is no systematic direction-level attack conflict across the fixed set. **Scale creates a concrete local risk:** at horse query `000133`, the same-lambda Sigmoid norm is **12.915×** the attack norm and its attack cosine is **-0.1221**. Its signed projection onto the attack direction is therefore **-1.578×** the attack norm, versus **-0.169×** for the current Softmax KD. Directly substituting Sigmoid KL with the unchanged lambda could dominate and oppose target suppression on that query. No new Sigmoid coefficient is inferred here.

## 3. Classwise function, assignment and localization

The largest three *decreases* in existing clean-VOC20 non-target AP50, KD-ObjBal minus Safe-Soft05 `last.pt`, are dog **-0.05298** (0.58005 → 0.52708), horse **-0.02547** (0.73372 → 0.70825), and tvmonitor **-0.01814** (0.72983 → 0.71169). Bicycle, requested separately, **improved +0.01483** (0.64068 → 0.65551). These AP values are from the already completed victim evaluation; the component measurements below are from the fixed clean *training* queries and one virtual surrogate update.

| Class | GT objects / Teacher positives | Softmax / Bernoulli KL, object mean | GT Sigmoid change | Top-1 agreement | assignment Jaccard | GT box IoU change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dog | 1 / 10 | 1.73e-5 / 3.80e-5 | -0.000543 | 1.000 | 1.000 | +0.000466 |
| horse | 6 / 60 | 0.00143 / 0.01375 | -0.001074 | 1.000 | 0.944 | +0.01745 |
| tvmonitor | 1 / 10 | 1.46e-6 / 5.74e-6 | +0.000756 | 1.000 | 1.000 | +0.000603 |
| bicycle | 3 / 30 | 1.98e-7 / 2.39e-6 | +0.0000936 | 1.000 | 1.000 | +0.000235 |

For the full fixed set, Teacher and Student non-target Top-1 classes agreed at **240/240** positive locations. The GT class was Top-1 at **221/240** for both; there were **0 new correct-to-wrong Top-1 flips**. The 19 shared errors were already present in the Teacher (nine horse→cow and ten chair→sofa). Thus the local update shows no newly introduced class confusion. Student retained assignment to the same GT at **238/240** Teacher-positive locations. Both lost locations came from horse object `000133:1`, where GT Sigmoid confidence fell by **0.13878** and **0.13988**, while aligned IoU increased by **0.06390** and **0.01727** respectively. This identifies a local confidence/assignment change, not a local box localization loss. The overall median aligned IoU change was **+0.000367** (Q10 **-0.000177**, Q90 **+0.00406**).

For dog and tvmonitor, the fixed queries contain only **one object each**, with stable Top-1, assignment and IoU. Their validation AP losses cannot be explained by those local measurements. Horse has the only marked sampled confidence decreases and two assignment changes, but six objects are insufficient to attribute its 200-epoch victim AP drop. Bicycle has no sampled damage and its validation AP increased. The classwise CSV covers all 19 non-target classes and records empty fixed-query classes explicitly.

**Component conclusion:** classification confidence change (A) is observed in two sampled horse objects; new Top-1 confusion (B) is not observed; assignment/foreground evidence change (C) is limited to one horse object; GT-aligned localization degradation (D) is not supported on the sampled positives. There is **insufficient evidence to identify the main cause of the independent victim's non-target AP decline** as classification, matching or localization. A direct, adequately covered component audit of the already trained victims would resolve that question before a new KD loss experiment.

## Answers and next decision

1. **Softmax absolute-confidence blind spot?** Yes, one predeclared horse location out of 240 meets the strict criterion; its GT Sigmoid drop is **0.06132** while Softmax KL is **3.35e-5**.
2. **Does Bernoulli KL capture it?** Yes at that location: **0.008151**. Across all locations the association is mixed: Bernoulli Spearman with absolute drift **0.762** versus Softmax **0.600**, but Pearson **0.894** versus **0.923**.
3. **Is current KD effective?** Its median weighted norm is only **0.171%** of attack and **0.224%** of retention, with a maximum **130.7%** of attack. It is highly uneven rather than consistently effective.
4. **Would Sigmoid KD conflict with attack?** Not systematically by unit direction (median cosine **+0.0061**, negative on **6/16** queries). At the unchanged coefficient, one query has a **-1.578× attack-norm** signed opposing projection, so scale is a serious local risk.
5. **Most damaged non-target classes?** Dog **-0.05298**, horse **-0.02547**, tvmonitor **-0.01814** AP50. Bicycle changed **+0.01483**.
6. **Main source of non-target detection loss?** Undetermined for the 200-epoch victim. The fixed surrogate queries show a local horse confidence/assignment event, no new Top-1 confusion, and no overall sampled IoU decline.
7. **Enough evidence for Sigmoid-KD experiment C?** No. One blind spot, sparse class coverage, largely aligned KD directions, and an uncalibrated same-lambda gradient tail do not establish likely benefit or safety.

The recommendation is **RECOMMEND_DETECTION_COMPONENT_AUDIT**. This is a diagnosis only; no loss, hyperparameter or formal experiment was changed. The full location-level vectors, query gradients and 19-class aggregate evidence are in the adjacent three CSV files. The measured correlations and classwise statements are descriptive for this one fixed train-query set and one seed; they are not causal claims about the validation AP change.
