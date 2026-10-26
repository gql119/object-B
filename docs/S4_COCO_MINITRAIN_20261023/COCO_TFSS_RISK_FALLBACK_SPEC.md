# COCO80 NT-risk boundary extension — isolated candidate specification

Protocol: `coco80-ntgt-native-tal-then-gt-risk-v1`.
Classification: **Level C / Methodological Extension**, not equality to strict TAL in its undefined cases. Validation authorized; full formal Fourier optimization is **not** authorized for this candidate.

## Scope and trigger

The unchanged original estimator is called directly first. Successful covered-image calls return its exact output and records, with no extra model forward. VOC20 and any non-COCO80/Person0 route always delegate to the original. An exception other than the two explicit missing-native-supervision errors propagates unchanged.

Only a valid non-target GT for which native all-GT TAL and existing independent same-TAL matching both supply no positive nonzero supervision uses the independent estimator. Zero-weight native assignments are also independently rematched before a fallback is considered. Invalid labels, nonfinite outputs and zero/nonfinite resulting gradients still fail; no epsilon creates supervision.

## Candidate detection locations

Let native pixel anchor centers be `a_i`, the actual GT box be `b_g=(x1,y1,x2,y2)`, center `c_g` and positive size `(w_g,h_g)`.

Use the candidate mask from the pinned native `select_candidates_in_gts` geometric rule, including its existing small-GT geometry handling. These locations are **independent risk candidates, not native TAL positives**. If that mask is empty, choose exactly one native location minimizing

`d_g(i)=((a_ix-c_gx)/w_g)^2+((a_iy-c_gy)/h_g)^2`.

Tie break by native anchor index. Compute this ranking in float64, with no added epsilon and no modified GT. Degenerate/nonfinite transformed boxes fail. The candidate rule uses true position and size, no tuned radius/TopK. Both selection cases are explicitly recorded.

## GT-conditioned detection-risk loss

For an uncovered GT with candidate set `A_g`, use uniform **risk-estimation** weights `1/|A_g|` and

`ell_fb(g) = mean_{i in A_g} [lambda_box*(1-CIoU(pred_box_i,b_g)) + lambda_cls*sum_{c!=Person} BCEWithLogits(z_ic,1[c=c_g]) + lambda_dfl*DFL(pred_distribution_i,bbox2dist(a_i,b_g))]`.

Use the same fixed native box/class/DFL gains, the same decoded predictions and the same native DFL target clamp as the original estimator. The classification target is the actual GT non-target class (hard one-hot), **not** the zero soft target returned by a failed TAL assignment. CIoU in the loss is the original unclamped loss expression. This therefore has ordinary localized detection-supervision meaning; it is not a loss chosen merely to create nonzero derivatives. It remains a proxy risk for an unrecognized GT, not evidence that Teacher detected it.

Covered GT terms and native weights remain unchanged, even in an image containing an uncovered GT. Each GT enters one loss exactly once. Compute the original mean of GT losses within each present NT class, then the original clean-input gradient norm, original per-class map normalization and original equal mean across present NT classes. The branch never creates a second class vote. Original `Q_t`, `W_U=1-.15*R_nt*(1-Q_t)` and `eta=.85` remain unchanged.

No-NT inputs keep the original empty estimator/zero-NT-risk semantics; this differs from assigning zero risk to a real uncovered GT, which is forbidden. No-Person images retain the original zero perturbation outside target support. Their GT coverage statistics and TFSS-active statistics must be reported separately.

## Isolation and cache identity

`risk_fallback.py` contains the independent estimator; original `tfss_nt_sensitivity.py` is unchanged. The ordinary original cache and original map builder also remain unchanged. A separate `RiskCleanMapCache` executes their original code objects with instance-local copies of globals that explicitly substitute only the NT estimator dependency. No original module global is patched. Thus `Q_t`, normalization, support and cache persistence arithmetic reuse the same original code, while provenance records this dependency change.

Cache identity adds policy version plus estimator, adapter and specification SHA. Old maps cannot masquerade as new-policy maps. Both canonical training and the export adapter select the same explicit cache/estimator policy. The original data/Teacher/calibration/TSFL/Safe-BG identities remain required.

## Required evidence and reporting

Normal images: actual original/candidate output and records exactness, covered-GT loss/gradient exactness, VOC20 regression. Uncovered images: actual Teacher finite nonzero detection-risk gradients, no fake native-positive count, complete GT/class accounting, overlaps/co-occurrence, noPerson/noNT, finite normalized risks/W, real Fourier coefficients and native export8/4/0 link. CPU evidence does not substitute for actual CUDA.

Full-data trigger counts/classes/proportions and TFSS-active image proportions must be actually measured. For uncovered inputs the original complete risk is **undefined/null**, not zero; no artificial old/new difference may be computed. Report exact equality on defined inputs and the new values/distributions on previously undefined inputs. All quantities remain pending until measured.

No AP tuning, new generator, skipped GT, altered labels/budget, extra seed, formal24epoch run or old12step resume is permitted. All long-term A/B repairs and this C extension must remain separately labeled.
