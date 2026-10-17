# S1 target-regression review — 2026-10-15

## Change / scope

Read-only review of completed S0/S1 source, frozen-budget summaries, existing noise diagnostics and archived independent-victim evaluations. No GPU forwards, new training, cache recomputation, parameter changes or mutation of either historical experiment or running S2. Existing server CSV/JSON were reduced on CPU; only small summaries were retrieved. This is main-agent analysis, not another independent review.

## Result — measured

Matched last.pt primary evaluations after random-init YOLOv8n 200 epochs:

| Metric | S0 | S1 | S1 minus S0 |
|---|---:|---:|---:|
| Person AP50 (%) | 2.8993419333 | 9.1670471011 | +6.2677051678 pp |
| NT mAP50 (%) | 69.1698089762 | 70.4576101876 | +1.2878012114 pp |
| NT mAP75 (%) | 51.5898126283 | 53.5902291633 | +2.0004165350 pp |
| S | 0.9334250734 | 0.9066492870 | -0.0267757865 |

S1 best.pt Person AP50 10.2131% is secondary; it must not be substituted for S1 last.pt in this comparison. The rise is 6.27 percentage points, not approximately ten percentage points. Both noise schedules completed 24 epochs / 9,144 optimizer steps. S1 full materialization/labels/clean-val/pixel-budget audits passed; all 13,256 Person GT boxes changed. Archived checks confirm matched victim protocol and unchanged Frozen/Adaptive reference models. Single seed; no statistical significance claim.

Full 6,095-image S1 map summary (byte units, i.e. numerator over 255): image-mean nominal budget 6.6954; mean change from old nominal map -0.09847. High-NT-sensitivity region mean budget 2.2504; low-sensitivity region 7.8159. Overlap image-mean budget 4.9997 versus old 4; non-overlap 7.0592 versus old 8. Different subgroup sample counts and image weighting apply; these are not one pooled pixel average. S1 and S0 feasible sets are not pointwise nested.

Matched epoch-24 means from existing full noise CSVs:

| Diagnostic | S0 | S1 | Relative change |
|---|---:|---:|---:|
| L_total | 1.0472627052 | 1.0477343326 | +0.0450% |
| L_BG | 0.4178099896 | 0.4176965799 | -0.0271% |
| L_rel | 0.5606753282 | 0.5660431739 | +0.9574% |
| L_ret | 0.0769288060 | 0.0579312880 | -24.6949% |
| mean_abs_delta | 0.0057881504 | 0.0057576124 | -0.5276% |
| Fourier gradient norm | 0.0541803245 | 0.0555564813 | +2.5400% |

mean_abs_delta is the trainer's padded full-image/channel mean of pre-export delta, not target-only RGB amplitude or decoded PNG strength. Identical metric definitions/order permit a bounded diagnostic comparison; this does not establish equal final perturbation strength. Existing S0 PNG perceptual audit samples 256 images with RGB/pixel-weighted statistics; S1 exhaustive audit reports a different per-image maximum-channel statistic. They must not be compared as the same amplitude measurement. No new image decoding was performed.

## Finding — MIXED

The observed outcome is a preservation/attack trade-off: NT performance improves while target suppression weakens. The budget constraint executed, but this does not demonstrate that the estimator isolated harmless pixels or preserved target-attack capacity. Similar total loss and mean delta cannot establish similar independent-victim AP. Lower L_ret is consistent with improved preservation; slightly higher L_rel is consistent with weaker relative target misdirection, but these surrogate losses do not causally explain the final AP difference.

### Code-established mismatch

1. The frozen budget is driven by magnitude of clean-teacher non-target detection input gradients, classwise normalized by Person-domain Q10/Q90, equally aggregated and smoothed. It contains no target-attack sensitivity score or constraint preserving the target-suppression objective. A low NT sensitivity is not evidence of a high target attack utility. Teacher-local sensitivity is not independent-victim learnability after 200 epochs. Magnitudes omit direction/sign, and per-class normalization removes absolute scale.
2. This is not merely a tighter cap everywhere. Old S0 multiplies synthesized amplitude by 1/.5 in exclusive/overlap regions. S1 undoes that final multiplier and applies a spatial hard clip. Overlaps can become less constrained while exclusive high-sensitivity Person regions become substantially more constrained. A small overall mean difference can hide removal of a useful spatial pattern.
3. In the CSB hard-clip branch, strictly saturated pixels have zero derivative with respect to the synthesized base. S0's final overlap multiplier instead has derivative .5 with respect to that base; upstream clamps still apply in both methods. Shared Fourier coefficients couple pixels/images, so this is a change to optimization geometry as well as budgets. Raw regularizers and unsaturated pixels still carry gradients. The measured total Fourier gradient norm is not lower; a claim of global gradient disappearance would be false.
4. Legacy-map equivalence was explicitly tested and achieved; the cache used by adaptive losses is replaced by the projected poison. Existing evidence does not identify an obvious label, outside-support, teacher-mutation or incomplete-schedule error. It does not prove absence of every possible defect.

### OPEN explanations, not demonstrated causes

- High NT sensitivity inside Person boxes may coincide with target-critical features/context, causing S1 to suppress disproportionately useful attack locations. No target-utility/sensitivity overlap measurement exists.
- Spatial clipping may remove effective components of the universal Fourier pattern and block their learning paths. The actual CSB-bound clipping fraction and lost attack-gradient share were not logged. The inherited saturation_ratio tests global 8/255, not each CSB budget; it cannot resolve this question.
- A frozen clean-teacher map may become a poor proxy as perturbations and the learner change. No controlled intervention isolates this effect.
- Quantized export floors fractional budgets; this is an additional representational difference, not an established dominant explanation.

No complete equal-strength-control victim exists. Therefore current evidence cannot separate spatial allocation, final strength, projection Jacobian and estimator mismatch causally. It does not establish convergence or that longer S1 training would recover suppression. S1's greater wall-clock duration is not more optimization steps.

## Next — proposals only

Keep the already-selected S0 Parent and immutable S1/S2/S3 protocol. Before redesign, obtain matched-definition existing-image amplitude breakdowns and, in a separately authorized diagnostic, measure CSB-bound clipping / passed attack-gradient fractions and overlap with target-useful locations. A later controlled study should change projection versus budget allocation separately and use matched final-strength controls; do not silently introduce a straight-through estimator, tune eta/gamma using validation AP, or rerun S1. Any such change is a disclosed design change requiring its own train-only correctness gates and authorization.

## Evidence

- `docs/s1_archive/raw/comparison.json`, `docs/s1_archive/raw/voc20_comparison.csv`
- `docs/s1_archive/verification_20261015.json`, `docs/s1_archive/s0_history_verification_20261015.json`
- `docs/s1_archive/raw/dataset_integrity.json`, `docs/s1_archive/raw/source_audit.json`
- `research_workspace/analysis/S1_S0_EXISTING_DIAGNOSTICS_AUDIT_20261015.json` (CPU aggregation of existing server CSV/JSON; records original paths)
- `ue_framework/methods/s1_csb.py:SpatialProjection`, `ue_framework/methods/csb_budget.py:construct_budget/png_project`
- `ue_framework/methods/legacy_kproto_ret.py:_compose_delta_batched`, `ue_framework/methods/b2_semantic.py:SemanticFrozenTrainer._batch_loss`
