# Safe-Soft05-KD-ObjBal preregistration

This candidate changes only the added non-target classification KD reduction from
positive-unit mean to mean-within-teacher-assigned-GT-object followed by mean over
objects with at least one positive. A zero-positive non-target GT is counted and
excluded from the denominator. Teacher TAL assignment, shared Student indices,
person GT/logit exclusion, classification-only KL and `tau=2` remain fixed.

The parent Safe-Soft05 computation, including `L_ret` and its weight `0.25`,
P5 selected 24 channels, TSFL Top40%/Q60, Safe-BG rank 8/alpha 0.5/active k 3,
orthogonality weight 0.0005, adaptive step, perturbation support and budgets,
is unchanged. The new KD has an independent coefficient `lambda_KD`.

Calibration uses exactly the existing seed-0 train-only fixed set: first 64
sorted mixed target/non-target train images, four disjoint batches of 16, query
positions 0/4/8/12 in each batch, zero perturbation optimizer updates, the
unchanged parent prototype cache, and the same frozen Teacher and adaptive base.
Measure 16 same-delta gradients with unit KD weight. Let `r_ret` be unit KD
gradient norm divided by weighted `L_ret` gradient norm, and `r_attack` be unit
KD gradient norm divided by weighted BG+rel+adaptive gradient norm. Set once:

```
lambda_from_ret = 0.25 / (median(r_ret) + 1e-12)
lambda_from_q90 = 0.10 / (Q90(r_attack) + 1e-12)
lambda_from_max = 0.50 / (max(r_attack) + 1e-12)
lambda_KD = min(lambda_from_ret, lambda_from_q90, lambda_from_max)
```

The resulting weighted 16-query gate requires median KD/ret at least 0.10,
Q90 KD/attack at most 0.10, maximum KD/attack at most 0.50, Q90 combined
utility/attack below 1, maximum below 10, finite values, and unchanged model
and perturbation state. The old and new KD/ret Q90-to-median ratios are
reported; tail reduction has no separate hard cutoff. A failed gate blocks
all formal work without adjusting this rule or resampling queries.

Only after that gate passes, bind the calibration artifact SHA into a new config
and run an isolated five-step production-entrypoint smoke. Audit all five
losses/gradients, Teacher and model freezing, selected channels, Safe-BG,
person exclusion, finite outputs, and decoded 8/4/0 budget. Then perform one
independent scientific implementation review. Only a passing gate and an
`allow_run` verdict admit the single fresh formal seed-0 experiment: 24 epochs,
9144 outer steps, complete 16,551-image materialization and data audit,
random-init YOLOv8n for 200 epochs, clean VOC20 evaluation with `last.pt`
primary and `best.pt` secondary. There is no result-based retuning or next KD
run.

No formal or validation-set result was available when this document was written.
