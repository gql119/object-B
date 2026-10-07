# Safe-Soft05-KDcls pre-run gate — blocked

Candidate `bb6f1f6cf4b527efed0539ebc7eeddfd5a4584d1`, isolated five-step RunID
`smoke5_candidate_bb6f1f6_seed0`. The production noise entrypoint completed five
outer optimizer steps with finite diagnostics. Official metrics and result ingest
were disabled. This is **not** a formal poison or victim result.

The latest user clarification overrides the attachment's proposed replacement of
`L_ret`: the inherited `L_ret` computation, layer, reduction, input and coefficient
remain unchanged. KD is a separate term on the existing Adaptive pre/post virtual
update clean query. The teacher's non-target-only TAL assignment supplies the same
prediction locations for teacher and student; person GT and person logits are
excluded. Only classification KL is used (`tau=2`, `beta=1`). The KD-only scale is
fixed from a hashed train-only, no-update probe. No validation AP was used.

The no-update probe passed: teacher/adaptive base remained frozen, 10 non-target
positive units were aligned, target GT/logit exclusion checks passed, losses were
finite, and the noise parameters did not change. The initial FP32 KL calculation
showed cancellation error near zero; the final candidate computes the selected
19-logit KL in FP64. The calibration artifact SHA256 is
`9f2990b3ce5678289572ec1f0959c32233101eb678672dbcfadfb8fff6ee2b1d`.
It records `L_ret=7.3163428169e-5`, `L_KD=2.3360243517e-8`, and fixed
KD-only scale `3131.96337890625`. All 14 copied coordinate artifact files were
verified byte-identical to formal Safe-Soft05.

| Diagnostic | First step | Last step | Five-step observation |
| --- | ---: | ---: | --- |
| `L_ret_raw` | 0.00007119 | 0.00157047 | Original computation unchanged |
| `L_KD_raw` | 0.0000000833 | 0.0000091667 | Non-target positives only |
| KD scaled, before `lambda_ret` | 0.000261 | 0.028710 | Fixed scale above |
| `L_util_diagnostic` | 0.000332 | 0.030280 | Informational sum; does not replace `L_ret` |
| `grad_ret_norm` | 0.002247 | 0.010219 | Weighted original `L_ret` gradient |
| `grad_KD_norm` | 0.006743 | 0.528789 | Weighted new KD gradient |
| `grad_util_norm` | 0.007968 | 0.529289 | Sum of the previous two gradients |
| `grad_attack_norm` | 0.306737 | 0.304518 | `BG + rel + adaptive` |
| Utility/attack gradient ratio | 0.0260 | 1.7426 | **Maximum 13.9756 at step 2** |
| cos(KD, ret) | -0.000142 | 0.034197 | No gradient surgery |
| cos(KD, attack) | 0.008549 | -0.026498 | Diagnostic only |
| Non-target top-1 agreement | 0.5 | 1.0 | EOT means |
| KD positive units | 5 | 10 | EOT means; range 5–25 |

`L_ret_norm` is intentionally absent: normalizing it would violate the latest
instruction. `L_BG`, `L_rel`, `L_orth`, `L_adapt`, `L_ret`, and KD were finite.
Person GT and person logits did not enter KD; teacher/student positive indices
were shared. Frozen teacher and adaptive base hashes matched before/after smoke.
An eight-image preview pixel audit passed: all target pixels changed, maximum
non-overlap change 3/255, overlap 2/255, outside target zero. This preview is
not a substitute for full dataset and label audits.

**Gate verdict: do not run.** The predeclared threshold requires the utility to
attack gradient ratio to remain at most 10 during sanity; step 2 reached 13.98.
Do not change beta, tau, `lambda_ret`, the parent losses, or proceed to formal
poison generation or victim training on this candidate. The likely issue is that
the single selected clean-query calibration item had an exceptionally small KD
value, so its fixed loss-scale factor magnifies gradients on later items. That is
an inference from the recorded numbers, not a demonstrated causal diagnosis.
Revising the calibration protocol requires a separate decision and a new candidate
and smoke RunID; no final-validation metric should guide it.
