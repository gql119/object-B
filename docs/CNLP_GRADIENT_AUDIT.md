# CNLP current training-development gradient evidence — 2026-10-15

All rows below use original Fourier variable; 38cases/33unique query images/67NTGT case-observations; no official noise/victim or validation AP. J1 chosen by preregistered smallest passing J.

| raw gradient | Q10 | Q25 | median | Q75 | Q90 | max |
|---|---:|---:|---:|---:|---:|---:|
|parent|0.050998813|0.055819266|0.066669945|0.085027495|0.10291185|0.14413996|
|cnlp|0|0|0.0010390676|0.0076571926|0.019560492|0.25517824|
|attack|0.04733084|0.054199266|0.065176703|0.083274456|0.10071242|0.14406896|
|ret|8.9054858e-05|0.00020775486|0.00038321229|0.00068471766|0.0011510447|0.0014295644|

Weighted CNLP/Parent norm median0.0066853987351454085,max0.25 on calibration only. Not a global clipping rule. Original unit attack=BG_sel+rel_sel proxy only; GPU production probe additionally measures full BG+rel+lambda_adapt*L_adapt_target.

Actual canonical Parent lambda0 comparison: five optimizer steps, loss/poison/gradient/coefficient max difference0, tolerance1e-6; reference source is actual original canonical directory, not candidate relabeled. Original weights8de8a0... same. S2/S3 real five-step plus3to5 resume checkpoint maxdiff0 and CSV byte-identical; decoded PNGs obey8/4/0 labels unchanged.

CUP real sink:12of20 batches have weighted-vs-uniform Fourier gradient difference; full attack cosine<-.5:0of15 valid batches, cosine range[-.2528443933,.5954961181]; max observed lambdaCNLP*gCNLP/gParent norm0.01999184489 in this CUP probe. Not a universal future bound.

CNLP coverage-v2:12fixed historical train-edge query images/51NTGT observations,9native-uncovered GT recovered, remainingzero0; no GT dropped. 38-case native-covered replay scalar/norm/cos maxdifference0; CPU full gradient difference0; do not claim complete GPU vector equality.

Source4d0c635 GPU calculations allpassed. rrctl completion artifact contract had stale unit filenames; separate artifact-only recovery copied12JSON byte-identically, no GPU rerun. Raw receipts in BASE/cnlp_small_20261015/cnlp_admission_artifacts_seed0_20261015_v1. Final source adds a separate matched Person proxy probe; final review pending.


## Complete formal execution evidence — 2026-10-17
S2 and S3 both completed all registered9144 outer updates/24epochs with retained checkpoints, originalFrozen/Adaptive unchanged. S3 final Adam/CNLP/CUP counters each9144, full RNG/query/source identity preserved; finite recorded key losses/gradients and outside0. Fullvictim200 primary evaluation exists. Runtime reachability and finite gradients do not prove causal protection or efficacy; S3 did not improve the S0 target/overallNT pair. See S2_S3_FINAL_REPORT.md. No new scientific reviewer or parameter change.
