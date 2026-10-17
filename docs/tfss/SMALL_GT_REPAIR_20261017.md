# Valid small GT / canonical P5 undercoverage — main-agent repair

Change: BUG_FIX / NUMERICAL_STABILITY routing under the existing invalid-Qt policy. The short run source2abe completed all six batch0 five-step trajectories, then exited1 in batch1 map construction before any batch1 update. Evidence/checkpoints retained. No scientific outcome is inferred from the stopped cohort.

Result: CPU geometry of all32/65PersonGT identifies exactly two valid boxes in2009_000455, pixel areas380/230 and canonical P5 feature-mask sums.5859375/.328125. Batch0 has zero such boxes. Original S0 `_pooled` uses min_pixels1; the failing Qt guard incorrectly merged valid pixel GT with invalid labels.

Finding: route canonical-pool-undercovered valid GT through the already declared undefined-Qt convention: full spatial strength on its retained pixel domain, valid:false, raw_norm:null, reason:canonical_P5_pool_undercoverage; exclude the domain from high-Qt statistics. No invented positive/sensitivity, GT drop, pixel zero-gradient claim, modified mask, loss, label, optimizer, eta, budget, precision, or NT assignment. Invalid/empty pixel GT and invalid/nonfinite feature masks still fail. This is not proof of small-object efficacy.

Next: current-source batch1 one-update smoke across six methods, verify both undercovered GT records and W=1/exclusion/real baseline equivalence/RMS/model identities, then main-agent reproducible closure. The ONE independent review at2abe remains the original verdict; do not invent an independent pass on repaired source or call another reviewer. Recover only last16 fixed images through --batch-index1 and fresh output. Original first16 complete trajectories are reused transparently with their source identity; this is a split-source recovery, never claim all60updates ran under one commit. Regular Qt branch is unchanged and CPU geometry proves no undercovered GT in batch0. All previous failed runs and completedbatch0 checkpoints remain untouched.

Local10CPUcontracts/AST/45originalSHA pass. Runtime smoke and closure are pending until observed.
