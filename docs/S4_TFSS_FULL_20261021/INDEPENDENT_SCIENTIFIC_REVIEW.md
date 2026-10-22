# PRERUN-REVIEW-S4-TFSS-Full

Reviewer: `/root/s4_full_scientific_review`, one independent read-only review with `fork_turns=none`; parent conversation not supplied. Reviewed snapshot `1d10ad620ef01aad20e8c85d49465473d72c9a4e`, diff base `4dd1cc370b0c93c888279ff949b1a5ebb7e15e9b`. Full original reviewer response remains in this conversation's reviewer completion message; this file records its findings and verdict.

Result: `scientifically_correct`. Decision: `allow_run`. `pre_run_result:pass`. Blockers: none. No blocking validation gaps. This concerns implementation correctness and does not predict efficacy or replace scheduling/admission.

| Dimension | Verdict | Finding |
|---|---|---|
| Intent alignment | correct | Binary Person union support; overlap .5→1; TFSS remains 1−.15 Rnt(1−Qt). |
| Canonical code location | correct | Production instantiates S4FullTrainer and S4FullGenerator; original S4 unchanged. |
| Critical data flow | correct | Frozen lock reaches support, composition, inherited loss, Fourier Adam, export, victim and clean validation. |
| Computation sink effect | correct | Full support multiplies composed delta; detached unchanged TFSS weights scale it; Adaptive consumes resulting poison. |
| Runtime scientific state | correct | Current GPU smoke reaches real losses/backward/two Adam updates; Frozen/Adaptive base states unchanged. |
| Baseline/disabled path | correct | Actual canonical/candidate no-overlap loss/delta/Fourier-gradient comparison passes tolerance0. |
| Experiment identity | correct | Source/config/teacher/prototypes/dataset/cache/noise/baseline/last attribution bound. |

Reviewer findings:

1. Original composition order is preserved: _TAUSBCommon Fourier/JND/EOT, original2.01/255 STE floor and boundary signs, followed by unchanged TFSS. ROI/context use nonzero geometric domain, so removing amplitude attenuation does not independently alter masks.
2. Original half-weighted area threshold is evaluated separately, preserving tiny-overlap sample admission while Full binary support reaches computation. The optimizer AST test identifies this as its only computational change.
3. Reviewer independently checked all45 canonical hashes (zero mismatches) and all8 implementation/test hashes in CPU evidence. Adam groups/EOT/frequency/cosine/loss/SafeBG/TSFL remain inherited/copied unchanged. Fourier initializes fresh, never from historical final noise.
4. Export validates new noise source/config/support, applies TFSS before clamp, checks decoded8/0, exact labels/clean validation and all Person coverage. Independent cache includes Full identity. GT overlap diagnostics do not depend on support==.5. Historical training-state and export-parameter hashes bind separate files correctly.
5. Noise24/9144/batch16/EOT2; victim arguments match completed historical S4, random YAML seed0/nativeAMP/SGD/200. Evaluation requires contiguous200 records, last primary/best secondary and matched clean validator.
6. Report uses verified S4/Clean metrics, fullVOC20 arrays, signed deltas and existing selectivity/collateral formulas. It avoids attributing AP solely to overlap or claiming single-seed significance.

Validation inspected: 28 CPU tests,0 failures/errors/skips, actual supplied log/source hashes; reviewer did not rerun write-producing tests. Current GPU smoke `s4_tfss_full_overlap_gpu_smoke_seed0_20261021_v2` exit0/two actual Adam updates/four finite EOT losses; metrics/ingest disabled and all artifacts retained. No-overlap image000035 loss0.7707809805870056 on both actual implementations; loss/delta/Fourier-gradient errors0 at tolerance0. Decoded overlap PNG000009 max2 bytes/outside0; CPU projection separately exercises8 bytes.

Full export/victim200/formal metrics remain unrun at review. Historical contemporaneous S0 runtime Git SHA unknown, truthfully represented; canonical bytes and measured S4 comparison are bound. No second/closing review requested.
