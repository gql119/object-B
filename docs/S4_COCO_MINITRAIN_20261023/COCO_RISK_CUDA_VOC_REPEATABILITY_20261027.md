# VOC CUDA comparison failure: measured harness diagnosis

Classification: Level A validation-harness repair. The independent COCO risk
estimator remains a Level C methodological-extension candidate, validation only.
The original failed run `coco_dim_20261027_risk_gpu01` remains FAILED, exit 1.

Diagnostic `coco_dim_20261027_voceq_diag01`, published source
`1124cf234868b57c1c22123c49352707f996929e`, completed with exit 0. It performed
10 native forwards total: two model-initialization CPU forwards and eight CUDA
reference/candidate forwards. It made zero parameter updates and computed no
official metrics. Its status explicitly says diagnostic-only and
candidate-validation-pass=false. Health, wait and the two-file pull completed.

On the exact VOC fixture from the failed validator, the original function compared
with itself under the default backend produced identical scalar losses and records,
but nonidentical input gradients and maps: maximum absolute difference
1.7763568394002505e-15. Reference versus candidate had map difference
2.1094237467877974e-15 and gradient difference 1.7763568394002505e-15.
Candidate versus itself also differed. Thus the zero-tolerance comparison domain
was not repeatable even without a candidate change. This does not localize the
effect to a specific CUDA operator, and does not prove global numerical equivalence.

In a separate subprocess with cuDNN deterministic=true, benchmark=false,
torch deterministic algorithms=true and CUBLAS_WORKSPACE_CONFIG=:4096:8, all three
pairs (reference/reference, reference/candidate, candidate/candidate) had exact
records, maps and full input gradients, maximum differences zero. The actual raw
outputs were finite on cuda:0, index 0; the model was unchanged. No tolerance was
relaxed. TF32 settings were recorded and were not changed by this diagnosis.

The validator now uses that common comparison domain for the two equivalence
checks, adds original-reference repeat controls and VOC full-gradient checks,
restores the production backend before testing the new COCO branch, and saves
partial stage evidence before later assertions. This changes only the test harness.
The risk estimator, cache adapter and original estimator bytes remain exactly
those of `0cd3730edef3fab46601eb3950bafa0227e502e0`.

The updated component still requires a new uniquely bound actual CUDA validation.
This diagnostic is not that pass. Full Fourier/TFSS/export validation, dataset-wide
trigger statistics and the one scientific review remain OPEN. Formal training
remains disabled.
