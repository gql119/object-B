# Original S4 — COCO80 Dimension-Compatible Implementation

Protocol: coco80-strict-tsfl-selected-dimension-v1. COCO80 uses the retained39 TSFL channels; VOC20 uses24. Both follow the same Strict TSFL selection rule. P5 SafeBG selected coordinates and independent P3 Orth coordinates remain separate.

| Check | Actual measured result |
|---|---|
| Frozen VOC24 CPU float32/64 outputs/diagnostics/projector | Exact, difference0 |
| Frozen VOC24 CUDA weightless kernel | Exact, difference0; RNG restored |
| Related VOC regression |55 tests,0fail/error/skip; synthetic random native VOC fixture |
| Reconstructed COCO39/all79 bases |Complete; class ranks12..19; retained P348 bytes exact; K8 unchanged |
| Actual learned-Teacher CPU single-image EOT2 |12 native forwards/one Adam update; L_BG/rel coefficient norms .05165540054/.05180947110 |
| Actual learned-Teacher CUDA single-image EOT2 |12 native forwards/one Adam update; L_BG/rel coefficient norms .05308785290/.05325238407; CUDA0 |
| First formal batch16 |FAILED before any completed update/CSV row: Invalid Safe-BG projector |

The sole independent review of e891024 rejected F01: image derivatives were mislabeled coefficient derivatives. Its negative verdict remains unchanged. Source34fee63 measures actual weighted BG_sel/rel_sel derivatives directly to the native64x3 Fourier coefficient parameter. Main-agent closure uses current CPU/CUDA evidence, with no second review. The original image diagnostics and failed evidence remain retained. Per-EOT integer class-ID sidecars address inherited categorical averaging without altering ranking/losses.

New recovery dimcompat_v1 was launched once with source34fee63a232d489ae54d61eb37a42483091c40d4 and release8614b4476a851d57c21fab8a6a5ba92c283f1c5969a555912b232b62f35e7a2d. Accepted Teacher/eval/calibration were reused. It failed on its first formal batch, then both controller and child exited. It must never be restarted/reused. Oldv3 and all original evidence remain unchanged. No export/victim/final evaluation ran.

A unique bounded diagnostic reproduced the failure on actual CUDA0, formal batch16 and original eligible list, with zero optimizer updates. Ordered active classes38,24,26 have rank8, but q Gram error1.0371208190917969e-5 exceeds unchanged1e-5. Double evaluation of the same q remains1.0364261922468643e-5: this is a true numerical violation, not an assumed TF32 false positive (TF32 was false, precision highest).

Independent CPU float64 factorization of the exact retained ordered basis gives Gram3.552713678800501e-15, cast-FP32 Gram5.960464477539063e-8 and rank8. Its projector differs from the failing CUDA-float32 projector by Frobenius .0009004869365338809. This is diagnostic evidence only. Production solver, thresholds and protocol have NOT been changed to force a pass.

Formal continuation is STOPPED on the numerical projector check. Any numerical implementation correction needs explicit source/version provenance and current production-batch evidence while preserving the same ordered low-rank SVD, rank8, alpha.5, active3, norm restoration and VOC24 compatibility. No threshold relaxation, silent rank reduction or disabled SafeBG is allowed. Do not treat single-image probes as full-batch readiness.

Original8/4/0, eta.85, Q/R, Fourier/FrozenAdaptive/loss weights, fixed official25K/COCO80/full originalval5K remain unchanged. No final S4 AP/PSNR/LPIPS results exist; no efficacy or single-seed significance/causal claim is made. The same sole monitor remains active.
