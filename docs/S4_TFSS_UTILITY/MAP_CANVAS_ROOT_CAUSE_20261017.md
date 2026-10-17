# S4 TFSS-Utility map-canvas root-cause audit

Status: implementation correction in progress; no formal S4 run has launched.

## Failure evidence

The retained `s4_tfss_utility_canvas_probe_seed0_20261017_v4` compared the same 32 fixed TRAIN images (65 Person GT, 46 non-target GT), image bytes, parsed annotations, reference model, selected TSFL channels, support masks, and original-image output coordinates. Model/Frozen/Adaptive/Fourier hashes were unchanged, valid-domain mismatches were 0, and optimizer updates were 0. The gate failed: Q max difference 1.0, R difference 0, and Utility-weight max difference 0.1499999761581421. Evidence: `execution/pull_v4/s4_tfss_utility_canvas_probe_seed0_20261017_v4/map_canvas_probe.json`; manifest SHA `3becd3c5c9338dfb887da6163df8063a8ed44303dbacc4351f15e5ba376be5a1`.

All 32 training calls used a 512x512 batch canvas. Their singleton export canvases varied (352x512, 384x512, 416x320, 512x288, 512x352, 512x384), so every image was evaluated on a different map canvas. Across the 32 *per-image maxima* (not a pixel-pooled distribution), Q difference mean/P95/P99/max was 0.9236138090 / 1.0 / 1.0 / 1.0; R was 0 / 0 / 0 / 0; Utility-weight difference was 0.0937031191 / 0.1497577429 / 0.1499999762 / 0.1499999762. The old probe did not retain full pixelwise difference arrays, so these summaries must not be represented as pixel-level quantiles. The corrected audit will report both pixel-pooled and per-image statistics and an aggregate difference heatmap.

## Root cause

This is a batch-dependent padding-context implementation mismatch, not different source images, GTs, TSFL channel selection, an independent `R_nt` formula error, or a `W_U` formula error. In training, `S4Trainer._batch_loss` passes the native batch to `CleanMapCache.get`. The cache selected the largest H/W in that batch, rounded each to 32, and supplied that canvas to `build_maps`. The loader shuffles examples, so an image could receive different companion-dependent canvases across epochs. Dataset materialization calls the same cache with a singleton image, which selected that image's own smaller canvas.

`Q_t` is not invariant to this difference. `build_maps` pads the clean image with 0.5, computes the selected `model.21` feature and input gradient, then pads the GT mask to the same canvas and projects it through `_feature_mask` to the feature grid. Thus the canvas changes both the feature-grid extent/context and the GT-mask resampling grid. The existing 3x3 masked smoothing and per-object Q10/Q90 normalization are identical in both paths, but they receive different gradients and feature masks. The coordinate output remains the original HxW image frame; the mismatch is the model-input padding and its feature-grid projection, not a changed annotation or inverse-coordinate transform.

`R_nt` is computed by `tfss_nt_sensitivity.gradients` on the original image with a fixed 640 letterbox, independent of this cache canvas; its observed difference is exactly zero. `W_U = 1 - 0.15 * R_nt * (1 - Q_t)` is unchanged. With R held fixed, Q differences propagate through the existing 0.15 coefficient; this explains the observed maximum near 0.15. No clipping, constant-map substitution, skipped GT, changed eta, or relaxed assertion is involved.

## Authorized correction

Use one image-local canonical canvas for every clean map: bottom/right 0.5 padding from that image's native HxW to `(ceil32(H), ceil32(W))`. `build_maps` now chooses this canvas per image even when passed a batch; `CleanMapCache` keys each entry by this same image-local canvas. Training's batched caller and export's singleton caller therefore run the same `build_maps` interface with the same image, GT, reference, selected channels, feature-mask projection, and padding. This is not output-value forcing: each path recomputes the map under the same specified input tensor and the Q/R/W outputs must then pass measured equivalence.

This correction changes the padded tensor context used for images whose former training batch canvas exceeded their singleton canvas. Consequently the affected Q/W values and prior TRAIN32 short-probe conditions are not carried forward as new-version evidence. The map version/canvas entry in `S4_CONFIG_LOCK.json` records the corrected per-image rule. The Q10/Q90 normalization, 3x3 smoothing, `W_U` formula, `eta_U=0.85`, all losses and weights, Fourier/Adaptive/Frozen behavior, TSFL channels, support/budget and victim protocol remain unchanged. No spatial parameter search is allowed.

## Verification still required

- Recompute all 32 maps through separate training-batch and export-singleton caches; compare Q, R, W and valid domains on identical inputs. Retain pixel-pooled mean/P95/P99/max and per-image statistics, plus difference heatmaps.
- Verify the original S0/no-op loss, Fourier-gradient, mask, delta and update gates; finite outputs; Person/outside/overlap/global pixel budgets; and exact checkpoint/resume gates.
- Rerun the affected fixed TRAIN32 map audit and the registered 5-step short validation, including Fourier and Adaptive gradient checks and the map-to-export perturbation correspondence.
- Preserve the old failed probe and all earlier short-run evidence. Do not start the formal 24-epoch run until these checks and the current resource gate pass.

At report creation, the source correction was not yet committed or GPU-validated; all listed future checks remain `NOT_RUN` until their artifacts exist.
