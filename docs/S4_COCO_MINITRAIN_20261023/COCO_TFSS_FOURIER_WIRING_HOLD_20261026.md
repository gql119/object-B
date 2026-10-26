# Original S4 COCO Fourier: scientific fidelity hold

The current dimcompat_v2 process is numerically running, but it is not eligible for scientific acceptance as complete Original S4. Its immutable source dcd91b33d90dfd2472b8eacb843a994a4801d1cf instantiates EligibleCOCOB2Trainer(B2SemanticTrainer), without S4Trainer/SoftSpatialProjection, TFSS-U mode, or CleanMapCache. The actual bound source SHA was read back from the server. No TFSS training diagnostics or clean-map cache exist in the current output.

The canonical Original S4 entrypoint scripts/run_s4_tfss_pipeline.py instantiates S4Trainer, sets TFSS-U and CleanMapCache, and optimizes the post-composition weighted image with the original adaptive loss. Adding W only during later export does not reproduce that optimization. A synthetic CPU counterexample executed the unchanged SoftSpatialProjection: W=.85 changes the pre-clamp delta by .0047058817 and the example coefficient derivative from 9 to 7.650000572. This is a mathematical counterexample with zero model forwards, zero dataset images, zero optimizer updates, and no initialized GPU; it is not a full learned-Teacher smoke.

## Disposition

- Suspend scientific acceptance and all downstream READY registration for this runtime. Preserve any later native accepted.json without treating it as independent scientific acceptance.
- export25k_READY.json was confirmed absent. No export, Victim, or official S4 metric was started. No process signal, restart, source hotpatch, or second controller was issued. The immutable current process may continue to its existing boundary; it has not been physically stopped.
- Retain the original single reviewer verdict and prior closures as historical evidence. They did not establish TFSS source-to-sink fidelity. Do not request a second or closing review for this same Fourier repair.
- Reuse the already accepted Teacher, Teacher evaluation, raw full25K calibration and 39D reconstruction. No data transfer, resampling, full calibration or Teacher retraining.

## Authorized repair dependency

Restore the actual canonical S4Trainer + detached clean-only W in every optimization EOT draw, including the adaptive consumer of the scaled image, the original spatial statistics and RNG handling. Keep original losses, weights, Fourier64 candidate/16 saved bases, selected39 P5/independent48 P3, SafeBG8/.5/top3 and 8/4/0/eta.85 fixed. A new unique recovery must bind this incident, exact source/config/data/checkpoints and the current predecessor's terminal identity. Run current learned-Teacher CPU then bounded actual GPU production validation after no active GPU conflict, with measured W/map source hashes, actual scaled-image sinks and direct BG/rel-to-Fourier gradients. Main closes this new blocker using evidence. No hotpatch/reuse of dimcompat_v2.

Also fix the future export artifact-schema check: native final params have 16x3 saved coefficients and no num_classes field. Trainable coefficients have 64x3 candidate shape; these are intentionally different original protocols, not a dimension bug. COCO80 must stay verified through bound Teacher/trainer/data identities. Add actual pre-reduction tensor CUDA evidence in the future export smoke instead of a configured-device assertion.

The earlier statement that numerical liveness established healthy complete Original S4 execution was too strong. This report corrects that attribution; current COCO S4 effectiveness remains unmeasured.
