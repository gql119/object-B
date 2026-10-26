# PROPOSED: explicit NTGT risk definition when native TAL has no supervision

**PROPOSAL ONLY — not implemented or an experimental result.** This requires an explicit method-scope decision because the existing native-TAL risk estimator is undefined for the reproduced valid GT.

## Option A: preserve the current definition

Keep the formal run blocked when a valid NTGT has no positive/nonzero native supervision after the existing independent same-TAL rematch. Retain all images, labels, Teacher, calibration and failures. No valid completion can be claimed for the fixed25K experiment under the currently incomplete definition.

## Option B: authorize a narrow, explicit fallback estimator

Preserve the existing native-TAL path byte/numerically unchanged whenever it supplies positive nonzero supervision. Only for an explicitly recorded valid NTGT for which independent native TAL still supplies none:

1. Keep the actual GT class and box, fixed Teacher, original prediction tensors, coordinate transform and pinned native geometric candidate rule. Do not drop a GT/class or enlarge its annotation.
2. If native geometric candidates exist, define their supervision weights explicitly as uniform over those candidates, and evaluate the existing per-GT box/CIoU, classification BCE and DFL terms using the original loss gains. This replaces undefined native assignment weights **only in this branch**; it is a new risk-estimator definition, not a precision fix.
3. Obtain the actual clean-input gradient, aggregate it with the original class/GT reduction and original normalization. Reject nonfinite/zero gradients rather than fabricating a risk map.
4. Record every fallback occurrence, GT identity, candidate set, native zero-evidence reason and gradient evidence. If no native geometric candidate exists or another required quantity is undefined, stop explicitly; do not cascade into additional undeclared rules.

This proposal changes the definition of the NT risk estimator in previously undefined cases. It cannot be called strict native-TAL supervision for those cases, or justified merely by keeping the8/4/0 budget. It does not change Safe-BG, selected39/VOC24, rank8, P3, target support, Q_t, eta, Fourier, optimization loss weights, datasets or victim protocol.

## Required validation before any formal recovery

- Version the new estimator policy and cache identity; never reuse incompatible clean-map caches or relabel the failed run as passed.
- Compare original and candidate on covered VOC24 and COCO inputs: the original branch must remain exact. No rerun of accepted old experiments or Teacher/calibration is needed to test this branch identity.
- Use the reproduced image/GT to verify real finite/nonzero per-GT and class risk gradients with all labels retained; then test their actual TFSS/Fourier consumption and8/4/0 export chain on bounded CPU/CUDA production inputs.
- Cover the0-positive,0-weight,covered-GT and no-geometric-candidate branches with explicit accept/reject contracts. No official metrics or AP tuning in probes.
- Retain the negative old review and all failure evidence; complete the appropriate scientific gate for the explicitly approved definition. Only then publish and create another unique formal recovery after dead-identity/capacity checks using the same controller mechanism and sole monitor.

No claim is made that this fallback improves protection or non-target utility. That remains an empirical question for independent victim evaluation. A different authorized policy would need its own equally explicit definition and validation.
