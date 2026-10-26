# TFSS optimization restoration: component candidate only

This candidate restores the COCO Fourier wrapper to the unchanged canonical `S4Trainer` and `ue_framework.methods.s4_optimization.optimize_s4`. It configures the original clean-only TFSS cache, eta .85 and locked map definition. Passive observers check that both the Frozen loss and Adaptive inner learner consume the actual weighted image, and that delta is the original pre-clamp delta multiplied by the detached W. The production path retains the original optimizer, EOT, frequency selection and checkpoint/RNG implementation.

The wrapper packages original optimizer artifacts in a retained native_s4 subdirectory. A bounded one-update probe creates an explicitly bounded parameter snapshot; this is never presented as a completed 24-epoch result. Final sixteen saved bases and sixty-four trainable candidates keep their original meanings. Canonical losses, formulas, 8/4/0, eta, selected P5 coordinates and independent P3 coordinates are unchanged. The original `s4_tfss.py`, `tfss_spatial.py` and `s4_optimization.py` have not been edited.

The admission proof now requires actual TFSS optimization evidence. An old B2-only successful smoke is rejected. The future exporter also rejects a Fourier receipt without this proof, so the current immutable v2 output cannot pass as complete Original S4.

## Completed local component checks

`python -B -m coco_s4.validate_canonical_s4_training_cpu --output execution/tfss_wiring_component_20261026_v3` passed. It compiled six affected implementation modules, rejected five invalid TFSS proofs and the old B2 class, and exercised the unchanged canonical optimizer adapter with a declared scalar-loss toy trainer: one Adam update, two EOT loss draws, exact original full-list seed0 first-sixteen ordering, retained optimizer/RNG checkpoint and 64x3 candidate / 16x3 output tensors. No learned Teacher model or actual dataset image was loaded; zero model forwards and no GPU initialized. This is component evidence only, not the full native CPU gate or CUDA readiness.

The first component invocation retained its completed toy update and then hit a Windows hard-link permission error while packaging artifacts. Packaging now uses exclusive retained byte copies; v2 passed and v3 added the adversarial proof cases. All three output roots remain intact. No experiment was restarted or accepted by these probes.

## Open gates and routing

The deployed dcd91b3 v2 process remains unchanged. No new remote code, bounded model probe, controller, READY receipt or formal training was launched. Full learned-Teacher CPU16 and actual bounded CUDA16 evidence are still OPEN, including actual clean map identity, TFSS-weighted Frozen/Adaptive inputs, per-loss Fourier gradients, exact rank8/79-class/P3 guards and the original 8/4/0 audit. Main must close the existing single-review findings plus TFSS01; no second or closing Fourier reviewer.

The unique recovery route has not yet been implemented or bound. Existing phase_contract still names the inspect-only v2; do not prepare/launch/reuse its SPEC. A subsequent unique recovery must bind the retained v2 method-fidelity hold and actual predecessor terminal/death evidence, reuse accepted Teacher/evaluation/calibration, and preserve all earlier output roots. GPU validation must wait for the active own GPU job to finish and fresh no-conflict/capacity checks. The future exporter metadata-field issue and its direct raw CUDA proof remain separate OPEN work.
