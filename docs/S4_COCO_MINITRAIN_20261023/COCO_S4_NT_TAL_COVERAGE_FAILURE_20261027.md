# TFSS non-target TAL coverage: formal failure and exact diagnosis

Status: **BLOCKED_METHOD_DEFINITION_NT_TAL_NO_EVIDENCE**. No current formal job/controller is running. This record supersedes whole-run readiness claims, while preserving the previously passed nine repair validations as historical facts.

## What actually ran

- Last exact export-device callback passed: one CPU rejection and one CUDA0 acceptance, zero updates. Prior real learned-Teacher CPU16/CUDA16 validation passed with44 native forwards and one Adam update each. No repeated Teacher/calibration/data work.
- Shared GPU coexistence was explicitly authorized and resource admission passed. Another UID alone was no longer a blocker. No action was taken against another user's process.
- New source `0eb162097ab07ea641a9b5167c0fe3d70d38d349` was normally pushed and independently found by GitHub API. The new148-file immutable release passed deployed identity checks; all145 reference files were verified (131 byte-exact,14 explicitly bound routing/authorization deltas; model/optimization AST unchanged).
- Unique new workflow `s4_coco_experiment_a_seed0_20261027_tfss_repaired_v1` started its actual Fourier child at2026-10-26T16:26:28Z. It completed **12 optimizer updates**, then failed in the next batch. Native diagnostics, latest continuation, Adam/scheduler/RNG, source/release/config and failure/exit are retained on the server. No poison25K export or victim training occurred.
- Controller2106847/start365647905/UID1014 and child2108510/start365666366/UID1014 were independently observed absent after native exit1. No restart, reset, second controller or RunID reuse.

The failure is `NO_VALID_POSITIVE: GT index=13, class=24`, raised at `ue_framework/methods/tfss_nt_sensitivity.py:87` through canonical `S4Trainer -> CleanMapCache -> build_maps -> gradients`.

## Exact sample and native evidence

The retained completed-step checkpoint has step12/cursor192. The next original seed0 batch begins with official image `000000247692` (640x458). Its valid GT13/class24 has normalized bbox `[0.4906640625,0.26548034934497816,0.006921875,0.013973799126637555]`, approximately4.43x6.40 pixels after the same native640 letterbox. Label SHA256 `b8e0567da839ed04c0ebd9eea69346aa47a9bb5cc546909c2f1fa6c23ce63718`.

One read-only CPU Teacher forward through the unchanged estimator confirmed both all-GT and independent same-TAL rematching give zero positive anchors for this GT. A second read-only CPU forward inspected the exact native assignment arithmetic. Both diagnostics had zero parameter updates, unchanged Teacher and no CUDA initialization; they are diagnostics, not passing smoke tests or official results.

Pinned native TAL's small-GT candidate rule provides **5** candidate anchors (the unexpanded raw GT contains one stride8 center). Every candidate's **unclamped CIoU is negative**, approximately−0.01294 to−0.01983. Native TAL clamps these overlaps to0, so every alignment metric and the resulting native supervision weight is0. Float64 recomputation retains the negative values; this is not the previous float32 SVD/Gram incident. Native positive count=0 and native score sum=0.

At the final completed update, actual TFSS weights ranged0.86705–1.0 (mean0.97778), with finite Fourier gradient norm0.06655. Thus the repaired TFSS optimization path did execute; the failure concerns a newly encountered uncovered legitimate NTGT. It is not a CUDA OOM or a GPU-admission failure.

## Why this cannot be silently repaired as numerical precision

The frozen estimator requires original native TAL positive supervision and nonzero normalized native weights. This valid GT supplies neither, even when it competes alone. Increasing precision does not provide supervision. Truncating/removing the GT, ignoring class24, using a zero sensitivity map, lowering a guard, or changing the native loss/assignment silently would violate the existing method constraints.

The previous dynamic-dimension authorization concerns selected feature coordinates; it does not specify an `R_nt` extension for zero native supervision. The user's constraint to stop/report substantive mathematical changes still applies. Automatic bug repair and formal autostart remain authorized **after legitimate gates close**, but are not interpreted as permission to invent a missing risk definition.

## Preserved evidence

Private execution records: `COCO_S4_REPAIRED_V1_FAILURE_EVIDENCE_20261027.json`, `COCO_DIMENSION_RECOVERY_LATEST.json`, `COCO_S4_NT_GT_COVERAGE_DIAGNOSIS_20261027.json`, `COCO_S4_NATIVE_TAL_ZERO_EVIDENCE_20261027.json`. Model/checkpoint/data files remain on the owned server; none were pulled for publication. All prior invalid/stopped runs remain unchanged.

A proposed explicit definition extension is separated in `COCO_S4_NT_GT_UNCOVERED_POLICY_PROPOSAL_20261027.md`; it has not been implemented, validated, or approved. No formal victim AP or complete COCO experiment is claimed.
