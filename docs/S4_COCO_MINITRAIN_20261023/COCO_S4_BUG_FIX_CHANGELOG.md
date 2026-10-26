# COCO S4 bug fix changelog — candidate, real gates OPEN

BEFORE_FIX archive commit:40f5890. Baseline433; stopped invalid runtime dcd91b3. Original sources are preserved; fixes only in COCO adapter/lifecycle/report consumers.

- S4-001/002: reuse original S4Trainer/SoftSpatialProjection/CleanMapCache/optimize_s4, preserve full native completed-update state. Real learnedTeacher CPU/CUDA gates remain OPEN.
- S4-003: validate original saved16x3 schema; COCO80 from actual bound Teacher/config, optional metadata may not conflict. No invented num_classes required.
- S4-004: actual nested raw export tensor device/index/finite proof.
- S4-005: compare canonical model-state digests; retain file SHA separately.
- S4-006: CPU-only adapter RNG context; all original8 source files byte-identical. Initialized CUDA path calls original helpers unchanged.
- S4-007: validation-only source release with exact emergency human task SHA; formal prepare/launch/registration explicitly blocked. Invalid native acceptance cannot override this scope.
- S4-008: trace TFSS poison through unchanged sharedEOT/nativeletterbox into actual Frozen forward; preserve actual Adaptive pre-EOT weighted input. Passive observers do not alter tensors or RNG.

No mathematical change:8/4/0,eta.85,Q/R formulas,39/P3 independent48,all79/rank8/alpha.5/top3,losses/weights/Fourier/EOT2/Adam/schedule unchanged. No formal READY/controller/job. No Teacher/calibration rerun.

Failures retained: first contract compile observer used decoded text on an existing BOM file; changed only compiler reader to original bytes. First55regression had1source-hash failure due to attempted core RNG guard; core bytes restored and isolation moved to adapter. Nothing was marked passed by suppressing a test.
