# Original S4 AT direct stop / CUDA optimization / restart report

State: OLD_AT_TERMINATED. Updated 2026-10-30T05:23:57.889233+00:00.

Old RunID s4_at_serial_seed0_20261029_v2 was explicitly abandoned by human at epoch87, 86 completed epochs / 39666 batches / 20265 actual SGD / 12 legal AMP skips. It is aborted, not accepted. Existing UID1014 full process group was reaped through scoped repaired RRCTL abort; no own GPU PID remains; next arms absent. Epoch86 checkpoint SHA 925dcaf995a6880d987e218b1e28a72c6008a93bbcf0cb26e10001167ea8e3ee and logs preserved.

Termination evidence: execution/direct_stop_20261030/20261030T052322Z_OLD_AT_TERMINATED.json. Single targeted control review and main repair closure: execution/direct_stop_20261030/TARGETED_CONTROL_REVIEW_20261030.json and TARGETED_CONTROL_MAIN_REPAIR_CLOSURE.json. This is not the subsequent candidate scientific review. Initial fingerprint and CPU01 setup failures remain retained.

Candidate CUDA equality, throughput, batch selection, new scientific review, new formal Release/RunSpec/controller are NOT COMPLETE yet. Actual values remain null until measured. The current authorized fallback is a fresh four-arm OriginalBatch36 protocol if no validated reliable speed gain. All4 seed0 randomYAML200epochs/PGD10/fullGT/640/fullVOC4952 remain fixed. Old checkpoint continuation is forbidden by latest scope.

## Actual probe receipts
Updated 2026-10-30T05:35:14.675092+00:00. These are bounded test components; formal four-arm training has not started.

| Probe | Case | Actual state | Real evidence |
|---|---|---|---|
| s4_at_eq02_clean4_20261030_direct01 | {'kind': 'eq', 'implementation': '02', 'arm': 'clean', 'epsilon': 4, 'budget': 32} | PASS_NATIVE_FIXED_INPUT_EQUALITY_COMPONENT | execution/direct_stop_20261030/probes/eq02c4/summary.json |
