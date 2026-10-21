# First smoke preflight failure and scoped correction
The unique v1 run at source34d8b105 exited before teacher construction or any GPU forward/Adam update. Its resource admission passed all three windows. baseline_identity incorrectly compared the SHA4413e45a… of the retained final training checkpoint against noise/global_params.pt.
Read-only server verification confirms:
- noise/checkpoints/step_009144_epoch_24_cursor_00000.pt SHA4413e45aa9c448bb768cac50e0ae40b5b0a1a580e38ff37cf66023bdab07c6e5.
- noise/global_params.pt SHAd731e2579ff58bfaa6bce67a2b4439ca9b6c182a3966413f8bb8c41d6049c2c2.
- original materialization_manifest and dataset_integrity hashes match the frozen S4 baseline.
The Full lock now names the final training checkpoint path explicitly and separately binds the export parameters SHA; both are checked. No original artifact was changed, no expected hash was replaced to accept changed data, and neither file initializes Full Fourier parameters.
The same correction records both real no-overlap loss scalars/weight paths and verifies the actually decoded PNG bytes so the single upcoming scientific review can inspect numerical evidence. Core Full computation, original45/S4 files, TFSS eta, losses, optimizers, schedules and victim protocol are unchanged. v1 artifacts/binding are retained; next current-source smoke must use a new v2 RunID. No reviewer has been called and no formal run has started.
