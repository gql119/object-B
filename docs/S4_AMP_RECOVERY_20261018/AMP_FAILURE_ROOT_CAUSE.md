# S4 Victim failure diagnosis — 2026-10-18 HKT

## Change

Diagnostic work only. The failed remote checkout, TFSS implementation, frozen configuration, exported dataset and checkpoints were not modified. No GPU replay, new training run, environment change or recovery was performed. Existing logs and installed implementation sources were read through RRCTL. A one-thread CPU scalar GradScaler contract test used no S4 model, image or checkpoint.

## Result

- Original RunID: `s4_tfss_utility_seed0_20261017_v1`; RRCTL terminal state `failed`, exit code 1. Pipeline failure time: 2026-10-18 09:08:15.696 HKT; control terminal record: 09:08:20.684 HKT.
- Noise status reports complete: 24 epochs, 9,144 updates, 33,074.985439 seconds. Final checkpoint SHA256: `4413e45aa9c448bb768cac50e0ae40b5b0a1a580e38ff37cf66023bdab07c6e5`. This diagnosis did not field-load that final checkpoint or infer convergence.
- Full dataset audit passed: 16,551 train, 4,952 validation, 6,095 target images and 13,256 changed Person GT. Labels, clean validation and non-target images are byte-identical; outside-target zero and original pixel budgets pass. Audit SHA256: `3eece1fc62df963f7d4aa15d378395273bd3963f812c0f7764ea9115dc82fe82`.
- Victim failed at the first logged batch (`0/460`), before the first native optimizer update. Completed epochs: 0. `results.csv`, `last.pt`, `best.pt` and victim state snapshot directory are absent. Final evaluation and efficacy are NOT_RUN.

## Finding: direct failure and confirmed implementation defect

The immediate exception is `FloatingPointError('Nonfinite victim gradient; stop before update')` in `ue_framework/s4_victim.py:28`. The preceding `_finite()` checks completed without raising: the checked loss/rolling-loss and model/EMA state did not trigger their nonfinite guards. The guard then examined `p.grad` before calling the native optimizer implementation.

The server's installed native trainer computes `self.scaler.scale(self.loss).backward()` and calls the overridden optimizer method. At that point the gradients are still AMP-scaled. Native `optimizer_step` starts with `scaler.unscale_(optimizer)`, then native clipping, `scaler.step(optimizer)`, `scaler.update()` and gradient clearing. PyTorch's scaler records Inf/NaN during unscale; when present it skips `optimizer.step()` and reduces its scale. The S4 wrapper raises before any of this logic runs, making every such overflow a terminal failure. This is a confirmed incompatibility between the safety wrapper and the frozen native AMP protocol.

Simply moving the assertion after `unscale_` would not fix the incompatibility: an already-infinite gradient does not become finite by division. The native skip-and-scale-backoff path must remain reachable.

Read-only source evidence:

- Remote S4 guard SHA256: `f93fa02d69df81b840dc06b1cc3702b3ee82551e6f1100ebc4ed592b424dc666`.
- Installed native trainer SHA256: `2eda410e0bfb8b65cb46eb975c465c5b62b0f3a3dad5a54967d6e9e1fcabecc7`; native `optimizer_step` line 874, scaled backward line 544.
- Installed PyTorch: `2.7.1+cu128`; actual victim arguments include `amp: true`, batch 36, workers 16, seed 0.

## CPU scalar contract test

This is a synthetic mechanism test, not a replay of the failed CUDA batch.

| CPU case | Original loss finite | S4 guard would raise | Native optimizer updates | Scale before → after | Model after |
|---|---|---|---|---|---|
| Finite gradient | yes | no | 1 | 65,536 → 65,536 | finite |
| Scaled overflow | yes | yes | 0 | 65,536 → 32,768 | finite and unchanged |

The second case establishes that the current guard can reject an overflow the installed native scaler safely handles. It does not establish that the real S4 batch would recover on the next batch or under a smaller native scale.

## Evidence limits and other observations

The actual failed batch IDs, gradients, offending parameter names, runtime loss-scale value and RNG snapshot were not retained. Therefore the original gradient's Inf-versus-NaN type and its underlying source cannot be conclusively recovered from these logs. AMP scaling overflow is the leading explanation for this first-step event; persistent unscaled numerical failure is not excluded. No evidence currently establishes a TFSS efficacy failure.

The console records that Ultralytics tried to fetch `yolo26n.pt` solely for its AMP self-check, could not do so offline, skipped that self-check, and kept AMP enabled. No environment or AMP setting was changed during this diagnosis. This is a validation gap, not proof that the device or dataset caused the failure.

The later pin-memory-thread `ConnectionResetError` follows the primary exception in the log; it is a shutdown-side error, not the first failure.

The pipeline's embedded `resource` object is its launch preflight snapshot, not a failure-time measurement. The separate post-failure resource observation cannot retrospectively assign resource conditions at the failure instant. No OOM exception is recorded; CPU saturation is not the recorded stopping condition.

The prior production smoke validated noise/map/PNG/resume behavior. Its `smoke` branch returns before `train_evaluate`, so it never exercised `S4VictimTrainer.optimizer_step` with real AMP victim training. That coverage gap allowed this wrapper defect to reach the formal victim stage.

## Next

The appropriate repair is to retain the exact native AMP optimizer/scaler sequence, preserve checks on actual loss/model/EMA state, and record overflow/skip/scale transitions instead of terminating merely on pre-unscale AMP gradients. This is a proposed implementation repair; it has not been applied or validated on the actual S4 victim path. Disabling AMP, changing batch/workers/optimizer, sanitizing gradients, lowering learning rate or regenerating the dataset is not justified by the present evidence.

Before any recovery, a bounded actual victim-path verification must establish native update/skip behavior, parameter/EMA finiteness and unchanged training protocol. A recovery must preserve the failed original output and reuse the completed, audited noise/dataset rather than regenerating them. No recovery run was launched by this diagnostic task.

Primary evidence: `execution/S4_VICTIM_FAILURE_DIAGNOSIS_20261018.json` and `execution/S4_AMP_GUARD_CPU_CONTRACT_20261018.json`. These contain the retained small failure log, exact installed function bodies, audit summaries and CPU scalar results.
