# Actual Victim AMP smoke

Status: **NOT_RUN — RRCTL SSH connection timeout**. No smoke binding or output has been created. Planned unique RunID is `s4_victim_amp_real_smoke_seed0_20261018_v1`.

Frozen design, max58 actual batches across isolated fresh processes:

1. Native DetectionTrainer: first5 actual S0 batches, seed0/random YOLOv8n/full200epoch schedule/batch36/workers16/AMPtrue.
2. Repaired Trainer: the same5 S0 batches and initialization. Require exact transformed input tensor hashes and loss/model/optimizer/scaler/scheduler/EMA state hashes; tolerance0.
3. REAL_S4_VICTIM_BATCH_TEST: first16 actual S4 batches with the same formal arguments. Require finite raw loss/model/EMA/optimizer and actual nonduplicate SGD updates. Native scaled overflow/backoff events are logged.
4. SYNTHETIC_OVERFLOW_TEST: isolated actual-S4/native-trainer process, max32 batches. After one real successful update, multiply one loss by1e20 solely in this trial; require native skip/backoff, unchanged trainable parameters on that skip, finite states and a subsequent normal actual update. Stop once recovered and at least5 batches have completed; otherwise fail at the bound.

No official validation metrics are computed. All trial states are retained. The bounded callback exits after optimizer/EMA handling; each trial runs in its own subprocess so worker/prefetch/RNG state cannot carry into formal training. The formal Victim starts from seed0 YAML initialization in a different output, not from a smoke checkpoint.

The CPU unit/scalar tests are not actual CUDA evidence. Any failed input/state equality, persistent native skip, raw nonfinite loss/model, malformed data or resource gate blocks the formal recovery.
