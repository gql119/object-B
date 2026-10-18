# Actual GPU victim AMP smoke

Status: **v1 FAILED at the bounded native S0 prefix; native AMP skip diagnosis confirmed.** This is not a TFSS/S4 algorithm failure and is not yet a completed AMP recovery validation.

The unique smoke RunID `s4_victim_amp_real_smoke_seed0_20261018_v1` launched once on source `5cd8158e8312c1fcebb9fa26ac551acc3cd3a669` after the CPU-only dataset reuse audit and fresh resource preflight passed. RRCTL recorded exit code 1 at 2026-10-18T06:37:38.790191Z in the first `native_S0` trial (workload PID 1660639; executor PID 1660631).

The native trainer entered epoch 1 and reached the five-actual-batch boundary. All five rows in the retrieved `native_S0/amp_audit/AMP_SCALE_AND_SKIP_LOG.csv` record an optimizer attempt, `native_scaler_nonfinite_skip=true`, `backoff=true`, no SGD update, and finite raw loss/model/EMA/optimizer state. The AMP scale sequence was `65536 → 32768 → 16384 → 8192 → 4096 → 2048`; raw losses were finite on all five batches. The optimizer hooks were working: the observer captured each scaler skip and scale reduction. At batch five, the smoke's unconditional “must have observed at least one update” assertion terminated the trial.

This establishes native GradScaler overflow skips on these five batches, consistent with scaled gradients containing nonfinite values. The specific offending gradient tensors were not retained, and these five rows do not establish whether the reduced scale later recovers. The paired `repaired_S0` comparison, actual S4 prefix, synthetic overflow test, and formal victim remain **NOT_RUN / NOT_LAUNCHED**.

AMP CSV: `C:\Users\20272\.codex\visualizations\2026\09\17\01a0afba-bca4-7582-815c-b2a5241bad4a\s4amp_small_20261018\s4_victim_amp_real_smoke_seed0_20261018_v1\native_S0_AMP_SCALE_AND_SKIP_LOG.csv`; SHA-256 `f8cb06f598aa331e9f0f4cebeec2db216b3c3e0cc1cb8e0dc0627b59408ff79a` (4215 bytes; five rows). RRCTL transport read succeeded at 2026-10-18T07:58:27.295490Z. The earlier bounded diagnostic snapshot remains archived; no remote files were altered.

The CPU-only dataset audit remains **PASS**: 16,551 train and 4,952 clean validation images; labels and clean validation byte-identical; outside/global/overlap budgets 0/8/4; source, noise, integrity, and manifest identities exact. No S4 algorithm, training argument, dataset, or frozen configuration changed.

Harness correction planned for a fresh RunID: allow the native and repaired S0 *equivalence controls only* to finish the exact five-batch prefix with zero updates when every row records a real GradScaler skip and backoff. Exact paired input/loss/model/optimizer/scaler/scheduler/EMA equivalence remains required. The real-S4 prefix must still show a successful update within its fixed 16-batch bound, and the isolated synthetic test must still show skip/backoff followed by a successful update. The fixed 16-consecutive-skip safety ceiling, total 58-batch smoke budget, and all formal victim arguments remain unchanged. RunID v1 will not be reused.


## Fresh smoke v2 result

Smoke `s4_victim_amp_real_smoke_seed0_20261018_v2` completed through RRCTL with exit 0 on source `ab626917f20917e58cf30033ac99df00938cb561`. Native and repaired S0 paired controls had exact transformed-input hashes and maximum difference 0.0. The real S4 prefix completed 16 batches (10 updates, 6 native scaler skips). The isolated synthetic overflow trial completed 8 batches (6 skips, 2 updates); a forced overflow at batch 7 was followed by scaler recovery and another update. Loss, model, EMA, and optimizer states were finite. Checkpoints were retained, official metrics and artifact ingestion were disabled. This passes the bounded smoke only; it is not an efficacy result. The separately authorized victim recovery is running under `s4_tfss_utility_seed0_20261017_v1_victim_recovery01`.
