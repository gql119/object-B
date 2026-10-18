# AMP overflow recovery evidence

CPU contract tests: PASS (7), including synthetic native scaler skip/backoff sequencing.

The first actual CUDA smoke RunID `s4_victim_amp_real_smoke_seed0_20261018_v1` failed within its first `native_S0` five-batch control. The per-batch RRCTL-retrieved AMP CSV contains five observed attempts, all skipped by native GradScaler with scale backoff, zero SGD updates, and finite raw losses/model/EMA/optimizer states. Scale fell `65536 → 2048`. This confirms real scaler skip/backoff behavior on the first five S0 batches and confirms the observer logged it; it does not show later recovery because the harness stopped at batch five.

`REAL_S4_VICTIM_BATCH_TEST`: NOT_RUN. `SYNTHETIC_OVERFLOW_TEST`: NOT_RUN. Formal 200-epoch victim recovery: NOT_LAUNCHED. Keep these outcomes separate from the CPU contract and the five-batch S0 overflow observation. The next bounded smoke may accept zero updates only in the paired S0 equivalence controls when every row proves native skip/backoff; exact equivalence remains mandatory, while real-S4 and synthetic trials still require successful updates/recovery. No victim parameters or AMP configuration may change.

AMP CSV: `C:\Users\20272\.codex\visualizations\2026\09\17\01a0afba-bca4-7582-815c-b2a5241bad4a\s4amp_small_20261018\s4_victim_amp_real_smoke_seed0_20261018_v1\native_S0_AMP_SCALE_AND_SKIP_LOG.csv`; SHA-256 `f8cb06f598aa331e9f0f4cebeec2db216b3c3e0cc1cb8e0dc0627b59408ff79a`.


## Real CUDA smoke v2

The current-source smoke preserved native GradScaler behavior. The actual S4 prefix produced 6 skips and 10 successful updates over 16 batches. The isolated synthetic overflow forced a skip at batch 7 and subsequently recovered to a finite update. All bounded smoke checks passed. The original S0 smoke v1 five-skip assertion remains archived as a harness-boundary failure.
