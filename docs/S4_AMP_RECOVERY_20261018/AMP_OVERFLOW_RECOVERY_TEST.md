# Overflow recovery evidence

CPU contracts: PASS7, including native skip/backoff, next finite update, accumulation accounting, duplicate-update absence and fixed16skip stop. Prior installed-server CPU scalar evidence is retained separately in S4_AMP_GUARD_CPU_CONTRACT_20261018.json.

SYNTHETIC_OVERFLOW_TEST (CUDA actual detector): NOT_RUN, RRCTL connection timeout. REAL_S4_VICTIM_BATCH_TEST: NOT_RUN. No claim that the original CUDA failure was benign or that real-batch recovery succeeded follows from CPU tests. See REAL_GPU_AMP_SMOKE.md for the bound and unchanged formal arguments.

The sixteen-consecutive-skip guard was fixed before any new GPU execution. Its basis is the full default65536→1 backoff ladder; persistent failure across that ladder requires stopping and a retained diagnostic snapshot. It does not alter native skip handling, scale initialization or backoff factors.
