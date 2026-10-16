# TFSS source snapshot — 2026-10-16

Authorized by the TFSS-Conservative/Utility task attachment bf6f4d1e-90c9-43c6-851b-b3b19d552791. This isolated branch archives source; it does not declare S3 complete or approve formal TFSS.

S3 actually runs committed source `017591686efb3675ee8d55debf1e2cd7da4e0e17` in its separate remote RunID checkout. Its live working directory equals the bound checkout, HEAD matches, and `git status --short` is empty. Local S2/S3 source bytes equal that commit; current disk differs in documentation and untracked receipts/results. Updated CNLP/CUP reports and locks are preserved here, with S3 explicitly INCOMPLETE. Diagnostic source is separately identified as `5bcae7233e03ebc6d2533f0ae311f67a68676899` and never described as the S3 computation.

Original main project Git metadata currently reports `fatal: cannot change to F:/autodl-tmp - 副本`; no repair or Git mutation was attempted there. Actual experiment worktree root, branch, HEAD and remote are verified independently. This snapshot was created using a separate Git worktree from the S3 source commit, without checkout/reset/rebase in any active or historical experiment.

`TFSS_SOURCE_SNAPSHOT_FILES.json` enumerates bytes/SHA for archived content and excluded local runtime/result files. `TFSS_S3_SNAPSHOT_STATUS.json` records the live state and exact source/config binding. No data, weights, poisoned datasets, caches, private profiles, tokens or large logs are added. No main change, force push, history deletion or merge is authorized. Snapshot content retains historical entrypoints for reproducibility; legacy launch_one reset/cleanup paths must not be used by TFSS.

S0 is the required parent for future TFSS. This snapshot ancestry is NOT parent selection. The separate development branch must restore exact historical S0 source and prove equivalence before any TFSS scientific run.
