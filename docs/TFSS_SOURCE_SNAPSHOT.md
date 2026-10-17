# 最新归档状态 — 2026-10-17

原a5aefbb快照为当时S3仍运行的历史快照。S3现在完整验收/归档，结果已发布object-B/codex/safe-soft05-s2-s3-counterfactual-20261014@583892c2f1f2a86770a3828d9512c66cd75f25bb。TFSS训练源码：原短测batch0@2abe6544ab0e2701b94c961d6e06c59a66ee87e3；小GT修复后的batch1@e9623c78a27314c5f6f733abea90602d9415fb4a。两批部分组合不是一个同commit的完整实验。源码e962已推送并复读；本报告为后继小结果归档，不改历史计算源码。main无变更，未推送数据/权重/秘密。

---

## 历史记录（以前的 NOT_RUN / 等待段落已被上文取代）

# TFSS source snapshot — 2026-10-16

Authorized by the TFSS-Conservative/Utility task attachment bf6f4d1e-90c9-43c6-851b-b3b19d552791. This isolated branch archives source; it does not declare S3 complete or approve formal TFSS.

S3 actually runs committed source `017591686efb3675ee8d55debf1e2cd7da4e0e17` in its separate remote RunID checkout. Its live working directory equals the bound checkout, HEAD matches, and `git status --short` is empty. Local S2/S3 source bytes equal that commit; current disk differs in documentation and untracked receipts/results. Updated CNLP/CUP reports and locks are preserved here, with S3 explicitly INCOMPLETE. Diagnostic source is separately identified as `5bcae7233e03ebc6d2533f0ae311f67a68676899` and never described as the S3 computation.

Original main project Git metadata currently reports `fatal: cannot change to F:/autodl-tmp - 副本`; no repair or Git mutation was attempted there. Actual experiment worktree root, branch, HEAD and remote are verified independently. This snapshot was created using a separate Git worktree from the S3 source commit, without checkout/reset/rebase in any active or historical experiment.

`TFSS_SOURCE_SNAPSHOT_FILES.json` enumerates bytes/SHA for archived content and excluded local runtime/result files. `TFSS_S3_SNAPSHOT_STATUS.json` records the live state and exact source/config binding. No data, weights, poisoned datasets, caches, private profiles, tokens or large logs are added. No main change, force push, history deletion or merge is authorized. Snapshot content retains historical entrypoints for reproducibility; legacy launch_one reset/cleanup paths must not be used by TFSS.

S0 is the required parent for future TFSS. This snapshot ancestry is NOT parent selection. The separate development branch must restore exact historical S0 source and prove equivalence before any TFSS scientific run.
