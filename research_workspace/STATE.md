# S4-TFSS-Full current research — 2026-10-21T15:03:17.017794+00:00

Change: independent ablation S4_TFSS_FULL_OVERLAP, only overlap support 0.5→1 with eta_U=.85. Original S4/robustness sources and outputs are immutable. User basis docs/S4_TFSS_FULL_20261021/USER_TASK_SOURCE.md SHA 084055907d7f4bcc391053120ef832aa3aa6df0cbe13729fe6779c64b70242cc.

Result: CPU components/regressions passed; see CPU_VALIDATION_REPORT.json. No GPU smoke, scientific review, full data, victim or AP result exists for this candidate. Cache test mocks only local disk-capacity return; it is not remote admission.

Finding (SUPPORTED only within CPU component protocol): no-overlap Fourier/JND/TFSS output and gradients match; overlap pre-clamp delta doubles with same W; PNG cap is8/outside0. Formal efficacy remains OPEN.

Next: existing recovery robustness pipeline must reach final COMPLETED_AND_ACCEPTED and subsequent queue be verified clear. Last live query failed RRCTL invalid JSON; do not infer training failure or idle admission. Then current-commit GPU smoke → exactly one independent scientific review → formal24/9144/new dataset/nativeAMPvictim200/last evaluation. No second monitor, no AT, no original S4 rerun.

Historical prior-task state follows.

# S4 鲁棒性研究 — 2026-10-19

当前唯一主方法 S4-TFSS-Utility，PhaseA/B全正式训练和评价已获人类授权；AT只做审计。独立分支codex/s4-robustness-evaluation-20261019。

读取 execution/S4_ROBUSTNESS_CURRENT_STATE.json 及 docs/S4_ROBUSTNESS_20261019/S4_ROBUSTNESS_PROTOCOL.md。远端只读已核验S4原始noise/dataset manifest/原正式last权重与完整200epoch。S0/S3/S5 poison不使用，不重生成噪声。四历史Clean有200epoch和实际lastSHA，已绑定待重评扩展指标，不重训。

本地3contracts和3AST通过，45原S0文件hash未变；当前GPU smoke/独立review/所有鲁棒性正式训练均NOT_RUN。不得从文档骨架或历史S0 Poison填写S4结果。
