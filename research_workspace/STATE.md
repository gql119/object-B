# S4 鲁棒性研究 — 2026-10-19

当前唯一主方法 S4-TFSS-Utility，PhaseA/B全正式训练和评价已获人类授权；AT只做审计。独立分支codex/s4-robustness-evaluation-20261019。

读取 execution/S4_ROBUSTNESS_CURRENT_STATE.json 及 docs/S4_ROBUSTNESS_20261019/S4_ROBUSTNESS_PROTOCOL.md。远端只读已核验S4原始noise/dataset manifest/原正式last权重与完整200epoch。S0/S3/S5 poison不使用，不重生成噪声。四历史Clean有200epoch和实际lastSHA，已绑定待重评扩展指标，不重训。

本地3contracts和3AST通过，45原S0文件hash未变；当前GPU smoke/独立review/所有鲁棒性正式训练均NOT_RUN。不得从文档骨架或历史S0 Poison填写S4结果。
