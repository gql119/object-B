# S4 结果拉取与比较汇总修复

## 根因

Victim 训练、200 轮 `results.csv` 和 last/best 两个 checkpoint 评价均已完成。原流水线随后在比较汇总阶段访问 `Recall_all` 时抛出 `KeyError('Recall_all')`。历史 S0/S1/S2/S3 指标归档保留了逐类 Recall/Precision 及 target/non-target 汇总，但没有 aggregate `Recall_all` / `Precision_all`。因此原任务终态为 failed，失败发生在报告汇总，而不是 victim 训练或验证。

RRCTL 的标准 `Controller.pull()` 需要终态 `artifact_manifest.json`。原流水线在异常后未写出比较文件，也未到达完成清单生成阶段，所以标准整包拉取没有 manifest 可读。这是失败路径上的工件清单缺口；RRCTL transport 本身仍可读取远端元数据和小文件。没有重启实验，也没有修改远端状态。

## 修复与验证

- `scripts/run_s4_tfss_pipeline.py` 现在只计算候选与历史 schema 共有的有限数值汇总指标；缺失的历史 aggregate 被显式记录为 unavailable，不再中止汇总。
- `scripts/recover_s4_failed_result_artifacts.py` 提供针对该唯一失败 RunID 的只读恢复路径：经 RRCTL transport 拉取固定清单中的小型 JSON/CSV 和控制状态，逐文件核对远端大小与 SHA256，再在本地重建报告。它不启动或改变远端任务。
- 使用该路径拉回 43,710 字节小型结果及控制记录。未拉取模型权重、图像/标签、约 65 MB AMP 明细 CSV 或大控制台日志。
- `tests/test_s4_metric_delta_schema.py` 覆盖旧历史 schema；pytest 1 项通过，相关 Python 文件 `py_compile` 通过。

## 核验结果

恢复记录确认 200 行连续 epoch 结果、last/best checkpoint 角色和 SHA 一致，完整 clean-val 评价已经保存。主结果必须以 `last.pt@200` 为准；best 仅为次要结果。完整汇总、逐类 20 类指标、相对 S0/S1/S2/S3 的差值和原始恢复记录随本目录报告 CSV/JSON 保存。

本次只修复汇总字段兼容与失败后的小文件取回路径。原远程 RunID 的 RRCTL 终态仍是 failed；本地重建报告不会把它伪装成原任务的 accepted/complete。比较只有一个 seed，不能据此声称统计显著或因果机制成立。
