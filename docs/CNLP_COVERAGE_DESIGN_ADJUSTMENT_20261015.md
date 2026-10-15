# CNLP coverage design adjustment（2026-10-15）

分类：CNLP_DESIGN_CHANGE，不是Parent修改或AP调参。用户任务Stage5明确允许有训练开发证据的有效GT匹配方式调整。

实际失败证据：RunID cnlp_production_admission_train32_seed0_20261015_v1，source65db658aac526de6d2f4fc2a0b4f4bef38eaf14d，exit1。首个matching_edges case为train/003396.jpg，GTindex1/class13 motorbike。真实clean-updated原生allGT TAL无positive，CNLP_NO_VALID_POSITIVE触发，formal_generation_enabled=false/victim_training_enabled=false，无任何正式生成或victim。原输出及完整console仍在原隔离RunID；不得重用它。

新候选显式matching_policy=native_shared_gt_coverage_v2。保持原生allGT竞争，已覆盖GT的anchor/softscore/GTbox/reduction和梯度完全保留；仅合法且未覆盖的NTGT，使用同一clean-updated预测和同一个native TAL，将该GT单独匹配。恢复的anchor/softscore/GTbox由clean/poison/initial三个query输出共享；不按Student单独匹配，不选择nearest-anchor，不设伪positive，不置零，不丢对象。仍无positive或非finite/zero evidence则fail。

这是GT-conditioned query损失的覆盖约定，可能令不同GT共享anchor，不等同原生整图检测损失。不是S1预算估计器，S0仍固定8/4/0，无CSB/cache。所有原Parent机制、原native support criterion、原初始化、BCE19/raw-logit、native box/DFL gains、对象/类别mean、J/lambda/CUP候选均不变。strict_native_all_GT仍为默认，旧RunID和旧标定保持严格模式，不能重新标成新设计。

冻结新模式后先CPU验证：原native covered loss/gradient严格相等、竞争导致空GT时sameTAL单独恢复且非零二阶梯度、恢复仍空时fail。之后freshRunID实际固定38case J1重放，比较旧已覆盖case值/mean-EOT梯度统计，核验沿用原权重规则是否成立；再fresh生产准入验证12GT边界、实际Parent等价、S2/S3五步/续跑/PNG。若任何证据失败保持阻断，不临时调参数。当前真实新模式GPU NOT_RUN，正式两锁/单次最终独立审查NOT_RUN。

CPU实际结果：8项native YOLO测试通过（含J1/3/5原梯度链、合法无NT、strict/new恢复仍空拒绝、已覆盖对象loss和完整delta梯度差0、单GT恢复两条虚拟路径loss一致且二阶梯度非零）。GPU新模式仍NOT_RUN。失败console/status/current_case已通过同一rrctl profile/transport取到BASE/cnlp_diagnostic_20261015_v1，字节SHA核验通过；CLI诊断拉取因Windows长路径失败，不曾删除原文件。

新增matching_policy checkpoint字段后，synthetic完整/3→5续跑/完成边界恢复用例再次1passed，CPU参数/Adam/scheduler/RNG/CUP/CSV相等。
