# S4-TFSS-Full：全 Person 框空间调控消融

当前交付是**独立实现与本地 CPU 验证**。正式 Fourier 优化、完整数据导出、GPU smoke、独立科学审查和 victim200 均未运行；不得填入正式 AP。

- 实验标识：`S4_TFSS_FULL_OVERLAP`。
- 独立分支：`codex/s4-tfss-full-overlap-20261021`。
- 唯一变量：Person–NT GT 交集的 support 从 0.5 改为 1.0，Person union 外仍为 0。
- TFSS：`W_U = 1 - 0.15 R_nt (1-Q_t)`，eta=0.85。
- 原始 S4 代码、噪声、数据、checkpoint 不覆盖；缓存也独立。
- 原鲁棒性流水线必须完整验收并核清后续队列，GPU 空闲本身不构成准入。

| 交付 | 文件 |
|---|---|
| 方法差异和源码审计 | [METHOD_DIFF_AND_SOURCE_AUDIT.md](METHOD_DIFF_AND_SOURCE_AUDIT.md) |
| 独立配置 | [S4_TFSS_FULL_LOCK.json](S4_TFSS_FULL_LOCK.json)、`ue_framework/configs/exp_voc_person_s4_tfss_full_seed0_20261021.yaml` |
| 代码修改清单 | [CHANGE_LIST.md](CHANGE_LIST.md) |
| CPU测试、预算、导出检查 | [VALIDATION_AND_BUDGET.md](VALIDATION_AND_BUDGET.md)、[CPU_VALIDATION_REPORT.json](CPU_VALIDATION_REPORT.json) |
| 实际测试原始日志 | [CPU_TEST_LOG.txt](CPU_TEST_LOG.txt) |
| 生产运行与评价命令、队列 | [RUN_COMMANDS_AND_QUEUE.md](RUN_COMMANDS_AND_QUEUE.md)、[QUEUE_STATUS.json](QUEUE_STATUS.json) |
| GPU时间估算 | [GPU_COST_ESTIMATE.md](GPU_COST_ESTIMATE.md) |
| 来源冻结与文件哈希 | [SOURCE_AUDIT.json](SOURCE_AUDIT.json) |
| 用户授权原文 | [USER_TASK_SOURCE.md](USER_TASK_SOURCE.md) |

现在的正式准入结论为 **NOT_READY：本地科学组件测试通过，当前教师的完整 GPU 生产链、一次独立审查、原流水线最终验收与新鲜队列审查尚未闭环**。不需要再次询问本任务已有的条件授权；条件未满足时继续保留候选，不插入现有服务器队列。
