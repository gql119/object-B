# S4-TFSS-Full：全Person框空间调控消融

当前状态（2026-10-22）：28项CPU测试、当前源码真实GPU smoke和唯一一次独立科学审查通过。已按条件授权唯一启动正式RunID s4_tfss_full_overlap_seed0_20261021_v1；正式AP和完整200轮结果尚未产生。

实际运行/审查源码为 1d10ad620ef01aad20e8c85d49465473d72c9a4e，配置锁SHA为 2bedd986d44dce40d3815093da76ef3d259aec83fab7f0c6518d0f7795b6675c。其后的文档提交仅更新发布记录，不替代不可变的运行源码SHA。

- 唯一变量：Person–NT GT交集的support从0.5改为1.0；Person union外为0。
- TFSS保持 W_U=1−0.15 R_nt(1−Q_t)，eta=0.85；原STE2.01/255 floor及原样本筛选独立保留。
- TSFL、SafeBG、全部损失、Frozen/Adaptive、Fourier/JND/TV、优化器和原训练协议不变；原S4代码、数据、噪声、checkpoint不覆盖。
- 原鲁棒性八项已全部验收并发布至 codex/s4-robustness-final-results-20261022；后续队列及无活跃本人训练核查通过后才启动Full。
- v1仅在计算前因历史checkpoint路径绑定错误失败；原件SHA未变。修正后新v2真实完成两次Adam更新、四个有限EOT loss。
- 无重叠样本原S4/Full loss均0.7707809805870056，loss/delta/Fourier梯度差均0；实际解码重叠PNG max2bytes/outside0，预算8/0通过。CPU单独检验最大8bytes投影。
- 独立reviewer /root/s4_full_scientific_review 给出 scientifically_correct/allow_run，无blocker。没有第二或closing review。
- 正式流程：新seed0Fourier24epochs/9144Adam/batch16/EOT2 → 新16551train及4952原样cleanval/完整8/0与GT审计 → fresh random-YAML YOLOv8n seed0/200/batch36/workers16/AMP/nativeSGD → last@200 primary、best secondary、完整VOC20与空间诊断。已有绑定不再启动。

| 文档 | 入口 |
|---|---|
| 用户授权原文 | USER_TASK_SOURCE.md |
| 锁定配置与历史身份 | S4_TFSS_FULL_LOCK.json、SOURCE_AUDIT.json |
| 方法差异与修改清单 | METHOD_DIFF_AND_SOURCE_AUDIT.md、CHANGE_LIST.md |
| CPU真实结果 | CPU_VALIDATION_REPORT.json、CPU_TEST_LOG.txt |
| 首次smoke失败及修正 | SMOKE_V1_PREFLIGHT_FAILURE_AND_FIX.md |
| 当前GPU和单次审查证据 | GPU_SMOKE_AND_REVIEW_ACCEPTANCE.json、INDEPENDENT_SCIENTIFIC_REVIEW.md |
| 运行入口、评价命令 | RUN_COMMANDS_AND_QUEUE.md |
| 时间估算 | GPU_COST_ESTIMATE.md（不是Full正式实测耗时） |

所有正式结论保持OPEN直至新数据和完整200轮评价验收。CPU随机教师不等于正式预训练教师；GPU smoke已使用锁定的预训练教师，但不提供正式AP。LPIPS没有合法现成权重绑定时保持不可用。单seed不作显著性、背景吸收或重叠中介因果宣称。权重、数据、profiles、原始日志和秘密不上传GitHub。
