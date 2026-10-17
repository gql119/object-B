# INSUFFICIENT_EVIDENCE

当前尚无完整真实5-step对照，不能选择TFSS-C或TFSS-U进入正式200epoch实验。源码/CPU正确性仅为部分Gate A，不能证明目标保持或非目标收益。eta_C/U均固定.85，无参数搜索。

最终16问当前答复：

1. 当前实验代码已隔离提交并推送object-B新分支，SHA a5aefbb1376cd5aacc0e2cefd73cebdbcf808f54。
2. git ls-remote重新读取同SHA；main仍b89147ee0e46cea3fbba10a8b6bd9af7e60a5b43。
3. S3目录/运行参数/检查点未修改，无新共享GPU任务；完整“零影响”不作因果保证。
4. 从按实际历史45个文件SHA恢复的S0开发；不是从S1/S2/S3方法开发。
5. 原loss/优化器计算体未改；source897cdfe真实首16单步 S0/noop 最大差0，非全部两batch/五步的等价证明。
6. C首16/单步/EOT2的保护梯度cos中位数.997833、范数比中位数.952332；仅不完整smoke描述，不能替代正式Gate B。
7. U减少NT干扰是否胜C：尚未完成匹配控制，不能判断。
8. 首16/单步两EOT的逐图配对RMS平均下降 C9.5593%、U1.4312%；测的是记录的pre-update前向，非五步结果/最终AP。S0平均RMS.00704838，C.00637463，U.00695410。
9. 优于各自Uniform：NOT_RUN；Uniform-C在首个forward的锁定数值配对门失败。
10. U风险额外开销：地图计时原始证据在maps_batch0.json，尚未完成规定对照分析；不从总smoke运行时间推断增量成本。
11. 首16单步描述中U梯度更近S0（cos中位数.999625、范数比.990035），无匹配Uniform和第二batch，不能据此选正式候选。
12. 谁NT效用更好：完整配对证据不足。
13. 跨样本稳定性：规定的两个独立batch/五步未完成。
14. 类别间得失：已有首16记录，但完整逐类配对分析未完成；不填成收益。
15. 当前不推荐任何正式200epoch候选；INSUFFICIENT_EVIDENCE。
16. 最低下一步：先由用户确认RMS数值协议是否修订，再做相应求解器局部验证/新current-source smoke/唯一科学审查/原训练32五步对照。无需跑完整victim来解决此数值门问题。具体未批准提案见tfss/RMS_GATE_BLOCKER_20261017.md；不自动放宽门槛或重试。

Change/Result/Finding/Next见SHORT_PROBE_RESULTS；所有可疑、undefined或冲突证据保留，可最终继续KEEP_S0/INSUFFICIENT_EVIDENCE，禁止靠反复调参制造成功。


## 2026-10-17 latest measured status — overrides earlier NOT_RUN / WAIT_FOR_S3

S3 fully accepted/archived/exited before TFSS started. Source897cdfe smokev2 passed actual canonical S0/noop equivalence maxdiff0; S0-reference/noop/C/U each completed1update/EOT2 on firstfixed16 with unchanged model hashes. It then failed the locked Uniform postclamp RMS error1e-9 gate; first complete six-method smoke and both-batch5step remain INCOMPLETE. No reviewer called and no formal TFSS. CPU saved-delta replay finds nearest float32 bracket error>1e-9 for11/16 first-EOT images, maximum7.5655e-9. Full evidence and a concrete **unapproved** numerical protocol proposal: [RMS_GATE_BLOCKER_20261017.md](tfss/RMS_GATE_BLOCKER_20261017.md). WAIT_FOR_HUMAN_NUMERICAL_PROTOCOL; no automatic retry or tolerance change. Original source45/formulas/budgets/eta/losses/optimizer/sample/EOT unchanged.
