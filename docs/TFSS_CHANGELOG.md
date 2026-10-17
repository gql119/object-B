# TFSS Change Log

- SOURCE_RESTORATION：从历史byte备份恢复45文件并验证其SHA，不把后来归档commit冒充运行commit；不修改原目录。
- TFSS_DESIGN_CHANGE：新增C=.85+.15Qt，U=1-.15R(1-Qt)，放在S0 a0后的软乘法；原计算损失/优化器不变。
- NUMERICAL_STABILITY / TFSS_DESIGN_CHANGE：Qt/R使用masked3x3平滑和GT域Q10/Q90；多Person max合并；无效Qt保留强度并从high-Qt统计排除，不伪造归因；正值flatR为域内1。相较旧诊断平滑属于明确设计变化。
- TFSS_DESIGN_CHANGE：NTGT estimator保留native allGT，未覆盖validGT使用同TAL独立匹配，positive/BCE19/box/DFL原gains及原对象/类归约。只用于clean风险和诊断，无CSB预算投影或新训练loss。
- TFSS_DESIGN_CHANGE：Uniform-C/U按各候选同step/EOT/图像在控制自身轨迹中匹配图像clip后实际RMS，而非只匹配理论权重；不可达则失败。
- BUG_FIX：新增诊断序列化使用Python二进制句柄修复本地受控Windows路径文件writer失败；无科学计算变化，原失败保留。
- EFFICIENCY_OPTIMIZATION：clean maps固定一次构建，候选共用；单独计时Qt与额外R，不能据此宣称运行加速。

这些改变的实际梯度/强度/目标/NT影响均待真实短测，未把假设记录为实验结果。没有引入eta.90/其他核心公式/中心渐弱/新模块或AP调参。


## 2026-10-17 latest measured status — overrides earlier NOT_RUN / WAIT_FOR_S3

S3 fully accepted/archived/exited before TFSS started. Source897cdfe smokev2 passed actual canonical S0/noop equivalence maxdiff0; S0-reference/noop/C/U each completed1update/EOT2 on firstfixed16 with unchanged model hashes. It then failed the locked Uniform postclamp RMS error1e-9 gate; first complete six-method smoke and both-batch5step remain INCOMPLETE. No reviewer called and no formal TFSS. CPU saved-delta replay finds nearest float32 bracket error>1e-9 for11/16 first-EOT images, maximum7.5655e-9. Full evidence and a concrete **unapproved** numerical protocol proposal: [RMS_GATE_BLOCKER_20261017.md](tfss/RMS_GATE_BLOCKER_20261017.md). WAIT_FOR_HUMAN_NUMERICAL_PROTOCOL; no automatic retry or tolerance change. Original source45/formulas/budgets/eta/losses/optimizer/sample/EOT unchanged.
