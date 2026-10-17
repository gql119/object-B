# TFSS-Utility

Risk=R_nt*(1-Q_t)；W_U=1-.15Risk；delta_U=delta_S0*W_U。eta_U=.85，与C同锁，不根据AP或短测结果反复搜索。

R_nt是原eval/frozen模型对真实非目标GT的检测损失输入敏感度；复用公开披露的NTGT估计器计算体，原文件SHA6ffa876785e318824d550ae363b486da7a606e51dda3522f114e714292ce7b18。只复用clean梯度估计，不加载S1预算/输出/硬投影，也不将风险损失加入优化。按GT soft-score加权positive box/BCE19/DFL、原native gains、对象均值/类别均值；它是GT-conditioned代理，不是完整all-anchor原生训练loss。Person logit与背景anchor排除。

保留native allGT匹配；仅valid未覆盖NTGT在同一native TAL中单独匹配，缺失positive/soft evidence/梯度仍失败。这个coverage选择与原strict-TAL不同，明确作为本轮TFSS_DESIGN_CHANGE记录，不伪装普通bug fix。每类L2梯度在Person支撑域masked3x3、Q10/Q90归一化后等类平均；这是明确的聚合设计，不宣称等价于总loss梯度的模。

合法no-NT图像R=0，因此U精确退化S0；正值平坦risk映射为域内1并披露，非法或零NT梯度不能用0回退。Q无效区域保留强度且不当成高归因证据。高Qt/高R冲突位置仍保留较大扰动，无理论解耦保证。

Qt与风险均在clean TRAIN上冻结，不读official val。缓存键由固定图像/标签/模型/源SHA、通道索引、batch padding和map version共同绑定。maps_batch记录单独Qt与额外NT风险耗时；真实额外成本、NT效用及是否优于C都NOT_RUN。


## 2026-10-17 latest measured status — overrides earlier NOT_RUN / WAIT_FOR_S3

S3 fully accepted/archived/exited before TFSS started. Source897cdfe smokev2 passed actual canonical S0/noop equivalence maxdiff0; S0-reference/noop/C/U each completed1update/EOT2 on firstfixed16 with unchanged model hashes. It then failed the locked Uniform postclamp RMS error1e-9 gate; first complete six-method smoke and both-batch5step remain INCOMPLETE. No reviewer called and no formal TFSS. CPU saved-delta replay finds nearest float32 bracket error>1e-9 for11/16 first-EOT images, maximum7.5655e-9. Full evidence and a concrete **unapproved** numerical protocol proposal: [RMS_GATE_BLOCKER_20261017.md](tfss/RMS_GATE_BLOCKER_20261017.md). WAIT_FOR_HUMAN_NUMERICAL_PROTOCOL; no automatic retry or tolerance change. Original source45/formulas/budgets/eta/losses/optimizer/sample/EOT unchanged.
