# TFSS-Conservative

W_C=.85+.15Q_t；delta_C=delta_S0*W_C。不扫描eta，不新增损失/训练参数，不改变S0的上游全局clip、2.01/255直通floor、a0=1/.5/0或最后图像clamp。W在a0之后应用，所以重叠区域仍最多4/255，Person外实际影响为0。

Q_t来自原Frozen干净训练输入、既有P5有序24通道的区域均方能量对RGB输入的L2梯度。使用canonical fractional GT feature mask；在原图坐标按Person分别做masked3x3平滑、Q10/Q90归一化，再max合并，不叠加多Person振幅。此版本与旧诊断的普通边界平均不同，明确标为TFSS_DESIGN_CHANGE/NUMERICAL_STABILITY；原方案保存在归档分支，不覆盖。

平坦/退化Q不伪造空间结构：该Person区域保留W=1并记录无效，且从high-TSFL统计排除。空/非法GT、非有限梯度或缺失真实监督必须失败。地图仅从clean TRAIN生成并detach，整个5-step重复固定，不根据优化噪声或验证集更新。

`tfss_spatial.SoftSpatialProjection`只接管原compose返回的最终delta和poison，原raw/JND/TV/budget项不变，更新_last_poison以保证原Adaptive内循环真正看到缩放图像。S0模式返回原tuple，不进入额外缩放。不存在新增局部hard clipping。

当前CPU公式/预算/梯度组件检查通过；真实目标保护梯度保持、5-step曲线、强度和最终AP均NOT_RUN。


## 2026-10-17 latest measured status — overrides earlier NOT_RUN / WAIT_FOR_S3

S3 fully accepted/archived/exited before TFSS started. Source897cdfe smokev2 passed actual canonical S0/noop equivalence maxdiff0; S0-reference/noop/C/U each completed1update/EOT2 on firstfixed16 with unchanged model hashes. It then failed the locked Uniform postclamp RMS error1e-9 gate; first complete six-method smoke and both-batch5step remain INCOMPLETE. No reviewer called and no formal TFSS. CPU saved-delta replay finds nearest float32 bracket error>1e-9 for11/16 first-EOT images, maximum7.5655e-9. Full evidence and a concrete **unapproved** numerical protocol proposal: [RMS_GATE_BLOCKER_20261017.md](tfss/RMS_GATE_BLOCKER_20261017.md). WAIT_FOR_HUMAN_NUMERICAL_PROTOCOL; no automatic retry or tolerance change. Original source45/formulas/budgets/eta/losses/optimizer/sample/EOT unchanged.
