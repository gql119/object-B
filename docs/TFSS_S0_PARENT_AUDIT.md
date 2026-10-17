# 真实短测等价补充 — 2026-10-17

历史S0的45原文件字节/SHA仍保持。两个固定TRAIN batch的S0-reference与S0-noop各5步实际loss/Fourier梯度/更新最大差0。原Frozen/Adaptive初始化、原loss/Adam/EOT/预算不变，未使用S1/S2/S3噪声作为初始化。S0历史运行时commit依旧未知；不可用后来档案commit替代。TFSS当前整体59/60更新，U控制未完成，不从等价检查推出正式效用。

---

## 历史记录（以前的 NOT_RUN / 等待段落已被上文取代）

# S0 父版本审计

Change：独立分支从后来的0d43da4归档建立，再从此前留存的原始source bytes恢复45个历史S0文件。每个实际磁盘文件SHA与历史parent_remote_identity逐一相符，见tfss/S0_CANONICAL_SOURCE_SHA256.json及S0_RESTORATION_AUDIT.json。.gitattributes保留原BOM/换行字节。0d43da4不是已知的历史运行时commit，运行时GitSHA仍未知。

Result：源码身份检查通过；真实模型/损失/梯度/Delta/单步更新等价性 **NOT_RUN**。不能把源码哈希或CPU组件测试写成完整父版本等价。

固定S0 YAML SHA80cffe012aa6bfefcbe66f7c8e880e680da1c1885f5d0a7fcb01a77be6697894；原Frozen/Adaptive checkpoint SHA8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89。P5/model.21通道有序24索引来自SOURCE_METADATA.json，不能重新选择。SafeBG rank8/alpha.5/active3；原L_BG/L_rel/L_ret/L_orth(.0005)、Fourier、Adam、EOT2、batch16、warmup5、cosine日程保持原计算。新入口不实例化CNLP/CUP/KD/XFP/Localization/CSB投影。

Finding：仓库归档标签不足以证明运行源码身份，已按字节恢复；但仍须以真实当前源码探针证明恢复代码数值等价。

Next：独立GPU资源门通过后，直接执行历史canonical B2SemanticTrainer与S0-noop包装器的相同模型/输入/RNG/优化器对照，测五项损失、Fourier梯度、参数单步、实际空间mask及delta；1e-7容差预先固定。失败不得继续候选比较。既有研究的等价性证据仅作来源参考，不冒充本轮新证据。


## 2026-10-17 latest measured status — overrides earlier NOT_RUN / WAIT_FOR_S3

S3 fully accepted/archived/exited before TFSS started. Source897cdfe smokev2 passed actual canonical S0/noop equivalence maxdiff0; S0-reference/noop/C/U each completed1update/EOT2 on firstfixed16 with unchanged model hashes. It then failed the locked Uniform postclamp RMS error1e-9 gate; first complete six-method smoke and both-batch5step remain INCOMPLETE. No reviewer called and no formal TFSS. CPU saved-delta replay finds nearest float32 bracket error>1e-9 for11/16 first-EOT images, maximum7.5655e-9. Full evidence and a concrete **unapproved** numerical protocol proposal: [RMS_GATE_BLOCKER_20261017.md](tfss/RMS_GATE_BLOCKER_20261017.md). WAIT_FOR_HUMAN_NUMERICAL_PROTOCOL; no automatic retry or tolerance change. Original source45/formulas/budgets/eta/losses/optimizer/sample/EOT unchanged.
