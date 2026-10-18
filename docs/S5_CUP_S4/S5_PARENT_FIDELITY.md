# S5 Parent 源码审计

状态：SOURCE_AUDIT_COMPLETE；CPU_SYNTHETIC_DECOMPOSITION_PASS；真实 TRAIN32/5-step/GPU/正式 S5 均 NOT_RUN。

本轮来源：同目录 `USER_TASK_SOURCE.md`，SHA256 `3f3521e3b82be7765ae35aa07850a01cbc12c8caf7c952233609f0e8f067947d`。独立分支 `codex/s5-s4-cup-adaptation-20261018` 从 S4 结果修复代码 `686eaacad069aac30f7fc7af4f10bebdeec1c390` 创建。S4 扰动运行源码为 `b1e29510e7b1ed5fa0fe361a27132d8790e65c14`；原生 AMP Victim 恢复源码为 `ab626917f20917e58cf30033ac99df00938cb561`。不把后来的报告提交当作扰动运行时源码。

Parent noise RunID `s4_tfss_utility_seed0_20261017_v1`；Victim RunID `s4_tfss_utility_seed0_20261017_v1_victim_recovery01`；last SHA256 `cc1b03eca939732599817abb97150907463ab10913456afa6cc6ae473b4bbdc3`。S4 config lock SHA256 `33d3d7af6c76c2c881db0dfc7d6fc66dafa9b5655026c07332701143ed52d6ec`，teacher SHA256 `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`，dataset fingerprint `29b131dfca2e3636e9c47502961caa6297acf0450f646ff6409b7b3bfb21b3fa`，seed 0。S5 config/checkpoint/AMP/runtime log 尚不存在，不能记为已冻结或已通过。逐文件来源在 `S5_SOURCE_AUDIT_PROVENANCE.json`。

## 实际调用链与 L_ret

实际链为 `S4Trainer -> B2SemanticTrainer -> B2AdaptiveTrainer -> SemanticFrozenTrainer -> LegacyKProtoTrainer`。`SemanticFrozenTrainer._batch_loss` 由 MRO 在 Adaptive 分支之前执行。正式 strict-P5 配置在 `b2_semantic.py` 将 `orth_feature_layer` 设为 `model.15`；这个变量不仅用于 orth，也决定了实际 Frozen `L_ret` 的来源。

`L_ret` 的来源是 **P3/model.15 的全部通道**，不是 P5 所选的 24 个 TSFL 通道。`legacy_kproto_ret.py:non_target_retention_loss` 为：

```text
z_j(x) = sum_p clamp(mask_j(p),0,1) * F_P3(x,p) / sum_p clamp(mask_j(p),0,1)
ell_j = ||z_j(poison) - stopgrad(z_j(clean))||_2^2
L_ret = (1/N) * sum_j ell_j
weighted_L_ret = 0.25 * L_ret
```

`masked_prototype` 是按 ROI 有效面积归一化的特征均值；没有通道 L2 归一化，没有除以通道数。ROI 来自真实 non-target GT 矩形，沿原 padded canvas、letterbox 与 area resize 映射到特征坐标。ROI 有效条件为特征 mask mass >=1。其监督对象是 Frozen 的配对输入表征，并非检测分类/定位损失。

原实现的资格规则也必须保持：只有至少存在一个有效 P3 Person/context 配对的图像，其有效 non-target P3 ROI 才进入 `non_target_pairs`；该 append 发生在 P5 target eligibility 的 `continue` 之前。整个 batch 至少有一个有效 P5 target/context 才返回有效 loss。不能把仅在 P5 有效的图像当作全部 retention 来源。

所有加入的有效 ROI 跨图像拼接后平均；不是每图平均、类别平均或像素平均。重叠 GT 各自池化，各保留一次实例贡献；不存在对重叠像素去重，也没有在 `L_ret` 中使用 overlap-orth 权重。无有效 non-target 时保持 graph-connected zero。

## 严格类别分解

令 I_k 为实际进入上述原始拼接序列的类别 k 的有效 GT ROI，n_k=|I_k|，N=sum n_k。则：

```text
L_ret,k = (1/n_k) * sum_{j in I_k} ell_j  (n_k>0)
pi_k = n_k/N
L_ret = sum_{k:n_k>0} pi_k * L_ret,k
```

缺席类别 n_k=0，pi_k=0，类别损失为 unavailable，不伪造观测 0。只有完全无有效 ROI 的 batch，原总 ret 为 0。不是将 19 个类别全部等权；CPU 反例中，等权类别均值与原损失差达 `6.898888386785984e-6`。

保留每个 mask 的 GT 原序号、class id、validity 和参与资格即可恢复类别身份；原 `_pooled` 会丢弃 invalid ROI，不能直接把过滤后的向量 zip 到未过滤的 GT 类别。clean/poison validity 必须逐 ROI 校验一致。

## 其他损失与梯度

- Frozen retention 的 clean 特征 detach，poison 特征经 Frozen detector 和 W_U/原 delta 路径向 Fourier 传播；teacher 参数冻结。不要 detach 类别损失本身。
- TSFL-P5 与 Safe-BG 用于 Person BG/relative 分支；Frozen retention 的通道没有被它们裁成 24 个。共享扰动意味着这些损失仍会在 Fourier 梯度上相互作用。
- `B2AdaptiveTrainer` 还计算 `lambda_adapt*(target + lambda_nt*non_target)`，其中 lambda_adapt=1，lambda_nt=.25。这个 Adaptive non-target 项不同于 Frozen `L_ret`，本轮不能一并重新加权。
- 已有 `feature_drift_<class>` 诊断来自 P5、并使用不同的图像 eligibility，因此不能直接拿来驱动按真实 Frozen `L_ret` 定义的 CUP-S4。

## 数值证据及边界

`scripts/audit_s5_parent_retention.py` 从原文件 AST 提取并执行真实 pooling/scalar-retention 函数。五组 CPU FP32 合成用例覆盖不均衡类别数、同/跨类别重叠、空/亚像素 ROI、19 类齐全和无 NT。类别分解最大 scalar 误差 `3.637978807091713e-12`，最大特征梯度误差 0；基准残差形式的 loss/gradient 差为 0。所有 45 个历史 S0 文件工作副本字节仍与 manifest 一致。

这证明代数与合成输入的局部实现，不证明真实 TRAIN32 的图像/GT资格恢复、Fourier 梯度、Adaptive update 或 5-step 数值等价。这些仍是后续生产 GPU 门槛，未执行。

## 六个源码审计回答

1. S3 的 EMA/权重状态机本身不需要 CNLP；其正式 damage 生产者及 active aggregate 则绑定 CNLP 的后学习 query loss。
2. 可复用 signed/clipped EMA、首观测初始化、缺席保留、step/EOT 事务、detach 权重、温度和有界 simplex 思想；原 uniform-centered 权重和 ReLU-damage aggregate 不能直接替换 S4 的实例平均 ret。
3. 原 Frozen L_ret 可以分解，类别粒度是上述实际有效 P3 ROI。
4. 必须以 pi_k=n_k/N 为基准；关闭 CUP 时返回原 ret，基准权重可用原 ret 加零残差保持浮点 no-op，同时独立核验分解恒等式。
5. 不需要更换监督损失；最小适配是保留有效 ROI 的 class/GT 身份并改变聚合。不能丢弃实例或把 reduction 改成无条件等类平均。
6. 该分解不改变单 ROI 保持定义。但将 S3 后学习 damage 换成 P3 drift 或同模型检测 loss 差，会改变 damage 的科学含义，必须明确披露并处理附件末节的确认条件。
