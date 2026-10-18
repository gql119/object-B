# S3 CUP 与 CNLP 依赖审计

状态：SOURCE_VERIFIED，不是 S5 runtime pass。通用 Parent/teacher/data/config/seed 元数据及 S5 未运行字段见 `S5_SOURCE_AUDIT_PROVENANCE.json`，授权全文见 `USER_TASK_SOURCE.md`。

审计使用 S3 正式 scientific commit `017591686efb3675ee8d55debf1e2cd7da4e0e17` 的 Git blobs；`counterfactual_utility.py`、`counterfactual_yolo.py`、`counterfactual_extension.py`、`counterfactual_trainer.py`、`counterfactual_optimization.py` 与后来归档分支中的相应 blobs 相同。锁文件并不在该 runtime code commit 中，单独绑定后续 archive 的 `docs/S3_CONFIG_LOCK.json`，SHA256 `bd89244e22795b85139d1dee793b13c1faf44306557c083f546639ec666b6fb8`；不将 archive commit 冒充 runtime SHA。

## 正式 damage 及梯度

`CounterfactualSGD.evaluate` 从相同 theta0 和 RNG 构造 clean/poison 支持集两条虚拟 SGD 路径。正式配置 J=1、lr=.0001；poison 路保留高阶计算图，clean 路与其参照 detach。两条更新后的模型处理同一 disjoint clean train query：

```text
d_k = L_query,k(theta_poison_plus) - stopgrad(L_query,k(theta_clean_plus))
L_CNLP_uniform = mean_valid_classes relu(d_k)
L_CNLP_CUP = sum_valid_classes w_k * relu(d_k)
total = Parent_loss + lambda_CNLP * L_CNLP_CUP
```

因此 S3 CUP **没有替换 Parent L_ret**；它重加权的是 CNLP 额外损失。

`YOLOCounterfactual.query_loss` 使用真实 non-target GT；clean-updated query logits 生成 detached native all-GT TAL 分配，并由两条路径共用。对 native 未覆盖的有效 GT，正式 coverage-v2 允许它在同一个 native TAL 中独立匹配；无正样本或无前景证据则失败。每 GT 的 softscore 正样本归一化后合并 native box-CIoU、19 类 BCE 和 DFL（原 native gains），先逐 GT 后逐类别平均。定位与分类混合，类别 GT 数量通过类内 object mean 归一化；没有 AP 或 validation 信息。

梯度经过 poison query loss 和 poison 虚拟更新回到 Fourier；clean loss detach。当前 d_k 的梯度进入 relu(d_k)；历史和 w_k detach，不通过 EMA 或权重生成反传。不能把这个后学习代理描述成完整 200-epoch victim AP 损伤。

## CUP 状态机

`CUPHistory` 独立于 model/learner/query provider，使用 20 槽 CPU float64 history，Person 槽保持 0。

- beta=.95；首观测直接初始化，后续 EMA；同一 Adam update 内所有 EOT 使用同一 pre-step history。
- step 成功后一次 `commit`，先对该步同类 EOT 观测平均，再裁剪到正负 damage_clip，更新 signed EMA。
- 未观测类别保持历史，不衰减、不新增观测；无 NT 不制造类别损伤。
- temperature=.0003758072853088379，damage_clip=.0010146975517272938。它们在 S3 后学习检测-loss 单位上校准，不能认定适用于 P3 表征漂移。
- 在**当前 query 有效类别集合**上 softmax(history/T)，通过 Euclidean bounded-simplex 投影使 sum w=1，边界为 `[.5/K, 2/K]`，K 是当前有效类别数。
- 不是“任意 batch 所有 19 类都有正权重”。缺席类无该次 loss，history 保留；有效类有下界。
- checkpoint 包括 mean、observations、steps、config、num_classes、target，恢复时校验身份和边界。

## 可独立复用与不可直接复用

可复用 EMA/EOT/Adam 边界、缺席保留、state validation、温度化有界重分配的机制。S5 仍需将权重下界/上界从 uniform 基准转换为真实 S4 pi_k，否则会把 instance mean 改成 class mean。

不能原样复用 `CUPHistory.aggregate()`：该函数聚合 `relu(d_k)`，不是 S4 的 feature-retention objective。也不能原样保留 S3 d_k 而去掉其计算依赖；若坚持同一定义就仍需两条虚拟更新，这与 S5 明确禁止完整 CNLP 双路径相冲突。

原 S3 源码与正式结果均未修改。S5 当前没有导入或实例化 CounterfactualSGD/CNLP。
