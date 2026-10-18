# CUP-S4 最小适配方案：待确认的 damage 定义

更新：`USER_TASK_LAYER_SELECTION_SOURCE.md` 已明确授权 `CUP_S4_DESIGN_ADAPTATION`，包括自主 P3/P5 选层和选定层 `d_k=stopgrad(L_ret,k)`。下文原“等待确认”保留为此前审计历史，现已解除；当前尚未选层，先执行 Stage B。最新权重记法为相对倍率 w_k：sum pi_k*w_k=1，w=1 恢复基准；下文绝对权重可写成 a_k=pi_k*w_k，两者不重复相乘。温度、标准化、EMA及边界的正式数值必须在实际固定训练开发证据后锁定，不能把原建议当作已实现或已验证。

状态：PROPOSED，未接入任何训练 loss，未冻结 S5 正式配置。相关授权、Parent、teacher、data、config、seed 元数据在 `S5_SOURCE_AUDIT_PROVENANCE.json`。S5 checkpoint/AMP/日志/正式结果为 NOT_RUN。

## 已由源码确定的部分

单 ROI 保持定义和有效实例集合沿用 `S5_PARENT_FIDELITY.md`。本轮只替换 Frozen L_ret 聚合；Adaptive 自带 NT 保持、L_BG/L_rel/L_orth、TFSS-U 地图/eta、原优化器及完整日程均不属于改动面。

基准 pi_k=n_k/N。拟使用：

```text
L_ret,S5 = L_ret,S4 + sum_valid_classes (w_k-pi_k)*L_ret,k
L_total,S5 = L_total,S4 + lambda_ret*(L_ret,S5-L_ret,S4)
lambda_ret = .25
```

这与 sum w_k*L_ret,k 在实数算术上等价；保留原 scalar 作为基准可使 w=pi 或 CUP-off 数值 no-op。需要在真实 GPU 独立比较直接原 S4 与候选，不能拿残差为 0 代替整体等价性验证。

## 必须明确的新 damage 定义

S3 damage 是经过两条虚拟训练后的 query 检测损失差；它无法在同时删除 CNLP 双路径的条件下保持同一数学定义。本方案建议的候选 A 是：

```text
d_A,k = stopgrad(mean_{j in I_k} ||z_P3,j(poison)-stopgrad(z_P3,j(clean))||²)
```

它使用原 S4 同一次 Frozen/EOT/有效 GT ROI，复用原 L_ret 的监督对象；只将 detach 后的值送给历史权重估计。L_ret,k 本身保留梯度。该量是“当前配对输入表征漂移”，不再是 S3 的“后学习 query 检测损伤”，更不能被称为 AP 损伤。将此替换称为纯兼容性修复不准确。

候选 B（仅供已授权训练开发诊断比较）是同一个 Frozen 模型在原配对 clean/poison 输入上的 GT 条件化 detector loss 差，使用共享 detached clean TAL、类内 GT mean、box/BCE19/DFL 和 native gains；无虚拟 SGD。它同样改变 S3 damage 的定义，并引入不同于 feature retention 的信号。不能把 B 作为新增可微训练损失叠加。

建议先获准该 damage 定义适配，再在原固定 TRAIN32 两独立 batch 上比较 A/B 的可用类/GT覆盖、符号/幅度、EOT波动、极值、裁剪率和权重响应。A 的优势是完全复用原 L_ret 监督对象；当前没有真实诊断证据，不能声称 A 更稳定或效果更好，也不提前锁定温度与裁剪数值。

## 权重与尺度（PROPOSED）

复用 beta=.95、首观测初始化、缺席 history 保留、成功 Adam 后一次 EOT 聚合提交。权重只看 prior-step detached history，不能让本步损伤通过 softmax 回传或消耗本步另一 EOT 的新历史。

令有效类集合 V={k:n_k>0}，拟计算：

```text
v = softmax(log(pi_V) + history_V/T)
w = EuclideanProjection(v; sum(w)=1, .5*pi <= w <= 2*pi)
```

所有有效类保留正下界，缺席类 pi=w=0 且历史不变；无有效 NT 时完全保持原 zero loss。所有 history 相等时实数上 v=pi。显式 CUP-off/base mode 返回原聚合，避免额外浮点差异。复用 S3 的 .5/2 相对边界与 bounded-simplex 思想，不能直接复用它的 1/K 中心。

S3 temperature 和 damage_clip 处在另一种损失单位，拟先用固定训练开发数据的预先记录统计规则校准一次；具体规则与数值尚未选择，正式 lock 不得填入猜测值。不做 validation AP 搜索。

sum w=1 保持聚合系数总量，但不保证 loss 或梯度范数不变。必须记录 lambda_ret*Lret 的有效范数、对 BG+relative+Adaptive-target 梯度的范数比与余弦、逐类梯度贡献及冲突比例；不通过动态整体梯度归一化或修改 lambda_ret 隐藏尺度变化。

## 附件条件与下一步

附件末节要求“如果需要更换 CUP 的核心损伤定义……必须停止并向我说明”。源码证明：从 S3 后学习损伤转为 A 或 B 都属于这一变化；虽然 A 保持原 S4 L_ret 监督对象，damage 的含义仍发生变化。因此当前在采用新 damage、真实诊断及正式 S5 之前提交此方案确认；这不是重新请求已授权的 seed0/24epoch/Victim200 启动许可。

确认后依次完成真实 TRAIN32 A/B 诊断、仅 Frozen L_ret 类聚合实现、基准和关闭路径等价、逐类权重与梯度审计、三路5-step/当前源码GPU smoke、一次新独立科学审查、配置冻结/预算/AMP/资源门槛。没有方法学待确认项且全部通过才执行已授权的一轮 S5。当前这些项目全为 NOT_RUN，不调用 reviewer，不启动正式实验。
