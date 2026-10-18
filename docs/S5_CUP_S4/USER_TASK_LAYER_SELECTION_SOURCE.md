# Codex 研究任务：S5 — P3/P5 特征保持层选择与 CUP-S4 自适应类别效用保持

## 一、总体研究目标

当前研究为目标检测模型中的类别选择性不可学习样本。

目标类别：Person（VOC ID=14）。

数据集：VOC20。

当前正式主方法：

S4 = Safe-Soft05 + TFSS-Utility。

S4 正式结果：

- Person AP50：0.028159
- Non-target mAP50：0.693787
- Non-target mAP75：0.515251
- Non-target AP50-95：0.480528
- Selectivity S：0.935330
- WCD：0.295951
- TCD3：0.264482

我们希望进一步提出：

**S5 = S4 + CUP-S4**

即：

Class-aware Utility Preservation adapted to S4。

研究目标是：

**在不明显削弱 S4 已有 Person 不可学习效果的前提下，通过类别感知的非目标特征保持，进一步提高非目标类别的检测效用，尤其减少受损严重类别的性能退化。**

与此前方案不同，本轮需要先解决一个重要问题：

我们通过 TSFL 选取的 Person 目标特异性特征位于 P5（历史配置为 model.21），但原 S4 的 L_ret 被审计为 P3（历史配置为 model.15）全通道 GT ROI 特征漂移。

这两个模块功能不同，因此不要求使用同一层。

但是：

**目前尚无充分证据证明 P3 是非目标特征保持的最佳层。**

因此本轮授权 Codex 在正式适配 CUP 之前，独立开展 P3/P5 特征保持层诊断，并根据训练开发集上的证据选择 S5 的非目标保持层。

不得只凭“P5 语义强”或“P3 分辨率高”作决定。

---

# 二、执行顺序

本轮严格分为六个阶段：

Stage A：原 S0/S4 特征层源码审计。

Stage B：P3/P5 非目标特征保持层对照。

Stage C：依据证据确定 S5 使用 P3 还是 P5。

Stage D：对选定层适配 CUP-S4。

Stage E：TRAIN32 与 5-step 正确性及机制验证。

Stage F：满足准入条件后执行正式 S5 实验。

在 Stage C 完成之前，不允许修改正式 S5 的特征保持层，也不允许启动完整扰动生成。

---

# 三、Stage A：核查 P3/P5 的真实历史实现

## 3.1 审计原 S0 和 S4

请确认：

1. S0 原始 L_ret 使用哪个特征层？
2. S4 L_ret 是否完全继承 S0？
3. P3 是否确实对应当前模型的 model.15？
4. P5 是否确实对应当前模型的 model.21？
5. S0/S4 的 L_ret 具体如何提取非目标 GT ROI 特征？
6. 是否使用全部通道？
7. 是否包含 GT mask、ROI pooling 或特定空间匹配？
8. L_ret 的原始 Reduction 和 lambda_ret 是什么？
9. L_ret 对 Fourier 和 Adaptive 参数的梯度路径如何？
10. P5 的 TSFL 选中通道是否参与 L_ret？

所有内容必须由真实源码、配置及运行记录确认。

如果当前 Ultralytics 模型结构与历史层编号不一致，必须以实际运行模型的模块结构和输出尺寸为准。

不得直接根据旧文档假设编号一定正确。

## 3.2 区分 TSFL 和 L_ret

TSFL：

目标是寻找 Person 敏感、非目标相对不敏感的通道。

其作用是目标类保护性干预。

L_ret：

目标是保持非目标 GT 的内部特征，减少非目标类别的训练效用损伤。

两者虽然可能作用于同一网络，但不要求使用同一个特征层。

不能因为 TSFL 选择 P5，就直接把 L_ret 切换到 P5。

也不能因为历史 L_ret 使用 P3，就默认 P3 必然最优。

输出：

P3_P5_SOURCE_AUDIT.md

记录全部源码证据和实际计算图。

---

# 四、Stage B：P3/P5 保持层对照实验

## 4.1 实验目的

确定在 S4 保护目标基本不变的情况下，P3 与 P5 哪一层更适合作为非目标特征保持层。

需要同时考虑：

- Non-target Feature Retention；
- Person Protection Gradient Conflict；
- 类别级保持效果；
- 小目标与大目标；
- 训练数值稳定性；
- 计算成本。

## 4.2 对比设置

第一组：

P3 Full-channel L_ret。

这是历史 S4 基准。

第二组：

P5 Full-channel L_ret。

用于单独研究保持层从 P3 更换到 P5 的效果。

第三组作为辅助诊断：

P5 Non-target-specific Channel Subspace。

只有现有 TSFL 信息能够可靠构造非目标相关子空间时才进行。

第三组不参与首轮 P3/P5 正式决策，除非前两组证据不足，并明确作为额外探索。

特别注意：

**P5 中 TSFL 已选的 24 个通道是 Person 目标特异通道，不能直接将它们当作 P5 非目标保持通道。**

P3 与 P5 必须首先采用相同的“全通道非目标 GT 特征保持”原则，避免同时改变层和通道选择规则。

## 4.3 固定实验条件

优先复用现有 TRAIN32 固定训练开发样本。

覆盖：

- Person-only；
- Person+Dog；
- Person+Horse；
- Person+Bicycle；
- Small Object；
- Medium Object；
- Large Object；
- 高度重叠；
- 多目标复杂场景。

两组使用相同：

- S4 Frozen Reference；
- S4 Adaptive；
- S4 TFSS-Utility；
- S4 TSFL；
- S4 Safe-BG；
- Fourier 初始化；
- EOT；
- Support/Query；
- Seed；
- GT；
- Optimizer；
- 短程更新步数。

唯一改变：

L_ret 的特征层。

## 4.4 P3/P5 损失定义

在每个层级 l：

h_i,l_clean

代表干净参考特征。

h_i,l_poison

代表受保护输入对应的特征。

定义：

L_ret,l =
mean_i D(h_i,l_poison, h_i,l_clean)

这里的 D、GT ROI 处理以及 Teacher/Adaptive 路径必须首先忠实继承原 S4。

P5 可以因输出尺寸不同而使用对应的 ROI 对齐操作，但不得改变 GT 语义。

注意不同层的原始 L_ret 数值可能具有不同尺度。

因此必须同时报告：

- Raw Loss；
- Per-instance Loss；
- Feature Dimension；
- Gradient Norm；
- 相对原保护梯度的比例。

如果 P3/P5 损失数值差异主要来自特征维度和 Reduction，不得直接将数值较低的层视为保持效果更好。

不得通过自由调节 lambda_ret 来制造某层优势。

---

# 五、Stage B 必须完成的诊断指标

## 5.1 非目标保持

分别计算：

- Non-target Feature Drift；
- Non-target GT Detection Loss Proxy；
- Classification Proxy；
- Localization Proxy；
- 每类别 Feature Drift；
- 每类别有效 GT 数量；
- Small/Medium/Large Object Drift。

尤其关注：

Dog、Horse、Bicycle 等既往出现明显损伤的类别。

但不能只根据这些类别作决定。

## 5.2 Person 保护影响

分别计算：

g_protect = grad_delta(L_protect)

g_ret_P3 = grad_delta(L_ret_P3)

g_ret_P5 = grad_delta(L_ret_P5)

实际主要优化参数如果是 Fourier 系数，则应在 Fourier 参数空间比较梯度。

计算：

- ||g_protect||；
- ||g_ret_P3||；
- ||g_ret_P5||；
- cos(g_protect,g_ret_P3)；
- cos(g_protect,g_ret_P5)；
- Gradient Conflict Ratio；
- 保护梯度反向分量；
- 5-step 后 L_BG；
- 5-step 后 L_rel；
- Target Suppression Proxy。

重点回答：

P3 或 P5 的保持约束是否更容易与 Person 的背景误导目标冲突？

不能只比较梯度范数，也必须比较梯度方向。

## 5.3 训练稳定性

检查：

- Forward/Backward；
- NaN/Inf；
- Feature Hook；
- ROI Alignment；
- GT Mapping；
- Empty GT；
- BN 状态；
- Fourier Gradient；
- Adaptive Update；
- GPU Memory；
- 单步执行时间。

所有诊断使用相同模型状态进行配对比较。

## 5.4 真实 5-step 对照

至少比较：

A. S4 原 P3 L_ret。

B. S4 仅改为 P5 L_ret。

两者均不加入 CUP。

从相同初始化独立执行最多 5-step。

观察：

- Person Protection Proxy；
- Non-target Feature Drift；
- Non-target GT Loss Proxy；
- 梯度冲突；
- 更新后的扰动强度；
- 有效训练梯度。

这里的 5-step 仅用于判断机制可行性，不代表完整 victim AP。

---

# 六、Stage C：让 Codex 自主选择 P3 或 P5

完成源码审计和短程对照后，请依据下列预先确定的原则选择层。

## 6.1 优先条件：保护目标类

首先检查候选保持层是否显著破坏 S4 原来的 Person 保护机制。

如果 P5 造成：

- 保护目标代理明显恶化；
- 梯度冲突明显增加；
- Fourier/Adaptive 更新不稳定；

即使其非目标特征漂移较低，也不应优先选择 P5。

## 6.2 第二条件：非目标保持

在 Person 保护代理未明显恶化的前提下，比较：

- 非目标特征保持；
- GT 检测损失代理；
- 类别间稳定性；
- Worst-class Proxy；
- 不同对象尺度。

不能只根据整体平均 Feature MSE 作决定。

## 6.3 第三条件：计算成本

如果 P3 与 P5 证据接近：

优先保留 P3。

因为 P3 是 S4 原始保持层，有利于保持单变量消融，并减少额外方法复杂度。

## 6.4 P5 的选择要求

只有出现清晰、可重复的正向证据，例如：

- 目标保护代理没有明显退化；
- 非目标保持指标具有一致优势；
- 梯度冲突没有明显加重；
- 5-step 结果与静态审计方向一致；
- 计算开销可以接受；

才建议选择 P5。

如果证据相互矛盾或者差异很小，默认选择 P3。

这里不要求 Codex 追求某个预设的最优结果。

**允许 Codex 根据实际证据决定使用 P3 或 P5，但必须给出完整理由。**

最终输出：

P3_P5_LAYER_DECISION.md

包含：

- 最终选择；
- 主要数值依据；
- 未选择另一层的原因；
- 对 Person 保护的影响；
- 对非目标类别保持的影响；
- 计算成本；
- 潜在风险；
- 是否属于方法学变化。

## 6.5 重要的消融边界

如果选择 P3：

S5 可以定义为：

S4 + CUP-S4。

如果选择 P5：

不能把 S5 相比原 S4 的变化全部归因于 CUP。

因为同时引入了：

- L_ret 层从 P3 转为 P5；
- CUP 类别加权。

因此必须保留：

S4-P5 = S4 + P5 L_ret，不使用 CUP。

S5-P5 = S4 + P5 L_ret + CUP。

后续才能通过 S5-P5 与 S4-P5 比较，分离 CUP 的增量价值。

P5 如果被选择，必须在实验设计中纳入这个独立层对照，而不是事后补解释。

---

# 七、Stage D：CUP-S4 最小适配

## 7.1 保留 CUP 核心思想

原 S3 CUP：

依据 CNLP 双路径学习损伤进行类别加权。

S5 CUP-S4：

允许将类别学习损伤代理替换为选定特征层的类别级 L_ret 特征漂移。

这属于：

CUP_S4_DESIGN_ADAPTATION。

不是普通 Bug Fix。

CUP 核心思想保持不变：

**根据非目标类别受到的干扰程度，动态分配效用保持优先级，使受损较大的类别得到更强的保持约束。**

## 7.2 保留原 L_ret 监督定义

根据 Stage C 选择的层：

L = P3 或 P5。

构造：

L_ret,k

表示类别 k 的 GT ROI 特征漂移。

除 Stage C 明确批准的保持层变化之外，必须保留：

- GT 对应；
- 原特征距离；
- 原 Reference/Adaptive 路径；
- 原 Reduction 原则；
- lambda_ret=0.25；
- 非目标实例集合；
- 原有 Stop Gradient 规则。

不得新增 CNLP 双路径。

不得修改 TSFL 或 Safe-BG。

## 7.3 保留类别基准聚合

根据源码审计，原 S4 L_ret 可以按类别分解，基准权重为：

pi_k=N_k/N。

其中 N_k 是当前有效类别 k 的实例数，N 是所有有效非目标实例数。

定义：

L_ret_base =
sum_k pi_k L_ret,k

CUP 改进：

L_ret_CUP =
sum_k pi_k*w_k*L_ret,k

要求：

sum_k pi_k*w_k = 1

当 w_k=1 时：

L_ret_CUP=L_ret_base。

必须在数值和梯度上验证等价性。

若 Stage C 选择 P3：

原 S4 L_ret 必须严格恢复。

若选择 P5：

必须先验证 P5 无 CUP 基准实现的等价性，再进行 CUP 加权。

不能用 P5 的基准损失与原 P3 的数值完全相等作为要求，因为两者是不同的特征层。

---

# 八、Stop-Gradient EMA 类别损伤机制

## 8.1 EMA 类别损伤

使用：

d_k=stopgrad(L_ret,k)

构造 EMA：

m_k^(s)
=
beta*m_k^(s-1)
+
(1-beta)*d_k^(s)

记录类别损伤历史。

优先复用 S3 已有：

- EMA；
- Temperature；
- Weight Bound；
- Numerical Stabilization；
- Missing Class Handling。

但要注意：

P3/P5 不同类别原始特征漂移的尺度可能不同。

不能仅因为某类别特征幅度天然更大，就长期给它更高权重。

需要检查类别内历史标准化或其他合理的损伤尺度处理。

所有新增处理必须记录数学定义和训练开发集证据。

## 8.2 权重约束

要求：

- 权重有界；
- 不出现单类别塌缩；
- 有效类别始终有基本保护；
- 空类别不产生虚假梯度；
- 类别权重使用 Stop Gradient；
- 权重变化具有稳定性；
- 保持原 L_ret 的基准尺度。

但注意：

sum_k pi_k*w_k=1

并不保证加权后梯度范数不变。

因此还要单独比较有效 lambda_ret 梯度强度。

---

# 九、Stage E：CUP-S4 真实 TRAIN32 和 5-step 验证

完成选层和 CUP 适配后，至少比较：

1. S4 原始基线；
2. 选定特征层的无 CUP 基准；
3. 选定特征层 + CUP。

若最终选择 P3，前两项应当数值等价。

若最终选择 P5，前两项属于不同的特征保持层，应保留其独立对照意义。

检查：

- S4/no-op；
- L_ret 数值；
- L_ret 梯度；
- L_BG；
- L_rel；
- Person Protection Gradient；
- Non-target Feature Drift；
- Class-wise Drift；
- CUP 权重；
- EMA；
- Weight Entropy；
- Gradient Norm；
- Gradient Cosine；
- Fourier/Adaptive Update；
- Delta RMS；
- Spatial Budget；
- AMP；
- NaN/Inf；
- Resume State。

还需额外比较：

Uniform Class Aggregation

与：

CUP Class-aware Aggregation。

检查 CUP 是否只是降低了整体效用梯度强度。

如有必要，加入不改变正式训练的 Gradient-Norm-Matched 诊断。

## 9.1 关键研究问题

必须回答：

1. CUP 是否真正提高了高损伤类别的保护优先级？
2. CUP 是否减少了原来较严重的类别特征漂移？
3. CUP 是否影响 Person 保护梯度？
4. CUP 是否改变总体 L_ret 梯度尺度？
5. P3/P5 的差异是否影响 CUP 权重稳定性？
6. CUP 的代理收益是否优于仅均匀改变 L_ret 权重？
7. 是否值得进入正式训练？

如果出现无效梯度、权重塌缩、严重梯度冲突或预算错误，暂停并报告。

---

# 十、Stage F：S5 正式实验

完成 Stage A–E 且全部正确性门槛通过后，允许启动已经授权的 S5 正式实验。

如果选定 P3：

正式方法：

S5 = S4 + CUP-S4(P3)。

如果选定 P5：

正式方法：

S5-P5 = S4 + P5 L_ret + CUP-S4(P5)。

同时必须保留 S4-P5 无 CUP 作为同层对照；若需要完整评估 CUP 的独立收益，应按锁定配置完成该对照，不得仅使用原 P3-S4 代替。

正式训练协议：

- 完整扰动优化 24 epoch；
- 9,144 step；
- VOC20；
- 16,551 张训练图；
- Random-initialized YOLOv8n；
- Seed=0；
- Victim 200 epoch；
- Batch=36；
- Workers=16；
- AMP=True；
- last.pt@200 主评价；
- best.pt 次评价。

除选定 L_ret 层及 CUP-S4 的类别权重适配外，其余均严格继承 S4。

不允许同时增加：

- CNLP；
- KD；
- XFP；
- Localization Preservation；
- 新梯度投影；
- 新空间预算；
- 新目标攻击 Loss。

使用 S4 已验证的原生 AMP 路径。

提前检查最终指标汇总对于缺失可选 Recall_all 等字段的处理，不允许再次因为报告接口问题导致训练完成后任务异常。

---

# 十一、正式评价与成功标准

必须完整报告：

- Person AP50；
- Person Recall；
- Person Precision；
- Person AP75；
- Person AP50-95；
- Non-target mAP50；
- Non-target mAP75；
- Non-target mAP50-95；
- Non-target Recall；
- Non-target Precision；
- All-class mAP50；
- VOC20 逐类别 AP50/AP75/AP50-95；
- P/U/S；
- WCD；
- TCD3。

主要比较：

S0；

S3；

S4；

S5。

如果选择 P5，额外比较：

S4-P5 与 S5-P5。

主要目标：

Person AP50 尽量维持 S4 的 0.028159 附近。

沿用此前预注册的保护容差上限：

Person AP50≤0.038993。

在满足目标保护前提下，希望：

Non-target mAP50>0.693787。

WCD<0.295951。

TCD3<0.264482。

Non-target mAP75 尽量改善。

以上是研究性判据，单种子实验不能证明统计显著性。

---

# 十二、允许的自主改进边界

我允许 Codex 根据真实审计选择 P3 或 P5。

允许在 CUP 适配中合理改进：

- 类别损伤标准化；
- EMA；
- Weight Temperature；
- Weight Bound；
- Missing Class Handling；
- 类别权重归一化；
- GT ROI Matching；
- 数值稳定性；
- 计算效率。

但不能改变：

CUP 的类别感知效用保持思想。

不得根据正式 VOC Validation AP50 反复调参。

所有调整都必须记录：

1. 原设计；
2. 修改原因；
3. 修改后的公式；
4. 改动的文件和函数；
5. 参数变化；
6. 梯度变化；
7. 训练开发集依据；
8. 是否影响 S4 可比性；
9. 是否改变原始 L_ret 的含义；
10. 对最终消融解释有什么影响。

如果选择 P5，必须明确这是一项额外的保持层设计改变，不能混同于 CUP 本身。

如需要改变 L_ret 的监督对象、引入新的训练信号或增加其他主要损失，应停止并申请进一步确认。

---

# 十三、代码与实验隔离

项目 GitHub：

https://github.com/gql119/object-B

请使用独立 S5 分支及独立工作目录。

不得修改：

- main；
- S0；
- S3；
- S4；
- 原 S4 数据集；
- 原 S4 checkpoint；
- 已有正式训练结果。

必须保存：

- S4 原始父版本 SHA；
- S5 源码 SHA；
- Stage C Layer Decision；
- CUP Adaptation Diff；
- 正式 Config；
- Train Sample Manifest；
- Gradient Audit；
- CUP Weight Log；
- Noise Integrity；
- Victim Checkpoints；
- VOC20 Per-class CSV；
- 最终完整报告。

历史失败或负向诊断不得覆盖删除。

---

# 十四、最终交付物

至少生成：

1. P3_P5_SOURCE_AUDIT.md
2. P3_P5_GRADIENT_COMPARISON.csv
3. P3_P5_NON_TARGET_RETENTION.csv
4. P3_P5_5STEP_RESULTS.md
5. P3_P5_LAYER_DECISION.md
6. CUP_S4_DESIGN.md
7. CUP_S4_ADAPTATION_DIFF.md
8. CUP_CLASSWISE_WEIGHTS.csv
9. CUP_GRADIENT_AUDIT.csv
10. S5_PARENT_EQUIVALENCE.md
11. S5_5STEP_SMOKE.md
12. S5_CONFIG_LOCK.json
13. S5_NOISE_TRAINING_REPORT.md
14. S5_NOISE_INTEGRITY.json
15. S5_VICTIM_TRAINING_REPORT.md
16. S5_VOC20_PER_CLASS.csv
17. S5_S0_S3_S4_COMPARISON.csv
18. S5_FINAL_REPORT.md

如果选择 P5，额外保存 P5 无 CUP 同层对照记录及其正式实验结果，避免遗漏关键消融。

---

# 十五、最终决策原则

本轮允许 Codex 自主完成：

P3/P5 审计 → 选择更合适的保持层 → CUP-S4 适配 → 短测 → 满足条件后正式训练。

但是，选层必须以真实训练开发集上的证据为依据。

不能因为 P5 是 TSFL 的目标特异性层，就默认 L_ret 应使用 P5。

也不能因为 P3 是原始默认层，就不进行对照分析。

如果没有充分证据证明 P5 优于 P3，选择 P3，以保持与 S4 的单变量可比性。

本轮最重要的研究目标：

**在不牺牲 S4 目标类别不可学习能力的前提下，通过合理的非目标特征保持层选择和 CUP 类别感知加权，减少非目标类别检测效用损伤。**

正式实验完成后停止。

不自动增加其他损失、不开展新的大规模调参、不自动启动其他 Seed。

最终请明确回答：

1. 为什么选 P3 或 P5？
2. 选层是否有可靠的梯度和非目标保持证据？
3. CUP 是否真正改善类别损伤分配？
4. 改善来自 CUP，还是改变保持层？
5. Person AP50 是否维持在 S4 附近？
6. Non-target mAP50 是否改善？
7. WCD/TCD3 是否改善？
8. S5 是否值得替代 S4 作为最终论文主方法？
