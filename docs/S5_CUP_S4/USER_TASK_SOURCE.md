# Codex 研究任务：S5 — S4-TFSS-Utility + Class-aware Utility Preservation

## 一、研究目标

我们决定进行最后一轮候选模块组合实验：

**S5 = S4 + CUP-S4**

Parent：

S4-TFSS-Utility。

本轮不直接加入 S3 的整个 CNLP+CUP 模块。

我们希望保留 S4 已经验证的目标类别不可学习能力，同时利用 CUP 的逐类别效用保护思想，减少不同非目标类别受到的不均衡损伤。

当前 S4 正式结果：

- Person AP50：0.028159；
- Non-target mAP50：0.693787；
- Non-target mAP75：0.515251；
- Non-target AP50-95：0.480528；
- Selectivity S：0.935330；
- WCD：0.295951；
- TCD3：0.264482。

评价协议：

- VOC20；
- Target=Person，ID=14；
- Seed=0；
- 随机初始化 YOLOv8n Victim；
- 200 epoch；
- last.pt@200 主评价；
- 4,952 张 clean validation。

本轮的核心问题：

**能否通过对 S4 已有非目标保持机制进行逐类别损伤感知加权，在保持原有 Person 抑制效果的情况下，提高非目标平均检测性能，并缓解损伤最严重类别的退化？**

---

# 二、第一阶段：审查 S3 CUP 与 S4 L_ret 的真实实现

先不要立即修改损失或启动完整训练。

请分别审查：

S3：

- CNLP；
- CUP；
- 每类别损伤 d_k；
- CUP EMA；
- CUP 权重生成；
- 权重归一化；
- 类别缺席处理；
- 每类别有效 GT；
- 梯度传播；
- 与 CNLP 的依赖关系。

S4：

- L_ret 的准确数学定义；
- 输入来自哪些特征层；
- 是否基于非目标 ROI；
- 是否使用 GT；
- 如何处理类别；
- 是否可以分解为每类别损失；
- 当前 reduction；
- 当前 lambda_ret；
- 与 TSFL、Safe-BG 的梯度关系。

必须以实际代码为准。

不要仅根据历史论文设想或之前的对话重构损失。

需要明确回答：

1. 原 S3 CUP 是否必须依赖 CNLP 才能运行？
2. CUP 权重机制中哪些部分可以独立复用？
3. 原 S4 L_ret 是否可以准确分解为 19 个非目标类别的损失？
4. 如果可以，如何保证均匀或基准加权时恢复原 S4？
5. 如果不能，应当如何进行最小范围的类别级适配？
6. 此种适配是否会改变原来的特征保持定义？

如果无法保留原 S4 L_ret 的基准等价性，应先停止并报告，不得自行替换为一个全新的非目标保持损失。

---

# 三、第二阶段：实现 CUP-S4

## 3.1 保持 CUP 思想不变

CUP 的核心思想：

不同非目标类别受到的损伤程度不同，因此需要根据类别损伤自适应调整效用保持权重。

不是对所有非目标类别一视同仁。

也不能根据正式验证集上 Dog、Horse、Bicycle 的 AP 下降结果，事后为这些类别手动设置更高权重。

类别重要性必须根据训练开发数据自动计算。

## 3.2 优先适配原 S4 L_ret

若原 L_ret 可以被严格分解，构造：

L_ret,k

表示类别 k 的非目标特征保持损失。

原 S4：

L_ret_S4 = sum_k pi_k * L_ret,k

其中 pi_k 必须来自原 L_ret 的真实聚合规则。

CUP-S4：

L_ret_CUP = sum_k w_k * L_ret,k

要求：

当 CUP 未启用、或者 w_k=pi_k 时，必须数值恢复原 S4。

本轮优先替换 L_ret 的类别聚合规则，不额外重复叠加同一保持损失。

总目标：

L_S5 =
L_S4 - lambda_ret*L_ret_S4
+ lambda_ret*L_ret_CUP

上述公式只是目标结构说明。

必须根据真实源码中的损失权重和计算图实现，不允许重复加权。

其他损失完全不变。

## 3.3 类别损伤代理

CUP-S4 不直接复制 CNLP 的双路径虚拟训练。

请先研究是否能从 S4 既有计算过程中获得稳定的类别损伤代理。

优先候选：

A. 类别级非目标特征漂移。

B. 同一模型在干净与加噪图像上的 GT 条件化非目标检测损失差异。

两者应先通过训练开发数据诊断，再选择一个明确、稳定的正式代理。

必须报告：

- 损伤代理的实际定义；
- 是否使用真实非目标 GT；
- 分类和定位是否混合；
- 类别样本数如何归一化；
- 是否存在零样本类别；
- 是否存在极端损伤值；
- 是否需要 Stop Gradient。

不能将输入特征漂移直接宣称为经过完整模型训练后的真实 AP 损伤。

不得引入完整 CNLP 双路径更新，否则本轮就不再是独立 CUP-S4 实验。

## 3.4 类别权重

尽量复用 S3 CUP 已经过验证的：

- EMA；
- Temperature；
- Weight Normalization；
- Weight Bounds；
- Missing Class Handling。

若原机制依赖 CNLP 专有状态，允许做最小必要适配。

权重原则：

- 类别损伤越大，通常获得越高保护优先级；
- 避免少数类别权重塌缩；
- 所有有效非目标类别都保留基本权重；
- 不因单个异常 batch 导致权重剧烈变化；
- 不改变原 L_ret 的有效整体尺度。

请特别验证：

当损伤权重退化为基准值时，是否恢复原 S4。

如果类别加权显著改变总效用梯度尺度，需要记录并检查，不应把单纯降低或增大 lambda_ret 的效果误认为 CUP 的类别选择作用。

---

# 四、第三阶段：必须保持 S4 不变的模块

S5 从正式 S4 创建独立分支。

建议：

codex/s5-s4-cup-adaptation-20261018

不得修改 S4 既有正式产物。

以下全部保持不变：

- TFSS-Utility；
- eta=0.85；
- Q_t；
- R_nt；
- W_U；
- 原 S0 overlap-half；
- Fourier 参数化；
- Frozen Reference；
- Adaptive Learner；
- TSFL P5 目标特异通道；
- Safe-BG；
- rank=8；
- alpha=0.5；
- active-k=3；
- L_BG；
- L_rel；
- L_orth；
- 其他原始保护损失；
- EOT；
- 优化器；
- 正式噪声生成日程；
- Victim 训练协议。

本轮只允许修改：

**非目标保持损失的类别级聚合与 CUP 权重机制。**

不允许重新加入：

- E1-XFP；
- E2-Localization；
- KD-ObjBal；
- CNLP 双路径训练；
- 新的梯度投影；
- 额外目标类别保护损失。

---

# 五、第四阶段：短程正确性与机制测试

正式运行前必须测试：

## A. S4 等价性

CUP 关闭时，S5 与原 S4 数值一致。

检查：

- Forward；
- Loss；
- Fourier Gradient；
- Adaptive Update；
- Spatial Weights；
- Delta；
- Budget；
- 训练数据和 GT 对应。

## B. 类别级分解

验证：

sum_k pi_k*L_ret,k

等于原 S4 L_ret。

若原损失包含复杂空间掩码或重叠实例，必须正确处理其贡献权重，不能简单按类别平均后声称与 S4 等价。

## C. CUP 权重

检查：

- 19 个非目标类别；
- 有效 GT 数量；
- EMA；
- 类别权重总和；
- 极端权重；
- 类别缺席；
- 少样本类别；
- 权重随 batch 的变化；
- 是否存在权重塌缩。

## D. 梯度审计

在相同训练开发 batch、相同模型状态和相同 Fourier 初始化下比较：

S4 原 L_ret 梯度；

CUP-S4 加权 L_ret 梯度；

原 Person 保护梯度。

记录：

- 梯度范数；
- 梯度余弦；
- 梯度冲突比例；
- 相对保护梯度的有效权重；
- 各类别梯度贡献。

重点判断：

CUP 的变化是否来自类别权重重新分配，还是主要来自整体效用约束尺度变化。

## E. 5-step 短测

使用固定训练开发样本。

比较：

1. S4；
2. S4 + 基准类别分解；
3. S4 + CUP-S4。

如果第 2 项不能恢复 S4，不能进入正式实验。

观察：

- L_BG；
- L_rel；
- L_ret；
- 每类别损伤；
- Person protection proxy；
- 非目标 feature drift；
- 有效梯度；
- 实际扰动强度。

短测只是正确性和机制代理，不能推断最终 AP 必然改善。

---

# 六、第五阶段：允许合理优化 CUP-S4

允许对 CUP 进行适配优化，但必须保证其核心思想不变。

允许修改：

- 类别损伤的稳定估计；
- EMA 更新；
- 类别样本数量归一化；
- Weight Temperature；
- Weight Bound；
- 类别缺席处理；
- 类别级 Loss Reduction；
- 数值稳定性；
- 计算效率。

所有修改必须主动说明：

1. 原来的实现是什么？
2. 为什么不适用于 S4？
3. 发现了什么具体问题？
4. 修改后的公式是什么？
5. 修改了哪些源码？
6. 是否改变 L_ret 定义？
7. 是否改变有效损失尺度？
8. 对 Person 保护梯度有什么影响？
9. 对非目标类别保持有什么影响？
10. 是否仍然属于 CUP 的逐类别效用保护思想？

参数设计必须在预先固定的训练开发集上完成。

不能根据正式 clean validation 的 AP50 反复调参。

必须保留原始 S3 CUP 代码，不得直接覆盖。

如果 CUP 需要更换监督对象或新增其他损失，属于方法学变更，必须先报告并获得确认，不能作为普通兼容性修复实施。

---

# 七、第六阶段：正式 S5 实验

只有在以下条件均满足后才能启动：

- S4 Parent 等价性通过；
- CUP 类别损伤计算有效；
- 基准权重恢复原 L_ret；
- 梯度审计通过；
- 5-step Smoke 通过；
- 原生 AMP Victim 路径通过；
- 完整预算检查通过；
- 服务器资源准入通过；
- S5 正式配置锁定。

如果 CUP 设计存在明显数值错误、不可恢复的基准偏差或无效梯度，先停止，不允许进入正式实验。

正确性及资源条件通过后，授权执行一轮正式 S5：

完整扰动生成：

24 epoch / 9,144 step。

完整训练数据：

VOC20，16,551 张。

Victim：

随机初始化 YOLOv8n。

Seed：

0。

Epoch：

200。

Batch：

36。

Workers：

16。

AMP：

True。

主评价：

last.pt@200。

次评价：

best.pt。

必须沿用 S4 已经修复的 YOLOv8 原生 AMP 更新路径。

不得重新加入此前导致 Victim 首批次失败的 AMP 梯度安全封装。

正式噪声生成与 Victim 的全部配置，除 CUP-S4 外，与 S4 保持一致。

必须使用独立 RunID、独立输出目录。

不得覆盖 S4 数据集或实验结果。

---

# 八、正式评价

必须保留：

- Person AP50；
- Person AP75；
- Person AP50-95；
- Person Recall；
- Person Precision；
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

额外记录：

- CUP 每类别权重；
- 每类别损伤代理；
- 类别损失；
- 类别权重熵；
- 最差类别；
- 各类别的实际 AP 改变量。

对比：

S0；

S3；

S4；

S5。

重点报告：

S5-S4；

S5-S3；

S5-S0。

其中最重要的是 S5-S4，因为这是 CUP-S4 的直接增量比较。

---

# 九、正式成功标准

S4 基础：

Person AP50=0.028159。

Non-target mAP50=0.693787。

S=0.935330。

WCD=0.295951。

TCD3=0.264482。

优先目标：

1. Person AP50 尽量接近 S4；
2. 不超过此前预注册的 0.038993 保护上限；
3. Non-target mAP50 高于 S4；
4. WCD 低于 S4；
5. TCD3 低于 S4；
6. Non-target mAP75 尽量不退化；
7. Selectivity S 高于 S4。

必须分开报告所有指标，不允许因为个别指标改善就忽略其他指标退化。

这里的阈值属于研究性判据。

单次 seed0 实验不能证明统计显著性。

---

# 十、独立审查与结果归档

请在 object-B 仓库创建独立 S5 分支。

不得修改 main。

不得覆盖 S4、S3、S0。

建议输出：

- S5_PARENT_FIDELITY.md
- S3_CUP_DEPENDENCY_AUDIT.md
- CUP_S4_DESIGN.md
- CUP_S4_CHANGELOG.md
- CUP_CLASSWISE_RETENTION_AUDIT.csv
- CUP_WEIGHT_DYNAMICS.csv
- CUP_GRADIENT_AUDIT.csv
- S5_5STEP_SMOKE.md
- S5_CONFIG_LOCK.json
- S5_NOISE_INTEGRITY.json
- S5_VICTIM_TRAINING_REPORT.md
- S5_VOC20_PER_CLASS.csv
- S5_S0_S3_S4_COMPARISON.csv
- S5_FINAL_REPORT.md
- S5_ARTIFACT_MANIFEST.json

所有报告必须包含：

- Git SHA；
- Config Hash；
- Parent RunID；
- Teacher Hash；
- 数据集 Hash；
- Seed；
- 训练日志；
- Checkpoint Hash；
- AMP 状态；
- 数据审计；
- 完整结果。

注意之前 S4 的报告汇总因 Recall_all 缺失而失败。

本轮必须提前验证汇总接口对于可选 aggregate 字段的兼容性，不能再次因为报告字段缺失导致训练完成后任务异常。

---

# 十一、最终要求

本轮只研究：

**S4 + CUP-S4 的增量效果。**

我们不希望重做 CNLP+CUP，也不希望继续增加新的损失模块。

CUP 的方法思想必须保持：

**对非目标类别的损伤进行类别感知的动态加权，让受损较大的类别获得更多保护。**

但允许对损伤代理、类别权重以及与 S4 原 L_ret 的适配做合理优化。

所有优化必须有明确的科学解释和可复现的代码证据。

先完成源码审计与短测，再决定最终实现。

如果只需要进行不改变数学定义的兼容性修复，可以直接实施。

如果需要更换 CUP 的核心损伤定义或改变原 L_ret 的监督对象，必须停止并向我说明。

当算法正确性、短测、AMP 和资源准入全部通过，且没有尚待确认的方法学变更时，可以启动已经授权的一轮完整 S5 正式实验。

完成后停止，不自动启动更多参数、第二个 Seed、CNLP 联合实验或新的空间方案。

最终需要回答：

**CUP 的逐类别保护思想，在没有完整 CNLP 双路径训练的情况下，是否能够改善 S4 的非目标检测效用，并维持 Person 的不可学习能力？**
