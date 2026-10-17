# Codex 正式实验任务：启动 S4-TFSS-Utility 完整扰动生成与 200 Epoch Victim 训练

## 一、任务目标与执行授权

现在正式授权启动：

**S4 — Safe-Soft05 + Target-Specific Feature-Guided Soft Spatial Scaling (TFSS-Utility)**

本轮不再开展新的 TFSS 参数搜索，而是使用已经完成短程机制诊断、静态同扰动对照和正确性审查的 TFSS-Utility 配置，进行一轮完整正式实验。

任务包括：

1. 审计并冻结 S4 正式代码、配置和模型来源；
2. 验证 S0 父版本与 TFSS-Utility 的实现正确性；
3. 完成 24 epoch / 9,144 step 扰动优化；
4. 生成完整 VOC20 加噪训练集；
5. 完成图像、标签与空间扰动预算审计；
6. 使用随机初始化 YOLOv8n 训练 victim 200 epoch；
7. 以 last.pt@200 为主评价；
8. 在同一份 clean VOC validation 上评价；
9. 与 Clean、S0、S1、S2、S3 统一对比；
10. 输出完整论文实验指标、逐类别结果、训练诊断和机制分析；
11. 保存完整实验产物、代码版本和结果报告。

**本次授权覆盖 S4 的完整噪声生成、victim 训练及正式评价。只要预设正确性门槛通过、资源安全，不需要再次请求启动许可。**

但不得自动启动额外参数实验、第二个 seed 或 S4 与 CUP 的组合实验。

---

# 二、正式实验身份

实验名称：

S4-TFSS-Utility

Parent：

S0-Safe-Soft05

Target：

Person，VOC class ID=14。

数据集：

VOC20。

预期数据规模：

- 训练图像：16,551；
- Person 训练图像：6,095；
- Clean Validation：4,952。

执行时必须重新核查实际数据数量。

实验 Seed：

0。

Victim：

Random-initialized YOLOv8n。

主评价：

last.pt @ 200 epochs。

次评价：

best.pt。

建议 RunID：

s4_tfss_utility_seed0_20261017_v1

每个正式阶段必须记录：

- Parent Git SHA；
- S4 Git SHA；
- Config Hash；
- Teacher Checkpoint SHA；
- Dataset Manifest Hash；
- RNG Seed；
- Noise Checkpoint；
- Victim Checkpoint Hash；
- Evaluation Protocol。

---

# 三、必须保持 S0 原始机制不变

本轮 S4 必须从真正的 S0 Safe-Soft05 源码和正式配置开始。

禁止以 S1、S2 或 S3 作为父版本。

保持以下模块完全不变：

### Frozen Reference

- 相同模型权重；
- 相同冻结策略；
- 相同特征提取；
- 相同预处理。

### Adaptive Learner

- 相同初始化；
- 相同 Support/Query；
- 相同虚拟训练更新；
- 相同 Inner Loop；
- 相同 Outer Loop；
- 相同 Optimizer；
- 相同 Learning Rate；
- 相同 EOT。

### TSFL

- P5 / model.21；
- 既有 24 个 target-specific channels；
- 原有通道选择索引；
- 不重新选择层；
- 不重新生成通道集合。

### Safe-BG

- 原 Background Misdirection；
- rank=8；
- alpha=0.5；
- active-k=3；
- 原背景原型；
- 原 Soft Removal；
- 原相对相似度约束。

### 损失函数

保持：

- L_BG；
- L_rel；
- L_ret；
- L_orth；
- 原有其他保护和正则损失。

原权重保持不变，包括：

lambda_ret=0.25

lambda_orth=0.0005

其他权重从正式 S0 配置读取。

不得添加：

- L_KD；
- L_XFP；
- L_Loc；
- L_CNLP；
- L_CUP；
- 新增非目标分类保持损失；
- 新增梯度投影优化器。

如果当前代码保留了历史实验模块，必须确认它们没有进入 S4 的实际计算图。

**S4 相对于 S0 的唯一方法变化是空间扰动缩放权重。**

---

# 四、TFSS-Utility 的研究思想

我们已经通过 TSFL 找到了对 Person 相对敏感、对非目标类别相对不敏感的特征通道。

现在希望进一步利用这些特征对应的输入空间信息，使扰动分布更具有类别选择性。

核心思想：

**尽量保留与 Person 目标特异性特征相关的扰动，在非目标检测敏感度较高、但 Person 特异性响应较低的位置适当削弱扰动。**

与 S1-CSB 不同：

S1 只根据非目标敏感度降低空间预算，并采用逐像素 Hard Clipping。

S4 同时考虑：

- TSFL Target Attribution；
- Non-target Detection Sensitivity。

并使用：

**Soft Multiplicative Scaling，而不是 S1 的 Spatial Hard Clipping。**

这样希望在不显著改变 S0 保护性优化方向的前提下，减少对非目标类别的不必要干扰。

---

# 五、S4 正式数学定义

## 5.1 目标特异性空间归因

使用冻结的干净参考检测器：

f_theta0

读取既有 TSFL 选定通道：

C_t。

对干净 Person GT 对应的目标特征响应进行输入归因：

G_t(p)
=
||∂z_Ct / ∂x(p)||_2

经过已经验证的归一化和平滑：

Q_t(p) ∈ [0,1]

其中：

Q_t 越高，说明该位置对既有 TSFL 目标特异性通道的局部影响越大。

必须沿用短测中已经通过检查的：

- 归因标量定义；
- 选定通道；
- 特征位置；
- GT 匹配；
- 梯度处理；
- 归一化；
- 空间平滑；
- 多 Person 合并；
- 小目标回退规则。

不得重新设计归因算法。

## 5.2 非目标检测敏感度

对于真实非目标 GT：

k != Person

使用 Frozen Reference 计算：

G_nt(p)
=
||∂L_det_nt / ∂x(p)||_2

经过原短测采用的归一化和处理，得到：

R_nt(p) ∈ [0,1]

要求：

- 只使用 clean training images；
- 教师模型冻结；
- GT 正确；
- 不使用 clean validation 设计权重；
- 不根据当前扰动动态修改敏感度图；
- 不根据 victim AP 重新归一化。

## 5.3 正式 TFSS-Utility 权重

沿用已验证公式：

W_U(p)
=
1 - 0.15 * R_nt(p) * (1-Q_t(p))

也就是：

eta_U=0.85。

不修改 0.15。

不额外引入 sigma、gamma、中心距离衰减或其他空间系数。

对应：

0.85 <= W_U(p) <= 1

含义：

- 非目标敏感度高、目标特异性低：适当降低扰动；
- 目标特异性高：尽量保留扰动；
- 非目标敏感度低：不主动削弱；
- 无有效非目标 GT：沿用短测已经验证的回退规则。

## 5.4 与 S0 原扰动结合

首先按照原 S0 生成基础扰动：

delta_base。

原 S0 经过固定区域缩放得到：

delta_S0。

S4：

delta_S4(p)
=
W_U(p) * delta_S0(p)

必须保留原 S0 的：

Person 独占区域预算 8/255。

Person 与 Non-target Bounding Box 重叠区域预算 4/255。

Person 框外严格为零。

这里的 overlap-half 仍保留，本轮不研究取消 overlap-half。

TFSS 只额外执行乘法缩放，不增加逐像素硬裁剪。

原有全局像素范围与预算安全处理仍然保留。

---

# 六、此前短测证据

已完成的固定 TRAIN32 诊断显示：

- 32 张训练开发图；
- 65 个 Person GT；
- 46 个 Non-target GT；
- 320/320 静态 RMS 配对通过；
- 真实 S0/no-op 等价性通过；
- S4 权重计算及相关梯度检查通过。

相同实际扰动 RMS 下：

### 高目标归因区域能量保留

Utility：

99.90%。

Uniform：

96.90%。

### 高非目标风险、低目标归因区域能量保留

Utility：

74.00%。

Uniform：

95.81%。

### L_ret 均值

Utility：

5.83398e-5。

Uniform：

5.95475e-5。

### 真实非目标 GT signed loss delta 均值

Utility：

0.00256339。

Uniform：

0.00320174。

这些结果支持进行完整实验，但不能证明最终 victim AP 一定改善。

尤其必须注意：

46 个真实非目标 GT 的变化并不一致，部分非目标类别代理损失反而升高。

因此正式评价必须保留 WCD、TCD3 和 VOC20 逐类别指标。

---

# 七、正式运行前的最后正确性审计

请在启动长程生成之前执行 preflight。

## 7.1 S0 父版本一致性

检查：

- 已恢复的正式源码；
- loss；
- Optimizer；
- Fourier；
- TSFL；
- Safe-BG；
- Adaptive；
- Space Mask。

关闭 TFSS 或设置 W_U=1 时，必须在固定输入和随机状态下恢复 S0 的计算结果。

比较：

- Loss；
- Fourier Gradient；
- Delta；
- Parameter Update；
- Pixel Budget。

如果不一致，不能启动正式生成。

## 7.2 TFSS 配置一致性

确认：

- eta_U=0.85；
- W_U 公式不变；
- Q_t 来源不变；
- R_nt 来源不变；
- 无 S1 hard clipping；
- 无额外中心衰减；
- 无 KD/CNLP/CUP；
- 无重叠预算修改。

## 7.3 空间预算检查

必须满足：

Global L∞ <= 8/255。

Overlap L∞ <= 4/255。

Person 外扰动=0。

最后还需检查实际导出的 PNG，而不只是内存张量。

## 7.4 标签与数据

- 全部图像 ID 唯一；
- 原始数据不覆盖；
- 标签字节一致；
- Person 类别 ID 正确；
- 无 Person 图像保持不变；
- validation 完全不变。

如检查失败：

记录错误并停止，不要自动跳过异常样本。

## 7.5 源码身份

核实短测通过的代码 SHA。

此前静态对照新增源码 SHA 记录为：

7cdcb382877496ee8be59b9099ede35fde672a56

但该 SHA 可能只对应诊断脚本，不一定是最终完整生成入口。

必须通过 Git 和文件 Hash 重新确认生产代码实际版本。

不得仅根据历史报告假定源码已完全一致。

如果生产入口需要最小接线修改，必须：

- 记录修改原因；
- 对旧 W=1 模式重新做等价性测试；
- 对 S4 重新做 smoke；
- 确认没有新增其他方法学变化。

---

# 八、正式噪声生成

全部 preflight 通过后启动：

S4-TFSS-Utility

完整扰动优化：

24 epoch。

完整更新：

9,144 step。

使用原 S0 正式生成日程。

不得根据早期 L_BG、L_rel、L_ret 曲线临时增减 Epoch。

保持：

- 相同 Fourier 参数化；
- 相同优化器；
- 相同初始学习率；
- 相同 LR Schedule；
- 相同 Adaptive 配置；
- 相同 EOT；
- 相同随机种子；
- 相同完整训练图像覆盖。

必须记录每轮：

L_BG；

L_rel；

L_ret；

L_orth；

L_total；

Protection Gradient Norm；

Fourier Gradient Norm；

Delta RMS；

Mean Absolute Delta；

Spatial Weight Mean；

High-Q_t Energy；

High-R_nt Energy；

实际预算饱和情况。

### 额外 TFSS 诊断

每轮记录：

- Q_t 分布；
- R_nt 分布；
- W_U 分布；
- 高 Q_t 区域扰动保留；
- 高 R_nt、低 Q_t 区域扰动衰减；
- Target-only 图像回退数量；
- 小目标回退数量；
- 异常归因图数量；
- 实际 L∞；
- 实际 Overlap L∞。

这些日志用于判断 TFSS 在整个生成训练过程中是否持续发挥预期作用。

不用于中途调参。

### Checkpoint

按照原日程定期保存：

- Fourier 参数；
- Optimizer；
- Scheduler；
- Adaptive 需要的状态；
- RNG；
- 当前 Epoch/Step；
- 已处理 Image IDs；
- Config Hash。

必须支持断点恢复。

中断恢复后不得从头重复优化已经完成的训练部分。

---

# 九、完整数据集导出

扰动优化结束后，生成完整 S4 poisoned training dataset。

需要输出：

- Images；
- Labels；
- Manifest；
- Noise Statistics；
- Integrity Report；
- Visualization Samples。

核查：

1. Train 图像总数；
2. Person 图像数量；
3. 实际受保护图像数量；
4. 未修改图像数量；
5. 图像 ID 一致性；
6. 标签字节一致性；
7. 非目标区域扰动；
8. Global L∞；
9. Overlap L∞；
10. PNG 量化后的实际扰动；
11. PSNR；
12. SSIM（如果已有计算工具）；
13. LPIPS（如果已有可靠实现）。

要求：

**完整数据集导出通过审计后，才能开始 Victim 训练。**

不得一边生成部分 S4 数据，一边提前训练 victim。

不删除任何旧数据集。

---

# 十、Victim 正式训练

使用：

Random-initialized YOLOv8n。

Epoch：

200。

Seed：

0。

训练数据：

完整 S4 poisoned VOC20。

Validation：

原始 4,952 张 Clean VOC validation。

严格与 S0 保持相同：

- Architecture；
- Initialization；
- Optimizer；
- Batch；
- Image size；
- LR；
- Scheduler；
- Warmup；
- Weight Decay；
- Momentum；
- Mosaic；
- Augmentations；
- NMS；
- Validator；
- 数据顺序及随机种子控制策略。

不使用预训练 victim 替代随机初始化 victim。

不得为了改善结果修改训练日程。

必须训练完整 200 epoch。

如果训练由于资源、安全或数值问题中断，使用一致性已验证的 checkpoint 恢复。

不能将未完成的 checkpoint 作为 last.pt@200 正式主结果。

---

# 十一、正式评价

主评价：

last.pt@200。

次评价：

best.pt。

所有模型在相同 clean validation 上执行。

## Target 指标

- Person AP50；
- Person AP75；
- Person AP50-95；
- Person Recall；
- Person Precision。

## Non-target 指标

- Non-target mAP50；
- Non-target mAP75；
- Non-target mAP50-95；
- Non-target Recall；
- Non-target Precision。

## Overall 指标

- All-class mAP50；
- All-class mAP50-95。

## 逐类别指标

VOC20 全部类别：

- AP50；
- AP75；
- AP50-95；
- Recall；
- Precision。

## Selectivity

P：

相对 Clean 的 Person AP50 下降比例。

U：

相对 Clean 的非目标 mAP50 保留比例。

S：

S=2PU/(P+U)。

同时计算：

WCD；

TCD3。

全部沿用之前已经锁定的定义。

不得事后改变统计口径。

---

# 十二、与历史实验统一对比

生成正式表：

| Method | Person AP50 ↓ | NT mAP50 ↑ | NT mAP75 ↑ | All mAP50 | S ↑ |
|---|---|---|---|---|---|
| Clean | 86.7943 | 76.6462 | 61.9326 | 77.1536 | — |
| S0 | 2.8993 | 69.1698 | 51.5898 | 65.8563 | 93.3425 |
| S1 | 9.1670 | 70.4576 | 53.5902 | 67.3931 | 90.6649 |
| S2 | 4.0368 | 68.9123 | 50.9453 | 65.6685 | 92.5494 |
| S3 | 3.4414 | 68.9391 | 51.0290 | 65.6642 | 92.8901 |
| S4 | 待测 | 待测 | 待测 | 待测 | 待测 |

以上统一使用百分点制表，计算时应以保存的原始小数为准，不得因为显示四舍五入产生错误。

另外生成：

- S4-S0 各项变化；
- S4-S1 各项变化；
- S4-S2 各项变化；
- S4-S3 各项变化；
- 20 类完整 AP50 差异；
- 20 类 AP75 差异；
- 20 类 Recall 差异。

不能只展示 S4 获得改善的类别。

---

# 十三、S4 成功标准

最主要的条件是：

**不能明显破坏 S0 已经实现的 Person 类别不可学习能力。**

预注册研究性门槛：

Person AP50_S4 <= 0.038993

即不比 S0 恶化超过 1 个百分点。

同时希望：

Non-target mAP50_S4 > 0.691698

也就是非目标平均 AP50 优于 S0。

在满足目标保护门槛的前提下，进一步比较：

- Non-target AP75；
- Non-target AP50-95；
- U；
- S；
- WCD；
- TCD3；
- Person→Dog 误检；
- Dog/Horse/Bicycle 检测表现。

需要区分：

### SUCCESS_CANDIDATE

Person 保持在预注册容差内，非目标 mAP50 高于 S0。

### TARGET_REGRESSION

Person AP50 超过容差。

### UTILITY_NOT_IMPROVED

Person 保持，但非目标 mAP50 未改善。

### MIXED_RESULTS

部分非目标指标改善，但整体改善不一致。

### INCOMPLETE

正式实验未完成或不能可靠评价。

这些状态只是 seed0 的实验结果分类，不代表统计显著性。

即使 S4 通过 SUCCESS_CANDIDATE，仍需后续独立多种子复现才能形成强学术结论。

---

# 十四、机制分析

除了最终 AP，必须回答：

1. TFSS 是否确实持续保留高 TSFL 目标特异性区域的扰动？
2. 是否持续减少高非目标风险、低 Qt 区域的扰动？
3. 实际平均扰动 RMS 相比 S0 降低多少？
4. Person 的目标保护损失轨迹是否与 S0 接近？
5. Non-target L_ret 是否比 S0 更低？
6. 是否出现部分类别改善、部分类别恶化？
7. Dog、Horse、Bicycle 哪些类别恢复？
8. Person→Dog 跨类别误检是否变化？
9. TFSS 是否只是整体减弱噪声？
10. 空间选择性短测优势是否转化为实际非目标 AP 改善？
11. 是否存在明显的目标保护与非目标效用权衡？
12. S4 是否优于 S0 的整体选择性？

注意：

短测中 S4 优于 Uniform 的空间能量分配，只能证明局部空间分配差异。

正式实验若有效，也不能仅凭一组 S4 与 S0 的 AP 变化证明全部改进来自特定空间归因。

报告中应区分：

- 源码确认事实；
- 训练日志证据；
- 实际检测性能；
- 机制推测；
- 尚未验证的因果解释。

---

# 十五、独立工作目录与资源安全

本轮代码仓库：

https://github.com/gql119/object-B

请在已有 TFSS 开发分支基础上创建独立 S4 正式实验分支。

建议：

codex/s4-tfss-utility-formal-20261017

禁止修改 main。

禁止覆盖 S0/S1/S2/S3 实验。

使用独立：

- Worktree；
- RunID；
- Config；
- Noise Output；
- Victim Output；
- Checkpoint；
- Log；
- Report Directory。

正式开始前检查：

- 服务器 GPU 是否有其他训练；
- 可用显存；
- 磁盘空间；
- 预计完整数据集占用；
- 是否存在冲突任务；
- 是否符合项目 AGENTS.md；
- 是否需要预先锁定可恢复的资源边界。

不能为了 S4 清理旧实验目录。

不能停止他人或本项目其他任务。

如资源不足，报告并等待，不得直接删除历史数据。

---

# 十六、自动监控与异常处理

正式运行需要监控：

- 当前 Epoch；
- 当前 Step；
- GPU Utilization；
- GPU Memory；
- Loss；
- Gradient；
- NaN/Inf；
- Checkpoint；
- Remaining Dataset Images；
- Data Integrity；
- Disk Free Space。

如出现：

- Loss NaN；
- 梯度非有限；
- 标签错位；
- 预算违规；
- 图像数据损坏；
- GT 对应错误；
- Resume 状态不一致；
- 模型更新异常；

立即停止当前阶段并保留证据。

如果出现普通运行中断：

只有在能够验证 checkpoint、optimizer、scheduler 和 RNG 完整一致的情况下，才能执行断点恢复。

不允许重新初始化后伪装成连续训练。

不允许为追求更好的结果而修改学习率、epoch、loss 或 TFSS 系数。

---

# 十七、报告与归档

建议输出：

docs/S4_TFSS_UTILITY/

包括：

1. S4_SOURCE_AUDIT.md
2. S4_CONFIG_LOCK.json
3. S4_PREFLIGHT_REPORT.md
4. S4_NOISE_TRAINING_REPORT.md
5. S4_NOISE_INTEGRITY.json
6. S4_TFSS_SPATIAL_STATS.csv
7. S4_VICTIM_TRAINING_REPORT.md
8. S4_LAST_EVALUATION.json
9. S4_BEST_EVALUATION.json
10. S4_VOC20_PER_CLASS.csv
11. S4_S0_S1_S2_S3_COMPARISON.csv
12. S4_SELECTIVITY_METRICS.json
13. S4_MECHANISM_ANALYSIS.md
14. S4_FINAL_REPORT.md
15. S4_ARTIFACT_MANIFEST.json

必须包含：

- Git SHA；
- Config；
- Seed；
- Checkpoint Hash；
- 数据集 Hash；
- 正式评价协议；
- 训练时长；
- GPU 资源；
- 运行完成状态；
- 全部错误及恢复记录。

完成后将本轮源码、配置与小型报告提交到 object-B 独立分支。

不要提交大型训练数据或模型权重。

如需要对 Git 进行任何可能覆盖历史的操作，必须停止并请求确认。

---

# 十八、最终执行顺序

请严格执行：

1. 核实 S0 原始代码；
2. 核实 TFSS-Utility 已验证实现；
3. 创建独立 S4 分支及工作目录；
4. 锁定配置；
5. 完成 S0/no-op 等价性检查；
6. 完成 TFSS 空间权重检查；
7. 完成 GPU smoke；
8. 完成预算、GT、标签正确性检查；
9. 通过 preflight；
10. 正式启动 24 epoch / 9,144 step 噪声优化；
11. 保存所有噪声训练日志与状态；
12. 导出完整 poisoned VOC20；
13. 完成完整数据审计；
14. 启动 random-init YOLOv8n victim；
15. 完整训练 200 epoch；
16. 保存 last.pt 和 best.pt；
17. 对 4,952 张 clean validation 执行正式评价；
18. 计算 VOC20 逐类别指标；
19. 计算 P/U/S、WCD、TCD3；
20. 与 S0/S1/S2/S3 统一比较；
21. 输出完整正式实验报告；
22. 提交代码及小型报告至独立 GitHub 分支；
23. 完成后停止，不自动启动其他实验。

---

# 十九、最重要的原则

本轮只验证一个研究假设：

**在不明显改变 Safe-Soft05 原有目标类保护机制的前提下，TFSS-Utility 能否通过更合理的空间扰动分配，提高非目标类别的正常检测能力？**

不能通过新增损失、调整 Adaptive、重新选择 TSFL 通道、改变 Soft Removal 或修改训练轮数来提高本轮结果。

TFSS-Utility 的正式参数已经由此前训练开发集短测确定。

本次完整实验必须使用冻结参数。

无论结果好坏，都必须真实记录。

如果 S4 没有改善 S0，也不代表短程空间归因计算错误；应当区分空间选择机制的正确性和最终不可学习效果的有效性。

请完成本次独立 S4 正式实验并提供完整结果，随后停止，等待进一步研究安排。
