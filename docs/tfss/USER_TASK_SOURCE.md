# Codex 研究任务：TFSS-Conservative 与 TFSS-Utility 短程对照实验、代码归档及后续正式实验决策

## 一、任务总目标

当前项目研究类别选择性不可学习样本（Class-Selective Unlearnable Examples for Object Detection）。

数据集：VOC20。

目标类别：Person，class ID=14。

检测器：YOLOv8n。

我们已有正式方法：

S0 = Safe-Soft05（无 KD、无 E1-XFP、无 E2-Localization）。

目前：

- S0 Person AP50 = 0.028993；
- S0 Non-target mAP50 = 0.691698；
- S0 Non-target mAP75 ≈ 0.5159；
- Clean Person AP50 = 0.867943；
- Clean Non-target mAP50 = 0.766462。

已有 S1-CSB 实验：

- Person AP50 ≈ 0.0917；
- Non-target mAP50 ≈ 0.7046。

S1 虽然改善了非目标检测效用，却明显削弱了 Person 不可学习能力。

我们的新研究目标是：

**保留 S0 已经实现的强 Person 抑制效果，并在此约束下，通过更合理的空间扰动分配减少非目标类别的检测损伤。**

本轮重点研究：

1. TFSS-Conservative；
2. TFSS-Utility。

先开展固定训练开发集上的短程机制实验，再根据证据选择下一阶段候选。

当前 S3-CNLP-CUP 正式实验仍在运行。

**S3 拥有最高资源优先级。本轮不得中断、修改、抢占资源或影响 S3。正式完整扰动生成和 victim 训练必须等 S3 完成后，且经用户确认才允许启动。**

---

# 二、Phase 0：先将当前实验代码安全归档至 GitHub

目标仓库：

https://github.com/gql119/object-B

仓库默认分支：

main。

请先检查：

- 当前项目 Git 根目录；
- 当前分支；
- HEAD SHA；
- 已修改但未提交的文件；
- Untracked files；
- 当前 S3 使用的代码目录；
- S3 进程工作目录；
- S3 使用的 checkpoint、配置与运行目录；
- Git remote；
- Git 状态是否干净。

特别注意：

**S3 可能正在运行当前工作目录中的 Python 代码。不能通过 checkout、reset、rebase、文件覆盖或其他 Git 操作修改该工作目录。**

## 2.1 新建归档分支

建议归档分支：

codex/tfss-source-snapshot-20261016

此分支用于保存当前实验代码的可复现快照。

请使用安全的隔离副本或独立 Git worktree。

如果原目录存在未提交改动：

先进行只读审查，并将需要归档的源代码复制到隔离目录。

不能为了方便归档而直接修改 S3 的工作目录。

需要区分：

- 当前磁盘上的代码快照；
- S3 实际启动时对应的代码 SHA 和配置。

如果两者不同，必须分别记录，不得声称当前工作目录就是 S3 实际运行代码。

## 2.2 允许归档的内容

包括：

- Python 源代码；
- YAML 配置；
- 实验入口脚本；
- 数据加载器；
- 损失函数实现；
- Frozen/Adaptive 代码；
- Safe-BG；
- TSFL；
- CSB；
- CNLP；
- CUP；
- 评价代码；
- 必要测试脚本；
- 项目依赖记录；
- 小型研究说明文档。

不上传：

- VOC 原始数据；
- 完整 poisoned datasets；
- 大型模型权重；
- GPU 缓存；
- 大型训练日志；
- 200 epoch checkpoint；
- 压缩归档；
- 凭据、密钥、token；
- 包含隐私信息的本地配置。

必须检查 .gitignore 和待提交文件大小。

## 2.3 上传验证

将代码快照提交并推送到 object-B 的新分支。

禁止：

- 修改 main；
- Force Push；
- 覆盖已有正式分支；
- 删除历史提交；
- 合并其他开发分支；
- 修改 S3 工作目录。

完成后输出：

- GitHub 分支链接；
- Commit SHA；
- 已归档文件清单；
- 未归档大文件清单；
- 当前实验状态说明。

必须通过 GitHub 远端重新读取验证提交存在。

如果鉴权失败、分支冲突或无法确认源代码完整性，立即停止并报告，不得通过强制覆盖解决。

---

# 三、Phase 1：建立独立 TFSS 开发分支

在 object-B 中新建：

codex/tfss-conservative-utility-20261016

该分支的实际实验基础必须是历史正式 S0 Safe-Soft05。

注意：

**前一个归档分支用于保存当前工作状态，不意味着 TFSS 要从当前 S3 的代码状态开始开发。**

请定位 S0 的真实正式源码、配置及运行记录。

要求恢复：

- Frozen Reference；
- Adaptive Learner；
- TSFL；
- Safe-BG；
- L_BG；
- L_rel；
- L_ret；
- L_orth；
- 原始 Fourier 扰动参数化；
- 原始优化器；
- 原始 EOT；
- 原始空间缩放；
- 原始完整噪声生成日程。

检查确认：

- 没有 E1-XFP；
- 没有 E2-Localization；
- 没有 KD；
- 没有 CNLP；
- 没有 CUP；
- 没有 S1-CSB 逐像素硬裁剪；
- 没有其他历史实验的新增优化分支。

TSFL 应恢复正式实验的既有 P5 目标特异通道及索引。

Safe-BG 应恢复 rank=8、alpha=0.5、active-k=3 等正式配置。

具体数值一律以 S0 实际归档为准。

不要凭历史描述强行覆盖正式参数。

## 3.1 父版本等价性验证

恢复 S0 后，必须用固定 seed 和相同输入比较：

- L_BG；
- L_rel；
- L_ret；
- L_orth；
- Total loss；
- Fourier gradient；
- Spatial mask；
- Delta；
- 单步更新。

要求恢复代码与正式 S0 在合理数值误差内一致。

如果无法证明恢复的代码与 S0 等价，不能启动 TFSS 实验。

---

# 四、Phase 2：明确 TFSS 的设计思想

TFSS：

Target-Specific Feature-Guided Soft Spatial Scaling。

中文：

目标特异特征引导的平滑空间扰动缩放。

我们已经通过 TSFL 找到了对 Person 相对敏感、对非目标类别相对不敏感的特征通道。

因此希望进一步回答：

**能否将这些目标特异性特征映射到输入图像空间，在尽量保留 Person 有效保护信号的位置维持较高扰动，同时平滑削弱对非目标类别有害的扰动？**

TFSS 不重新选择特征层，也不重新设计背景误导损失。

TSFL：

决定哪些网络特征是目标特异性的。

TFSS：

决定这些特征所对应的输入空间中，不同位置的扰动应该缩放多少。

本轮重点不是增强目标攻击能力，而是：

**在 Person 保护能力接近 S0 的条件下提高 Non-target utility。**

---

# 五、Phase 3：固定 S0 原始扰动机制

设原始 Fourier 扰动为：

delta_base(p)

S0 的空间缩放抽象为：

delta_S0(p)
=
M_target(p) * a0(p) * delta_base(p)

其中：

a0(p)=1：

Person 独占区域。

a0(p)=0.5：

Person 与非目标 Bounding Box 重叠区域。

Person 外：

0。

具体实现需以正式 S0 为准，包括已有的上游幅度限制和裁剪顺序。

新 TFSS 不替换 Fourier 模块。

不改变：

- Fourier basis；
- Fourier 参数初始化；
- 优化目标；
- 优化器；
- Inner Loop；
- Outer Loop；
- EOT；
- Loss weights；
- Surrogate；
- Victim。

只在原空间缩放基础上引入新的连续权重。

delta_TFSS(p)
=
M_target(p)*a0(p)*W(p)*delta_base(p)

要求：

0 < W(p) <= 1

在 Person 区域外 W 的实际影响必须为零。

最终全局扰动上限仍为 8/255。

Person 与其他类重叠区域仍遵守 S0 原有的 4/255 局部限制。

不得重新引入 S1-CSB 的逐像素 Hard Clipping 作为主要缩放机制。

如果原 S0 本身存在全局 clip，应继续保留，不得误将全局预算安全裁剪与 S1 新增的局部硬裁剪混为一谈。

---

# 六、Phase 4：TFSS-Conservative 方案

## 6.1 设计动机

此前 TSFL-only 诊断中：

实际扰动 RMS 约为 S0 的 69.84%。

高保护区域能量保留约 60.42%。

保护梯度范数保留约 79%。

虽然相比等强度均匀缩放更能保留目标保护相关信息，但整体扰动仍下降约 30%。

我们担心这会再次削弱 Person 不可学习能力。

因此提出：

**TFSS-Conservative：保持 TSFL 引导的空间选择性，但减少额外扰动衰减。**

## 6.2 基础公式

通过已有 TSFL 通道计算输入归因：

Q_TSFL(p) ∈ [0,1]

定义：

W_C(p)
=
eta_C + (1-eta_C)*Q_TSFL(p)

建议初始：

eta_C=0.85

即：

W_C(p)
=
0.85+0.15*Q_TSFL(p)

这样最高归因区域权重接近 1，最低归因区域权重不低于 0.85。

它比此前 eta=0.5 的 TSFL-only 更保守。

最终：

delta_C(p)
=
delta_S0(p)*W_C(p)

注意：

这里 eta_C 是新 TFSS 的空间缩放参数，不是 Safe-BG alpha，也不是 S1-CSB 的预算参数。

## 6.3 核心假设

TFSS-Conservative 希望：

- 保留绝大多数 Person 保护性扰动；
- 在较低 TSFL 归因位置进行有限削弱；
- 避免硬裁剪导致的局部零梯度；
- 尽可能降低非目标特征漂移；
- 不显著恶化 Person 保护代理。

初版不使用中心渐弱。

不引入新检测损失。

不使用 S1 非目标梯度预算图。

---

# 七、Phase 5：TFSS-Utility 方案

## 7.1 设计动机

TFSS-Conservative 仅考虑：

某个像素对目标特异性通道的重要程度。

但是：

TSFL 高敏感不必然代表该位置完全不影响非目标。

因此我们提出：

**TFSS-Utility：同时考虑目标特异性信息和非目标检测敏感度，重点削弱“目标保护相关性较低、非目标风险较高”的空间位置。**

这里不是再建立一个新的 TSFL。

TSFL 已经完成目标特征选择。

TFSS-Utility 只是将现有 TSFL 的输入空间归因与非目标风险信息用于空间缩放。

## 7.2 两类空间信息

目标特异性空间归因：

Q_t(p)

来自：

已选 TSFL 通道对干净输入像素的归因。

非目标检测敏感度：

R_nt(p)

来自：

Frozen Reference 对真实非目标 GT 检测损失的输入梯度。

例如：

G_nt(p)
=
||∂L_det_nt/∂x(p)||_2

经过鲁棒归一化和平滑：

R_nt(p) ∈ [0,1]

注意：

Q_t 高表示局部特征敏感度较高。

R_nt 高表示非目标损失对该位置的输入变化较敏感。

二者均为代理指标，不等于最终 Person AP 和 Non-target AP 的因果贡献。

## 7.3 初始空间设计

建议先采用：

Risk(p)
=
R_nt(p)*(1-Q_t(p))

即：

只有非目标敏感度较高，且 TSFL 目标特异性相对较低时，风险分数才较高。

再定义：

W_U(p)
=
1-(1-eta_U)*Risk(p)

建议初始：

eta_U=0.85

得到：

W_U(p)
=
1-0.15*R_nt(p)*(1-Q_t(p))

最终：

delta_U(p)
=
delta_S0(p)*W_U(p)

该公式具有以下性质：

- R_nt 高、Q_t 低：适度降低扰动；
- R_nt 低、Q_t 高：接近保留原 S0 扰动；
- R_nt 高、Q_t 高：优先保留与目标特异性相关的位置，避免像 S1 一样单纯压低全部高非目标敏感区域；
- R_nt 低、Q_t 低：不主动削弱，因为缺少明确的非目标风险证据。

这里采用的是保守策略，不追求大范围降低噪声。

## 7.4 非目标风险处理

要求：

- 仅使用训练数据；
- Frozen Reference 不更新；
- GT 正确映射；
- 不使用 official clean validation；
- 正确处理没有非目标 GT 的图像；
- 检查敏感度极值；
- 检查少数高梯度点；
- 检查多 Person 重叠；
- 检查对象尺寸；
- 避免归一化造成权重塌缩。

没有非目标 GT 时：

R_nt=0

因此：

W_U=1

即退化为原始 S0 空间缩放。

如 Codex 认为该回退方式不合理，必须先提供数学解释和固定训练样本证据，不得静默替换。

## 7.5 方案的局限

当前公式是一个初始候选。

它并没有保证目标保护与非目标损伤完全解耦。

特别是：

目标与非目标均高度敏感的位置，仍然可能发生冲突。

因此必须通过实际梯度和特征诊断判断这个位置选择规则是否有效。

---

# 八、Phase 6：允许 Codex 提出更合理的设计调整

本轮允许 Codex 根据已有诊断和固定训练开发样本，对 TFSS-Conservative 和 TFSS-Utility 做适当改进。

允许改进：

- TSFL 归因的数值稳定性；
- 通道归因聚合；
- 多 Person 的空间权重合并；
- 空间平滑；
- 鲁棒归一化；
- 非目标风险图；
- 低样本或空 GT 处理；
- 空间权重映射；
- 归因图与原图坐标对齐；
- 梯度显存开销；
- 敏感度计算缓存。

允许考虑更高的最低缩放比例，例如：

eta=0.90。

但不得直接根据正式 validation AP 搜索 eta。

参数变更必须先在预先固定的训练开发样本上完成对照、记录和锁定。

如果修改核心公式，必须保留原候选版本，并将修改明确标注为新候选，不得悄悄替换。

每项实质调整必须说明：

1. 为什么修改；
2. 原设计是什么；
3. 新设计是什么；
4. 修改了哪个文件和函数；
5. 修改了哪些数学表达式；
6. 是否新增参数；
7. 对扰动强度产生什么影响；
8. 对目标梯度和非目标代理指标产生什么影响；
9. 是否破坏与 S0 的可比性；
10. 为什么该调整更符合研究目标。

不得为了获得更漂亮的短程指标而同时修改其他模块。

---

# 九、Phase 7：实验必须包含的对照

至少比较以下四种方法：

| Method | Spatial Strategy |
|---|---|
| S0 | 原 Safe-Soft05 固定缩放 |
| TFSS-C | Conservative TSFL Scaling |
| TFSS-U | Target/Non-target Aware Soft Scaling |
| Uniform-Matched | 与新方案实际噪声强度匹配的均匀缩放 |

Uniform-Matched 必须针对 TFSS-C、TFSS-U 分别构造相应对照。

匹配时需要区分：

- 平均绝对扰动；
- RMS；
- Person 区域扰动强度；
- 重叠区域扰动强度。

主匹配指标建议为相同有效 Person 支撑域内的实际扰动 RMS，其他指标辅助记录。

均匀缩放应沿用 S0 相同的预算和处理次序。

不能采用不同随机初始化或不同训练样本。

需要回答：

TFSS 的效果究竟来自更合理的位置选择，还是仅仅来自降低整体噪声？

---

# 十、Phase 8：建立固定短程实验协议

## 8.1 样本选择

优先复用此前 TSFL 空间诊断已经锁定的训练开发样本。

此前诊断规模：

32 张训练图像，65 个 Person GT。

如果需要扩大样本，必须在查看短测结果前确定新增样本清单。

建议覆盖：

- Person-only；
- Person+Dog；
- Person+Horse；
- Person+Bicycle；
- 高度重叠；
- 相邻但不重叠；
- 多 Person；
- Small Person；
- Large Person。

正式 clean validation 不参与参数选择。

## 8.2 保持所有模型条件一致

所有候选必须使用：

- 相同 S0 Frozen Reference；
- 相同 Adaptive 初始化；
- 相同 24 个 TSFL 通道；
- 相同 Safe-BG；
- 相同 Fourier 参数初始化；
- 相同优化器；
- 相同 EOT；
- 相同 Support；
- 相同 Query；
- 相同 inner steps；
- 相同 seed；
- 相同损失及权重。

唯一变化是空间权重 W。

## 8.3 短程优化

第一轮只运行最多 5 个实际扰动优化 step。

必须确保执行的是真实的 Fourier/Adaptive 优化链路，而不只是把固定扰动乘上新权重后计算一次 loss。

建议在多个固定 batch 上独立重复 5-step。

不要将前一候选的优化状态延续到下一候选。

不得生成完整 poisoned dataset。

不得启动 200 epoch victim。

如短测需要 GPU，严格遵守后面的 S3 资源隔离规则。

---

# 十一、Phase 9：短程诊断指标

## 9.1 目标保护指标

记录：

- L_BG；
- L_rel；
- Target suppression proxy；
- Target foreground confidence proxy；
- Protection gradient norm；
- Protection gradient cosine；
- High-protection-gradient energy retention；
- TSFL high-attribution region retention；
- Fourier parameter gradient；
- 5-step 内各项变化曲线。

重点判断：

TFSS 是否显著削弱原 S0 的背景误导机制。

## 9.2 非目标效用指标

记录：

- L_ret；
- Non-target feature drift；
- Non-target classification proxy；
- Non-target localization proxy；
- 非目标逐类别诊断；
- Dog/Horse/Bicycle 对象的特征变化；
- 非目标 GT positive matching 情况。

这些只是代理指标，不能将其变化直接宣称为最终 mAP 改善。

## 9.3 空间扰动指标

记录：

- Spatial Weight min/mean/max；
- Weight Q10/Q25/Q50/Q75/Q90；
- Mean Absolute Delta；
- Delta RMS；
- L∞；
- Effective Perturbation Area；
- Overlap Delta；
- Non-overlap Delta；
- High-TSFL-region Delta；
- High-NT-risk-region Delta；
- Gradient Retention；
- Weight Smoothness；
- 数值饱和比例。

必须记录真正的有效扰动强度，而不是仅记录理论权重。

## 9.4 训练机制指标

记录：

- Inner model update norm；
- Adaptive model state；
- Outer loss；
- Meta-gradient norm；
- Gradient cosine；
- NaN/Inf；
- GT matching；
- 目标梯度和非目标梯度的冲突情况。

不得把普通推理阶段 Person 分类梯度直接等同于 UE 训练动态梯度。

---

# 十二、Phase 10：短测决策逻辑

我们本轮最重要的要求是：

**保护 S0 的 Person 不可学习能力优先。**

正式结果中：

S0 Person AP50≈2.90%。

因此 TFSS 的目标不是进一步降低 Person AP，而是尽量维持 S0 的目标抑制能力，并减少非目标检测损伤。

短程代理无法直接输出最终 AP50，因此要使用代理准入，而不是声称短测能够证明最终性能。

建议先检查三个层面。

### Gate A：正确性

必须通过：

- 固定 S0 源码等价性；
- 正确 TSFL 通道；
- 正确空间坐标；
- 固定 Frozen/Adaptive；
- 正确预算；
- 不改变非目标 GT；
- 无 NaN/Inf；
- Fourier/Adaptive 梯度有效。

失败则停止。

### Gate B：目标保护保持

与 S0 相同条件下，检查：

- 目标背景误导代理是否明显退化；
- 保护梯度方向是否严重偏离；
- 高目标保护梯度位置是否仍保留大部分有效扰动；
- 5-step 后是否出现持续性的保护目标恶化。

本轮不预设某一个梯度比例能够保证最终 Person AP。

若采用数值门槛，必须在实验前固定，并说明其只是代理筛选阈值。

### Gate C：非目标效用改善

在通过目标保护准入的候选之间，比较：

- L_ret；
- 非目标特征漂移；
- 真实非目标 GT 检测损失代理；
- 各类别损伤；
- 与 Uniform-Matched 的差异；
- 非目标风险高区域的实际扰动减少程度。

如果某候选只是整体扰动更小，而且非目标改善不优于等强度均匀缩放，不应将其解释为特异性空间分配收益。

---

# 十三、Phase 11：给出最终候选建议

最终只需要推荐：

A. TFSS-Conservative

B. TFSS-Utility

C. 二者均不推荐，继续使用 S0

D. 当前短程证据不足

决策优先级：

第一：机制与预算正确。

第二：目标保护代理尽量接近 S0。

第三：非目标效用代理有改善。

第四：优于等强度均匀缩放。

第五：计算成本、额外参数及实现复杂度尽可能低。

如果 TFSS-C 与 TFSS-U 的结果十分接近：

优先选择机制更简单、需要额外超参数更少、运行成本更低的方案。

如果 TFSS-U 在非目标代理指标上稍好，但保护梯度明显退化：

不应仅凭平均非目标损失选择 TFSS-U。

如果两个方案都弱于 S0：

如实报告失败，不得通过连续修改候选参数制造成功结论。

---

# 十四、S3 实验资源隔离

当前 S3-CNLP-CUP 正在正式运行。

本轮必须遵守以下规则：

**S3 的 GPU、训练进程、工作目录、配置、checkpoint、训练日志和数据目录具有最高优先级。**

在 S3 运行期间，允许：

- 只读源码审计；
- GitHub 归档；
- 独立 Worktree 创建；
- TFSS 代码实现；
- 静态测试；
- 不影响 S3 的 CPU 单元测试；
- 小规模不占用共享训练资源的数值检查。

默认不允许在 S3 使用的 GPU 上运行 TFSS 短测。

只有确认存在完全独立的空闲 GPU、充足资源且不会影响 S3 时，才能在该 GPU 上执行轻量短测。

否则：

完成代码和 CPU 测试后进入 WAIT_FOR_S3。

不得通过降低 S3 优先级、暂停 S3、减小 S3 batch、抢占显存或修改 S3 的运行参数来为 TFSS 让出资源。

禁止：

- kill S3；
- 重新启动 S3；
- 重写 S3 的 Python 文件；
- 覆盖 S3 checkpoint；
- 修改当前训练环境依赖；
- 删除实验数据；
- 清理 S3 CUDA cache；
- 修改其他进程使用的文件；
- 共享不安全的输出目录；
- 在 S3 目录执行 Git checkout/reset。

如果 S3 状态不明，先核实，不得假设 S3 已结束。

## 14.1 S3 完成判定

必须核实：

- 完整噪声生成结束；
- 完整数据审计通过；
- victim 200/200 epoch；
- last.pt 保存；
- 正式评价完成；
- 指标归档完成；
- 训练进程退出。

只有日志显示结束但进程仍在运行，不代表可以立即抢占资源。

---

# 十五、正式实验的后续安排

本轮主要任务是短测与候选筛选。

S3 完成之后：

先完成尚未执行的 GPU 短程对照。

再依据短测报告，确定下一版 TFSS。

**在用户确认正式候选之后，才允许运行完整噪声生成和 victim 200 epoch。**

正式实验使用：

S0 作为父版本。

不使用 S1。

不使用 S2/S3 作为父版本。

保持：

- S0 Frozen；
- S0 Adaptive；
- S0 TSFL；
- S0 Safe-BG；
- S0 L_ret；
- S0 L_orth；
- 原 Fourier 参数化；
- 原 24 epoch / 9,144 step 生成日程；
- 相同 victim；
- 随机初始化；
- 200 epoch；
- last.pt 主评价；
- 相同 VOC20 clean validation。

如果最终需要比较多个 TFSS 方案，先向用户提交明确的正式实验矩阵，不要自动运行所有候选。

正式指标：

- Person AP50；
- Person Recall；
- Person AP75；
- Person AP50-95；
- Non-target mAP50；
- Non-target mAP75；
- Non-target mAP50-95；
- Non-target Recall；
- Non-target Precision；
- All-class mAP50；
- 全部 VOC20 逐类别 AP50；
- P/U/S；
- WCD；
- TCD3。

不得根据 official validation AP50 反复搜索 TFSS 参数。

---

# 十六、代码与报告归档

所有新代码只存放在独立 TFSS 分支。

建议生成：

docs/TFSS_SOURCE_SNAPSHOT.md

docs/TFSS_S0_PARENT_AUDIT.md

docs/TFSS_CONSERVATIVE_DESIGN.md

docs/TFSS_UTILITY_DESIGN.md

docs/TFSS_CHANGELOG.md

docs/TFSS_SHORT_PROBE_PROTOCOL.md

docs/TFSS_CORRECTNESS_AUDIT.md

docs/TFSS_SHORT_PROBE_RESULTS.md

docs/TFSS_CANDIDATE_DECISION.md

同时保存：

- 完整固定样本清单；
- JSON 配置；
- 逐样本指标 CSV；
- 梯度统计；
- 空间热图；
- 扰动图；
- 训练曲线；
- S0/TFSS-C/TFSS-U/Uniform 对照；
- Git SHA；
- 原模型 SHA；
- 环境版本；
- RNG seed；
- 全部实验变更。

所有实质修改必须明确标注：

BUG_FIX

NUMERICAL_STABILITY

EFFICIENCY_OPTIMIZATION

TFSS_DESIGN_CHANGE

不得将方法学调整伪装成普通代码修复。

---

# 十七、最终必须回答的问题

1. 当前实验代码是否已安全提交到 object-B 新分支？
2. GitHub 远端是否验证成功？
3. S3 是否完全不受影响？
4. TFSS 是否确实从原 S0 开发？
5. 原 S0 所有损失和优化器是否保持不变？
6. TFSS-C 是否保留了 S0 大部分有效保护梯度？
7. TFSS-U 是否比 TFSS-C 更好地减少非目标敏感区域的干扰？
8. 两种方法的实际扰动 RMS 分别下降多少？
9. 两种方法是否比对应的 Uniform-Matched 更有效？
10. TFSS-U 是否因为使用非目标风险图而增加明显计算成本？
11. 哪种方案的短程目标保护代理更接近 S0？
12. 哪种方案的非目标效用代理更好？
13. 这些差异是否跨样本稳定？
14. 是否存在某些类别得到改善、另一些类别反而恶化？
15. 哪个方案值得进入正式 200 epoch victim 实验？
16. 如果证据不足，还需要什么最低成本的补充诊断？

完成后给出：

RECOMMEND_TFSS_CONSERVATIVE

RECOMMEND_TFSS_UTILITY

KEEP_S0

或：

INSUFFICIENT_EVIDENCE

并说明理由。

---

# 十八、执行顺序与停止条件

严格按下面顺序执行：

1. 核对 S3 当前运行状态。
2. 检查 object-B 仓库。
3. 只读审计当前代码。
4. 创建隔离代码副本。
5. 上传当前实验代码快照至 object-B 新分支。
6. 验证 GitHub 提交。
7. 创建独立 TFSS 开发分支。
8. 恢复 S0 并通过等价性审计。
9. 实现 TFSS-Conservative。
10. 实现 TFSS-Utility。
11. 实现对应 Uniform-Matched 控制。
12. 完成 CPU 单元测试和静态分析。
13. 核实独立 GPU 资源是否可用。
14. 如不会影响 S3，执行小规模 GPU 5-step 对照；否则等待 S3 完成。
15. 完成全部短程机制诊断。
16. 选择最合理的 TFSS 正式候选。
17. 输出推荐方案及完整审计报告。
18. 等待用户确认。
19. S3 完成且用户批准后，才开始正式 TFSS 噪声生成与 victim 训练。

本轮不允许修改 S3，也不自动启动 TFSS 的完整正式实验。

**核心原则：目标类不可学习效果必须尽量维持 S0 水平；TFSS 的真正价值在于恢复非目标类别的检测性能，而不是通过普遍降低扰动强度来换取表面上的非目标改善。**
