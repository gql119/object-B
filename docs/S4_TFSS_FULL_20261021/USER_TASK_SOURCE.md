# Codex 任务：S4-TFSS-Full 全目标框空间调控消融实验

## 1. 研究目标

当前项目：

Class-Selective Unlearnable Examples for Object Detection

正式基线：

S4 — Safe-Soft05 + TFSS-Utility

本次需要建立一个独立消融版本：

**S4-TFSS-Full**

研究问题：

原始 S4 在 Person 与非目标 GT 重叠区域设置固定 0.5 区域系数，使重叠区扰动上限从 8/255 降为 4/255。

本次希望取消这一固定重叠区域减半机制，让整个 Person GT 框内均采用 8/255 基础扰动预算，并完全通过原有 TFSS-Utility 权重进行位置相关的扰动缩放。

需要验证：

1. 去掉 4/255 重叠区限制后，是否进一步降低 Person AP50。
2. TFSS 能否在没有重叠区固定减半机制的条件下，保持非目标类别的检测效用。
3. 性能变化是否主要来自重叠区域扰动强度的增加。
4. 哪些非目标类别受到的影响最大。

## 2. 修改原则

只改变 Person 与非目标框重叠区域的基础扰动系数。

原始 S4：

support =
  1.0：Person 独占区域
  0.5：Person 与非目标 GT 重叠区域
  0.0：Person 框外

新版本：

support =
  1.0：整个 Person GT union
  0.0：Person 框外

包括：

- Person 独占区域；
- Person 与 Dog 重叠区域；
- Person 与 Horse 重叠区域；
- Person 与 Bicycle 重叠区域；
- Person 与任意其他非目标 GT 重叠区域。

统一使用 binary Person GT support。

但必须保留原始 TFSS：

W_U = 1 - 0.15 * R_nt * (1 - Q_t)

delta_new = W_U * delta_base_full

不得改变：

- Q_t 的计算方式；
- R_nt 的计算方式；
- eta_U=0.85；
- TSFL；
- Safe-BG；
- L_BG；
- L_rel；
- L_ret；
- L_orth；
- Adaptive；
- Fourier 结构；
- JND；
- TV；
- 优化器；
- 训练轮数；
- 原始 GT；
- 原有模型权重。

不要为了增强差异，擅自把 eta_U 改成 0.5 或 0。

本次只研究取消重叠区域固定减半约束的影响。

## 3. 必须审查的代码位置

先审计：

- legacy_kproto_ret.py
- tfss_spatial.py
- tfss_clean_maps.py
- tfss_nt_sensitivity.py
- s4_tfss.py
- 正式 S4 配置
- 数据导出代码
- 数据与 Checkpoint 身份校验代码

特别注意：

原始 _bbox_half_support() 将重叠区域系数设置为 0.5。

不能只修改该函数就认为实验已经完成。

还需要检查：

1. 训练过程是否使用该 support。
2. 数据导出是否再次独立构造 support。
3. S4 是否存在 8/4/0 的硬编码预算校验。
4. 原有 TFSS 是否假设重叠区域一定等于 0.5。
5. overlap_linf、saturation_fraction 等诊断是否依赖 support==0.5。
6. 图像量化前最小幅度逻辑是否与重叠减半耦合。
7. Fourier 原始场、JND 和最终区域缩放的执行顺序。
8. GT ROI、上下文提取和正交损失是否不应受到 support 改变的额外影响。
9. 缓存、配置身份和导出元数据是否需要独立版本标识。

新版本不能再使用原始 8/4/0 规则作为合法性判断。

新的合法预算为：

- Person union 内：绝对扰动不超过 8/255；
- Person union 外：严格为 0。

但是仍然需要单独计算原有重叠区域的诊断指标，不能因为新 support 不再含有 0.5 就失去 overlap 统计。

重叠区域应根据真实 Person GT 与非目标 GT 的几何交集独立构建诊断掩码。

## 4. 独立实现，不覆盖原始 S4

建立新的实验标识：

S4_TFSS_FULL_OVERLAP

使用独立配置、数据目录、日志和结果目录。

保持正式 S4 所有数据、Fourier 参数、Checkpoint 和已有结果不变。

建议使用显式配置：

support_mode: bbox_full_target

不得通过覆盖原有 bbox_half_overlap 的语义实现新实验。

训练器、生成器、Manifest 和 Checkpoint 必须使用一致的新 support_mode。

不允许直接复用原始 S4 的最终加噪训练数据作为新实验输出。

如果某些干净图像归因缓存能够复用，必须先证明其对 support 的具体幅度值不敏感，且缓存身份有效；否则使用独立缓存。

## 5. 新版本扰动优化

使用与正式 S4 相同的：

- 教师模型；
- 初始 Fourier 参数策略；
- 随机种子；
- 原始训练集；
- 训练轮数；
- Batch；
- EOT；
- Loss；
- Optimizer；
- Scheduler。

重新优化 Fourier 参数。

唯一实验变量为：

Person 与非目标重叠区域的基础 support 从 0.5 改为 1.0。

如因实现需要修改基础支持函数，必须证明非重叠区域计算行为没有被额外改变。

## 6. 先进行科学正确性测试

在正式运行前，完成：

- 非重叠 Person 图像测试；
- Person + Dog 重叠测试；
- Person + Horse 重叠测试；
- 多个 Person 交叠测试；
- 多个非目标共同重叠测试；
- 无 Person 图像测试；
- 小目标和边界目标测试；
- Fourier 梯度回传测试；
- TFSS 缓存一致性测试；
- 数据导出与训练端一致性测试；
- uint8/PNG 量化后的扰动检查。

特别验证：

对于完全没有 Person—非目标重叠区域的样本，在相同 Fourier 参数、相同 EOT 和相同输入条件下，新旧 support 实现的输出应保持一致，允许正常浮点误差。

对于存在重叠区域的样本：

- 不得再施加固定 0.5 区域系数；
- 应保留原始 Q_t/R_nt 空间缩放；
- 不得超出 8/255；
- 框外仍然必须为零；
- 梯度必须正确传回 Fourier 参数。

输出独立的测试报告。

## 7. 先完成代码，避免干扰服务器现有任务

当前服务器仍在进行 S4 原有鲁棒性实验。

本次先完成：

1. 代码审计；
2. 独立实验实现；
3. 配置；
4. CPU 单测；
5. 低资源 Smoke Test；
6. 实验运行入口；
7. 评价脚本；
8. 指标统计脚本。

不得抢占正在运行的 GPU 训练。

如果当前服务器资源已经恢复空闲，但已有流水线尚未全部完成，仍然不能仅凭 GPU 空闲就启动本实验。

先报告原流水线状态，并确认后续任务队列是否已有其他计划。

本实验作为独立候选消融运行，不能覆盖或重排已启动的流水线。

## 8. 正式实验流程

当现有 S4 鲁棒性流水线完成验收，且后续调度不存在冲突时，启动：

第一阶段：

S4-TFSS-Full Fourier 扰动优化。

第二阶段：

导出新的完整 Poisoned VOC 训练集。

第三阶段：

运行数据正确性与扰动预算审计。

第四阶段：

使用与正式 S4 完全一致的 Victim 训练协议，训练随机初始化 YOLOv8n，200 epoch，seed=0。

第五阶段：

使用 last.pt@200 在未经修改的干净 VOC Validation 上进行正式评估。

best.pt 可作为次评价。

需要重新读取锁定 S4 Victim 配置，不能只根据记忆自行构造训练参数。

如果希望先进行短轮数实验，只能将其标注为 PILOT，不得直接用短轮数结果与 200 epoch S4 的正式结果进行优劣判断。

## 9. 评价指标

至少报告：

- mAP50_target；
- mAP50_non_target；
- mAP75_non_target；
- mAP50-95_non_target；
- mAP50_all；
- Selectivity S；
- WCD；
- TCD3；
- VOC20 per-class AP；
- Person Recall。

重点补充新的空间诊断：

- overlap_area_ratio；
- overlap_mean_abs_delta；
- overlap_RMS_delta；
- overlap_Linf_delta；
- non_overlap_RMS_delta；
- overlap_mean_W_U；
- overlap_mean_Q_t；
- overlap_mean_R_nt；
- Person 整体 RMS；
- PSNR；
- LPIPS（如项目已有对应实现）；
- 扰动区域面积；
- 不同 GT 重叠情况下的扰动强度分布。

如果某项指标没有现成合法实现，需要说明，不能伪造数据。

## 10. 核心对照

使用历史正式 S4 作为对照：

S4 Person AP50 = 0.028159

S4 Non-target mAP50 = 0.693787

S4 Selectivity S = 0.935330

这些只是历史参考值，正式对照必须先核对原始报告和评价协议。

输出：

S4 vs S4-TFSS-Full

包括：

- 目标类别检测能力变化；
- 非目标平均检测能力变化；
- 各非目标类别 AP 变化；
- 最差类别损伤变化；
- 重叠区域扰动增强程度；
- 可视化样例。

特别分析：

如果 Person AP 进一步降低，但非目标 AP 或 WCD 明显恶化，说明取消重叠减半虽然强化了目标攻击，却不能证明 TFSS 已充分替代原始非目标重叠保护。

如果 Person AP 下降且非目标保持改善，才支持完全基于 TFSS 的空间控制策略可能优于原有固定重叠限制。

## 11. 交付要求

现在先交付：

- 代码修改清单；
- 独立配置；
- 实验方法差异报告；
- 单元测试结果；
- 预算检查结果；
- 数据导出检查；
- 正式运行命令；
- 预计 GPU 时间；
- 当前实验队列状态；
- 独立结果目录；
- 是否已具备正式训练条件。

代码验证通过后再根据服务器流水线验收与任务队列状态启动正式训练。

不要只完成代码而没有测试，也不要因为当前阶段暂不允许 GPU 训练就擅自跳过科学正确性验证。

本次任务核心要求：

**取消所有 Person—非目标 GT 重叠区域的固定 4/255 上限，改为整个 Person GT union 统一 8/255 基础预算，再通过原有 TFSS-Utility 空间权重调节扰动强度。其他 S4 方法保持完全不变。**
