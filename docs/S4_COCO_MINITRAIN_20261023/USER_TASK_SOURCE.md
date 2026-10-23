# Codex 任务：原始 S4 跨数据集方法泛化实验（VOC20 → COCO minitrain 25K / COCO80）

## 一、任务目标

当前研究项目：

**Class-Selective Unlearnable Examples for Object Detection**

研究目标：让指定目标类别 Person 变得不可学习，同时尽可能保持其他类别的正常目标检测能力。

本次新增：

**Experiment A — Cross-Dataset Method Generalization**

实验数据集：

**COCO minitrain · 25,000 Training Images · 80 Detection Classes**

正式评估集：

**COCO val2017 · 5,000 Images**

本次必须继续使用原始 S4：

**S4 — Safe-Soft05 + TFSS-Utility**

尤其需要强调：

### 本次实验的父线是原始 S4，不是 S4-TFSS-Full

原始 S4 扰动支持规则：

- Person 独占区域：8/255；
- Person 与任意非目标 GT 框重叠区域：4/255；
- Person 框外：0。

保留原始重叠区域减半机制。

不得将 Full 版本中整个 Person 框统一 8/255 的规则引入本次实验。

本次只验证：

**原始 S4 方法从 VOC20 扩展到 COCO80 后，是否仍然能够实现目标类别不可学习与非目标类别效用保持。**

不要引入新的神经网络扰动生成器，不要重新设计方法，不要改进现有损失函数。

---

# 二、执行顺序：必须先上传当前实验代码到 GitHub

这是本任务的第一优先级。

在下载 COCO 数据、修改 S4 或开始 COCO 适配前，先完成现有实验代码的版本归档。

## Step 1：审计当前项目

只读检查服务器上正在使用的实际项目目录，确认：

1. Git 仓库根目录和 remote。
2. 当前分支、HEAD Commit。
3. 当前未提交的代码变更。
4. 是否存在未推送到 GitHub 的实验实现。
5. 当前正在运行的 S4 / Full / 鲁棒性流水线。
6. 正式 S4 的配置和实现文件。
7. 当前实验结果对应的实际代码版本。

主要 GitHub 仓库：

https://github.com/gql119/object-B

必须区分：

- 正式原始 S4；
- S4-TFSS-Full；
- S5；
- Cutout/CutMix/MixUp/OP 开发分支；
- 其他正在执行的实验。

不要把 Full 或 S5 的方法改动误当作原始 S4。

## Step 2：备份并推送当前实验代码

将当前服务器实际存在、且尚未归档的实验代码整理为 GitHub 快照。

建议创建：

`codex/current-experiments-snapshot-20261023`

以实际 Git 状态决定安全的分支和提交方式。

要求：

- 归档当前必要的 Python 源码；
- 归档实验配置模板；
- 归档运行入口；
- 归档指标计算和评价脚本；
- 归档必要的依赖版本说明；
- 保存可复现的实验协议说明；
- 对路径敏感的配置进行脱敏；
- 保存实际运行配置的文件清单和校验信息。

不要上传：

- VOC 或 COCO 原始图像；
- 毒化训练数据；
- 大型模型权重；
- Checkpoint；
- 大型实验日志；
- Python 环境目录；
- 缓存；
- VPN 密钥、Token、密码或其他敏感文件。

如果某些大型产物已有独立存储位置，只需在 Manifest 中记录其路径、文件身份和 Hash。

如果存在未提交变更，不得直接丢弃。

优先在隔离的 Git worktree 或完整项目快照目录中完成归档，避免影响正在运行的实验。

禁止：

- `git reset --hard`；
- `git clean -fd`；
- `git push --force`；
- 覆盖他人提交；
- 未经检查直接提交全部未跟踪文件；
- 为归档而修改正在执行的脚本。

完成 GitHub Push 后，必须验证远程 Commit 已真实存在，并输出：

- GitHub 仓库地址；
- Snapshot 分支；
- Commit SHA；
- 文件清单；
- Push 成功证据；
- 未归档文件及原因。

**只有完成当前实验代码的 GitHub 归档，才允许继续 COCO 适配开发。**

如果 GitHub Push 失败，不要假称上传成功，也不要绕过该前置步骤。

---

# 三、创建独立 COCO 实验分支

完成归档后，从已核实的**正式原始 S4 版本**创建独立开发分支。

建议命名：

`codex/s4-coco-minitrain25k-generalization`

不允许从 S4-TFSS-Full 的修改版本继承其重叠区域规则。

不允许直接修改现有正式 S4 的实验目录。

本次 COCO 实验所有新代码、配置、数据与结果使用独立目录。

原始 VOC20 S4 必须保持完全可复现。

---

# 四、数据集必须严格来自 COCO minitrain 官方发布划分

## 4.1 唯一指定的数据集

官方作者仓库：

https://github.com/giddyyupp/coco-minitrain

论文：

HoughNet: Integrating near and long-range evidence for bottom-up object detection，ECCV 2020。

必须阅读作者仓库：

- README.md；
- STATS.md；
- 官方 Object Detection & Segmentation 下载说明；
- 官方 JSON 标注文件；
- 官方类别定义。

本次使用：

**作者已经发布的 COCO minitrain 25K 固定划分。**

不得自行重新随机采样 25,000 张图像。

不得使用：

- COCO128；
- COCO8；
- 任意 Hugging Face 随机 COCO 子集；
- 自定义 COCO10K；
- COCO Keypoint 15K 子集；
- 其他同名 Mini-COCO；
- 完整 COCO train2017 直接替代 minitrain。

必须是官方作者发布的 Object Detection & Segmentation 25K Image IDs。

## 4.2 官方文件优先级

首先从 GitHub README 提供的官方 JSON 链接获取固定划分标注。

优先使用作者发布的 COCO 格式 JSON。

若无法访问官方原始文件，应尝试 GitHub README 中作者认可的备用链接。

使用镜像文件时，必须能够证明 image_id、annotation_id 和对应内容与官方划分一致。

如果无法证明镜像与作者固定划分一致，停止数据准备并报告原因。

**不允许使用 `sample_coco.py` 重新抽样来冒充原始官方划分。**

COCO 原始图像应使用 COCO2017 的 train2017 图像文件，并根据官方 minitrain Image ID 精确选择对应图像。

正式验证集：

- 原始 COCO val2017；
- 5,000 张图像；
- 原始 instances_val2017.json；
- 不得使用任何训练图像代替验证图像。

## 4.3 数据正确性审核

必须确认：

1. 训练集恰好 25,000 个唯一 Image ID。
2. 所有图像来自 COCO train2017。
3. 训练集包含 80 个检测类别。
4. 类别名称与 COCO 官方类别顺序一致。
5. 实例数量与作者公布的约 184K 一致，记录精确统计。
6. 训练集与 val2017 没有重合的 Image ID。
7. 无缺失图像。
8. 所有图像宽高合法。
9. 所有 bbox 合法。
10. `iscrowd` 和忽略标注的处理规则清楚、可复现。
11. 所有类别 ID 映射正确。
12. 转换前后目标类别及实例数量可核查。
13. 无 Person 图像也必须保留在训练集内，不得因为目标类别不存在而被全部删除。
14. 经过标签转换后，训练图像总数仍为 25,000。

必须输出：

`COCO_MINITRAIN_DATA_AUDIT.md`

`COCO_MINITRAIN_MANIFEST.json`

`COCO_CATEGORY_MAPPING.json`

以及类别、目标尺寸、Person 共现比例等统计。

所有数据文件应保存在符合当前服务器规范的静态数据目录下，不覆盖已有 VOC20 文件。

---

# 五、实验定义：仅进行 Cross-Dataset Method Generalization

本次不是 VOC 扰动直接迁移实验。

因此允许：

- 在 COCO minitrain 上构建新的背景原型；
- 在 COCO minitrain 上重新估计目标特异性特征通道；
- 在 COCO minitrain 上建立非目标类别语义子空间；
- 在 COCO minitrain 上计算 TFSS 所需的归因图；
- 使用 COCO80 教师重新优化 S4 共享 Fourier 参数；
- 导出 COCO minitrain 25K 的 S4 Poisoned 训练图像。

但是：

**必须保留原始 S4 的算法、公式、主要超参数及优化目标。**

重新估计数据集相关统计不视为修改算法。

不能因为 COCO 数据更复杂，就擅自重新设计空间控制、Safe-BG、正交损失、Adaptive 或 Fourier 生成结构。

本次不训练新的 U-Net、GAN、Diffusion 或其他扰动生成网络。

---

# 六、原始 S4 方法必须保持不变

## 6.1 扰动支持与预算

原始规则：

Person-only：

8/255

Person ∩ Non-target：

4/255

Outside Person：

0

即：

M(u,v) =
  1.0, Person-only
  0.5, Person ∩ Non-target
  0.0, Outside Person

这里 Non-target 指 COCO80 中除 Person 外的其余 79 类。

无论与几个非目标框重叠，原始 S4 的有效支持规则必须保持一致，不得重复乘多个 0.5。

## 6.2 TFSS-Utility

保留：

W_U = 1 - 0.15 R_nt(1-Q_t)

其中：

eta_U = 0.85

保留：

- Q_t 目标特征能量归因；
- R_nt 基于非目标真实 GT 的检测损失敏感度；
- 干净训练图像上的归因计算；
- 相同归一化机制；
- 同一 TFSS 空间缩放逻辑。

不得替换为新空间权重公式。

不得使用 Full 版本的统一 8/255 支持规则。

## 6.3 Target-Specific Feature Localization

保留原始 Strict TSFL 规则。

原始 S4 在 P5 使用目标特异性通道。

应在 COCO80 下按照原有筛选算法重新选择特征通道。

不能直接复制 VOC 中选出的通道索引，并假设在 COCO 教师中仍然具有相同语义。

不能在没有实验依据的情况下更换特征层。

如果严格筛选无法得到原始协议需要的有效通道数量，应报告问题，而不是偷偷降低筛选标准。

## 6.4 Contextual Background Misdirection

保留：

- K=8 背景原型；
- 原始背景原型匹配机制；
- 原始背景误导方向定义；
- L_BG；
- L_rel；
- 原始损失权重；
- 原有上下文范围与归一化机制。

背景原型需要使用 COCO minitrain 的干净训练图像重新构建。

不得使用 COCO 验证集构建背景原型。

## 6.5 Safe-BG

保持：

- 原始选定特征空间；
- Top-3 active non-target classes；
- Low-rank projection；
- rank=8；
- removal_alpha=0.5；
- 原始范数恢复；
- 不新增 Safe-BG Loss。

从 19 个非目标类别扩展到 79 个类别时，允许改造投影缓存的数据结构和计算方式。

但必须保持 Safe-BG 的数学定义、类别冲突排序与数值语义不变。

## 6.6 L_orth 与 L_ret

保留：

- P3 正交位移约束；
- P3 非目标特征保持；
- 原始 L_orth；
- 原始 L_ret；
- 原始 Top-3 冲突类别选择；
- 原始类别重叠加权；
- 原始损失权重。

不得加入 S5 CUP。

不得引入新的逐类别动态权重。

不得更换正交目标函数定义。

## 6.7 Adaptive

保留：

- 原始 Frozen / Adaptive 双分支；
- 一次可微虚拟更新；
- 相同学习率；
- 相同目标学习反馈；
- 相同非目标特征保持反馈；
- 相同外层损失权重。

不得把单步更新改为多步。

不得引入与原始代码不一致的独立 Support/Query 数据集。

## 6.8 Fourier

保留：

- 原始共享 Fourier 参数化；
- 原始频率候选与选择机制；
- 原始 EOT；
- 原始优化器；
- 原始优化轮数；
- 原始 JND；
- 原始 TV 正则；
- 相同随机种子原则。

COCO 上的 Fourier 参数必须重新优化。

不得直接使用 VOC 训练得到的最终 Fourier 参数作为 Experiment A 的正式结果。

---

# 七、COCO80 兼容性改造

这部分是本次开发的重点。

## 7.1 类别映射

VOC20：

Person class ID = 14

COCO80 的连续 YOLO 类别编码：

Person class ID = 0

注意原始 COCO JSON 的 category_id 与连续 YOLO 类别索引不同。

必须使用完整的 COCO 官方 80 类映射。

全项目检查所有：

- target_class_id；
- num_classes；
- `cls == 14`；
- `cls < 20`；
- 19 个非目标类别；
- VOC 类别名称数组；
- 固定长度分类统计；
- VOC 类别索引；
- 检测头类别数量；
- class-wise AP。

原则上采用配置驱动：

target_class_id = 0

num_classes = 80

non_target_classes = all COCO classes except Person

但不允许因此破坏原始 VOC20 代码行为。

## 7.2 Safe-BG 子空间扩展

原始 Safe-BG 的三类别缓存面向 19 个非目标类别。

COCO80 有 79 个非目标类别。

原始缓存实现可能不适合直接扩展。

需要：

1. 审计现有缓存逻辑。
2. 估算 79 类时的时间、内存和磁盘开销。
3. 在保持原始数学公式的前提下设计按需缓存。
4. 验证不同调用顺序下结果一致。
5. 验证 rank=8、alpha=0.5。
6. 验证 active top-3 类别选择。
7. 验证特征坐标和数据类型。
8. 使用 VOC19 的旧实现进行数值等价测试。

不要通过降低 rank、减少类别、改变权重或移除 Safe-BG 来解决性能问题。

COCO 各类别子空间样本数量可能不均衡，必须报告有效样本量和秩。

如果某类无法建立有效子空间，应将其视为兼容性阻塞，不要静默跳过。

## 7.3 TFSS 扩展

保留 Q_t 和 R_nt 原有计算公式。

针对 COCO 需要测试：

- Person ID=0；
- 79 个非目标类别；
- 多 Person 实例；
- 无 Person 图像；
- 多类别共现；
- 多个重叠框；
- 密集目标；
- 小目标；
- 高宽比不同的图片；
- 归因图和原图像素坐标的一致性。

R_nt 必须仍然是基于 COCO 非目标 GT 的检测敏感度，而不是把目标类或其他损失错误混进去。

归因缓存只可以利用 COCO minitrain 的干净训练图像和正式教师权重。

## 7.4 图像尺寸与坐标

当前 S4 使用 YOLOv8 输入尺寸 imgsz=640。

COCO 原始图像可能具有不同分辨率与纵横比。

需要保留原始等比例预处理与填充语义。

审计：

- 图像尺寸读取；
- bbox 坐标转换；
- LetterBox；
- padding；
- P3/P5 特征对齐；
- 像素级 TFSS 权重；
- 原始坐标支持区域；
- Fourier 扰动合成；
- PNG/图像量化；
- 最终数据导出。

禁止把所有 COCO 图像先强行非等比例压缩成 640×640 再冒充原始 S4 处理流程。

## 7.5 数据导出与训练标签

导出的 Poisoned 数据必须仍然包含全部 80 类检测 GT。

只允许 Person 区域承载目标保护扰动。

不得删除非目标类别的 GT。

不得修改 bbox。

不得把无 Person 图像错误丢弃。

需要验证：

- Clean/Poisoned 一一对应；
- Image IDs 一致；
- 类别标签一致；
- Bounding Boxes 一致；
- 图像尺寸一致；
- 扰动仅作用于规定区域；
- 重叠区不超过 4/255；
- Person-only 不超过 8/255；
- Person 框外严格为零；
- 导出文件重新读取后仍符合预算。

预算审计时应区分浮点优化空间和 uint8 图像存储带来的量化行为。

---

# 八、Teacher 和 Victim 训练协议

## 8.1 Teacher

必须先确定一套与 COCO80 匹配的 YOLOv8n Teacher。

原始 VOC20 Teacher 不能直接用作 COCO80 正式教师。

优先采用：

在 COCO minitrain 25K 的干净训练数据上训练 COCO80 Teacher。

禁止使用 COCO val2017 的标注或图像进行 Teacher 参数更新、TSFL 校准、背景原型构建或扰动优化。

如果想使用已在完整 COCO train2017 上预训练的公开模型作为 Teacher，必须明确报告这会引入 minitrain 之外的额外训练数据，并将其列为不同实验协议；不得悄悄混入当前严格 25K 实验。

正式 Teacher 的训练方案和权重身份必须保存。

## 8.2 Victim

分别训练：

**Clean COCO minitrain Victim**

**S4 Poisoned COCO minitrain Victim**

使用相同：

- YOLOv8n；
- 随机初始化协议；
- 训练轮数；
- Batch；
- Workers；
- 输入分辨率；
- Optimizer；
- LR Scheduler；
- AMP；
- 数据增强；
- 随机种子；
- 验证集。

优先沿用 VOC 主实验锁定的：

epochs=200

imgsz=640

seed=0

batch=36

workers=16

实际配置先从已归档的原始 S4 运行记录中核对，不能凭空重建。

如果资源原因必须改动 Batch 等训练参数，Clean 与 S4 Victim 必须同时采用相同的新设置，并在跨数据集实验报告中记录原因。

正式评价使用：

last.pt@200

best.pt 仅作为辅助结果。

---

# 九、实验分阶段执行

## Stage 0：GitHub 归档

完成当前实验源码 GitHub Snapshot。

远程 Push 成功之前，不进入后续阶段。

## Stage 1：COCO minitrain 数据审计

下载官方固定 25K JSON。

获取精确对应的 25K 图像和完整 val2017 5K。

完成数据一致性测试与 Manifest。

不要启动 GPU 训练。

## Stage 2：COCO80 兼容性开发

在独立分支修改类别映射、数据加载、语义子空间、TFSS 缓存及相关组件。

严格保持原始 S4 方法不变。

完成 CPU 单元测试。

## Stage 3：兼容性验收

重点测试：

- VOC20 原始行为未回归；
- COCO80 的正确类别映射；
- 原始 S4 8/4/0 空间支持；
- Safe-BG 数学等价；
- TFSS 梯度与坐标；
- 80 类标签；
- Clean/Poisoned 一一对应；
- 导出预算审计；
- 各类别 AP 计算。

必须通过后才能进入训练。

## Stage 4：Teacher

资源允许且不影响已有流水线后，训练正式 COCO80 Teacher。

保存完整权重、配置和校验信息。

## Stage 5：S4 扰动优化

在 COCO minitrain 上：

- 重建背景原型；
- 重新选择目标特异性通道；
- 重建 79 个非目标语义子空间；
- 计算 TFSS 归因；
- 重新优化共享 Fourier 参数；
- 导出 25K Poisoned 训练数据。

## Stage 6：Victim

训练：

Clean Victim

S4 Victim

均完整执行 200 epoch。

## Stage 7：统一评价

使用未经扰动的完整 COCO val2017 5K 评价。

生成最终实验报告。

---

# 十、服务器运行与资源保护

当前服务器可能仍存在：

- 原始 S4 鲁棒性实验；
- S4-TFSS-Full；
- Cutout/CutMix/MixUp/OP 后继队列。

不得因为本次 COCO 项目中断它们。

开发和 CPU 单测可以立即执行。

下载数据之前先检查：

- 磁盘剩余空间；
- 项目目录；
- 数据存储规范；
- 当前服务器 I/O；
- 文件权限；
- 是否已有完整 COCO 数据可以安全复用。

如果正在运行 GPU 实验，不得抢占同一 GPU。

COCO 的 Teacher、Fourier 优化与 Victim 训练必须独立排队。

不要同时部署多个互相冲突的 GPU 调度器。

如果现有服务器调度器支持安全依赖队列，则接入现有机制；否则只准备独立的待运行配置，不能破坏原有队列。

所有 COCO 结果使用独立的 RunID、目录、日志和配置 Hash。

---

# 十一、正式评价指标

核心指标：

1. mAP50_target（Person）
2. mAP50_non_target（79 类平均）
3. Selectivity S
4. mAP50_all
5. mAP50-95_all
6. mAP50-95_target
7. mAP50-95_non_target
8. mAP75_non_target
9. VOC20 与 COCO80 共有非目标类别的 AP
10. COCO 新增非目标类别的 AP

必须保存 COCO80 全部逐类别 AP50 与 AP50-95。

原有选择性指标继续使用：

P = 1 - AP50_target(S4) / AP50_target(Clean)

U = mAP50_non_target(S4) / mAP50_non_target(Clean)

S = 2PU / (P+U)

边界情况按项目原始锁定计算协议处理。

进一步记录：

- WCD；
- TCD3；
- Person Recall；
- PSNR；
- LPIPS；
- 扰动区域比例；
- 重叠区扰动 RMS；
- 重叠区扰动 L∞；
- TFSS 权重分布；
- Safe-BG 方向保留率；
- 正交损失；
- 非目标保持损失。

注意：

VOC20 与 COCO80 的 mAP 绝对值不能直接视为公平的性能高低比较。

跨数据集主要比较每个数据集内部 Clean 与 Poisoned 的性能差异及 Selectivity。

正式 COCO AP 评估应确保 COCO category_id、忽略规则与官方评价协议一致。

---

# 十二、输出报告

至少生成：

1. `CURRENT_EXPERIMENT_GITHUB_SNAPSHOT.md`
2. `COCO_MINITRAIN_DATA_AUDIT.md`
3. `COCO_MINITRAIN_MANIFEST.json`
4. `COCO_CATEGORY_MAPPING.json`
5. `S4_VOC_TO_COCO_COMPATIBILITY_AUDIT.md`
6. `S4_COCO_METHOD_PARITY.md`
7. `S4_COCO_TEST_REPORT.md`
8. `S4_COCO_EXPERIMENT_CONFIG.yaml`
9. `S4_COCO_RESULTS.csv`
10. `S4_COCO_PER_CLASS_AP.csv`
11. `S4_COCO_FINAL_REPORT.md`

其中 `S4_COCO_METHOD_PARITY.md` 必须逐项比较：

原始 VOC20 S4

与

COCO80 适配 S4

确认核心算法公式、目标、预算和优化协议没有被改变。

如果某项实现无法在 COCO80 下保持数学等价，必须明确记录问题，禁止静默替换为新方法。

---

# 十三、GitHub 最终提交

完成 COCO80 兼容性开发与测试后，将新代码推送到独立 GitHub 分支。

必须保留两项不同的 Git 历史身份：

**A. 原始当前实验代码 Snapshot**

用于保存开发前的真实代码版本。

**B. COCO80 Adaptation Branch**

用于保存本次跨数据集的新增兼容性修改。

提交时报告：

- Snapshot Commit；
- S4 父线 Commit；
- COCO 适配 Commit；
- 实际 Diff；
- 测试结果；
- 兼容性变化；
- 配置变化；
- 方法未改变的证据。

禁止直接覆盖 main。

禁止覆盖 VOC20 的原始代码与结果。

---

# 十四、最终强制要求

本次实验必须满足：

**1. 严格使用 HoughNet 作者发布的 COCO minitrain 25K 固定图像划分和完整 COCO80 类别。**

**2. 父线必须是原始 S4，而非 S4-TFSS-Full。**

**3. 保留 Person-only 8/255、Person 与非目标重叠 4/255、Person 外 0 的原始预算机制。**

**4. TFSS、TSFL、Safe-BG、L_BG、L_rel、L_orth、L_ret、Adaptive 与 Fourier 原始核心方法均保持不变。**

**5. 在 COCO minitrain 上重新优化 S4，不引入额外神经网络扰动生成器。**

**6. 使用 COCO minitrain 25K 构建 Clean/Poisoned 配对训练，在完整 COCO val2017 5K 上评价。**

**7. 必须先将当前实验代码成功归档并推送到 GitHub，再进行任何 COCO 代码修改。**

**8. 当前服务器已有实验不能受到干扰。**

请立即开始 GitHub 归档、数据集审计及 COCO80 兼容性开发。完成各阶段必须留下可审计证据，遇到方法学不等价、数据划分无法验证或运行环境冲突时停止对应阶段，不得自行改变原始 S4 算法来绕过问题。
