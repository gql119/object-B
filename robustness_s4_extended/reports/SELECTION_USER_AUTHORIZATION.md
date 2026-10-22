# Codex 任务：根据 S4-TFSS-Full 实验结果自动选择最优方法，并继续四项鲁棒性实验

## 一、总体目标

当前研究项目：

Class-Selective Unlearnable Examples for Object Detection

当前正式方法：

S4 — Safe-Soft05 + TFSS-Utility

候选改进版本：

S4-TFSS-Full

Full 版本取消了 Person 与非目标 GT 重叠区域固定 4/255 的扰动上限，改为整个 Person GT 区域统一 8/255 基础预算，由 TFSS-Utility 决定不同位置的扰动缩放。

本次任务：

**在 Full 正式实验结束后，自动比较 S4 与 Full 的类别选择性效果，选择效果更好的版本，随后使用该版本完成新增的 Cutout、CutMix、MixUp、Orthogonal Projection（OP）四项鲁棒性实验。**

不再要求 Full 效果更好时暂停并等待人工决定。

只要 Full 通过正式验收且满足预设优势标准，就自动将 Full 选为本轮新增鲁棒性实验的使用方法。

---

## 二、最高优先级规则

### 情况 A：Full 优于 S4

自动选择：

**SELECTED_METHOD = S4_TFSS_FULL**

随后使用 Full 正式导出的 Poisoned VOC 训练集开展：

1. Full + Cutout
2. Full + CutMix
3. Full + MixUp
4. Full + Orthogonal Projection（须已完成 OP 协议审批）

每项实验分别训练匹配的 Clean Victim 和 Full Poisoned Victim。

不得继续使用原始 S4 的 Poisoned 数据进行这四项新增实验。

### 情况 B：Full 未证明优于 S4

自动选择：

**SELECTED_METHOD = S4_ORIGINAL**

随后使用原始 S4 正式 Poisoned VOC 训练集开展：

1. S4 + Cutout
2. S4 + CutMix
3. S4 + MixUp
4. S4 + Orthogonal Projection（须已完成 OP 协议审批）

保留 Full 所有实验结果，作为独立消融。

### 情况 C：Full 实验异常或结果无效

设置：

**SELECTION_STATUS = BLOCKED_FULL_VALIDATION**

不得因为 Full 训练失败、指标缺失或实验中断，就把它判定为效果差于 S4。

必须保留错误信息和实验现场，不自动进入新的鲁棒性训练。

---

## 三、Full 和 S4 的比较标准

首先读取两个版本的正式实验结果。

必须核查：

- Fourier 优化是否完整完成；
- Poisoned VOC 数据集是否完整导出；
- Victim 是否完整训练 200 epoch；
- 是否使用相同 YOLOv8n 训练配置；
- 是否使用相同随机种子；
- 是否使用 last.pt@200；
- 是否在相同干净 VOC Validation 上评价；
- VOC20 逐类别 AP 是否完整；
- Dataset Manifest、Checkpoint、Config Hash 是否通过校验。

正式比较指标：

1. Person AP50（越低越好）
2. Non-target mAP50（越高越好）
3. Selectivity S（越高越好）
4. WCD（越低越好）
5. TCD3
6. Non-target mAP75
7. 每个非目标类别的 AP50

原始 S4 历史参考值：

Person AP50 = 0.028159

Non-target mAP50 = 0.693787

Selectivity S = 0.935330

WCD = 0.295951

这些数字仅供核对，正式比较必须读取原始结果文件。

### 预设方法选择规则

以 Selectivity S 为主要选择指标，同时控制目标抑制和非目标效用。

只有满足以下全部条件时，判定 Full 更好：

- Full 的 S 高于 S4，且超过固定数值比较容差；
- Full 的 Person AP50 不高于 S4；
- Full 的 Non-target mAP50 不低于 S4；
- Full 的 WCD 不高于 S4；
- 两组评估协议完全一致。

容差必须在读取 Full 结果前确定并写入配置，不允许事后调整。

如果 Full 仅提升目标攻击强度，但明显损害非目标效用，不能仅凭 Person AP50 下降就判定 Full 更好。

如果 Full 的优势不满足上述全部条件，则默认选择原始 S4。

注意：这是本轮自动调度采用的保守方法选择标准，不代表单种子实验具有统计显著性。

---

## 四、自动选择方法及数据集

完成正式比较后生成：

S4_FULL_COMPARISON.md

S4_FULL_COMPARISON.csv

S4_FULL_DECISION.json

其中 Decision JSON 必须包含：

- selected_method；
- selected_poison_dataset；
- selected_dataset_manifest；
- selected_dataset_sha256；
- comparison_metrics；
- decision_rule_version；
- selection_reason；
- decision_timestamp；
- validation_status。

后续四项鲁棒性实验必须统一从 Decision JSON 读取方法和数据身份。

**禁止在 Cutout、CutMix、MixUp、OP 的各自脚本中手动硬编码 S4 或 Full。**

所有后续实验应使用同一个选定版本。

不得出现：

- Cutout 使用 Full；
- CutMix 使用 S4；
- MixUp 又使用 Full；

这种不同 Poisoned 方法混用的情况。

方法选择结果一旦用于启动正式训练，应固定整个新增鲁棒性实验批次。任何重新选择都必须产生新的 Batch ID 和独立结果目录，不得悄悄覆盖已经运行的结果。

---

## 五、四项新增鲁棒性实验

新增实验：

Cutout、CutMix、MixUp、OP。

参考：

https://github.com/hala64/mi-ue

OP：

https://github.com/psandovalsegura/learn-from-unlearnable

### Cutout

使用适配目标检测的 Cutout。

必须保证：

- 遮挡区域和概率可复现；
- 多实例 GT 标签合理；
- 不发生错误的类别标签丢失；
- Clean 与 Poisoned 使用相同增强参数；
- 不修改原始训练集文件。

### CutMix

优先复用当前 Ultralytics 支持的检测版 CutMix。

必须保证 bbox 和 class labels 正确转换。

禁止直接复制分类版本的 CutMix 标签混合逻辑。

### MixUp

优先使用当前 Ultralytics 原生检测版 MixUp。

必须核对图像混合、GT 合并、类别标签及与基础数据增强之间的关系。

### Orthogonal Projection

OP 与当前 S4 中的 L_orth 不是同一方法。

OP 的原始实现面向单标签分类，不能直接套用到 VOC 多实例目标检测。

只有在 OP 检测适配方案明确获批、代码测试通过后，才能执行正式 OP 实验。

如 OP 尚未通过协议审批：

OP_STATUS = BLOCKED_PROTOCOL_APPROVAL

前三项可以继续完成，不得因此停滞。

同时，不允许擅自采用未经批准的 OP 实现启动训练。

---

## 六、Clean 与 Poisoned 的配对方式

根据 SELECTED_METHOD 自动选择对应 Poisoned 数据。

### 如果选择 Full

运行：

- Clean + Cutout / Full + Cutout
- Clean + CutMix / Full + CutMix
- Clean + MixUp / Full + MixUp
- Clean + OP / Full + OP

### 如果选择 S4

运行：

- Clean + Cutout / S4 + Cutout
- Clean + CutMix / S4 + CutMix
- Clean + MixUp / S4 + MixUp
- Clean + OP / S4 + OP

必须确保：

- 相同 YOLOv8n 网络；
- 相同训练配置；
- 相同初始化协议；
- 相同 Epoch；
- 相同 Batch；
- 相同数据增强设置；
- 相同随机种子；
- 相同验证集；
- 相同指标计算协议。

如已有完全匹配且身份可验证的 Clean+Defense 结果，可以在确认配置、数据、权重和评价协议一致后复用；否则重新训练匹配的 Clean 对照。

严禁复用不匹配的历史 Clean 结果作为 Selectivity S 的计算分母。

所有正式评价均使用未经处理的干净 VOC Validation。

主评价权重：

last.pt@200

---

## 七、服务器调度规则

当前服务器可能同时存在：

1. 原有 S4 Phase A / Phase B 鲁棒性流水线；
2. S4-TFSS-Full 实验；
3. 新增四项鲁棒性实验的待运行队列。

不得停止或抢占已有实验。

新增实验必须满足：

- 原有 Phase A / Phase B 已 COMPLETED_AND_ACCEPTED；
- Full 实验已完成正式验收；
- S4 和 Full 已完成自动比较；
- SELECTED_METHOD 已写入并锁定；
- 新增实验代码通过测试；
- 当前没有 GPU 训练冲突；
- 数据与结果目录完全隔离。

必须使用服务器端持久化调度机制。

不能依赖 Codex 对话、VPN、SSH 或本地电脑持续在线。

### 自动流水线

FULL_ACCEPTED
→ COMPARE_S4_VS_FULL
→ SELECT_BEST_METHOD
→ LOCK_SELECTED_DATASET
→ VERIFY_EXISTING_ROBUSTNESS
→ VERIFY_NEW_DEFENSE_CODE
→ CUTOUT_CLEAN
→ CUTOUT_POISON
→ CUTMIX_CLEAN
→ CUTMIX_POISON
→ MIXUP_CLEAN
→ MIXUP_POISON
→ OP_APPROVAL_GATE
→ OP_CLEAN / OP_POISON（仅在获批后）
→ FINAL_REPORT

如果 Full 不优于 S4，自动使用 S4。

如果 Full 优于 S4，自动使用 Full。

这两种情况均不需要我再次确认方法选择。

如果 Full 实验结果无效，则停止方法判定与后续实验。

如果 OP 尚未审批，则前三项实验照常完成，OP 维持阻塞状态。

---

## 八、训练、验收与异常恢复

每个防御实验必须：

1. 执行 GPU Smoke Test。
2. 运行匹配的 Clean Victim。
3. 运行选定方法的 Poisoned Victim。
4. 完成 200 epoch。
5. 检查 last.pt@200。
6. 在干净验证集上评估。
7. 计算 VOC20 各类别 AP。
8. 计算 Selectivity S。
9. 验证 Dataset Manifest 和 Checkpoint Hash。
10. 保存日志和结果。
11. 标记 ACCEPTED。
12. 自动进入下一项。

禁止同时启动多个竞争同一 GPU 的训练任务。

不得修改已经完成的历史实验。

不得在正式实验运行失败时自动改变训练超参数掩盖问题。

支持阶段恢复，但不得重复执行已通过验收的训练。

---

## 九、最终评价指标

每项防御至少记录：

- mAP50_target；
- mAP50_non_target；
- mAP75_non_target；
- mAP50-95_non_target；
- mAP50_all；
- Person Recall；
- VOC20 per-class AP；
- Selectivity S；
- WCD；
- TCD3。

计算：

P_D = 1 - AP50_target(Poison+D) / AP50_target(Clean+D)

U_D = AP50_non_target(Poison+D) / AP50_non_target(Clean+D)

S_D = 2 P_D U_D / (P_D + U_D)

边界情况使用项目既有锁定指标协议。

所有报告中必须明确标注：

Selected Method: S4 或 S4-TFSS-Full

不能仅写 Poisoned 而不说明使用哪个版本。

如果选择 Full，应额外保留 Full 与原始 S4 的非目标逐类别差异和空间扰动预算审计结果。

---

## 十、最终结果与文件

需要输出：

S4_FULL_COMPARISON.md

S4_FULL_DECISION.json

EXTENDED_ROBUSTNESS_RESULTS.csv

EXTENDED_ROBUSTNESS_PER_CLASS.csv

EXTENDED_ROBUSTNESS_REPORT.md

PIPELINE_STATUS.json

结果报告需要说明：

1. Full 是否优于 S4。
2. 自动选择了哪个方法。
3. 为什么选择该方法。
4. Cutout/CutMix/MixUp/OP 的执行状态。
5. 每项 Clean 与 Poisoned 的指标。
6. 各非目标类别 AP。
7. 所有数据、配置及权重的 Hash。
8. 尚未完成的阶段和阻塞原因。

如果 OP 未获批准，应输出部分完成报告，明确标记 OP 未运行，不允许伪造四项均已完成的结果。

---

## 十一、现在的执行要求

请现在开始：

1. 只读检查现有实验状态。
2. 检查 Full 是否已经完成。
3. 检查新增四类实验代码是否已经实现。
4. 编写 S4 与 Full 自动比较脚本。
5. 实现最优方法自动选择逻辑。
6. 将选定方法的数据集身份传递给后续四项实验。
7. 修改或接入已有后继任务队列。
8. 模拟 Full 优于 S4、Full 不优于 S4、Full 实验失败三种情况。
9. 测试是否会错误切换数据集或重复执行实验。
10. 部署服务器端自动调度。
11. 报告当前任务队列和状态查询命令。

**不要立即抢占 GPU，也不要干扰当前正在运行的实验。**

如果已有调度器已经具备对应功能，应优先增量修改，不要重复部署多个自动控制器。

## 十二、最终原则

**Full 效果更好 → 自动使用 Full 完成后续四项鲁棒性实验。**

**Full 未证明更好 → 自动使用原始 S4 完成后续四项鲁棒性实验。**

**Full 实验异常 → 暂停并保留问题，不允许错误回退。**

方法选择完成后，整个新增鲁棒性实验批次必须使用同一个 Poisoned 方法与数据集。

不再要求在 Full 优于 S4 时暂停等待人工确认。

OP 仍然保留独立的检测协议审批门槛。

保证所有实验可复现、可审计、可恢复，并且不影响已有服务器任务。
