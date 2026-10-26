# Codex 紧急任务：停止错误 COCO Fourier 优化，保全有效实验产物，全面审计并修复 S4 方法一致性 Bug

## 一、任务背景与最高优先级

当前研究项目：

Class-Selective Unlearnable Examples for Object Detection

当前实验：

Experiment A — Cross-Dataset Method Generalization

数据集：

COCO minitrain 25K / COCO80 / val2017 5K

正式方法：

Original S4 — Safe-Soft05 + TFSS-Utility

当前发现一个严重的科学实现错误：

**正在运行的 COCO Fourier 优化没有在训练计算路径中使用原始 S4 的 TFSS-U 空间权重，而是使用基础 B2SemanticTrainer 优化未经 TFSS 缩放的扰动。**

具体表现：

错误训练入口使用：

`EligibleCOCOB2Trainer(B2SemanticTrainer)`

正确入口应包含：

`S4Trainer(SoftSpatialProjection, B2SemanticTrainer)`

正确 S4 应当在每一次 Fourier 优化前向中使用：

W_U = 1 - 0.15 R_nt (1 - Q_t)

delta_S4 = W_U * delta_base

并让 Frozen 和 Adaptive 分支使用实际 TFSS 加权后的加噪图像计算损失。

**当前错误训练产生的 Fourier 参数不能作为正式 Original S4 实验结果。**

本次任务授权：

1. 立即停止错误的 Fourier 优化。
2. 防止调度器自动重启错误任务。
3. 保全所有有效前期实验成果。
4. 对整个 COCO Experiment A 进行端到端方法一致性审计。
5. 先生成详细 Bug 报告。
6. 再修复所有确认的代码问题。
7. 完成必要验证。
8. 暂不恢复正式 Fourier 优化，等待本次完整审计验收。

不需要因为普通代码审计和修复再次向我请求许可，但不能擅自改变 S4 的科学方法或超参数。

---

## 二、第一阶段：立即停止当前错误 Fourier 进程

这是最高优先级。

### 2.1 精确识别运行中的进程

首先检查：

- 当前 COCO Fourier Job ID；
- PID / PGID；
- 所属用户；
- 运行脚本；
- Python 可执行文件；
- Git Commit / Source SHA；
- 使用的配置文件；
- 输出目录；
- 调度器任务状态；
- 父子进程关系；
- GPU 占用情况。

必须确认正在运行的进程确实属于本次错误的 COCO Fourier 优化。

不得根据进程名称粗略匹配后批量终止。

### 2.2 停止错误任务

按以下顺序处理：

1. 通过现有串行控制器或任务管理机制设置当前 COCO Fourier 阶段为取消状态。
2. 禁止原有控制器自动重试或启动后续 Poisoned Export、Victim Training。
3. 对已经核实身份的错误 Fourier 进程发送正常终止信号 SIGTERM。
4. 等待进程正常退出，并核对退出状态。
5. 如果无法正常退出，再检查进程身份和剩余子进程。
6. 仅在确认所有权和任务范围后，必要时强制终止该错误任务的残留进程。
7. 确认没有同一错误任务的 GPU 子进程继续运行。
8. 确认不会被调度器再次拉起。

严禁使用：

`pkill -f python`

`killall python`

或任何可能影响其他用户、其他实验的全局终止命令。

如果当前控制器还管理其他有效任务，必须仅停止本次 COCO 错误 Fourier 阶段，不得粗暴杀掉共享控制器。

如果无法确定进程归属，应停止发送终止信号并报告，不得冒险操作。

### 2.3 停止后的状态记录

将本次错误 Fourier 任务标记为：

`STOPPED_SCIENTIFIC_INVALID`

原因：

`MISSING_TFSS_OPTIMIZATION_WIRING`

确保：

- 错误 Fourier 结果不进入正式数据导出；
- 不生成下游 READY；
- 不被用于 S4 Poisoned Victim；
- 不被登记为有效 Original S4；
- 不因旧有 native ACCEPTED 文件而绕过新的科学验收。

保留旧状态文件原件，通过独立的 superseding invalidation receipt 标记本次科学验收失效。

不得篡改历史日志或删除既有验收证据。

---

## 三、第二阶段：保留所有仍然有效的数据与校准成果

当前错误发生在 Fourier 优化计算路径，不意味着前面的所有数据都无效。

请逐项核实并保全：

### A. 数据集

- COCO minitrain 25K 原始训练图像；
- COCO80 训练标注；
- COCO val2017 5K；
- 官方 Image ID Manifest；
- COCO 类别映射；
- Clean 数据身份和 SHA256。

### B. Teacher

- 已训练完成的 YOLOv8n COCO80 Teacher；
- last.pt@200；
- Teacher 训练配置；
- Teacher 评价报告；
- Teacher Checkpoint Hash；
- 已验收的 CUDA 计算证据。

不得重新训练已经验收的 Teacher。

### C. 校准结果

- 完整 25K 干净训练图像校准统计；
- P5 Strict TSFL 39 通道选择结果；
- P3 独立正交特征坐标；
- 非目标类别语义子空间；
- Safe-BG 79 类投影数据及缓存；
- K=8 背景原型；
- 原始 Person 支持区域清单；
- 13,630 张已有有效 Person 支持的图像清单（以实际 Manifest 核验）；
- 所有身份与哈希记录。

这些成果只有在审计确认未受错误代码影响后才能标记为 REUSABLE。

不要直接假设全部可复用。

### D. 错误 Fourier 运行产物

将本次错误训练的以下内容单独封存：

- 当前 Fourier 参数；
- 训练曲线；
- 优化日志；
- Checkpoint；
- 运行配置；
- Source SHA；
- 相关梯度统计；
- 调度器状态；
- 实际运行的训练入口。

这些产物应标记为：

`INVALID_FOR_ORIGINAL_S4`

可以用于 Bug 分析，但不得用于最终论文结果。

不得删除，也不得把错误 Fourier 参数作为新修复训练的正式续训起点。

修复后的正式优化应按照原始初始化策略建立新的 Run ID。

完成保全后生成：

`COCO_S4_EMERGENCY_STOP_REPORT.md`

`COCO_S4_ARTIFACT_REUSE_MANIFEST.json`

---

## 四、第三阶段：全线方法一致性审计

**本阶段只读分析和运行隔离测试，不修改正式实验代码。**

不能只检查本次发现的 TFSS 接线问题。

请以已归档的 VOC Original S4 正式版本为方法基准，逐项对照当前 COCO80 版本的实际执行路径。

必须覆盖：

数据输入
→ Teacher
→ TSFL
→ 背景原型
→ Safe-BG
→ P3 Orth/Retention
→ Fourier 扰动
→ TFSS
→ Frozen/Adaptive
→ 外层梯度更新
→ 参数保存
→ Poisoned 数据导出
→ Victim 训练
→ 干净验证集评价。

重点检查以下内容。

### 4.1 训练器继承与入口

检查：

- 当前实际运行的训练器类型；
- S4Trainer；
- B2SemanticTrainer；
- SoftSpatialProjection；
- 训练器 MRO；
- `_compose_delta_batched()` 的实际调用顺序；
- `_batch_loss()` 的实际调用顺序；
- Frozen 与 Adaptive 对加噪图像的实际引用；
- 是否存在绕过原始 S4 的替代训练入口。

必须证明：

每一个正式 Fourier EOT draw 都使用完整 S4 计算路径。

不能只检查代码文件里是否存在 `S4Trainer` 字样。

### 4.2 TFSS 空间控制

检查：

Q_t、R_nt、W_U 的计算与原始公式完全一致。

固定：

eta_U = 0.85

检查：

- Q_t 是否来自正确的目标特征响应能量输入梯度；
- R_nt 是否来自非目标 GT 对齐检测损失；
- 是否使用干净训练图像计算；
- 归因图归一化是否一致；
- CleanMapCache 是否正确绑定 Teacher 与图像身份；
- 缓存是否使用正确图像坐标；
- TFSS 权重是否在优化中真正应用；
- EOT 每次抽样是否一致地应用空间权重；
- 权重是否保持 detached；
- Frozen 分支是否使用真实加权图像；
- Adaptive 内循环是否使用真实加权图像；
- 梯度是否通过 TFSS 后的扰动正确传播至 Fourier 参数；
- 导出时是否使用与训练相同的 TFSS 公式及空间坐标。

必须提供真实训练器前向与梯度证据，而不仅是配置文件截图。

### 4.3 空间支持和扰动预算

Original S4 必须保持：

Person-only：8/255

Person ∩ Non-target：4/255

Outside Person：0

检查：

- GT support 构造；
- 重叠区域掩码；
- 多实例重叠；
- JND；
- Fourier 扰动合成；
- TFSS 缩放；
- 裁剪；
- uint8 量化；
- PNG 保存；
- 导出后重新读取。

确保没有混入 S4-TFSS-Full 的统一 8/255 规则。

### 4.4 TSFL 与特征坐标

检查：

- COCO80 的 Person ID=0；
- P5 Strict TSFL；
- 39 个选择通道；
- 所有通道索引；
- 通道敏感度统计；
- P3 独立 48 通道正交坐标；
- P5/P3 的独立性；
- GT 与特征池化对齐；
- 小目标处理；
- 是否错误复用了 VOC20 的通道索引；
- 是否出现隐藏的固定 24D 或 Person14 假设。

COCO39D 属于经过记录的维度兼容性适配，不能擅自改回 24D。

### 4.5 Safe-BG

检查：

- 79 个非目标类别完整性；
- active top-3 选择；
- 非目标投影比例；
- rank=8；
- removal_alpha=0.5；
- P5 39D 坐标；
- SVD / 正交基；
- 范数恢复；
- 按需缓存；
- 不同类别顺序的结果；
- 投影矩阵有效秩；
- 是否存在非目标类别静默遗漏；
- Safe-BG 修正目标是否真正进入 L_BG 和 L_rel。

必须证明 Safe-BG 没有被意外替换成原始未经修正的背景方向。

### 4.6 原始损失函数

逐项审计：

- L_BG；
- L_rel；
- L_orth；
- L_ret；
- L_A,t；
- L_A,nt；
- L_TV。

检查：

- 损失实际输入；
- 正负号与优化方向；
- 损失权重；
- 特征层；
- GT 实例对齐；
- 实例/图像平均方式；
- 类别权重；
- Frozen/Adaptive 分支归属；
- stop-gradient；
- 数值稳定性；
- 梯度是否真正进入 Fourier 参数。

必须防止出现：

损失函数存在于代码中，但没有参与外层优化；

或指标记录了一项损失，但该损失使用的是错误图像或错误特征。

### 4.7 Adaptive 虚拟学习

检查：

- inner_steps=1；
- inner_lr=1e-4；
- 正确的初始 Teacher 权重；
- Poisoned 训练输入；
- 同批干净对应图像；
- 可微参数更新；
- 目标学习反馈；
- 非目标保持反馈；
- 外层梯度传播；
- 是否意外永久修改 Teacher；
- 是否出现图像加权前后不一致。

### 4.8 Fourier 优化

检查：

- 64 个候选频率基；
- 最终保存 16 个活动频率基；
- Fourier 系数优化；
- 频率选择；
- 原始 EOT；
- Batch=16；
- 24 个优化 epoch；
- 优化器；
- 学习率；
- 调度；
- RNG 保存恢复；
- Checkpoint；
- 梯度统计；
- 训练图像采样顺序；
- 13,630 张 Person 支持样本与实际迭代计划是否一致。

注意：

64×3 是优化时的候选参数形状；

16×3 是最终选择并保存的参数形状。

两者不同属于原始方法设计，不是需要修复的参数维度错误。

### 4.9 Poisoned 数据导出

全面检查：

- 正式 Fourier 参数加载；
- 导出时的 TFSS；
- 训练/导出归因图一致性；
- Person support；
- 原始图像与加噪图像对应；
- 非 Person 图像保持干净；
- COCO80 全部标签保留；
- 原图尺寸；
- GT 坐标；
- PNG 量化；
- 扰动预算；
- 数据成员数量；
- 导出文件 SHA256。

尤其检查正式导出是否使用训练时真正优化的同一种图像构造机制。

### 4.10 Victim 与最终评价

虽然当前尚未进入 Victim 阶段，也必须提前审计：

- Clean200 与 Poisoned200 配置一致；
- YOLOv8n 80 类；
- 随机初始化；
- 原始 SGD 配置；
- 200 epoch；
- seed=0；
- imgsz=640；
- batch=36；
- workers=16；
- 不加载 Teacher 权重初始化 Victim；
- 使用正确 Clean/Poisoned 数据；
- 使用未经扰动的完整 val2017；
- COCO category_id 映射；
- COCO 官方 AP；
- Person AP50；
- Non-target mAP50；
- Selectivity S；
- 79 类逐类别 AP；
- best.pt 与 last.pt 的区分。

不得将 Teacher 的 AP 直接当作 Clean200 基线。

---

## 五、审计必须先形成正式 Bug 报告

这是强制的执行分界点。

在开始修复任何确认的错误之前，必须先生成：

`COCO_S4_FULL_PIPELINE_BUG_AUDIT_BEFORE_FIX.md`

`COCO_S4_FULL_PIPELINE_BUG_AUDIT.json`

每一项发现应包括：

1. Bug ID；
2. 严重等级；
3. 所在文件和具体函数；
4. 原始 VOC S4 预期行为；
5. COCO 当前实际行为；
6. 不一致的根本原因；
7. 是否影响损失或梯度；
8. 是否影响最终 Poisoned 数据；
9. 是否影响 Person AP 或非目标 AP 的解释；
10. 证据、最小复现和测试结果；
11. 修复方案；
12. 修复是否可能改变科学方法；
13. 是否需要重新计算已有校准；
14. 是否影响下游实验。

明确区分：

- Confirmed Bug；
- Suspected Bug；
- Intentional COCO Compatibility Change；
- Performance Optimization；
- No Issue。

不能把尚未验证的怀疑当作已确认 Bug。

必须在修复之前保存当前代码与审计证据的版本快照。

审计报告完成后，允许继续执行下一阶段修复，不必再次等待我确认。

---

## 六、第四阶段：隔离修复所有已确认 Bug

审计报告完成后，在独立修复分支中开始修改代码。

不得直接修改正在运行的旧进程使用的不可变源码。

修复优先级：

P0：影响实际 S4 方法定义的错误。

P1：影响损失、梯度或数据输出的错误。

P2：影响复现性、评价和结果记录的错误。

P3：不影响科学正确性的工程问题。

本次首先修复所有 P0/P1，再处理必要的 P2。

对于涉及科学方法变化的情况，不允许自行引入新方法或改变原始超参数，应保留阻塞并说明。

### TFSS 已确认 Bug 的修复要求

优先恢复完整的：

`S4Trainer + SoftSpatialProjection`

以及正确的：

- CleanMapCache；
- TFSS-U；
- EOT 加权图像；
- Frozen Loss；
- Adaptive Loss；
- Fourier 真实梯度；
- 训练与导出计算一致性。

当前 GitHub 已有候选修复：

Commit `433a9dd35198bc5c007b849a34a6f4b42fbb81fc`

请先检查其实际内容、测试范围与当前服务器已部署代码是否一致。

不得因为候选提交存在，就认为 Bug 已经修复完成。

此前的 Toy CPU 测试仅作为组件证据，不能替代真实 Teacher 和 CUDA 训练验证。

如果候选修复满足要求，可以复用；如果存在其他 Bug，应在同一隔离修复分支中统一处理。

不要叠加多个互相覆盖的临时热修补。

---

## 七、修复后的验证要求

### 第一级：静态与配置检查

- 所有相关模块可以导入；
- 配置通过；
- 无过期硬编码；
- 正式 COCO80 类别映射正确；
- 原始 S4 损失、预算与优化器保持一致。

### 第二级：CPU 数学和单元测试

- TFSS 权重有效；
- 权重真正改变前向扰动；
- 权重真正改变 Fourier 梯度；
- Safe-BG 投影合法；
- 24D VOC 回归通过；
- 39D COCO 兼容通过；
- P3 Orth 独立；
- 训练与导出一致；
- 错误 B2-only 路径被验收器拒绝。

### 第三级：真实 Teacher 端到端测试

必须使用真实 COCO80 Teacher 和实际 COCO 训练图像，不能仅用合成张量。

检查：

- TSFL；
- Safe-BG；
- TFSS；
- Frozen；
- Adaptive；
- Fourier 梯度；
- 原始支持预算。

### 第四级：有界 GPU Smoke Test

仅在没有其他 GPU 冲突时运行。

使用真实 YOLOv8n Teacher、正式训练器和完整损失，执行有限的实际优化步。

必须采集：

- TFSS 图像与缓存来源；
- W_U 实际范围和分布；
- 加权前后的真实扰动差异；
- Frozen 实际输入；
- Adaptive 实际输入；
- L_BG 到 Fourier 的梯度；
- L_rel 到 Fourier 的梯度；
- 总 Fourier 梯度；
- CUDA 上实际运算张量的设备证据；
- 8/4/0 预算；
- Teacher 权重未改变的证据。

不能只用 `torch.cuda.is_available()` 证明 GPU 计算正常。

### 第五级：完整流水线的无训练模拟

验证：

Fourier
→ 参数保存
→ Poisoned Export
→ Clean Victim
→ S4 Victim
→ Official Eval

所有阶段的：

- 输入身份；
- 输出 Schema；
- Hash；
- 依赖；
- READY；
- ACCEPTED；
- RunID；

保持一致。

重点防止训练通过，但在导出或评价阶段才发现参数格式和标签错误。

---

## 八、修复后的正式实验恢复策略

本轮任务只要求：

**停止错误训练 → 保全数据 → 全面审计 → 报告 → 修复 → 验证。**

暂时不要自动重新开始完整 24 epoch Fourier 训练。

修复完成后将新版本设为：

`REPAIRED_AWAITING_FORMAL_RELAUNCH`

必须保留：

- 错误运行的独立身份；
- 修复版本的新 Commit；
- 新配置 Hash；
- Teacher Hash；
- 校准 Hash；
- 39D 兼容身份；
- 完整测试报告。

新的正式 Fourier 运行应使用独立 RunID，并从原始初始化协议开始。

不能直接从错误训练的 Fourier Checkpoint 续训。

只有后续明确恢复正式实验时，才能建立新的有效优化结果。

Clean200 如有独立于错误扰动生成的正常训练成果，可以经审计后复用；Poisoned200 必须使用修复后完整 S4 生成的数据。

---

## 九、GitHub 与实验产物管理

保持以下版本彼此独立：

1. 原始 VOC20 S4；
2. COCO80 维度兼容版本；
3. 当前错误 Fourier 版本；
4. 本次审计修复版本。

建议新建：

`codex/s4-coco-full-pipeline-bugfix-20261026`

所有修复代码和测试报告提交到新分支。

正常 Push 至 GitHub。

禁止：

- force push；
- 覆盖历史 Commit；
- 删除错误实验日志；
- 覆盖 Teacher；
- 覆盖校准；
- 删除旧 Fourier 参数；
- 上传完整图像、毒化数据和大型权重到 GitHub；
- 修改其他正在运行的实验任务。

如果 GitHub 暂时无法连接，应保留本地提交与待推送记录，不得声称已成功发布。

---

## 十、最终交付内容

完成本次任务后至少输出：

1. `COCO_S4_EMERGENCY_STOP_REPORT.md`
2. `COCO_S4_ARTIFACT_REUSE_MANIFEST.json`
3. `COCO_S4_FULL_PIPELINE_BUG_AUDIT_BEFORE_FIX.md`
4. `COCO_S4_FULL_PIPELINE_BUG_AUDIT.json`
5. `COCO_S4_BUG_FIX_CHANGELOG.md`
6. `COCO_S4_METHOD_PARITY_REPORT.md`
7. `COCO_S4_END_TO_END_TEST_REPORT.md`
8. `COCO_S4_RECOVERY_READINESS.json`

最终向我报告：

- 错误 Fourier 训练是否已完全停止；
- 是否仍有错误任务自动重启风险；
- 保留了哪些有效数据和校准结果；
- 全线发现了多少个确认 Bug；
- 每个 Bug 的根因；
- 每个 Bug 的修复情况；
- 哪些模块通过了真实 CPU/GPU 测试；
- 是否保持原始 S4 数学方法不变；
- GitHub 修复分支及 Commit；
- 是否已经具备重新启动正式 Fourier 优化的条件。

不得因为所有单元测试通过，就跳过真实 Teacher 和 GPU 端到端验证。

---

## 十一、最终强制原则

**第一，立即停止错误 Fourier 优化，不继续消耗 GPU 资源生成科学上不合格的结果。**

**第二，保留并核验全部仍可复用的 Teacher、校准、背景原型、语义子空间及数据成果，不重复完成有效阶段。**

**第三，不只修复 TFSS 一个已知问题，而是以原始 VOC S4 为基准，对 COCO80 的整个计算、优化、导出和评价链路进行方法一致性审计。**

**第四，发现其他 Bug 必须先形成完整报告，再集中修复。**

**第五，保留 Original S4 的 Person-only 8/255、重叠 4/255、框外 0、TFSS eta=0.85，以及全部原始损失和优化协议。**

**第六，修复和验证完成后先汇报，不自动重启正式 Fourier 训练。**

现在立即执行：精准停止错误训练、保全数据、生成完整 Bug 审计报告，然后修复和验证。
