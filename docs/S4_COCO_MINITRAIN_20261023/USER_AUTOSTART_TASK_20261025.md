# Codex 任务：COCO minitrain 测试完成后自动启动 S4 正式跨数据集实验

## 一、任务目标

当前研究项目：

**Class-Selective Unlearnable Examples for Object Detection**

本次实验：

**Experiment A — Cross-Dataset Method Generalization**

数据集：

- COCO minitrain：25,000 张训练图像
- COCO80：80 个目标检测类别
- COCO val2017：5,000 张验证图像
- Target：Person，YOLO class ID=0
- Non-target：其余 79 个类别

正式方法：

**Original S4 — Safe-Soft05 + TFSS-Utility**

必须保留原始 S4 扰动预算：

- Person 独占区域：8/255
- Person 与非目标 GT 重叠区域：4/255
- Person 框外：0

**禁止使用 S4-TFSS-Full 的全目标框 8/255 版本。**

本次指令的核心要求：

**当前 COCO minitrain 相关代码适配、数据审核和测试全部完成且通过验收后，自动继续正式实验，无须再次等待我确认。**

不要重复已经完成的测试、下载或数据准备工作。

---

## 二、先检查当前进度

请读取当前 COCO 实验真实状态，包括：

1. COCO minitrain 25K 数据准备状态。
2. COCO val2017 5K 数据准备状态。
3. 官方标注与类别映射验证情况。
4. COCO80 兼容性代码修改情况。
5. Safe-BG 79 类子空间适配情况。
6. TSFL 与 TFSS 兼容性测试情况。
7. 原始 S4 8/4/0 预算验证情况。
8. 已完成的 CPU/GPU Smoke Test。
9. 当前 Git 提交与 GitHub 备份状态。
10. 现有 GPU 训练进程和调度队列。

如果某项已完成并且结果可验证，直接复用，不要重复执行。

如果某项测试仍在执行，让其正常完成。

---

## 三、正式训练的自动启动条件

只有以下条件全部满足，才能启动 COCO 正式实验：

### 数据验收

- COCO minitrain 恰好 25,000 张官方固定划分训练图像。
- 完整 COCO80 类别及合法 GT。
- COCO val2017 恰好 5,000 张原始验证图像。
- Train/Val 没有错误的数据重叠。
- Person 类别正确映射为 ID=0。
- 训练与验证标注转换无错误。
- Clean 数据保持原始身份和内容。

### 方法验收

- COCO80 教师和检测头的类别数量正确。
- TSFL 在 COCO80 下正确运行。
- Safe-BG 能处理 79 个非目标类别。
- P5 Safe-BG 和 P3 Orth/Retention 的原始实现保持一致。
- TFSS 能正确计算 Q_t 和 R_nt。
- 目标框与非目标框重叠区域预算仍为 4/255。
- Person 独占区域预算为 8/255。
- Person 框外扰动为零。
- Fourier 梯度正常回传。
- Frozen/Adaptive 双分支均能正确计算损失。
- 数据导出和数据读取一致。
- VOC20 的原始 S4 实现没有受到回归影响。

### 工程验收

- CPU 单元测试全部通过。
- 必要的 GPU Smoke Test 全部通过。
- 无 NaN、Inf 或非法 GT。
- 显存使用和数据加载正常。
- 配置与代码已保存为可追溯版本。
- 原始 S4 方法和配置身份已锁定。
- 当前不存在冲突的 GPU 正式训练任务。

**只要以上条件全部满足，就自动开始正式实验，不需要我再次确认。**

如果测试失败，应暂停正式训练，保留错误日志并报告原因；不得把测试未完成当作测试通过。

---

## 四、正式实验执行顺序

### Stage 1：训练 COCO80 Teacher

使用已经验证的 COCO minitrain 25K 干净训练集，准备正式的 COCO80 Teacher。

要求：

- 模型架构 YOLOv8n。
- 使用 80 类检测头。
- Person 类别 ID=0。
- 不使用 COCO val2017 参与模型参数训练或方法统计估计。
- 保存训练配置、权重和 Checkpoint Hash。
- 在干净验证集上评价 Teacher 的 Person AP 和各类别 AP。
- 验证 Teacher 的检测能力足以支撑后续 TSFL、Safe-BG 和 TFSS。

如果已有通过验收、且满足本实验数据来源与类别协议的 COCO80 Teacher，可以直接复用。

不得将 VOC20 Teacher 当作 COCO80 Teacher。

### Stage 2：原始 S4 扰动优化

使用正式 COCO80 Teacher，在 COCO minitrain 上重新优化 S4 共享 Fourier 扰动。

必须保持：

- 原始 TSFL 算法；
- 原始背景原型匹配；
- K=8 背景原型；
- 原始 Safe-BG；
- rank=8；
- removal_alpha=0.5；
- 原始 L_BG 和 L_rel；
- 原始 L_orth 和 L_ret；
- 原始 Adaptive 单步更新；
- 原始 TFSS-Utility；
- eta_U=0.85；
- 原始 Fourier 参数化和优化策略；
- 原始 S4 损失权重；
- 原始 8/4/0 空间扰动预算。

COCO 适配只允许改变数据集相关的统计结果、类别映射和必要的工程实现。

**不得为了提高 COCO 结果而擅自修改 S4 的算法、超参数或损失函数。**

保存优化曲线、梯度统计、模型配置和最终 Fourier 参数。

### Stage 3：导出 Poisoned COCO minitrain

导出完整的 25K Poisoned 训练集。

必须确保：

- 训练集图像数仍为 25,000。
- 全部 80 类标签保持一致。
- 不修改类别 ID。
- 不修改 GT Bounding Boxes。
- 没有 Person 的图像正常保留。
- 不丢失非目标类别实例。
- Person 独占区域不超过 8/255。
- Person 与非目标 GT 重叠区域不超过 4/255。
- Person 框外没有方法引入的扰动。
- 数据导出及量化后的预算通过验证。

必须对原始图像和 Poisoned 图像建立一一对应的 Manifest。

导出审计通过后再开始 Victim 训练。

### Stage 4：训练 Clean Victim

使用干净 COCO minitrain 25K 训练 YOLOv8n。

优先沿用现有 VOC 主实验锁定协议：

- Random initialization
- epochs=200
- imgsz=640
- seed=0
- batch=36
- workers=16

其余 Optimizer、Scheduler、AMP 和基础增强从真实锁定配置读取。

如果因硬件兼容性必须调整某项训练设置，必须在 Clean 和 Poisoned Victim 中保持一致，并记录差异。

主结果使用：

**last.pt@200**

不要使用 COCO 预训练权重初始化 Victim。

### Stage 5：训练 S4 Poisoned Victim

使用 Stage 3 导出的完整 Poisoned COCO minitrain 25K。

要求：

- YOLOv8n。
- 与 Clean Victim 相同初始化协议。
- 与 Clean Victim 相同随机种子。
- 相同训练轮数和训练配置。
- 相同 80 类检测头。
- 相同数据增强。
- 相同验证集。

必须完整训练 200 epoch。

不能因为 Person AP 提前下降而提前停止训练。

主结果同样使用：

**last.pt@200**

### Stage 6：统一评价

使用未经扰动的完整 COCO val2017 5K 评价：

Clean Victim

S4 Poisoned Victim

输出：

- Person AP50
- Non-target mAP50
- Person AP50-95
- Non-target mAP50-95
- All-class mAP50
- All-class mAP50-95
- Non-target mAP75
- VOC20/COCO80 共有类别 AP
- COCO80 全部类别 AP50
- COCO80 全部类别 AP50-95
- Selectivity S
- WCD
- TCD3
- PSNR / LPIPS
- 空间扰动面积及强度统计

Selectivity 使用匹配的 COCO Clean 基线计算：

P = 1 - AP50_person(S4) / AP50_person(Clean)

U = mAP50_non_target(S4) / mAP50_non_target(Clean)

S = 2PU / (P+U)

WCD/TCD3 按现有项目正式定义计算。

不得随意更改指标定义。

---

## 五、服务器自动运行与异常恢复

本次要求正式实验可以在服务器后台独立持续运行。

不应依赖：

- Codex 对话窗口持续在线；
- 本地电脑持续开机；
- VPN 长期保持连接；
- SSH 会话持续存在。

优先使用当前已有的服务器端 Pipeline Orchestrator。

如果已有 COCO 后继流水线，应恢复和完善它，而不是创建第二个相互冲突的调度器。

要求：

1. 同一 GPU 上不并发执行冲突的正式训练。
2. 不干扰其他已经启动的实验。
3. 每个 Stage 有独立状态。
4. 每个 Stage 完成后自动验收。
5. 验收通过后自动进入下一阶段。
6. 已完成阶段不得无故重复运行。
7. 失败时保存日志和 Checkpoint。
8. 支持合法的断点恢复。
9. 数据与结果目录不得覆盖 VOC 或旧 S4 实验。
10. 检测到数值错误、数据错误、配置污染时停止对应阶段，不擅自改变超参数。

自动流水线：

COCO_TESTS_ACCEPTED

→ TEACHER_READY

→ S4_FOURIER_OPTIMIZATION

→ POISONED_25K_EXPORT

→ POISONED_DATA_ACCEPTED

→ CLEAN_VICTIM_200E

→ POISONED_VICTIM_200E

→ CLEAN_VAL2017_EVALUATION

→ FINAL_REPORT

→ COMPLETED_AND_ACCEPTED

如果某个阶段已经存在经过验证的有效结果，可以从下一阶段继续。

如果 Teacher 或其他阶段需要运行 GPU，而当前 GPU 正被原有正式实验占用，应进入等待队列，之后自动继续，不要抢占资源。

---

## 六、GitHub 与代码管理

本次 COCO 实验必须基于已验证的原始 S4 代码。

继续使用独立 COCO 适配分支。

在正式训练开始前：

- 确认当前修改已提交。
- 确认 GitHub Push 成功。
- 记录 Commit SHA。
- 保存运行配置。
- 保存数据 Manifest。
- 保存 Ultralytics 和 PyTorch 版本。
- 保存随机种子与环境信息。

不得上传大型数据集、模型权重、隐私文件或实验缓存到 GitHub。

不得覆盖原始 S4 分支或 Full 实验分支。

---

## 七、正式结果目录与报告

为 Experiment A 创建独立结果目录。

至少生成：

- `COCO_EXPERIMENT_A_STATUS.json`
- `COCO_EXPERIMENT_A_CONFIG.yaml`
- `COCO_DATA_ACCEPTANCE.md`
- `COCO_S4_METHOD_PARITY.md`
- `COCO_TEACHER_REPORT.md`
- `COCO_S4_OPTIMIZATION_REPORT.md`
- `COCO_POISONED_DATA_AUDIT.md`
- `COCO_CLEAN_VICTIM_REPORT.md`
- `COCO_S4_VICTIM_REPORT.md`
- `COCO_PER_CLASS_AP.csv`
- `COCO_CROSS_DATASET_FINAL_REPORT.md`

最终报告必须对比：

VOC20 原始 S4

与

COCO80 原始 S4

但需要明确：

本次是 **Cross-Dataset Method Generalization**，而不是 VOC Fourier 扰动参数直接迁移。

COCO 上重新构建统计和优化扰动是允许且必要的。

---

## 八、立即执行的要求

请从当前 COCO 项目的实际进度继续：

1. 检查哪些兼容性测试已经完成。
2. 继续执行尚未完成的测试。
3. 验证全部启动条件。
4. 自动准备正式实验配置。
5. 完成代码与配置的 GitHub 归档。
6. 建立或恢复持久化实验流水线。
7. 通过全部验收后自动开始 COCO 正式实验。
8. 不需要再次向我询问是否开始训练。
9. 定期保存阶段状态、完整日志和结果。
10. 完成所有阶段后生成正式对比报告。

不要只完成测试就停止工作，也不要只给出正式训练命令而不执行。

**本次明确授权：当 COCO minitrain 25K、val2017 5K、COCO80 兼容性和原始 S4 方法一致性全部通过验证，且 GPU 资源及现有任务依赖条件满足后，直接启动完整 Experiment A。**

原始 S4 的重叠区域 4/255 约束是不可改变的实验条件。

现在开始执行并设置后继任务。
