# Codex 正式任务：S4 Victim AMP 兼容性修复、真实训练路径验证与 200 Epoch 恢复实验

## 一、任务背景

当前研究项目：

Class-Selective Unlearnable Examples for Object Detection。

正式实验：

S4 — Safe-Soft05 + TFSS-Utility。

原始 RunID：

`s4_tfss_utility_seed0_20261017_v1`

当前情况：

S4 扰动生成已经完整完成，但 Victim 训练在第一个 Batch 因非有限梯度安全检查而终止。

本轮目标：

**修复 S4 新增的 AMP 梯度安全检查与 YOLOv8 原生训练流程之间的兼容性问题，验证真正的 Victim 优化路径，然后使用已经生成并通过审计的 S4 数据集重新完成随机初始化 YOLOv8n 的 200 epoch 训练。**

不得重新生成扰动，不得修改 TFSS 数学公式，不得改变原有实验配置。

本次授权包括：

1. 代码与训练流程审计。
2. 最小范围 AMP 兼容性修复。
3. CPU 单元测试。
4. 真实 CUDA Victim AMP Smoke。
5. 正式 Victim 重新训练。
6. last.pt@200 正式评价。
7. 与 S0/S1/S2/S3 结果比较。
8. 代码和小型报告归档。

只要正确性与资源准入通过，可以直接执行完整 Victim 训练，无须再次请求启动授权。

---

# 二、首先确认已完成的 S4 产物

请读取正式状态文件与失败报告：

`S4_VICTIM_FAILURE_ANALYSIS_20261018.md`

以及：

`execution/S4_VICTIM_FAILURE_DIAGNOSIS_20261018.json`

`execution/S4_AMP_GUARD_CPU_CONTRACT_20261018.json`

已知记录：

- 原始 RunID：s4_tfss_utility_seed0_20261017_v1
- Noise Epoch：24/24
- Noise Updates：9144/9144
- 训练图像：16551
- Clean Validation：4952
- Person 训练图像：6095
- 完成扰动的数据集审计：PASS
- Victim 完成 Epoch：0
- Victim last.pt：不存在
- Victim best.pt：不存在
- 正式 AP50：NOT_RUN

原始噪声 checkpoint SHA256：

`4413e45aa9c448bb768cac50e0ae40b5b0a1a580e38ff37cf66023bdab07c6e5`

数据完整性审计 SHA256：

`3eece1fc62df963f7d4aa15d378395273bd3963f812c0f7764ea9115dc82fe82`

必须重新读取实际文件并核对 Hash，不得只相信历史日志。

确认扰动数据集仍满足：

- 所有训练图像数量正确；
- Person 目标类别正确；
- 原始标签 byte-identical；
- Validation 未修改；
- 非目标图像未修改；
- Person 区域之外扰动为零；
- 全局 L∞≤8/255；
- 原始 S0 重叠区域局部预算≤4/255；
- 不存在重复 Image ID 或遗漏图像。

**若完整性验证通过，直接复用该数据集，不允许重新运行 9144 步扰动优化。**

---

# 三、诊断已确认的 AMP 实现问题

已有证据：

S4 新增文件：

`ue_framework/s4_victim.py`

报告指出第 28 行附近存在：

`FloatingPointError("Nonfinite victim gradient; stop before update")`

当前检查发生在 AMP-scaled backward 之后、GradScaler unscale 之前。

实际安装的 YOLOv8/Ultralytics 训练路径会执行：

```python
self.scaler.scale(self.loss).backward()
```

然后由原生 Optimizer Step 处理：

```python
self.scaler.unscale_(optimizer)

# Native gradient clipping

self.scaler.step(optimizer)

self.scaler.update()
```

实际函数顺序必须以服务器安装版本的源码为准。

当前环境记录：

- PyTorch 2.7.1+cu128
- AMP=True
- Victim Batch=36
- Workers=16
- Seed=0

已有 CPU 标量机制测试证明：

有限原始 Loss 经 AMP 放大后可以产生 Inf 梯度。

原生 GradScaler 可以：

- 检测梯度溢出；
- 跳过当前参数更新；
- 将 scale 从 65536 降为 32768；
- 保持模型参数有限；
- 继续后续训练。

但 S4 额外安全检查会在这之前直接终止。

这属于新增安全封装与原生 AMP 协议不兼容。

注意：

CPU 标量测试不能证明实际失败 Batch 一定只是暂时 AMP 溢出。

仍必须使用真实 CUDA Victim 训练路径验证。

---

# 四、修复原则：优先恢复 S0 的原生 Victim 训练行为

这是本轮最重要的要求。

## 4.1 对照 S0 和 S4 的 Victim 实现

先定位 S0 完成 200 epoch 时使用的：

- Victim Trainer；
- Ultralytics 版本；
- Optimizer Step；
- AMP；
- GradScaler；
- EMA；
- LR Scheduler；
- Gradient Clipping；
- Checkpoint；
- 数据加载。

再对比 S4 新增的 Victim Wrapper。

输出完整 Diff。

必须回答：

**既然 S4 只修改训练数据中的扰动，为什么需要修改已经在 S0 中验证过的 Victim 训练更新路径？**

如果 S4 的额外 Wrapper 没有不可替代的必要性，优先恢复 S0 已验证的原生 Optimizer Step。

只在外部增加不改变训练语义的检查、日志和完整性监测。

不要重新实现一套 AMP 优化器。

## 4.2 必须保留原生 GradScaler 行为

确保：

- AMP backward 语义不变；
- GradScaler 能正常 unscale；
- 原生 GradScaler 可以检测溢出；
- 合法 AMP overflow 可以触发 skip；
- GradScaler scale 可以 backoff；
- 原生 Gradient Clipping 保持不变；
- 参数更新和梯度清零顺序保持一致；
- 原生 EMA 与 Scheduler 处理不被错误 Wrapper 改写。

严禁：

- 在 unscale 前对 scaled gradient 的 Inf/NaN 直接抛出终止异常；
- 人工把 Inf/NaN 梯度改成零；
- 使用 nan_to_num 掩盖错误；
- 直接关闭 AMP；
- 擅自降低 GradScaler 初始 Scale；
- 修改 Batch；
- 修改 Workers；
- 修改学习率；
- 修改优化器；
- 通过跳过整个 Batch 假装训练成功；
- 重复 unscale 同一个 Optimizer；
- 手动调用额外的 optimizer.step；
- 使用 STE 或自定义梯度替代原生 AMP。

注意：

单纯把原梯度检查移动到 unscale 之后，也不能修复原生 AMP skip/backoff 路径被阻断的问题。

## 4.3 区分两类异常

### A. 合法 AMP Scaling Overflow

特点：

- 原始 Loss 有限；
- 模型状态有限；
- 放大梯度出现 Inf；
- GradScaler 检测；
- 当前 Optimizer Update 合法跳过；
- Scale 降低；
- 后续训练可以恢复正常更新。

应允许原生 AMP 处理，并记录事件。

### B. 持续真实数值异常

例如：

- 原始未缩放 Loss 本身非有限；
- 模型参数已经 NaN；
- EMA 非有限；
- 持续出现非有限梯度，Scale 降低后仍无法恢复；
- 数据或 GT 异常导致真实计算错误。

这些情况不得被自动忽略。

必须停止、保留证据并诊断。

如果需要设置连续异常更新上限，应在真实 GPU 验证前固定，并说明依据。

不能为了让 Victim 跑完 200 epoch 而无限忽略更新失败。

---

# 五、增加 AMP 诊断日志，但不修改训练语义

希望能够记录：

1. Batch Index；
2. Epoch；
3. Raw Loss；
4. AMP Scale Before；
5. AMP Scale After；
6. 是否检测到 Overflow；
7. 是否实际执行 Optimizer Update；
8. 是否发生 GradScaler Backoff；
9. 是否存在连续跳过更新；
10. Model Parameters Finite；
11. EMA Parameters Finite；
12. Gradient Norm（仅在适当的 AMP 阶段计算）；
13. CUDA Memory；
14. CUDA Peak Memory。

特别注意：

不能只根据 Loss 有限就认定梯度一定有限。

也不能把 AMP 缩放后的梯度范数直接当作实际梯度范数。

如果原生 Trainer 没有稳定的 public API 提供 Skip 事件，可以使用经过验证的 Optimizer Step 计数或其他不改变训练行为的诊断方式。

避免依赖未经验证的私有 API。

所有诊断代码不得改变梯度、参数或优化器更新结果。

---

# 六、重新补齐真正的 Victim AMP Smoke

此前 Smoke 的覆盖范围存在明确缺口：

原来的 Smoke 只验证：

- Noise；
- Q/R/W Maps；
- PNG；
- Resume；
- Fourier；
- Adaptive。

但在进入真实 Victim `train_evaluate` 之前已经结束，没有执行真实 AMP Victim Optimizer Step。

因此这次必须新增：

**Real Victim AMP Training Smoke。**

## 6.1 必须使用真实 S4 数据

不是只使用随机张量或 CPU 标量模型。

必须使用：

- 当前 S4 已生成的加噪图像；
- 正确的 VOC20 标签；
- 正式 YOLOv8n；
- 正式 Victim 配置；
- 正式 AMP=True；
- 正式检测 Loss；
- 正式 Optimizer；
- 正式 Trainer 更新链路。

建议测试至少前 5 个实际训练 Batch，或在必要时增加少量预先登记的 Batch，以确认出现 Overflow 后能否恢复。

不得修改实验核心参数来通过测试。

## 6.2 必须验证正常 AMP 更新

在梯度有限时：

- Loss 正常；
- Backward 正常；
- GradScaler 正常；
- Optimizer Update 正常；
- 参数确实发生有效更新；
- Model/EMA 保持有限；
- 没有重复更新。

## 6.3 必须验证 AMP Overflow 恢复

在独立的隔离测试中，可以通过可控的人工 Loss Scaling 构造溢出场景。

但这个人工测试不能污染正式 Victim。

检查：

- Overflow 能被检测；
- 当前更新被合法跳过；
- Scale 下降；
- 模型参数保持有限；
- 梯度得到正确清理；
- 下一次正常 Batch 可以继续训练。

必须分别报告：

SYNTHETIC_OVERFLOW_TEST

和：

REAL_S4_VICTIM_BATCH_TEST

不能把合成测试当成真实 S4 故障已经恢复的证据。

## 6.4 与 S0 原生训练行为比较

在相同初始化、相同小批次、相同 seed 下，比较：

- 原 S0 Victim Trainer；
- 修复后的 S4 Victim Trainer。

在相同输入条件下：

Loss、Optimizer Update、Scheduler、EMA 和 AMP 行为应数值一致，或明确说明合理的浮点容差。

然后再用真正 S4 数据测试。

不得将 S0 clean 输入与 S4 poison 输入的 Loss 不同错误归因于 Trainer 差异。

---

# 七、真实训练前必须通过的准入条件

以下项目均通过才能正式恢复：

### A. Source Fidelity

S4 TFSS-Utility 算法未修改。

Q/R/W 地图计算未修改。

Safe-BG、TSFL、Fourier、Adaptive 未修改。

### B. Data Integrity

原 S4 加噪数据全部通过 Hash 与预算审计。

### C. Victim Protocol

与 S0 的正式 Victim 训练配置一致。

### D. AMP Compatibility

真实 AMP 正常更新路径通过。

### E. Overflow Recovery

隔离合成测试与原生 GradScaler 行为一致。

### F. Parameter Integrity

Model、Optimizer、EMA 无非有限状态。

### G. Resource Safety

GPU、CPU、RAM、Swap、磁盘准入通过。

### H. Provenance

修复源码、配置、环境版本和数据 Hash 全部锁定。

任何一个关键项目失败，暂停正式恢复并报告。

---

# 八、正式 Victim 恢复策略

目前原 S4 Victim：

- 第一个 Batch 失败；
- 0 个 Epoch 完成；
- 未完成正式参数更新；
- 无可用 last.pt；
- 无 best.pt；
- 无完整训练状态快照。

因此：

**不能把新 Victim 训练称为从原训练 Batch 精确断点续训。**

正确做法是：

使用原始 S4 已完成的加噪训练数据，从正式随机初始化重新启动 Victim 训练。

不得重新生成 Noise。

不得重新训练 Frozen/Adaptive。

不得改变 S4 空间地图。

## 8.1 新恢复 RunID

原始失败 RunID 保持：

`s4_tfss_utility_seed0_20261017_v1`

状态：

FAILED_VICTIM_AMP_GUARD

建议新 Victim RunID：

`s4_tfss_utility_seed0_20261017_v1_victim_recovery01`

新 RunID 必须明确关联原始 Noise RunID。

记录：

- parent_noise_run_id；
- poisoned_dataset_path；
- poisoned_dataset_manifest_sha；
- victim_config_sha；
- victim_source_sha；
- repair_commit_sha；
- original_failure_report；
- recovery_reason。

不可覆盖原始失败日志。

## 8.2 Victim 训练配置

使用与 S0 相同：

Model：

Random-initialized YOLOv8n。

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

其他参数：

- Optimizer；
- LR；
- Warmup；
- Scheduler；
- Momentum；
- Weight Decay；
- Image Size；
- Mosaic；
- Augmentations；
- DataLoader；
- NMS；
- EMA；

均以原正式 S0 配置为准。

不能修改这些参数来规避 AMP 问题。

---

# 九、正式执行 200 Epoch Victim

完成全部准入后：

直接启动完整 Victim 训练。

必须使用：

已经完整生成并通过审计的 S4 TFSS-Utility 数据集。

训练：

200 Epoch。

主 Checkpoint：

last.pt@200。

次 Checkpoint：

best.pt。

每个 Epoch 正常保存训练指标。

按照现有保存周期进行 checkpoint。

必须正确记录：

- 实际 Optimizer Updates；
- AMP Overflow Events；
- Skipped Updates；
- Loss Scale；
- Model Finite；
- EMA Finite；
- Epoch；
- Step；
- GPU Memory；
- 训练耗时。

如果只发生少量合法 AMP overflow，且 GradScaler 能恢复正常更新，则按原生 Trainer 语义继续。

如果持续无法正常更新，或出现未缩放 Loss/模型参数非有限，则停止并保留证据。

不得为了完成 200 Epoch 而修改训练算法。

---

# 十、CPU 资源问题独立处理

此前 S4 扰动生成阶段出现过 CPU 持续饱和及反复暂停。

但现在已经明确：

**Victim 首 Batch 失败的直接原因是 AMP 安全 Wrapper，不是 CPU OOM 或 CPU Saturation。**

这两个问题必须分别处理。

目前不要为了 AMP 修复而修改：

- DataLoader workers；
- Batch；
- CPU affinity；
- TFSS 预处理；
- Victim Loss。

需要继续使用既有资源准入规则：

- CPU 实际使用率；
- I/O Wait；
- RAM；
- Swap；
- GPU；
- 磁盘；
- 其他任务争用。

不再仅凭 Linux Load Average >24 拒绝启动。

但持续 CPU 饱和仍应按照已有资源安全规则处理。

不得停止其他用户的进程或删除其他实验数据。

---

# 十一、完成正式评价

Victim 完成 200 Epoch 后：

使用：

last.pt@200。

在相同的 4952 张 clean VOC validation 图像上完成评价。

完整保留：

### Target

- Person AP50；
- Person AP75；
- Person AP50-95；
- Person Recall；
- Person Precision。

### Non-target

- Non-target mAP50；
- Non-target mAP75；
- Non-target mAP50-95；
- Non-target Recall；
- Non-target Precision。

### Overall

- All-class mAP50；
- All-class mAP50-95。

### Per-class

VOC20 全部类别：

- AP50；
- AP75；
- AP50-95；
- Recall；
- Precision。

### Selectivity

- P；
- U；
- S；
- WCD；
- TCD3。

采用原有锁定定义与 last.pt 主评价协议。

不能使用 best.pt 替代主结果。

---

# 十二、与历史实验对比

生成：

| Method | Person AP50 ↓ | NT mAP50 ↑ | NT mAP75 ↑ | S ↑ |
|---|---:|---:|---:|---:|
| S0 Safe-Soft05 | 2.8993 | 69.1698 | 51.5898 | 93.3425 |
| S1 CSB | 9.1670 | 70.4576 | 53.5902 | 90.6649 |
| S2 CNLP | 4.0368 | 68.9123 | 50.9453 | 92.5494 |
| S3 CNLP+CUP | 3.4414 | 68.9391 | 51.0290 | 92.8901 |
| S4 TFSS-Utility | 待测 | 待测 | 待测 | 待测 |

历史数据使用已归档原始评价记录，重新核对数值与相同数据集条件。

S4 主要研究条件：

Person AP50 ≤3.8993%。

并希望：

Non-target mAP50 >69.1698%。

另外保留 AP75、WCD、TCD3、逐类别检测结果。

单 Seed 结果不得声称统计显著。

---

# 十三、报告与归档

建议新建：

`docs/S4_AMP_RECOVERY_20261018/`

包括：

1. AMP_FAILURE_ROOT_CAUSE.md
2. S0_S4_VICTIM_DIFF.md
3. AMP_NATIVE_COMPATIBILITY_FIX.md
4. AMP_CPU_CONTRACT_TEST.md
5. REAL_GPU_AMP_SMOKE.md
6. AMP_OVERFLOW_RECOVERY_TEST.md
7. VICTIM_PROTOCOL_EQUIVALENCE.md
8. S4_DATASET_REUSE_AUDIT.md
9. VICTIM_RECOVERY_CONFIG.json
10. VICTIM_RECOVERY_PROGRESS.json
11. AMP_SCALE_AND_SKIP_LOG.csv
12. S4_LAST_EVALUATION.json
13. S4_VOC20_PER_CLASS.csv
14. S4_S0_S1_S2_S3_COMPARISON.csv
15. S4_VICTIM_FINAL_REPORT.md
16. RECOVERY_PROVENANCE.json

每份报告记录：

- Git SHA；
- Python/PyTorch/Ultralytics 版本；
- Config Hash；
- Noise Dataset Hash；
- Victim Checkpoint Hash；
- RunID；
- Seed；
- AMP Scale；
- 实际 Optimizer Update；
- Overflow/Skip 事件；
- 原始故障；
- 修复内容；
- 实际运行状态。

将修复后的源代码和小型报告提交到 object-B 的独立 S4 修复分支。

建议：

`codex/s4-victim-amp-repair-20261018`

不得修改 main。

不得将大规模 poisoned dataset 或模型权重直接提交到 Git。

---

# 十四、最终执行顺序

请依次执行：

1. 读取 S4 正式失败报告。
2. 核查原始故障证据。
3. 确认噪声和数据完整性。
4. 对照 S0 原生 Victim 训练路径。
5. 定位 S4 AMP Wrapper 的不兼容实现。
6. 最小范围修复。
7. 运行 CPU AMP Contract Test。
8. 运行真实 CUDA Victim AMP Smoke。
9. 验证 Overflow Skip/Backoff。
10. 验证模型参数与 EMA 有限性。
11. 验证 S0/S4 训练语义一致。
12. 锁定修复代码与正式配置。
13. 建立独立 Recovery RunID。
14. 复用原 S4 poisoned dataset。
15. 从相同随机初始化启动 Victim 200 epoch。
16. 保存 last.pt 与 best.pt。
17. 完成 clean VOC20 正式评价。
18. 计算全部逐类别和 Selectivity 指标。
19. 与 S0/S1/S2/S3 统一比较。
20. 归档修复证据和实验结果。
21. 提交独立 GitHub 分支。
22. 完成后停止。

---

# 十五、最高优先级原则

本轮是：

**Victim AMP Compatibility Repair**

不是：

TFSS Algorithm Improvement。

S4 的扰动生成和数据审计已经完成。

不得修改：

- TFSS-Utility；
- Q_t；
- R_nt；
- W_U；
- η=0.85；
- TSFL；
- Safe-BG；
- Frozen Reference；
- Adaptive Learner；
- Fourier；
- 原始 S0 保护损失；
- 训练数据；
- 噪声 checkpoint。

只允许修复 Victim AMP 安全检查与测试覆盖缺陷。

优先恢复已经验证的 Ultralytics 原生训练语义，而不是自行编写一套新的 AMP 更新流程。

只要修复、真实 Smoke、数据完整性和资源准入通过，就直接启动已授权的完整 S4 Victim。

如果真实 GPU Smoke 仍然出现持续非有限 Loss、参数异常或 AMP 无法恢复，则必须停止并报告根因，不得通过关闭 AMP 或修改正式训练超参数强行完成实验。

**最终目标是得到与 S0 严格同协议、可信且可复现的 S4 Victim 检测结果。**