# Codex 正式任务：S4 鲁棒性实验服务器端全自动流水线

## 一、总体任务

当前研究为 Class-Selective Unlearnable Examples for Object Detection。

正式主方法：

**S4 — Safe-Soft05 + TFSS-Utility**

目标类别：Person，VOC class ID=14。

本次要求将后续鲁棒性实验改为**服务器端独立运行的自动化流水线**，而不是由 Codex 手动启动每一项实验。

执行顺序必须为：

Gray → BDR → Phase A 完整验收与报告 → Gauss → MeanF → MedianF → Data Augmentation → Phase B 完整验收与最终报告。

此前 JPEG50、JPEG30 的产物属于 Phase A，需要先审计并复用已完成结果，不得无故重复训练。

本轮唯一允许使用的 Poison 数据来源为正式 S4-TFSS-Utility。

禁止混用 S0、S3、S5 的扰动数据、Victim 权重和鲁棒性结果。

本次授权覆盖后续上述防御实验的完整训练、评价、归档和自动恢复。

**本次不授权执行 AT 对抗训练。AT 保持待执行状态，后续单独安排。**

---

## 二、最重要的要求：实验必须在服务器独立运行

不允许使用以下方式作为正式流水线：

1. Codex 手动依次输入多条训练命令；
2. Codex 每完成一项后再人工启动下一项；
3. 依赖 Codex 对话持续在线才能运行；
4. 依赖 VSCode、SwordAgent 或用户电脑保持连接；
5. 依赖 ChatGPT 定时任务触发下一阶段；
6. 仅通过一个前台 Shell 会话维持长时间训练。

请设计并部署一套服务器端自动执行机制。

优先采用：

- 独立 Python Pipeline Orchestrator；
- 持久化状态文件；
- 进程锁；
- Checkpoint 和阶段恢复；
- 服务器本地日志；
- systemd user service（如果当前用户环境支持）或可靠的 tmux/nohup 守护方案。

优先选择不需要 root 权限的方案。

如果不能可靠保证服务器重启后的自动恢复，必须明确说明这一限制，不能声称已经实现重启自恢复。

部署完成以后，即使 Codex 对话关闭，流水线也应该继续执行。

Codex 只需要一次性完成部署和启动验证。

---

# 三、正式流水线阶段

请建立以下任务队列。

| Stage | 任务 | 前置条件 |
|---|---|---|
| 0 | 已有 JPEG50/JPEG30 结果审计 | S4 数据身份确认 |
| 1 | Gray | Stage 0 通过 |
| 2 | BDR | Gray 完成、BDR 协议锁定 |
| 3 | Phase A 验收与报告 | JPEG50/JPEG30/Gray/BDR 全部完成 |
| 4 | Gauss | Phase A 验收通过 |
| 5 | MeanF | Gauss 完成 |
| 6 | MedianF | MeanF 完成 |
| 7 | Data Augmentation | MedianF 完成 |
| 8 | Phase B 验收与最终报告 | Phase A/B 全部防御完成 |
| 9 | 自动归档并退出 | 最终审计通过 |

阶段顺序必须严格执行。

每个防御阶段内部包含：

- 确认防御参数；
- 确认 S4 数据集；
- 检查匹配的 Clean 对照；
- 数据变换与完整性检查；
- 真实训练 Batch Smoke；
- 完整 Victim 200 epoch；
- last.pt@200 正式评价；
- best.pt 次评价；
- VOC20 逐类别指标；
- P/U/S、WCD、TCD3；
- 数据与结果归档；
- 阶段完成验收。

如果当前 JPEG50/JPEG30 已经完成，优先直接读取并校验历史结果。

不得仅凭文件存在就判定完成。

需要检查 checkpoint、训练记录、评价指标、SHA256 和数据来源。

---

# 四、锁定 S4 实验协议

正式 S4 无防御结果：

- Person AP50：0.028159
- Non-target mAP50：0.693787
- Non-target mAP75：0.515251
- Non-target AP50-95：0.480528
- All-class mAP50：0.660506
- Selectivity S：0.935330

训练设置：

- VOC20；
- Random-initialized YOLOv8n；
- Seed=0；
- Epoch=200；
- Batch=36；
- Workers=16；
- AMP=True；
- 原 S4 Optimizer/Scheduler；
- Clean Validation=4,952 images；
- last.pt@200 为主评价；
- best.pt 为次评价。

必须重新读取正式 S4 的配置、RunID、Manifest 和数据 Hash。

**所有鲁棒性实验必须复用 S4 已经完整生成并经过审计的受保护训练数据。**

不得重新生成 Fourier 扰动。

不得重新训练 Frozen/Adaptive。

不得修改 TFSS-Utility、TSFL、Safe-BG、Lret、Lorth 或其他 S4 方法模块。

---

# 五、各防御的处理协议

## Gray

将训练图像转换为灰度图，再转为三通道 RGB。

Clean 与 S4 使用同一转换函数和图像保存协议。

保持：

- 图片尺寸；
- Bounding Box；
- Class ID；
- 原始标签；
- 数据划分。

## BDR

必须先核查此前锁定的 BDR 定义。

如 BDR 为 Bit-Depth Reduction，则按对应已验证代码和参考实验协议执行。

不允许在缺少证据时自行猜测：

- Bit Depth；
- 量化公式；
- 量化范围；
- 舍入规则。

若服务器已有明确锁定且经过测试的 BDR 协议，则自动使用。

若协议尚未确定，部署前应优先完成来源审计。

如果确实存在必须由研究者决策的参数缺口，提前明确报告，不得以未经授权的配置完成正式实验。

## Gauss

复用原 S0 已验证的 Gaussian Filtering 代码。

核查是否确实为 Gaussian Filtering，而不是 Gaussian Noise。

保持原来的：

- Kernel；
- Sigma；
- Padding；
- 图像输入输出范围。

仅将正式受保护数据来源替换为 S4。

## MeanF

复用此前锁定的 Mean Filtering。

保持原有核大小、边界处理和图像处理方式。

## MedianF

复用此前锁定的 Median Filtering。

保持原有 Kernel 和边界处理。

## Data Augmentation

复用原 S0 已验证的数据增强防御方案。

先确认具体增强操作及概率：

- Flip；
- Resize；
- Crop；
- Mosaic；
- Color；
- Scale；
- 其他已有操作。

如果防御采用在线随机增强，就必须在 Victim 数据加载和训练管线中复用同一逻辑，而不是擅自转换为离线增强。

不得因为复用旧代码而额外重复执行标准 YOLO 增强。

所有防御的正式参数必须在训练前冻结。

---

# 六、Clean/Poison 配对实验

每一种防御 D 必须有：

Clean + D

S4 + D

两个对应条件。

所有结果在相同的未经修改的 clean VOC validation 上评价。

如果历史 Clean+D 已经完成，允许复用，但需要严格核查：

- 防御参数；
- 训练图像身份；
- 数据处理实现；
- Model；
- Seed；
- Epoch；
- Batch；
- Optimizer；
- Scheduler；
- Augmentation；
- Ultralytics 版本；
- Checkpoint；
- Evaluation Protocol。

若无法证明严格等价，则自动安排新的 Clean+D 训练。

每个防御不得只训练 S4 模型而缺少有效 Clean 对照。

---

# 七、服务器自动调度设计

这是本次任务的重点。

请实现一个独立服务器 Orchestrator。

建议入口：

`python scripts/run_s4_robustness_pipeline.py`

实际名称可根据项目结构调整。

要求：

## 7.1 单实例运行

整个流水线只能有一个主控制实例。

使用文件锁或 flock 等机制防止重复启动。

如果再次执行启动命令，应优先检测并连接已有任务状态，而不是再创建第二条完整流水线。

同一 GPU 同时最多运行一个正式 Victim 训练进程。

不得出现多个防御的 200 epoch 训练同时争抢 GPU。

## 7.2 自动阶段衔接

Gray 正式完成并通过审计后，自动启动 BDR。

BDR 完成后自动执行 Phase A 验收和报告。

Phase A 通过后自动启动 Gauss。

依此类推。

**阶段之间不得依赖 Codex 再次发送命令。**

## 7.3 持久化任务状态

建议保存：

`pipeline_state.json`

记录：

- Pipeline RunID；
- 当前阶段；
- 阶段状态；
- 开始时间；
- 完成时间；
- RunID；
- PID；
- Config Hash；
- Dataset Manifest Hash；
- Checkpoint Hash；
- 已完成 Epoch；
- Last Valid Checkpoint；
- Retry Count；
- Failure Reason；
- Evaluation State；
- Acceptance State。

每个阶段状态至少包括：

PENDING

RUNNING

PAUSED_RESOURCE

RETRYABLE_FAILURE

BLOCKED

FAILED

COMPLETED

ACCEPTED

状态更新需要使用原子写入。

不得因为异常退出而生成虚假的 COMPLETED 状态。

## 7.4 已完成阶段不重复运行

支持幂等执行。

恢复时首先读取现有产物和状态。

对于已经通过验收的阶段，自动跳过。

对于运行到一半的阶段，检查可恢复状态。

不得覆盖已经完成的正式训练结果。

---

# 八、自动资源管理

服务器为共享训练环境。

需要持续监测：

- CPU 利用率；
- CPU Load Average；
- I/O Wait；
- GPU 利用率；
- GPU 显存；
- RAM；
- Swap；
- 磁盘；
- 其他训练进程；
- 当前阶段耗时。

不允许只根据 Load Average 大于 24 就暂停实验。

但如果出现持续 CPU 饱和、GPU 资源冲突、内存不足或严重 I/O 堵塞，必须按已有安全策略处理。

要求：

1. 资源正常时自动运行；
2. 临时资源不足时自动等待；
3. 资源恢复后自动继续；
4. 不因为短时 CPU 峰值频繁暂停/恢复；
5. 不停止他人进程；
6. 不修改他人任务优先级；
7. 不删除其他用户数据。

建议使用原有 RRCTL 的资源判断逻辑，避免重新编写一套不一致的准入规则。

如果需要改动 RRCTL，必须保证不影响历史 S4/S5 产物。

---

# 九、异常与断点恢复

流水线必须区分两类问题。

## 可恢复的运行环境问题

例如：

- 临时 CPU/GPU 资源冲突；
- 远程连接断开；
- 可验证的进程意外退出；
- 暂时性 I/O 错误。

允许在资源恢复后自动重试或继续。

但必须设置有限重试次数和退避间隔，不允许无限失败重启。

## 不可自动忽略的科学正确性问题

例如：

- 非有限原始 Loss；
- 持续 NaN/Inf；
- 数据标签错误；
- S4 数据身份错误；
- 预算或图像完整性违规；
- Checkpoint 损坏；
- GT 匹配错误；
- 防御参数改变；
- 训练结果无法复核。

这些问题必须停止对应阶段并记录证据。

如果问题涉及整个实验协议，则暂停整条流水线。

不得通过修改学习率、Batch、AMP 或训练 Epoch 强行完成。

## Resume

如果 Victim 中断，需要核查：

- Model；
- Optimizer；
- Scheduler；
- AMP Scaler；
- EMA；
- RNG；
- Epoch；
- DataLoader/数据顺序相关恢复条件。

只有通过一致性检查，才能声明为精确续训。

否则必须以新的恢复 RunID 如实记录，不能把不完整训练伪装成连续 200 epoch。

本轮全部沿用 S4 已修复的原生 YOLO AMP 行为。

---

# 十、完整训练验收

每种防御只有满足全部必要条件才允许标记 ACCEPTED：

1. 训练图像来源正确；
2. S4 数据身份正确；
3. 防御参数正确；
4. 图片与 GT 一致；
5. 数据集完整；
6. Victim 完成 200 epoch；
7. 训练记录连续且可审计；
8. last.pt 为第 200 epoch；
9. best.pt 可用时作为次评价；
10. Clean Validation 完整；
11. VOC20 逐类别结果齐全；
12. P/U/S 计算正确；
13. Checkpoint SHA 可复核；
14. 没有未解决的数值异常；
15. Clean 对照有效。

任何一个正式指标缺失，不得用估算结果代替。

---

# 十一、Phase A 验收

当 JPEG50、JPEG30、Gray、BDR 全部完成时，自动启动 Phase A 验收。

输出：

`PHASE_A_FINAL_REPORT.md`

`PHASE_A_RESULTS.csv`

`PHASE_A_PER_CLASS.csv`

`PHASE_A_ACCEPTANCE.json`

报告包括：

- 四种防御的 Person AP50；
- 非目标 mAP50；
- 非目标 AP75；
- 非目标 AP50–95；
- Person Recall；
- Non-target Recall；
- 全类 mAP50；
- P/U/S；
- WCD；
- TCD3；
- VOC20 逐类别结果；
- Clean 对照来源；
- 数据与模型 SHA；
- 训练总耗时。

如果四种防御均通过，Phase A 状态改为 ACCEPTED，并自动进入 Phase B。

如果部分失败，应明确失败项，不得将不完整 Phase A 宣称为完整验收通过。

---

# 十二、Phase B 实验与验收

依次运行：

Gauss → MeanF → MedianF → Data Augmentation。

继续使用 S4。

保持与 Phase A 相同的：

- Victim；
- Seed；
- Epoch；
- 评价协议；
- 数据审计；
- 结果计算；
- Checkpoint 保存；
- 自动恢复。

完成后输出：

`PHASE_B_FINAL_REPORT.md`

`PHASE_B_RESULTS.csv`

`PHASE_B_PER_CLASS.csv`

`PHASE_B_ACCEPTANCE.json`

随后生成八种防御的统一汇总。

---

# 十三、P/U/S 统一计算

对每个防御 D，使用该防御下的 Clean 和 S4 正式 AP50。

目标保护能力：

P_D =
1 - AP_person(S4+D) / AP_person(Clean+D)

非目标效用保持：

U_D =
mAP_nt(S4+D) / mAP_nt(Clean+D)

综合选择性：

S_D =
2P_D U_D / (P_D+U_D)

必须保留原始 AP 和计算所用分母。

不能跨防御使用错误的 Clean 对照。

若公式出现零分母或其他边界情况，按锁定评价协议处理并说明，不得生成无依据结果。

---

# 十四、统一论文表格

最终输出与当前论文相对应的鲁棒性表：

| Method | Clean | Gauss | JPEG50 | JPEG30 | MeanF | MedianF | Gray | BDR | AT | Data Aug |
|---|---|---|---|---|---|---|---|---|---|---|

每个方法包含：

- mAP50_target ↓；
- mAP50_non_target ↑；
- Selectivity S ↑。

当前只填写：

Clean、Gauss、JPEG50、JPEG30、MeanF、MedianF、Gray、BDR、Data Aug。

AT 标记 NOT_RUN。

不要凭空补充 AT 数值。

另输出完整 VOC20 逐类别 AP50/AP75/AP50–95、Recall、Precision。

Clean 行在 Selectivity S 不适用时使用 —。

---

# 十五、日志和监控

需要在服务器持久化保存：

`pipeline.log`

`pipeline_state.json`

`pipeline_progress.json`

`pipeline_events.jsonl`

每个阶段独立保存：

- stdout；
- stderr；
- Epoch；
- Step；
- Loss；
- GPU；
- CPU；
- Checkpoint；
- Evaluation；
- Duration；
- Status。

监控建议每 15 分钟记录一次。

如果实验正常运行，不需要反复创建新的通知任务，也不需要人为暂停。

如果出现：

- 阶段完成；
- 阶段失败；
- 恢复成功；
- 资源持续不足；
- 数据正确性错误；
- Phase A 验收完成；
- Phase B 验收完成；

应将事件写入持久化状态和日志，便于之后读取。

不要求依赖 Codex 在线报告进度。

---

# 十六、代码与服务器目录

代码仓库：

https://github.com/gql119/object-B

使用独立实验分支。

不修改 main。

不覆盖：

- S0；
- S1；
- S2；
- S3；
- S4；
- S5。

建议建立：

`runs/robustness_s4_pipeline/`

目录结构：

- pipeline/
- jpeg50/
- jpeg30/
- gray/
- bdr/
- gauss/
- meanf/
- medianf/
- data_aug/
- reports/
- logs/
- checkpoints/

实际使用项目现有目录规范，不必机械照搬。

每项防御独立保存结果。

确保大型数据和 checkpoint 留在服务器，不提交 GitHub。

GitHub 只提交：

- 流水线源码；
- 配置；
- 运行说明；
- 小型结果与审计报告。

---

# 十七、一次性部署和启动

Codex 现在需要执行：

1. 检查当前服务器及现有实验状态。
2. 确认当前 S4 数据来源。
3. 检查 JPEG50/JPEG30 的已有结果。
4. 审计并锁定 Gray、BDR、Gauss、MeanF、MedianF、Data Aug 参数。
5. 确认 BDR 协议是否完整。
6. 检查每种防御 Clean 对照的有效性。
7. 编写并测试流水线 Orchestrator。
8. 加入单实例进程锁。
9. 加入持久化状态管理。
10. 加入阶段正确性验收。
11. 加入资源等待和安全恢复机制。
12. 加入断点继续及结果归档。
13. 检查完整任务队列和阶段依赖关系。
14. 使用低成本测试验证调度器可以自动从一个模拟阶段进入下一阶段。
15. 验证关闭控制终端后，服务器上的流水线仍可运行。
16. 验证重复执行启动命令不会启动第二实例。
17. 配置可靠的服务器端后台运行方式。
18. 正式启动唯一的 S4 鲁棒性流水线。
19. 确认 Gray 阶段已经由服务器独立启动。
20. 输出运行 PID、RunID、状态文件、日志路径和检查命令。
21. Codex 完成部署任务后即可退出，不需要持续参与阶段调度。

如果某一必要方法学参数尚未锁定，必须先解决或报告阻塞，不得自行使用未经验证的参数。

---

# 十八、最终自动完成后的动作

当 Data Augmentation 正式完成后，服务器应自动：

1. 复核全部八项防御结果；
2. 检查全部 Clean 对照；
3. 检查全部 last.pt@200；
4. 重新计算 P/U/S；
5. 汇总 WCD/TCD3；
6. 生成 VOC20 逐类别表；
7. 生成 S4 Robustness 论文主表；
8. 生成最终完整报告；
9. 生成 Pipeline Acceptance；
10. 保存全部文件 SHA 和来源；
11. 将任务状态设为 COMPLETED_AND_ACCEPTED；
12. 释放流水线自己的运行资源；
13. 正常退出。

不得自动启动 AT。

不得自动启动新的 Seed。

不得自动调整 TFSS 或 S4 方法。

不得重新执行已经完成并通过验收的实验。

---

# 十九、最终需要向我提供什么

部署完成后，Codex 只需要向我报告：

1. 自动流水线是否已经真实启动；
2. 是否能够在 Codex 退出后独立继续；
3. 服务器重启后能否自动恢复，以及具体依赖条件；
4. 当前唯一 Pipeline RunID；
5. 当前阶段；
6. 已完成 JPEG50/JPEG30 的验收情况；
7. BDR 参数是否已经锁定；
8. 当前 PID 和后台运行方式；
9. 状态文件与日志路径；
10. 监测命令；
11. 预计任务顺序与资源预算；
12. 是否存在尚未解决的阻塞。

请给出一条只读的状态查询命令，让我以后可以直接查看实验进度。

---

# 二十、最高优先级原则

**本次工作不是让 Codex 帮我逐项运行实验，而是让 Codex 在服务器上部署一套可独立运行、可恢复、可审计的 S4 鲁棒性自动实验系统。**

正式执行顺序：

Gray → BDR → Phase A 验收 → Gauss → MeanF → MedianF → Data Augmentation → Phase B 验收 → 最终报告。

JPEG50/JPEG30 已完成的有效结果应直接复用。

所有 Poison 数据均来自 S4-TFSS-Utility。

所有防御均有正确的 Clean 对照。

所有 Victim 正式训练为 200 epoch。

全部使用 last.pt@200 主评价。

流水线负责自动推进，不需要 Codex 再次手动启动下一阶段。

如果服务器出现临时资源问题，自动等待并安全恢复。

如果出现真正的科学正确性问题，停止并保留证据，不得隐瞒或擅自调参。

本次授权允许完成上述全部流水线部署及正式实验。

现在请开始实现、验证并启动服务器端自动流水线，完成一次性部署后报告状态。
