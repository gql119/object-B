# 任务：立即终止当前Original S4 AT实验，完成CUDA加速测试，并使用最优配置重新启动四组正式实验

## 一、最新人工授权（最高优先级）

当前研究项目：

SMIC-Det — Class-Selective Unlearnable Examples for Object Detection。

正式方法：

Original S4 = Safe-Soft05 + TFSS-Utility。

当前正在开展VOC20 Original S4对抗训练鲁棒性实验：

1. Clean-AT4
2. Poison-AT4
3. Clean-AT8
4. Poison-AT8

**最新人工决定：允许立即终止当前正在运行的AT正式实验，释放GPU资源，进行真实CUDA性能优化测试，然后统一采用经过科学验收的最快有效配置，从epoch 0重新运行四组完整AT实验。**

本次授权明确覆盖之前以下限制：

- 不再要求等待Clean-AT4训练到200epoch。
- 不再要求等待195–199epoch自然阶段边界。
- 不再要求先实现可逐位恢复的训练暂停机制。
- 不再要求通过恢复旧checkpoint继续当前实验。
- 不再要求在停止旧实验之前证明重训具有10%以上的剩余总时间净收益。
- 不再要求为已明确授权的停止、测试及重新训练重复请求人工确认。

原因是用户已决定放弃当前未完成训练的进度，优先获得真实CUDA优化数据，并采用更高效的正式训练方案。

**这不是暂停后续训，而是终止旧实验、保留历史证据、重新建立正式实验。**

但科学正确性、运行身份、数据完整性和服务器权限限制仍然必须遵守。

---

## 二、首先检查实际运行状态

读取：

- CURRENT_STATE.json
- AT_CONTINUE.md
- S4_AT_SPEED_OPTIMIZATION_AUDIT.md
- S4_AT_SCIENTIFIC_SPEEDUP_AND_RESTART_REPORT.md
- S4_AT_CUDA_BATCH_OPTIMIZATION_AND_RESTART_REPORT.md
- PENDING_BATCH_CUDA_CONTRACT.json
- Candidate 02/03相关源码和验证记录
- 当前正式RunSpec
- 当前RRCTL控制器及监控状态

历史参考：

RunID：
s4_at_serial_seed0_20261029_v2

历史正式源码：
51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb

历史CUDA测试源码：
dbdc05f938d1e553fbd62af32b2fa36f908c68d0

这些只是定位线索。

必须以最新真实运行状态为准。

重点确认：

1. 当前是哪一组AT。
2. 当前已完成多少epoch。
3. 是否已有正式完成并验收的组。
4. 当前PID、start-ticks、UID。
5. RRCTL运行身份。
6. 唯一串行队列监督器。
7. 当前GPU计算进程。
8. 当前checkpoint及日志状态。
9. 是否有其他属于本人的独立正式任务。
10. 是否存在其他用户的GPU进程。

不得仅凭历史截图判断当前状态。

---

## 三、直接终止当前AT实验

本次明确授权停止当前Original S4 AT正式队列。

### 1. 停止前保存证据

在终止之前，尽可能以只读方式保存：

- 当前RunID。
- 当前epoch与batch。
- 已完成的真实SGD更新数。
- AMP跳过记录。
- 现有checkpoint。
- 训练日志。
- 源码SHA。
- RunSpec SHA。
- 控制器身份。
- 当前进程PID/start-ticks。
- GPU资源状态。

不要求为了保存新的checkpoint而继续运行大量训练。

如果已有合法checkpoint，保持原样。

不得修改、删除或覆盖旧checkpoint。

### 2. 防止串行队列自动启动下一组

停止当前任务之前，先通过现有合法RRCTL机制阻止同一队列自动拉起后续AT任务。

必须检查队列监督器与科学训练子进程之间的控制关系。

不能只结束Clean-AT4，却意外自动启动Poison-AT4。

不得创建第二个竞争性监控器。

### 3. 执行正式终止

优先通过RRCTL现有受控终止接口执行。

确认目标为本人拥有的当前AT实验进程。

允许使用RRCTL已有的正常终止流程，即使该流程不支持恢复训练。

如果正常TERM无法及时退出，可以使用RRCTL受控终止流程已经定义的升级机制，但必须确认目标进程身份准确、信号作用范围明确且不会影响其他任务。

禁止：

- sudo。
- 终止其他用户进程。
- 对不明PID发送信号。
- 删除锁文件伪造任务结束。
- 绕过RRCTL直接操作未经核验的远端进程。
- 重启GPU驱动。
- 影响其他实验。

若RRCTL终止接口存在可修复的工程问题，允许在本项目授权范围内自行排查解决，但不能降低运行身份校验要求。

### 4. 验证终止

确认：

- 原科学训练进程确实退出。
- 原队列不会继续自动启动下一组。
- GPU资源已经释放。
- RRCTL真实终态已记录。
- 旧日志和checkpoint仍完整。
- 旧实验被明确标记为提前终止/被新实验取代。

不得将旧实验标记为COMPLETED或ACCEPTED。

保留旧RunID和所有历史记录。

---

## 四、第一阶段：Candidate 02和03真实CUDA科学等价验证

当前已有两个优化候选。

Candidate 02：

合并PGD循环中的诊断标量CPU回读，减少GPU同步开销。

Candidate 03：

在Candidate 02基础上向量化GT匹配计数审计。

此前CPU小规模测试已经通过。

现在必须进行真实CUDA验证。

使用：

- YOLOv8n。
- 真实VOC训练图像。
- 完整检测GT。
- Batch36。
- imgsz640。
- PGD10。
- AT4与AT8。
- FP32 PGD内层。
- AMP外层。
- 原始模型初始化。
- 固定输入和随机状态。
- 原生完整YOLO检测损失。

比较：

1. PGD10每次迭代的输入梯度。
2. 检测损失。
3. 完整动态TAL。
4. 每步扰动。
5. 最终L-infinity预算。
6. 外层参数梯度。
7. 实际SGD更新。
8. AMP与GradScaler。
9. BN。
10. EMA。
11. optimizer。
12. scheduler。
13. RNG。
14. checkpoint恢复状态。

延续已有冻结验证合同。

不能因希望通过测试而放宽验收阈值。

如果出现差异，允许自行分析和修复。

如果无法证明某个候选正确，则将其排除，不得进入正式训练。

---

## 五、第二阶段：真实CUDA工程提速测试

通过CUDA科学等价验证后，比较：

- Original + Batch36
- Candidate02 + Batch36
- Candidate03 + Batch36

必须使用真实PGD10和完整外层训练。

不能使用虚假的模型前向测试代替。

测试要求：

- 相同GPU。
- 相同真实训练数据。
- 相同GT。
- 相同模型。
- 相同AMP与PGD设置。
- CUDA预热。
- 多次稳定测速。
- 尽可能使用ABBA交替对照。
- 排除明显的GPU资源竞争。
- 记录GPU温度、功率和时钟。

统计：

- seconds/batch。
- images/s。
- seconds/epoch。
- PGD内层时间。
- 外层训练时间。
- GPU显存峰值。
- 实际optimizer更新。
- AMP跳步情况。
- 稳定速度提升百分比。

如果Candidate02/03无法证明明显提速或出现异常，继续使用原始实现。

不为了使用新代码而强行选择候选。

---

## 六、第三阶段：CUDA Batch Size性能测试

测试：

- Batch36
- Batch48
- Batch64
- Batch72

优先比较Batch36与Batch72。

GPU为RTX A6000 48GB，但实际可用显存和算力必须重新检查。

当前原始协议：

Batch36。

nbs64。

稳定阶段accumulate2。

近似有效Batch72。

历史审计得到：

Batch36：accumulate2，460 batch/epoch。

Batch48：accumulate1，345 batch/epoch。

Batch64：accumulate1，259 batch/epoch。

Batch72：accumulate1，230 batch/epoch。

必须通过实际原生Trainer验证以上规则，而不能将它们视为固定不变的硬编码值。

### 测试要求

全部保持：

- YOLOv8n。
- imgsz640。
- PGD10。
- AT4/AT8原预算及步长。
- 完整GT。
- 动态TAL。
- 原生检测损失。
- 原始数据增强。
- FP32内层。
- AMP外层。
- 200epoch正式训练协议。

进行真实短程CUDA训练。

统计：

- images/s。
- seconds/batch。
- 预计seconds/epoch。
- 实际accumulate。
- 有效Batch。
- optimizer更新频率。
- GPU峰值显存。
- GPU计算利用率。
- GPU温度和时钟。
- AMP跳步率。
- OOM/NaN。
- 真实尾Batch处理。

### 学术要求

Batch变化在机器学习实验中可以接受，但属于新的训练超参数协议。

不能声称Batch72与Batch36严格数学等价。

必须核查：

- BatchNorm统计差异。
- 梯度累积差异。
- Warmup。
- Weight decay。
- Optimizer step数量。
- EMA更新。
- Scheduler。
- 动态TAL。
- 数据增强随机性。
- 最后一批图像覆盖。

如果原生Ultralytics会因为Batch变化自动缩放实际weight decay，需要明确记录并决定新协议。

优先保持原主要优化器超参数和实际正则化强度，不能偷偷改变训练条件。

若Batch48或64导致优化器更新数量、有效Batch或学习率相关行为发生显著变化，必须明确披露并完成科学审查。

所有正式Clean与Poison组必须采用相同的新Batch协议。

---

## 七、第四阶段：选择实际最快且科学有效的方案

对所有通过验证的方案生成对照表：

Implementation | Batch | Images/s | Epoch time | GPU memory | Scientific status

选择标准：

1. 科学定义正确。
2. 真实PGD10完整执行。
3. 检测GT及TAL正确。
4. 训练稳定。
5. 没有异常数值行为。
6. 真实CUDA吞吐更高。
7. 显存有合理余量。
8. 适合200epoch长程训练。
9. Clean与Poison能够严格配对。

优先选择具有稳定、可重复且明显吞吐提升的配置。

可以继续采用既有15%稳定加速作为优选门槛。

但旧实验已经获得新的终止授权，因此不再需要计算“能否通过挽回旧epoch进度获得10%净收益”来决定是否允许开展测试和重启。

如果没有任何候选获得可靠性能提升，则选择Original + Batch36作为保底正式配置。

**不得因为旧实验已经终止，就强行采用未经验证的Candidate或新Batch。**

无论最终选择加速版还是原始版，都要确保新的正式四组实验能够从头完整执行。

---

## 八、第五阶段：冻结并重新启动正式AT

完成上述测试及一次必要的科学审查后，建立新的正式源码Release和RunSpec。

记录：

- Original S4父方法。
- 旧AT RunID和源码SHA。
- 新AT源码SHA。
- 最终Batch。
- 实际accumulate。
- 完整优化器参数。
- scheduler。
- PGD10参数。
- AMP与BN规则。
- 候选工程优化。
- CUDA等价性结果。
- 实测吞吐。
- 显存及训练稳定性。
- 新RunID。

使用新的独立实验目录。

不能覆盖原正式训练数据和checkpoint。

### 四组训练统一从epoch0运行

严格串行：

1. Clean-AT4，200epoch。
2. Poison-AT4，200epoch。
3. Clean-AT8，200epoch。
4. Poison-AT8，200epoch。

所有四组采用同一经过验收的实现和Batch配置。

Victim仍为YOLOv8n随机初始化、seed0。

如果最终选择Batch72，则四组统一使用Batch72。

如果最终选择Candidate03 + Batch72，则四组统一使用经过验证的Candidate03 + Batch72。

不能出现Clean和Poison采用不同Batch或不同未经验证的Trainer。

不允许从旧实验中断checkpoint继续训练。

新实验必须从约定的随机初始化开始。

---

## 九、正式科学协议不可改变

保持：

- Original S4 = Safe-Soft05 + TFSS-Utility。
- VOC20 train16,551。
- Clean val4,952。
- Person class14。
- Non-target19类。
- 原S4保护图像及完整GT。
- YOLOv8n。
- seed0。
- imgsz640。
- 200epoch。
- AT4 epsilon4/255。
- AT8 epsilon8/255。
- PGD10。
- AT4 step size1/255。
- AT8 step size2/255。
- 随机PGD初始化和L-infinity投影。
- 原生YOLO完整检测损失。
- FP32内层PGD。
- 经审计的AMP外层。
- 完整clean val评价。
- last.pt@200主评价。

Batch和必要梯度累积是本次唯一允许独立重新确定的核心训练超参数。

禁止：

- PGD10改为PGD5。
- 减少200epoch。
- 降低分辨率。
- 更换模型。
- 删除GT。
- 修改TAL机制。
- 使用FGSM代替PGD。
- 改变原始S4扰动。
- 使用ST checkpoint微调代替随机初始化。
- 关闭必要科学验证。
- 为提高速度而改变攻击预算。

---

## 十、正式实验完成后评价

完整执行四组200epoch。

在干净VOC val4952上评价全部20类。

主评价last.pt@200。

必须输出：

- Person AP50。
- NT19 mAP50。
- All mAP50。
- Person AP50:95。
- NT AP75。
- NT mAP50:95。
- 全20类逐类AP。
- WCD。
- TCD3。
- 每组模型与RunSpec身份。

Clean-AT4与Poison-AT4配对。

Clean-AT8与Poison-AT8配对。

计算：

P_AT = 1 - PersonAP50(Poison_AT) / PersonAP50(Clean_AT)

U_AT = NTmAP50(Poison_AT) / NTmAP50(Clean_AT)

S_AT = 2 * P_AT * U_AT / (P_AT + U_AT)

检查分母合法性。

不依据验证曲线挑选有利checkpoint替代last.pt。

如果新Batch不同于原标准训练Batch36，应在论文中明确披露。

---

## 十一、允许自主解决工程问题

本次授予Codex较完整的本项目工程执行权限。

允许：

- 修复CUDA测试入口。
- 修复Candidate02/03实现。
- 修复Batch传参。
- 修复合法的GT处理问题。
- 优化CPU/GPU同步。
- 修复测试RunSpec和源码绑定。
- 修复RRCTL本人任务终止和重新注册问题。
- 建立新的不可变源码Release。
- 执行必要的真实CUDA测试。
- 检查显存和OOM。
- 重新进行失败的有限测试。
- 自动选择经验证最快的合理配置。
- 自动启动新的正式四组训练。

出现工程问题时，先定位原因、自行尝试修复，再运行必要验证。

不能通过降低科学要求来解决问题。

不允许：

- sudo。
- 操作其他用户文件或进程。
- 修改共享驱动。
- 修改其他Conda环境。
- 修改已经验收的COCO结果。
- 删除旧实验产物。
- 绕过RRCTL。
- 强制覆盖Git历史。
- 创建多个竞争性GPU长程控制器。
- 修改Original S4数据。
- 使用未经确认的第三方模型。
- 擅自引入新的AT防御算法。

如果确实遇到无法解决的阻塞，必须报告真实状态，不得虚报正式实验已启动。

---

## 十二、任务执行纪律

本次任务不是单纯写一份优化计划。

应真实执行：

1. 核实当前任务身份。
2. 通过授权控制接口终止旧AT队列。
3. 验证GPU释放。
4. 运行CUDA等价性测试。
5. 运行真实Batch性能测试。
6. 选择最优有效配置。
7. 完成科学审查。
8. 冻结新版本。
9. 启动四组新的正式AT实验。

性能测试应有合理、明确的执行上限，避免无限测试延迟正式训练。

如果工程测试结果已足以确定最佳方案，不必为了测试更多Batch而无限扩展搜索。

如果某候选不稳定或不能验收，应尽早排除。

不创建第二个独立监控体系，现有唯一监控器应与新RunID正确衔接。

---

## 十三、最终报告

生成：

S4_AT_DIRECT_STOP_CUDA_OPTIMIZATION_AND_RESTART_REPORT.md

记录：

1. 旧AT终止前实际状态。
2. 旧实验结束回执。
3. GPU释放证据。
4. 原始checkpoint及日志保留情况。
5. Candidate02/03 CUDA等价性。
6. Candidate02/03提速百分比。
7. Batch36/48/64/72真实吞吐。
8. 实际最佳Batch。
9. 显存峰值及稳定性。
10. Batch修改带来的科学协议差异。
11. 最优配置选择原因。
12. 新正式源码SHA和RunSpec。
13. 新RunID。
14. 四组实验的实际启动状态。
15. 当前真实运行的训练任务。
16. 是否存在未解决的科学风险。

结论使用实际状态：

- OLD_AT_TERMINATED
- CUDA_BENCHMARK_COMPLETED
- OPTIMIZED_PROTOCOL_VALIDATED
- NEW_AT_CONTROLLER_STARTED
- FALLBACK_ORIGINAL_PROTOCOL_STARTED
- BLOCKED

不能将尚未执行的测试或训练描述为已完成。

---

## 十四、最终明确指令

**用户现在明确决定终止旧AT实验并重新开展正式实验，不需要再次请求批准。**

不要再等待Clean-AT4训练完成200epoch。

不要继续等待原先的自然阶段边界。

先确认本人进程身份，再使用现有合法RRCTL机制关闭当前AT队列。

旧实验数据和checkpoint保留，但不要求恢复旧训练进度。

GPU释放后，立即开展Candidate02/03 CUDA等价测试和Batch36/48/64/72真实性能比较。

选择科学验收通过、稳定且速度最优的配置。

然后从随机初始化epoch0统一重新启动四组200epoch AT实验。

若没有任何新配置通过科学验证或获得可靠加速，则使用原始Batch36、原始已验收Trainer重新启动四组，而不是让项目长期停滞。

保持Original S4、PGD10、完整GT及200epoch科学要求不变。

**现在开始执行，不再停留在只读审计和CUDA窗口等待阶段。**