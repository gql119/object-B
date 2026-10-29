# 任务：验证S4-AT工程加速候选、测试CUDA Batch Size，并在满足条件后自动重新运行四组正式实验

## 一、任务目标与授权范围

当前科研项目：

SMIC-Det — Class-Selective Unlearnable Examples for Object Detection。

唯一科学父方法：

Original S4 = Safe-Soft05 + TFSS-Utility。

目前正在进行VOC20数据集的Original S4对抗训练鲁棒性实验：

- Clean-AT4
- Poison-AT4
- Clean-AT8
- Poison-AT8

现有AT实验运行时间较长。

此前的性能审计发现两个工程优化候选，同时GPU显存容量仍有较多余量。

本次授权Codex完成以下工作：

1. 读取当前AT正式运行状态与历史性能报告。
2. 验证现有Candidate 02和Candidate 03的CUDA科学等价性。
3. 进行真实GPU Batch Size性能测试。
4. 对比不同实现和Batch配置的实际训练吞吐。
5. 选择科学上有效、性能上最优的候选方案。
6. 判断新配置是否显著缩短全部四组实验的剩余完成时间。
7. 如果全部验收通过，允许安全停止现有AT实验。
8. 使用统一的新配置，从随机初始化重新运行四组完整200epoch AT实验。
9. 完成评价与结果归档。

**对于已有科学定义内的工程问题，允许Codex自行分析、修改、测试、修复和重新验证，不需要每次询问人工。**

但是，涉及新的科学方法定义、数据泄漏、无法证明公平性的实验协议修改，以及不能安全处理的运行控制问题，不允许通过绕过验收强行解决。

本次任务不是简单寻找最大Batch，而是寻找实际总运行时间更短、训练设置明确、实验结果可靠的配置。

---

## 二、首先读取既有报告与运行状态

先读取以下文件：

1. S4_AT_SPEED_OPTIMIZATION_AUDIT.md
2. S4_AT_SCIENTIFIC_SPEEDUP_AND_RESTART_REPORT.md
3. CURRENT_STATE.json
4. AT_CONTINUE.md
5. Candidate 02和03的源码补丁。
6. PENDING_CUDA_BENCHMARK_CONTRACT.json
7. 当前正式AT的RunSpec与RRCTL绑定。

从实际文件中查找真实路径，不凭名称猜测文件已存在。

此前性能审计中的参考结果：

- 正式Batch：36。
- imgsz：640。
- PGD：10步。
- 每Batch平均耗时：约1.6884秒。
- 每Epoch平均耗时：约785.34秒。
- GPU平均利用率：83.5%。
- GPU进程显存：约10 GiB。
- 总GPU显存：24 GiB。
- 内层PGD使用FP32。
- 外层训练使用AMP。
- nbs=64，稳定阶段原生accumulate=2。
- DataLoader已使用缓存、pin_memory及预取。
- GPU存在部分软件热限制和功率限制记录。

这些是历史参考，不代表当前实时状态。

必须重新核实当前：

- RunID。
- PID/start-ticks。
- 训练阶段。
- 当前epoch。
- 已完成及剩余实验。
- 源码SHA。
- 实际训练配置。
- 当前GPU使用情况。
- 是否有其他正式任务运行。
- 当前RRCTL是否允许安全停止。

不允许重复执行已验收的COCO实验。

---

## 三、保持不变的核心科学要求

Original S4数据集不修改：

- VOC20。
- Person类别ID14。
- Train 16,551张。
- Clean val 4,952张。
- 原S4扰动图像及GT完全不变。

Victim核心设置：

- YOLOv8n。
- 随机初始化，seed0。
- imgsz640。
- 200epoch。
- PGD-10。
- AT-4预算4/255。
- AT-8预算8/255。
- 原生完整YOLOv8检测损失。
- 原始GT全部参与训练。
- 原优化器算法及已冻结的主要学习率调度。
- 干净验证集评价。
- last.pt@200为主评价。

PGD步长、随机初始化分布、投影、BN处理和AMP细节，以已经验收的正式RunSpec为准。

禁止通过以下方式获得速度提升：

- 减少PGD步数。
- 缩减训练epoch。
- 降低图像分辨率。
- 减少数据集。
- 更换YOLOv8n架构。
- 删除合法GT。
- 简化TAL。
- 去掉box、cls、DFL损失。
- 使用旧梯度代替PGD实时梯度。
- 使用FGSM代替PGD。
- 取消原生数据增强。
- 关闭原有验证流程以制造更有利的速度。
- 擅自改变GradScaler或优化器更新规则。
- 使用旧AT checkpoint微调冒充从头训练。

**唯一额外允许调整的核心训练超参数是Batch Size及其必要的、经过审计的梯度累积配置。**

需要明确：调整Batch属于新的完整实验协议，而不是数学上与旧Batch36完全等价的代码优化。

---

## 四、第一阶段：验证Candidate 02与03

现有两个工程候选：

### Candidate 02

将PGD过程中分散的诊断标量CPU回读合并，减少CPU/GPU同步。

### Candidate 03

在Candidate 02基础上，将审计使用的GT匹配数量计算进一步向量化。

历史报告显示：

- 两项候选分别通过5个CPU测试。
- CPU对照atol=rtol=0。
- PGD输入与梯度一致。
- 动态TAL一致。
- 检测损失一致。
- 扰动与外层SGD更新一致。
- BN、EMA、优化器及scheduler状态一致。

但历史测试为：

batch2、img160、CPU、AMP关闭。

因此，这些候选尚未得到正式CUDA等价性证明。

### CUDA验证要求

在安全GPU窗口内，以原始实现作为参考，分别验证：

1. Original vs Candidate 02。
2. Original vs Candidate 03。
3. Candidate 02 vs Candidate 03（如有必要）。

使用：

- CUDA真实YOLOv8n。
- imgsz640。
- batch36。
- PGD10。
- 原始Clean和Poison实际样本。
- AT4和AT8两种预算。
- FP32内层。
- AMP外层。
- 完整GT。
- 相同初始模型。
- 相同输入数据。
- 相同随机PGD初始化。

检查：

- PGD10每一步的输入梯度。
- 检测损失。
- 动态TAL分配。
- 最终对抗扰动。
- L-infinity预算。
- 外层参数梯度。
- optimizer更新。
- AMP跳步。
- BN统计。
- EMA。
- scheduler。
- RNG状态。
- checkpoint恢复。

对CUDA数值差异使用测试开始前锁定的合理容差，并保留逐项结果。

如果出现不一致，先定位原因，允许自行修复并重新测试。

不得通过扩大容差、删除验证项或忽略异常来强行PASS。

科学等价性通过后，再评估真实性能。

---

## 五、第二阶段：真实CUDA Batch Size性能测试

本次重点比较：

| 配置 | Batch Size | 梯度累积 |
|---|---:|---|
| Baseline | 36 | 原生规则 |
| Candidate A | 48 | 由实际协议核验 |
| Candidate B | 64 | 由实际协议核验 |
| Candidate C | 72 | 由实际协议核验 |

可以在完成上述测试后，根据真实性能补充其他合理的Batch，但必须说明理由。

### 特别注意有效Batch

当前原配置：

batch36、稳定阶段accumulate2。

稳定阶段近似有效Batch为72。

候选batch72、accumulate1也可能得到近似相同的有效Batch。

但是：

**有效Batch相同不意味着训练行为严格等价。**

必须检查：

1. BatchNorm的统计变化。
2. 每次PGD的实际输入Batch。
3. TAL在不同Batch下的行为。
4. Mosaic和其他数据增强。
5. optimizer实际更新次数。
6. 原生nbs与accumulate计算。
7. warmup期间梯度累积变化。
8. 学习率调度的实际单位和触发时机。
9. weight decay是否被原生逻辑自动重标定。
10. 尾Batch处理及每epoch图像覆盖。
11. AMP GradScaler行为。
12. EMA更新节奏。

不得认为batch72天然比batch36等价或更优。

如果调整Batch导致优化器超参数被Ultralytics自动改变，必须识别具体变化。

对于这些变化：

- 可以通过显式配置保持原主要优化器协议。
- 不能绕过原有数值正确性验证。
- 不能偷偷修改学习率、权重衰减或warmup以追求更高性能。
- 无法保持的差异必须列入新的训练协议并判断其学术可接受性。

若batch72显存不足或更慢，不要强行使用。

### 性能测试必须采用真实训练计算

测试包含：

- 真正的PGD-10。
- 完整检测损失。
- 原始GT。
- 原生数据增强。
- 真实外层训练更新。
- CUDA数据传输与DataLoader。
- AMP及既定状态管理。

不能使用随机张量、空模型、单次前向传播或PGD-1代替真实训练基准。

每种候选须进行：

1. 预热。
2. 多次稳定训练batch。
3. 多种真实样本。
4. Clean与Poison样本覆盖。
5. AT4和AT8预算覆盖。
6. 有代表性的多目标、重叠GT和小目标检查。
7. CUDA同步计时及端到端墙钟计时。
8. 显存峰值统计。
9. GPU温度、功率及频率记录。

不要在GPU明显热降频或受到其他任务干扰的情况下，直接认定某种Batch更快。

建议交替测量候选配置，避免测试先后顺序导致偏差。

### 必须记录的性能指标

对每种候选记录：

- Batch Size。
- 实际accumulate。
- 稳定阶段有效Batch。
- 每epoch batch数。
- 每epoch真实SGD更新数。
- 每batch耗时。
- images/s。
- 每epoch预计耗时。
- 内层PGD耗时。
- 外层训练耗时。
- GPU显存峰值。
- GPU计算利用率。
- 温度及是否发生功率/热限制。
- CPU/RAM/I/O开销。
- AMP跳过更新数。
- OOM、NaN等异常。

优先使用实际images/s与epoch端到端耗时进行比较，不能只看单个batch耗时。

---

## 六、第三阶段：比较组合优化方案

完成单项测试后，对下列组合进行实际比较：

- 原始实现 + Batch36。
- Candidate 02 + Batch36。
- Candidate 03 + Batch36。
- 最佳已验收实现 + Batch48。
- 最佳已验收实现 + Batch64。
- 最佳已验收实现 + Batch72。

如果Candidate 02和03没有形成可证明的显著提速，可以保留原始实现，仅比较Batch。

如果某项优化增加显存使用、引起数值不一致或使运行不稳定，即使单batch速度更快，也不能直接采用。

选择最佳方案时综合考虑：

1. 实际吞吐。
2. 完整epoch耗时。
3. 数值正确性。
4. 训练状态稳定性。
5. 显存安全余量。
6. 有效optimizer更新效率。
7. Clean/Poison配对的公平性。
8. 长程运行的可恢复性。

如性能差异落在测量噪声范围内，优先保持原配置。

---

## 七、第四阶段：独立科学协议验收

必须明确分开两种情况。

### 情况A：只有工程代码优化

例如只改变诊断同步方式，不改变Batch。

要求：

- PGD数学定义不变。
- 检测损失不变。
- 数据顺序及GT不变。
- 外层训练更新一致。
- CUDA数值验证通过。

此类优化可标记：

ENGINEERING_EQUIVALENT。

### 情况B：改变Batch Size

例如Batch36改为Batch72。

这种修改在机器学习实验中是允许的，但应视为新的训练超参数配置。

不能要求Batch36和Batch72的完整训练轨迹逐位一致。

应验证：

- 新Batch设置能够正确训练。
- 优化器、调度及累积变化明确。
- Person及19个非目标类别监督完整。
- 两组防御预算下Clean/Poison协议一致。
- 所有训练都从随机初始化开始。
- 所有四组均重新运行200epoch。
- 最终论文完整披露Batch、accumulate与其他必要参数。

此类修改应标记：

NEW_VALIDATED_TRAINING_PROTOCOL。

如果存在无法合理解释的训练定义变化，标记：

SCIENTIFIC_REVALIDATION_REQUIRED。

不允许把新Batch训练曲线与旧Batch训练曲线称为严格数值等价。

---

## 八、第五阶段：判断是否值得从头重新训练

本次明确授权在满足全部条件后安全停止旧AT并重启。

但必须有实际性能证据。

### 速度门槛

建议新配置相对于原正式配置具有：

稳定实际训练吞吐提升至少15%。

该15%是本项目的工程决策阈值，不是学术论文规定。

### 总时间门槛

不仅比较单epoch速度，还要比较完成四组实验所需的实际剩余时间。

方案A：

继续当前AT运行，完成剩余所有实验。

方案B：

停止当前AT，统一使用新配置重新完成四组200epoch。

计算包括：

- 原方案已完成epoch。
- 原方案剩余训练。
- 新配置每epoch时间。
- 四组各200epoch完整运行。
- Clean/Poison及AT4/AT8之间的运行成本差异。
- 数据加载及验证。
- checkpoint保存。
- 初始化及可能的编译成本。
- 科学验收成本。
- 停止与重新启动成本。

必须考虑不同Batch下的epoch迭代次数变化。

不能直接用旧batch耗时乘以新batch数量估计正式结果。

建议方案B在考虑合理不确定性后，净剩余时间节省至少10%，才触发重启。

如果只有少量速度提升，或者当前已有大量已完成进度，应继续原实验。

### 必须同时满足

1. 新配置科学上允许。
2. 真实CUDA测试通过。
3. 关键工程优化正确。
4. 性能提升达到门槛。
5. 四组总剩余时间确实减少。
6. 训练可完整重新开始。
7. 旧运行可以被安全关闭。
8. 不影响其他现有正式任务。
9. 不影响Original S4或COCO数据。
10. 新版本的源码、RunSpec、配置和测试报告已冻结。

任何条件不满足，不得为了执行重启授权而强行停止现有任务。

---

## 九、第六阶段：安全停止旧实验与重新启动

如果重启条件全部满足：

### 1. 先冻结新版本

记录：

- Original S4父方法身份。
- 旧AT源码SHA。
- 新AT源码SHA。
- 新Batch Size。
- 实际accumulate。
- 优化器和scheduler配置。
- PGD配置。
- 所有科学验证结果。
- 实际CUDA测速数据。
- Git commit及RunSpec身份。

新旧代码不得混用。

### 2. 安全停止原有AT

现有报告指出：

- RRCTL缺少已经验证的非终态pause/continue机制。
- Controller.resume不等同于恢复训练。
- abort可能导致TERM之后SIGKILL。
- 旧队列的输出目录和锁不能简单重复使用。

因此，请Codex先分析如何通过现有控制机制实现安全、可验证的停止。

如果现有系统无法安全暂停，可以在不干扰当前生产进程的前提下，研究并实现合作式停止机制或通过现有合法控制器在安全边界停止。

允许Codex自行修复运行控制工程问题，但不得：

- 修改当前正在执行的科学计算。
- 向其他用户进程发送信号。
- 直接破坏正在写入的checkpoint。
- 清除旧队列锁。
- 伪造进程退出状态。
- 使用未经验证的强制终止路径冒充安全暂停。

如果无法保证安全停止，则不得重启，应继续原实验并报告BLOCKED_SAFE_STOP。

### 3. 保留旧实验记录

必须保存：

- 旧RunID。
- 原始源码SHA。
- RunSpec。
- checkpoint。
- 完整训练日志。
- 已完成epoch。
- 中止位置。
- 控制器停止回执。
- 实际GPU进程退出状态。

不得删除或覆盖历史结果。

### 4. 从epoch 0重训全部四组

统一采用同一个经过验收的新实现与Batch配置。

严格串行：

1. Clean-AT4，200epoch。
2. Poison-AT4，200epoch。
3. Clean-AT8，200epoch。
4. Poison-AT8，200epoch。

所有模型均从原先约定的随机初始化协议重新开始。

禁止加载旧AT训练到一半的权重作为初始模型。

同一预算下Clean与Poison的初始化、Batch、optimizer、scheduler、GT、数据增强和评价规则必须一致。

如使用Batch72，就四组全部使用Batch72。

不得出现Clean使用Batch36、Poison使用Batch72的混合配置。

---

## 十、第七阶段：正式结果评价

四组全部执行200epoch。

完整VOC20干净验证集4,952张。

主评价last.pt@200。

记录：

- Person AP50。
- Non-target 19类mAP50。
- All mAP50。
- Person AP50:95。
- NT AP75。
- NT mAP50:95。
- 全20类逐类AP。
- WCD。
- TCD3。
- 完整checkpoint身份。

计算：

P_AT = 1 - AP50_person(Poison_AT) / AP50_person(Clean_AT)

U_AT = mAP50_NT(Poison_AT) / mAP50_NT(Clean_AT)

S_AT = 2 * P_AT * U_AT / (P_AT + U_AT)

检查所有分母有效性。

Clean-AT4只与Poison-AT4比较。

Clean-AT8只与Poison-AT8比较。

现有Original S4标准训练数据保留为参考：

Clean-ST：

Person AP50 = 86.7943%。

NT mAP50 = 76.6462%。

Original S4 Poison-ST：

Person AP50 = 2.8159%。

NT mAP50 = 69.3787%。

如果新的AT使用不同Batch，而ST基线仍使用Batch36，必须在论文中披露该配置差异，不得暗示整个实验矩阵使用完全相同的训练超参数。

---

## 十一、允许Codex自主解决工程问题

本次给予Codex较充分的工程自主权。

遇到问题时，可以自行：

- 阅读项目代码。
- 检查相关依赖API。
- 分析异常堆栈。
- 在隔离分支修复实现。
- 改善DataLoader性能。
- 优化CPU/GPU同步。
- 调整安全的测试入口。
- 设计真实CUDA对照测试。
- 修复Batch配置传递错误。
- 修复候选优化的正确性问题。
- 完善checkpoint与RunSpec身份校验。
- 修复本项目合法范围内的RRCTL控制代码。
- 完成多次必要的测试和验证。

对于无法解决的问题：

先记录具体问题、失败证据及尝试措施。

允许在不改变科学定义的前提下采用其他工程解决方案。

不能通过降低数据规模、跳过计算、删除GT、更改PGD次数或取消测试来制造成功。

禁止：

- sudo。
- 修改他人文件或进程。
- 修改共享CUDA驱动。
- 修改系统级环境。
- 未授权安装全局依赖。
- 重置GPU驱动。
- 修改VPN/TLS认证。
- 删除正式数据。
- 强制覆盖Git历史。
- 创建多个并发正式控制器。
- 干扰已经验收的COCO项目。

如遇无法通过合法工程措施解决的阻塞，保持当前正式实验运行并报告。

---

## 十二、最终报告

请生成：

S4_AT_CUDA_BATCH_OPTIMIZATION_AND_RESTART_REPORT.md

报告至少包含：

1. 当前实际AT运行状态。
2. 两个工程优化候选的CUDA等价性结果。
3. Candidate 02与03的真实测速。
4. Batch36/48/64/72性能对照表。
5. 不同Batch下的实际梯度累积。
6. 各配置images/s与每epoch耗时。
7. 显存峰值、GPU温度及异常。
8. 是否存在AMP跳步或训练数值风险。
9. 选择最佳配置的理由。
10. 旧配置剩余总耗时。
11. 新配置四组重训总耗时。
12. 预计净节省时间及测量依据。
13. Batch变化的科学协议说明。
14. 安全停止可行性。
15. 是否实际重启。
16. 新旧RunID及源码SHA。
17. 当前四组正式实验的实际状态。

最终给出以下结论之一：

- OPTIMIZATION_VALID_RESTART_EXECUTED
- OPTIMIZATION_VALID_RESTART_NOT_WORTHWHILE
- ENGINEERING_EQUIVALENCE_FAILED
- BATCH_CHANGE_SCIENTIFICALLY_UNACCEPTABLE
- BLOCKED_SAFE_STOP
- NO_SIGNIFICANT_SPEEDUP
- INCONCLUSIVE

只报告实际完成的测试、操作与结果。

---

## 十三、最终执行要求

本次不是只做静态审计。

请尽可能推进到真实CUDA性能测试，并根据实测结果完成决策。

执行顺序：

**读取现有报告和运行状态 → 验证Candidate 02/03 → 完成真实CUDA Batch测试 → 完成科学审查 → 计算四组总耗时 → 满足门槛则安全停止旧AT → 冻结新配置 → 从epoch0串行重启四组AT。**

重要：

- 不允许为获得测试GPU而冒险强制停止当前实验。
- 优先寻找不与正式运行冲突的合法GPU测试窗口。
- 如果无法获得安全测试窗口，保留当前运行并记录下一次可执行的测试条件。
- 若旧AT进度已经变化，必须使用最新实际进度重新计算重启收益。
- 如果新版本不能证明更快或更可靠，则维持原方案。
- 如果科学验收、性能门槛和安全停止条件全部通过，允许自动完成重启，不需要重复人工确认。

**现在开始执行。优先完成科学正确性和真实CUDA测速；只有结果证明整体收益显著时，才统一使用新配置重新训练四组AT。**