# 任务：基于 Original S4 开展 AT-4/255 与 AT-8/255 鲁棒性实验

## 一、任务目标与严格边界

当前项目为 Class-Selective Unlearnable Examples for Object Detection，论文方法名 SMIC-Det。

**当前唯一正式父方法：Original S4 = S0 Safe-Soft05 + TFSS-Utility。**

请基于已经验收的VOC20 Original S4受保护训练集，实现目标检测Victim的Adversarial Training（AT）鲁棒性评估，包括AT-4/255和AT-8/255。

本次只扩展**下游Victim训练与鲁棒性评估**，不修改S4扰动生成方法，不重新优化Fourier参数，不改变S4的科学定义。

研究目的：判断采用对抗训练后，受保护Person类别的可学习性是否恢复，以及其余19类检测性能如何变化。

先审计现有代码和实验产物，再实现独立AT评估模块，执行有限预检。预检通过且资源允许后，可串行启动正式实验。

## 二、需要继承的Original S4实验协议

VOC20：

- 目标Person：YOLO class ID 14。
- 非目标类别：其余19类。
- 训练集：16,551张。
- 干净验证集：4,952张。
- Victim：YOLOv8n，随机初始化。
- Seed：0。
- Epochs：200。
- 原标准训练配置：batch36、workers16。
- 主评价：last.pt@200。
- best.pt仅作为次要参考，不能代替last.pt主结果。

已验收参考结果：

| 数据 | Person AP50 | NT mAP50 |
|---|---:|---:|
| Clean ST | 86.7943% | 76.6462% |
| S4 Poison ST | 2.8159% | 69.3787% |

原始S4毒化训练集只对Person相关区域加入扰动，且满足：

- Person独占区域≤8/255。
- 与非目标GT框重叠区域≤4/255。
- Person框外不扰动。
- 无Person图像保持不变。

**禁止改变S4已导出的任何训练图像、标签、checkpoint及历史实验产物。**

请从原始实验的manifest、源码、config、Git SHA、数据SHA中查找实际身份，不得凭文件名推断其科学有效性。

## 三、AT科学协议

采用PGD-based Adversarial Training，即在Victim训练阶段对当前训练输入生成对抗样本，然后用这些样本正常更新YOLOv8n。

防御预算：

- AT-4：epsilon_AT = 4/255
- AT-8：epsilon_AT = 8/255

建议先采用统一、明确标注为本项目自定义检测适配的PGD-10协议：

- PGD steps：10。
- AT-4 step size：1/255。
- AT-8 step size：2/255。
- 随机初始化：Uniform(-epsilon_AT, epsilon_AT)，使用受控RNG。
- 约束：L∞，每一步投影回对应epsilon_AT-ball。
- 最终图像clamp至[0,1]。
- 攻击目标：最大化YOLOv8完整GT监督检测损失，包括box、cls、DFL。
- 内层PGD不修改GT、不更新模型权重和优化器状态。
- 外层训练每个正常optimizer step执行一次真实模型更新。

这组PGD具体参数属于本项目拟采用的检测适配设置，不能声称是MI-UE论文Table 4的原始完整参数。论文明确报告AT-4/AT-8，但未充分披露其所有PGD细节。

对抗训练的数学过程使用如下纯文本表达式：

**内层PGD：**

delta_star = argmax_delta L_det(f_theta(clip(x + delta, 0, 1)), Y)

约束：norm_inf(delta) <= epsilon_AT

**外层Victim参数更新：**

theta_new = theta - lr * grad_theta[L_det(f_theta(clip(x + delta_star, 0, 1)), Y)]

其中：

- x：当前Victim训练管线经过正常数据增强、LetterBox及归一化后的实际输入。
- Y：同步变换后的完整合法GT。
- delta_star：通过PGD优化得到的输入对抗扰动。
- theta：当前Victim模型参数。
- lr：当前训练学习率。
- L_det：YOLOv8完整目标检测训练损失。

AT不是修改静态受保护数据，而是Victim训练中的动态内层对抗优化。

必须特别区分：

**S4原始8/4/0保护预算与AT防御预算是两个独立概念。**

AT允许在当前训练输入整图内加入≤epsilon_AT的对抗扰动，包括非目标和背景区域。这不表示S4原始毒化图像预算被放宽，也不得覆盖原始导出数据。

## 四、需要完成的四组正式实验

必须分别训练：

1. `S4AT4_Clean_seed0`：干净训练集 + AT-4。
2. `S4AT4_Poison_seed0`：Original S4受保护训练集 + AT-4。
3. `S4AT8_Clean_seed0`：干净训练集 + AT-8。
4. `S4AT8_Poison_seed0`：Original S4受保护训练集 + AT-8。

四组均从随机初始化YOLOv8n开始，不能使用ST训练完成的last.pt继续微调来冒充完整AT实验。

同一预算下，Clean和Poison应固定相同的模型结构、初始化种子、数据增强、训练超参数及评估规则。

原则上保留原Victim的200epoch、优化器、学习率调度、imgsz640、batch36和workers16等真实配置。

如果AT显存开销使原batch不可运行，先检查能否使用等效的microbatch + gradient accumulation保持有效batch及optimizer更新节奏。任何改变必须记录；特别要说明梯度累积不保证BatchNorm等行为严格等价。不可悄悄调整batch、epochs、学习率或模型结构。

原YOLO的Mosaic、MixUp、HSV、随机翻转、关闭Mosaic的调度等，应按照原ST Victim实际配置继承，Clean/Poison严格一致。

数据增强应在AT计算前完成，PGD基于实际进入检测器的训练输入施加。

## 五、YOLOv8 AT实现要求

请先检查当前Ultralytics版本及项目中Victim训练的实际调用链，再决定最小侵入式修改位置。

禁止直接大规模修改第三方Ultralytics源码。优先使用项目内独立Trainer、封装模块或明确的hook。

重点检查：

1. YOLOv8检测损失是否可以对输入图像正确反向传播。
2. PGD内部使用torch.autograd.grad获取图像梯度，避免累积模型参数梯度。
3. 每一步PGD结束都detach扰动，防止跨步计算图和显存持续增长。
4. 内层PGD不能执行optimizer.step、scheduler.step或EMA update。
5. 内层PGD前向传播不得意外多次更新BatchNorm running statistics；如需临时冻结BN统计，必须恢复原模型训练状态，并单独记录该处理。
6. 外层正式训练保持原有AMP、梯度缩放、optimizer、EMA和scheduler行为；内层PGD应保证数值稳定且输入梯度有效。
7. 标签类别、框坐标、TAL目标分配及box/cls/DFL监督必须正确，禁止静默丢弃合法GT。
8. 如果内层PGD出现NaN、无梯度、形状不匹配、异常空监督等问题，应输出真实错误证据并停止，不能将攻击改成随机噪声后继续声称AT有效。
9. 验证集不进行PGD或其他攻击，始终使用完全干净的VOC validation。
10. 训练不调用S4的Frozen/Adaptive生成模块，不更新Fourier系数，也不重新计算TFSS W。

确保AT-4与AT-8通过同一套实现，仅由配置指定预算和步长。

## 六、工程整理与版本隔离

请检查当前仓库目录结构，建立独立实验分支，建议：

`codex/s4-at-robustness-20261027`

该分支以已验收的Original S4为科学父线。

可以通过新增独立的配置、Trainer、启动入口和测试文件来实现AT，不得为了方便而修改Original S4方法数学逻辑。

建议将新增内容组织为：

- AT训练器及PGD核心实现。
- AT实验统一配置。
- Clean/Poison四组RunSpec。
- 数据与checkpoint身份校验。
- AT预检及有限smoke test。
- 训练和恢复调度。
- 完整评估及结果汇总。
- 运行说明与科研审计报告。

路径应服从当前项目实际结构，不必强行采用新的顶层目录。

所有正式运行使用独立RunID；不同预算、Clean/Poison不得共享checkpoint输出目录。

记录Git commit SHA、实际运行源码hash、配置SHA、数据清单SHA、S4受保护数据身份、Ultralytics/PyTorch/CUDA版本和随机种子。

checkpoint、日志、metrics和报告不能覆盖。

## 七、正式运行前必须完成的验证

先进行单元测试与真实YOLOv8n小规模训练验证，至少覆盖：

- epsilon=4/255与8/255的L∞预算严格正确。
- PGD产生非零且有效的输入扰动。
- 扰动后训练图像保持有效像素范围。
- GT类别和检测框信息完全保持。
- 内层PGD不更新基础模型权重。
- BN状态和优化器状态不会被内层PGD意外污染。
- 完整YOLO原生检测损失能正常求导。
- CPU/CUDA实际执行路径可用（按当前环境能力检查）。
- AMP或FP32训练路径数值有效。
- 数据增强、训练、验证的实际调用顺序正确。
- 无意外的Person-only目标损失或非目标GT遗漏。
- 相同RunSpec及checkpoint可以恢复训练。
- Clean与Poison使用一致的训练协议。

执行真实小批量PGD10 + 一个外层optimizer update，记录实际L_det、输入梯度、PGD最终扰动范数、显存峰值和执行结果。

不要用随机张量单元测试替代真实YOLO检测路径验收。

预检失败时，不允许启动正式200epoch训练。

## 八、资源控制与正式执行

远端学校服务器只允许操作本人账户文件与本人进程，不使用sudo，不修改其他用户环境，不干扰他人的GPU任务。

检查当前是否有COCO25K正式实验或其他现有任务正在使用GPU和独立串行控制器。

如果已有正式GPU实验运行：

- 可以完成不占用生产GPU的代码审计与静态准备。
- 不得抢占该实验资源。
- 不启动第二个竞争性长程控制器。
- 将AT运行标记为Ready/Pending，报告资源阻塞原因。

只有在正式预检全部通过、四组RunSpec锁定、资源允许、不会与现有任务冲突时，才能依次执行：

Clean-AT4 → Poison-AT4 → Clean-AT8 → Poison-AT8。

每组200epoch结束后完成checkpoint哈希、last.pt@200验证和完整指标归档，再进入下一组。

不允许因为best.pt提前达到较高性能而提前结束训练；不允许从其他组checkpoint继续训练；恢复仅限相同RunID、相同协议、相同训练状态的合法断点。

## 九、指标与最终结果报告

使用原VOC验证集4952张图，完整20类评价。

四组结果均至少记录：

- Person AP50。
- 19NT mAP50。
- All mAP50。
- Person和19NT AP50:95。
- NT AP75。
- 20类逐类AP。
- WCD与TCD3。
- last.pt@200的完整checkpoint身份。

对于每个AT预算，使用**相同AT预算下的Clean Victim**作为主要分母。

**1. 目标类别保护率 P_AT**

P_AT = 1 - [AP50_person(Poison, AT) / AP50_person(Clean, AT)]

含义：相同AT预算下，Person AP50的相对下降比例。

**2. 非目标类别保留率 U_AT**

U_AT = mAP50_NT(Poison, AT) / mAP50_NT(Clean, AT)

含义：相同AT预算下，非目标19类mAP50的相对保留比例。

**3. 综合评价指标 S_AT**

S_AT = (2 * P_AT * U_AT) / (P_AT + U_AT)

含义：目标保护率与非目标保留率的调和平均。

说明：

- P_AT、U_AT、S_AT计算时均使用0到1之间的比例，展示为百分数时乘以100。
- 计算P_AT前必须确认Clean-AT的Person AP50非零。
- 计算U_AT前必须确认Clean-AT的NT mAP50非零。
- 计算S_AT前必须确认分母非零。
- 遇到无效分母时标记为N/A并说明原因，禁止隐式替换数值或人为设置为零。

同时报告Person和NT的绝对百分点变化，不能只报告自定义S分数。

Clean ST与S4 ST原始结果作为独立参考行保留，不能与同预算Clean-AT分母混用。

另外计算：

- AT-4相对S4 ST的Person恢复幅度。
- AT-8相对S4 ST的Person恢复幅度。
- AT-4/AT-8对非目标效用的影响。
- Clean-AT自身相对Clean-ST的性能变化。
- 两种AT预算下各类别最严重的附带性能变化。

不要预设AT-8一定比AT-4更能恢复Person，也不要预设S4一定能抵御AT。

结果必须通过实际训练和干净val得出，不能由训练损失、短程测试、Frozen模型或已有ST数值推算。

## 十、交付要求

请按以下顺序执行工作：

第一，审计仓库中现有S4数据、Victim训练代码、正式指标和GPU任务，不修改任何已验收产物。

第二，提交独立S4-AT代码及四组冻结配置，确保只扩展下游训练端。

第三，完成PGD-10原生YOLO检测损失梯度验证、预算验证、一个真实optimizer update和恢复测试。

第四，形成正式preflight审计结果，列明PASS/FAIL、源码身份、科学协议及任何差异。

第五，在资源与预检条件全部满足时，串行执行四组正式AT实验；若有现存任务或科学协议阻塞，则停留在Ready/Pending，不强制启动。

第六，导出四组完整报告及Clean-ST/S4-ST对比表，明确未完成的项目，不编造结果。

整个过程保持Original S4作为唯一科学父方法。AT-4/AT-8是Victim侧防御评估，不是S4的新生成版本。

**现在开始审计、代码整理、实现和预检；不要未经检查就直接启动四组长程训练。**