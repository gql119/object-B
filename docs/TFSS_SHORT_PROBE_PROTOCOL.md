# 固定短程协议 — PREREGISTERED / WAIT_FOR_S3

机器可读锁：configs/tfss_short_probe.json。复用旧固定32 train/65PersonGT清单及原顺序，前16与后16为两个独立trial；不扩大样本、不读验证集。

每trial：S0-reference、S0-noop、C、U、Uniform-C、Uniform-U分别从seed0原始Fourier/Frozen/Adaptive开始，epoch0原warmup/频率探索、原Adam/Fourier-only参数组、原lr/cosine、原batch16/EOT2、原Adaptive1step，最多5个外层更新；不继承另一候选优化状态。参考与noop是等价性审计，不是两个baseline正式实验。原Adaptive干净query/poison support及轮转顺序保持一致，不引入CNLP的外部query。

Uniform-C/U分别匹配对应候选同batch/step/EOT/图像、Person有效域中 **图像clamp后的实际RMS**。控制自身轨迹的alpha在[0,1]二分求解、detach，不能放大；目标不可达则停止并报告，不能静默换指标或放宽预算。平均绝对值、重叠/独占强度同时记录。轨迹分化后的比较是描述性配对，不能声称每步反事实因果分解。

Gate A：真实S0/noop损失/梯度/参数更新差<=1e-7，实际通道/坐标/模型状态/预算/labels正确，梯度非空且finite。
Gate B：实验前固定的工程代理筛选阈值见JSON：保护objective相对S0均值增幅<=2%、任何batch<=5%；保护梯度cos中位数>=.95、范数保持>=.90、高保护域operator能量保持>=.90。零/平坦/undefined单独列出，不填成成功。全部仅为代理，不能保证最终PersonAP。
Gate C：在A/B通过者之间比较Lret/NT drift/GT box/cls/DFL、逐类分布、与其Uniform对照；两batch仅描述稳定性，不作统计显著或最终AP推论。不优于等强度Uniform就不能声称空间选择性获益。证据冲突可选择INSUFFICIENT_EVIDENCE。

执行顺序：本地静态/CPU检查 -> 完全独立空闲GPU或S3完整接受/归档/进程退出后 -> 当前commit真实1step smoke（首batch，metrics/ingest disabled）-> 单次独立TFSS科学实现审查及必要主代理修复闭合 -> 真实两batch各5step。无资源时WAIT_FOR_S3，不在S3 GPU上试跑。尚无新TFSS审查或GPU pass，不沿用旧TSFL诊断/S23审查冒充。

运行入口scripts/tfss_short_probe.py，必须通过rrctl使用隔离新RunID；输出需在JSON指定独立边界下。脚本同时校验原45文件、本地恢复与历史目录、模型/数据/原型SHA、实际GPU occupancy/CPU load、review receipt。保留所有checkpoint/RNG/Adam/scheduler/频率状态，绝无cleanup。

trace每step/EOT保存原loss、元梯度、selected image ID、逐图强度与保护梯度、原模型update范数。NTGT原生匹配/box/cls/DFL在最后记录的pre-update图像测量；不误称5步更新后的最终结果。NPZ仅保留derived Qt/R/delta/weight供热图；不转移原RGB/数据或权重。结束后仍须分析CSV/图形，成功smoke不会自动产生候选推荐。

2026-10-17 配对数值重放修复：所有独立 reference/candidate/Uniform trial 一致设置 CUBLAS_WORKSPACE_CONFIG=:4096:8、cudnn benchmark=false/deterministic=true、deterministic_algorithms=true；不支持的算子必须报错。原首个 smoke 因 Gate A 差异1.4722e-5>1e-7失败并保留证据，不能当成正式结果。修复不改变公式/原45源码/精度/样本/超参数或 Gate A 容差；当前源码须重新 smoke 后才能调用唯一审查。


2026-10-17 NUMERICAL_STABILITY/PROTOCOL_AMENDMENT（人类明确批准）：Uniform选择最接近的可达FP32 alpha端点，保持[0,1]与原FP32前向，裁剪后实际RMS误差同时满足absolute<=1e-7及relative<=1e-4，配置值必须到达实际求解器；实际误差逐图保存，任一门失败即停止。原1e-9检查及未批准/等待文字仅为保留的历史；配置/样本/eta/预算/损失/优化器及S0/noop1e-7不变。新测试覆盖浮点台阶、绝对与相对门独立失败。源码需新RunID smoke通过后再一次独立审查。


2026-10-17 BUG_FIX / NUMERICAL_STABILITY: valid pixel GT below unchanged canonical S0 P5 pooling minimum follows the existing invalid-Qt preservation convention (W=1 on retained GT, validfalse/raw_normnull, excludedhighQt), not label rejection/zero evidence. No data/optimizer/loss/eta/budget change. Fresh --batch-index1 recovery avoids rerunning completedbatch0; results must disclose split-source provenance. Current-source batch1 smoke/main closure required, no second scientific reviewer. See tfss/SMALL_GT_REPAIR_20261017.md.
