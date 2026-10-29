# Original S4 AT 训练速度审计

结论：**INSUFFICIENT_EVIDENCE**（尚无经过正式 CUDA 对照测量的提速百分比）。运行决策：**CONTINUE_ORIGINAL_AT**。

本报告区分已测量事实、静态代码证据和待验证候选。正式运行、RunSpec、Release、checkpoint、GradScaler 与数据均未修改；无新 GPU 作业、Profiler、信号或并发控制器。

## 1. 当前真实运行身份

- 最后只读核验：2026-10-29T15:09:27.160038+00:00（香港时间加 8 小时）。
- RRCTL RunID：`s4_at_serial_seed0_20261029_v2`，状态 `RUNNING`。
- 当前组：`S4AT4_Clean_seed0`；已完成 20 epoch，正在 epoch 21。
- 累计训练 batch：9515；实际 SGD 更新：5195；合法 AMP 跳过：6。当前尚无一组正式 200-epoch 验收。
- 实际源码：`51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb`；RunSpec SHA：`e9f80f82d4288a9b08ed9877b900911f813854e390479eb6a453176c907c87b3`。
- 协议 SHA：`3bce32a3dc0d907f446535d3c92ca02c8ac04b98a79d342e866bcdba77690b3d`；初始化 SHA：`0ff81f5755379765bfb1c7080b73178b88da200c1cdd2fd9f551cadd2e3b9eb6`。
- executor PID/start：`3338788/389474995`；workload：`3338796/389475008`；训练 CUDA 进程：`3339007/389475899`。最后均为 UID1014，身份一致且存活。
- 实际 Release 的 7 个相关源码/配置文件与本地只读副本 SHA 一致；当前环境 12 个 pinned native 文件 SHA 均匹配冻结协议。
- COCO 已完成并发布；此次没有读取其权重、重跑或更改 COCO。

正式配置：VOC20 train16551、clean val4952、Person14、随机 YOLOv8n seed0、batch36、640、workers16、200epoch、PGD10、AT4/AT8步长1/255与2/255、完整原生检测损失。额外冻结事实：**PGD 内层 FP32，外层 AMP；nbs64，原生 accumulate=2（warmup 期间按原生规则变化）**。cache=true、val=true、plots=true、compile=false、profile=false。

## 2. 资源实测

采样窗口：2026-10-29T14:54:21.600237+00:00 至 2026-10-29T14:54:50.090474+00:00，8 次，间隔约 4 秒，共 28.49 秒。

| 项目 | 实测 | 含义 |
|---|---|---|
| GPU计算利用率 | 48–100%，均值 83.5% | 显存较低不能解释为算力空闲；短时有波动 |
| NVIDIA显存活动指标 utilization.memory | 39–100%，均值 81.75% | 是显存读写活动采样指标，不是显存容量占用或实测带宽百分比 |
| 全设备显存 / 本训练进程显存 | 10,347 / 10,018 MiB | 容量有余量 |
| AMP CSV记录的正式 CUDA分配峰值 | 8990.74 MiB（8.78 GiB） | 当前运行真实历史峰值；不等同于 allocator reserved 或 nvidia-smi总占用 |
| 温度 / 功率 / SM时钟 | 85–86°C / 267.01–280.16W / 1770–1800MHz | 软件热限制 6/8 次 Active，软件功率限制 4/8 次 Active；硬件热限制和 power-brake 为 Not Active |
| 主机CPU busy / 本训练进程树 | 45.36% / 1.18 核 | 本次窗口不支持训练由 CPU 算力耗尽主导的判断 |
| RAM available | 71.50–73.86 GiB | 未见内存容量压力 |
| Swap used / 本窗口换入换出 | 3.93 GiB / 4页入、0页出 | 换入仅 16 KiB，未见持续交换压力 |
| 系统I/O wait | 0.0132% | 非磁盘I/O拥塞主导的证据 |
| 训练进程树块读写 | 0 / 300 KiB | 缓存路径下当前磁盘读取很少 |
| 其他UID CUDA compute-app | 所有8次均为0 | 仅报告聚合数量；未操作其他用户进程 |

训练进程的 CPU PSS 单次读数约 12.99 GiB。不能将 49 个进程的 RSS 直接相加作为真实 RAM 用量，因为缓存/共享页面会被重复计算。采样时 48 个子进程通常处于 poll sleep，这与预取队列已有数据相容，但**没有直接测量 DataLoader 队列长度或 next() 等待时间**。

## 3. 实际耗时分解与缺失证据

| 时间项 | 已测量结果 | 精确范围 |
|---|---|---|
| 训练 batch完整墙钟间隔 | warm batch 8798 个，均值 1.6884s；p50 1.6557s；p90 1.7901s；p95 1.8187s | AMP日志相邻batch末尾间隔，含数据获取、PGD、外层和审计；排除跨epoch与每epoch前10个batch |
| 最近32条训练间隔 | 均值1.6809s、p50 1.6587s | 同一正式实现，无性能候选混入 |
| PGD结束至batch progress时间戳 | 均值 0.1302s，范围 0.1045–0.1628s | 包含外层前后向、可能的SGD/EMA、原生显示和AMP审计；不是纯外层CUDA耗时，且不含随后progress fsync |
| 10次PGD纯GPU耗时 | **未测量** | 日志无每步CUDA event；不能将其余约1.5s直接命名为PGD时间 |
| 每次PGD前向/输入梯度反向 | **未测量** | 不向当前进程附加Profiler |
| 外层前向/反向/SGD各自耗时 | **未分别测量** | 上面的联合时间窗不能分开三者 |
| DataLoader独立等待 | **未测量** | 低I/O和高GPU活动支持其非主要瓶颈，不能据此声称等待为零 |
| CUDA同步与日志独立开销 | **未测量** | 静态发现多处同步来源，尚无隔离A/B成本测量 |
| epoch端到端 | 19个完成epoch的均值 789.84s；最近5个均值 785.34s（13.09min） | 原生results.csv累计time相减；指标记录边界不等于纯训练或checkpoint耗时 |
| 控制台epoch训练显示 | 最近5个完成epoch约12:29–13:12 | 460个batch/epoch，TQDM显示值有秒级取整 |
| 控制台验证显示 | 最近5轮69/69 batch约7.7–13.4s | 保留clean val4952，不能据此关闭val改变冻结协议 |
| epoch末至下一epoch首batch末 | 9.83–30.96s | 合并验证、保存、回调、下轮启动及首batch；不能单独当作保存时间 |
| checkpoint保存纯耗时 | **未测量** | 文件mtime间距不是序列化耗时计时器 |

已在CPU读取封存 `epoch_019.pt`：19 epoch、8740 batch、4808真实SGD、6次AMP跳过，原生phase last_opt_step8739/accumulate2/460batch，safe_optimizer_boundary=true；文件大小和mtime未变化，读取前后 CUDA_initialized=false。模型/优化器/Scaler/RNG/EMA完整状态存在；workers16预取轨迹验证标记仍为false。该事实不构成“任意中断可逐位恢复”的证明。

## 4. AMP与有效更新

正式AMP CSV本次统计到 **8998 batch、4943 次尝试、4937 次实际SGD、6次跳过**。

- 跳过占尝试比例：0.1214%；占batch比例：0.0667%。6次均在epoch1，后续已完成epoch未再出现跳过。
- 4055 个batch没有优化器尝试，是冻结 nbs64/batch36 导致的原生梯度累积，**不是AMP丢弃这些训练batch**。
- 最新已读scale4096，模型/EMA/优化器有限；内层不计算参数梯度，且保留外层已累积梯度。
- 不修改Scaler、更新频率或梯度累积规则。

## 5. 主要瓶颈判断

当前证据最支持 **PGD10 FP32计算与审计同步共同构成主要训练路径成本，并受到已出现的软件热/功率限制影响**。磁盘I/O、swap或增加worker不是本次数据支持的主要方向。GPU计算与显存活动较高，但缺少每段CUDA event，因此无法量化计算、内存传输、同步各自的百分比。

源码中每次PGD均重新调用当前模型和完整检测损失、动态TAL、当前输入梯度。没有缓存预测、TAL或梯度。参数 requires_grad 在内层关闭，BN统计冻结并核验，已不存在“内层计算全部参数梯度”这一可删项。

## 6. 优化候选与风险

| 候选 | 实际实现/证据 | 风险与可用性 |
|---|---|---|
| 02：合并诊断标量CPU回读 | 保留每步即时finite/GT/梯度守卫与全部计算，暂存detached loss/parts/norm，PGD10结束后统一回读；CPU5例零容差PASS | 不改损失/更新数学；CUDA及真实性能待测，尚不能正式使用 |
| 03：再向量化“匹配GT数量”审计 | 将每步逐图36次unique改成图像ID与localGT联合键一次unique；仍逐步计算动态TAL；CPU5例完整TAL张量和audit零容差PASS | 仅审计计数重组；CUDA动态形状及性能待验证，尚不能正式使用 |
| 合并GPU finite flags回读 | 目前raw预测、loss、gradient与BN多处bool/float触发同步 | 尚未实现；需保持每个原有失败守卫，不能减少检查或延迟到失效梯度已被采用后 |
| 日志缓冲/异步落盘 | 每batch PGD append、AMP CSV flush、latest JSON，progress atomic fsync | 尚未实现；需验证崩溃证据、顺序与异常传播，不可简单关闭审计 |
| 同一PGD内静态GT预处理复用 | 可进一步核查标签打包；本轮未替换native criterion | 不得缓存动态TAL、模型输出、loss或梯度 |
| FP64预算端点/最终delta计算重组 | 当前严格预算检查有大张量FP64运算 | 历史precision修复必须保留；不得直接改FP32，需额外科学复验 |
| 输入layout、compile、cuDNN benchmark、TF32 | 未修改、未测试 | 可能改变数值路径，列为额外科学验证方向，不直接采纳 |
| 数据加载参数增加 | cache=true；native pin_memory开启、训练prefetch_factor4；InfiniteDataLoader重用iterator与workers | 已有机制有效，无证据支持增加workers。不能仅因persistent_workers字段未显式true而判断每epoch都重建 |

线程数已由worker设为1；当前native Linux CUDA模型路径按channels_last默认生效，已有current-source baseline布局证明可复用。不能从进程外直接读取活模型stride，未声称当前所有输入张量均为channels_last。代码未开启profile/compile/anomaly detection；未在PGD循环找到empty_cache，原生内存清理仅在epoch等阶段按阈值调用。

## 7. 已完成隔离验证与真实性能

候选02与03各执行5例：真实Clean/Poison训练fixtures的AT4、AT8，以及显式标注的空GT边界。使用同一已增强输入、相同初始模型、完整原始GT、PGD RNG、预算/步长，逐步比较10次输入与输入梯度、loss、完整TAL输出、最终扰动和所有diagnostic字段；再比较原生裁剪/SGD更新、模型/BN、优化器、Scaler、EMA与scheduler。全部 **atol0/rtol0** PASS。一个案例还比较了内存fullstate恢复后的下一次PGD；不冒充原生workers16中断恢复测试。

限制：CPU batch2、img160、workers0、AMP关闭；真实小目标覆盖未由这两个fixture建立。**尚无batch36/img640/CUDA/AMP完整等价结论，尚无GPU Old/New秒/batch、提速百分比、编译/启动成本或CUDA峰值变化**。这些字段均为null/未测量，而非0%或无差异。

初次CPU调用因sandbox无法读取已有NumPy依赖，在模型/测试开始前退出；记录保留。经授权的CPU-only依赖读取后，候选02和03分别正常exit0。原生scheduler在epoch开始step的CPU警告如实保留，参考与候选使用相同调用次序，未据此更改冻结调度。

## 8. 正式配对实验与下一步门槛

当前Clean与后续Poison保持原始51ce790实现。候选没有进入正式worker、Release或RunSpec，也未只对Poison启用新实现。只有完整current-source CUDA等价/真实吞吐测量、合法恢复、一次科学审查和主闭环全通过，才能判断替换与配对重做范围。

现有GPU仍由正式AT使用；RRCTL同UID互斥保留。`Controller.resume`仅重建本地索引；没有受控非终态pause/continue接口。`abort`含tmux终止、TERM后可能SIGKILL兜底，不是合作式checkpoint暂停。当前queue创建fail-on-existing输出与保留锁，不支持直接重入恢复整条队列。本次未调用abort或发出任何信号。

因此不能为了取得性能数据先不安全地中断当前实验，亦不能并行跑GPU基准。完整测试合同已准备，保持所有GPU/科学门槛。建议继续当前正式AT；优化结论暂为 **INSUFFICIENT_EVIDENCE**，没有可报告的实测GPU提速百分比。

## 9. 证据

- [LIVE_SPEED_AUDIT.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/observations/20261029T145420Z/LIVE_SPEED_AUDIT.json)：实时采样、完整AMP聚合、PGD尾部、源哈希、checkpoint stat。
- [RRCTL_INSPECT.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/observations/20261029T145420Z/RRCTL_INSPECT.json)：冻结binding。
- [CHECKPOINT_AND_NATIVE_METADATA.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/observations/20261029T145420Z/CHECKPOINT_AND_NATIVE_METADATA.json)：CPU checkpoint元数据、native哈希、真实训练/验证console片段。
- [FINAL_LIVE_CHECK.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/FINAL_LIVE_CHECK.json)：最后PID/start与进度。
- [CPU_EQUIVALENCE.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/cpu_candidate02/CPU_EQUIVALENCE.json)；[CPU_EQUIVALENCE.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/cpu_candidate03/CPU_EQUIVALENCE.json)。
- [CANDIDATE_02_AST_DIFF.patch](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/CANDIDATE_02_AST_DIFF.patch)；[CANDIDATE_03_AST_DIFF.patch](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/CANDIDATE_03_AST_DIFF.patch)；[CANDIDATE_MANIFEST.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/CANDIDATE_MANIFEST.json)。
- [PENDING_CUDA_BENCHMARK_CONTRACT.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/PENDING_CUDA_BENCHMARK_CONTRACT.json)。


## 后续控制工程进展（2026-10-30香港时间）
原先“无已验证暂停通路”的诊断现已部分推进：独立CPU04真实RRCTL探针验证了仅保持队列监督器、不暂停科学子进程，并在子进程自然结束后释放同一监督器的机制。当前正式队列未操作，CUDA/速度仍未测量；详见 [S4_AT_CUDA_BATCH_OPTIMIZATION_AND_RESTART_REPORT.md](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/docs/S4_AT_ROBUSTNESS_20261027/S4_AT_CUDA_BATCH_OPTIMIZATION_AND_RESTART_REPORT.md)。旧报告原始测量结论保留。
