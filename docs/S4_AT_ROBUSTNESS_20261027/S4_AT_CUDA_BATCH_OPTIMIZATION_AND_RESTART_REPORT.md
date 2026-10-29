# S4-AT CUDA Batch优化、科学协议及条件重启报告

**结论：INCONCLUSIVE。** 当前没有CUDA候选等价或真实A/B吞吐结果，未选择新Batch、未重启。安全窗口问题已推进为通过真实RRCTL CPU验证的“自然阶段边界预留”方案；正式队列尚未暂停。

## 1. 授权和当前实际状态

最新授权：[USER_CUDA_BATCH_OPTIMIZATION_AUTHORIZATION_20261030.md](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/docs/S4_AT_ROBUSTNESS_20261027/USER_CUDA_BATCH_OPTIMIZATION_AUTHORIZATION_20261030.md)，SHA `96fd4eae66bf9105e92f32fa609470942c0a8be78b56576f67b06cd50c44af77`。Batch及必要累积可作为新完整协议验证；科学参数仍以冻结RunSpec为基准。

最后核验为北京时间/香港时间 **2026-10-30 01:00:57**（原始UTC 2026-10-29T17:00:57.299322+00:00）。原RunID `s4_at_serial_seed0_20261029_v2`，`S4AT4_Clean_seed0` 正在epoch30，已完成29轮，累计13455batch、7165SGD、6次合法AMP跳过，模型/EMA/优化器和loss有限。

源码仍为 `51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb`，冻结RunSpec SHA `e9f80f82d4288a9b08ed9877b900911f813854e390479eb6a453176c907c87b3`。executor `3338788/389474995`，队列监督器 `3338796/389475008`，科学GPU子进程 `3339007/389475899`；UID1014、PID/start均匹配。原绑定worker zip仍为 `696cb314eb11784a9099728b117eac69cd2e01315d7732154d1ddeab609eff6a`，与控制测试隔离worker不同。

当前四组：CleanAT4 RUNNING；PoisonAT4、CleanAT8、PoisonAT8未启动，尚无200epoch正式验收组。COCO保持已验收发布，无重跑或修改。

## 2. 当前资源与GPU窗口

本次实际只有一个配置过的RRCTL profile `school-s1`，目标节点仅一张 **RTX A6000 48GB，驱动可见46068MiB**。附件中的“24GiB”不适用于当前节点。

00:27附近5次只读采样：GPU89–100%，86°C，功率274–288W，SM1755–1815MHz，4/5软件热限制Active、3/5功率限制Active；仅有本人的正式GPU进程。GPU空闲显存约34GiB，但并非空闲计算卡。最新AMP记录的CUDA分配峰值 9431.94MiB。

00:36资源门槛读数：RAM available 75.16GiB、磁盘空闲884.19GiB，3个I/O wait窗口均0%，swap无换入换出；该短窗口CPU busy为100%，load约15.6/11.8/11.5，不能据此声称CPU始终空闲。原生产准入唯一blocker为 `OWN_GPU_JOB_ALREADY_ACTIVE`。未修改此规则、功率、频率、散热、驱动或其他用户任务。

## 3. Candidate02/03 CUDA等价和测速

既有各5例CPU零容差证据保留；不重复运行。Candidate02合并诊断标量回读，03再向量化审计匹配GT计数。两者尚未完成CUDA batch36/img640/AMP的逐步梯度、loss、完整TAL、扰动、参数梯度/SGD、AMP、BN、EMA、scheduler/RNG和生产resume验证。

工程等价测试继续预锁定 `atol=rtol=0`，不以新授权为由事后扩大容差。若CUDA差异出现，定位原因而非忽略。不同Batch不要求跨Batch训练轨迹相等，必须另外验证新协议的正确性、公平性和稳定性。

| 实现 | Batch | 秒/batch证据 | images/s / epoch正式对照 | 新GPU峰值 |
|---|---:|---|---|---|
| Original | 36 | 历史正式日志约1.6884s/batch；当前尾部1.6168s/batch；不是配对基准 | 未完成受控GPU对照 | 未测量 |
| Candidate02 | 36 | 未测量 | 未完成受控GPU对照 | 未测量 |
| Candidate03 | 36 | 未测量 | 未完成受控GPU对照 | 未测量 |
| Best_pending | 48 | 未测量 | 未完成受控GPU对照 | 未测量 |
| Best_pending | 64 | 未测量 | 未完成受控GPU对照 | 未测量 |
| Best_pending | 72 | 未测量 | 未完成受控GPU对照 | 未测量 |

当前旧实现最近5个已完成epoch平均 780.40s。上述尾部/历史读数含环境与热状态差异，不能直接当作候选A/B的Old单元。纯PGD和纯外层CUDA耗时仍未独立测量。所有候选GPU异常/OOM/NaN及AMP跳步结果为“未测试”，不能填0次。

## 4. 已实际执行的原生Batch协议审计

在CPU上执行从SHA匹配的native Trainer提取的真实表达式，并构建真实YOLOv8n原生SGD参数组；未前向、未更新、未初始化CUDA。这是配置审计，不是GPU训练或性能测试。

| Batch | 稳定accumulate | 满batch有效图像数 | batch/epoch | 尾batch图像数 | 稳定期尝试/epoch | 原nominal0.0005得到的实际decay | 保持旧实际decay需显式nominal |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 2 | 72 | 460 | 27 | 230 | 0.0005625000 | 0.0005000000 |
| 48 | 1 | 48 | 345 | 39 | 345 | 0.0003750000 | 0.0007500000 |
| 64 | 1 | 64 | 259 | 39 | 259 | 0.0005000000 | 0.0005625000 |
| 72 | 1 | 72 | 230 | 63 | 230 | 0.0005625000 | 0.0005000000 |

所有配置的native尾batch保留规则在compile=false时为drop_last=false，算术覆盖16551张；这不是已完成新DataLoader全数据运行的声明。SGD参数组的57组norm、64组decay weight及63组bias数量一致，LR0.01、momentum0.937未改。CPU验证显式nominal可使实际decay精确保持原值0.0005625000000000001；这些只是提案，没有写入正式配置。

- nbs保持64；36稳定累积2，而48/64/72均为1。48的稳定有效batch48，64为64，72为72；72与36×2相同的数值不代表BN或训练等价。
- warmup仍原3epoch，由原生公式得到1380/1035/777/690个batch；36的warmup累积按1→2变化，其余保持1。没有偷偷缩短warmup或强设稳定累积。
- 原生优化器尝试次数第一至第四epoch：36为460/444/230/230；48为345/345/345/345；64为259/259/259/259；72为230/230/230/230。200epoch尝试总数分别46444/69000/51800/46000，**不是预知的真实SGD次数**；仍须扣除实际AMP跳步。
- cosine scheduler仍每epoch触发一次、200epoch、lr0/lrf不变。Batch改变会改变BN统计、每次PGD分组/增强组合、每图像更新与EMA节奏；动态TAL必须照常重新计算。
- 原48/64的实际decay分别较旧降低33.33%/11.11%，不可隐瞒。未来测试拟显式保持旧实际decay，并记录nominal差异供一次科学验收。
- 新Batch仅在完整CUDA/协议验收后才能标为 `NEW_VALIDATED_TRAINING_PROTOCOL`；目前为NEW_TRAINING_PROTOCOL_PENDING。若最终采用，全四组统一从随机初始化重训，披露与Batch36 ST参考的差异。

## 5. 不打断当前训练的阶段间测试窗口

正式监督器当前在`do_wait`等待科学子进程。经用户最新允许的控制工程范围，已实现独立的PID/start/UID/argv/cwd及RRCTL绑定检查，使用Linux pidfd对**单一监督器PID**作STOP/CONT，绝不发给子进程或进程组；不使用TERM/KILL，不改活Release。

真实学校RRCTL CPU04探针通过，RunID `s4_at_queue_window_20261030_cpu04`：4类错误身份（uid/start/argv/pid）均在信号前拒绝；监督器被保持后，子进程进度从tick20继续到tick29并自然完成；第二阶段未开始；同一监督器CONT后才启动下一阶段，最终正常exit0/health/wait/pull一次。0模型、0GPUcontext；正式AT信号数0。

保留了CPU01新输出父目录缺失、CPU02自然退出子进程回收后释放检查过严、CPU03原RRCTL启动身份捕获竞态的全部证据。CPU02已仅恢复其测试监督器并自然完成。隔离控制器副本只修复空cmdline的有界重读，不猜身份；CPU04最终版本完整通过。原在跑worker zip未替换。

正式队列仅做只读inspect/assess，确认精确PID、source和子进程；尚未hold。计划为：

1. 当前CleanAT4继续原样连续训练和评价。先完成可执行、已发布、限步的CUDA验证入口；入口未准备好时不预留窗口。
2. 同一个现有监控器在接近阶段结束时临时缩短检查间隔，不创建第二个监控器；捕获completed_epoch195..199，在监督器仍等待当前科学子进程时，仅hold该监督器。
3. 科学子进程不暂停，继续到200epoch、last/best完整评价、accepted回执和正常退出。独立验收并确认GPU进程真实退出，全部资源与无竞争条件通过后，再执行CUDA等价/Batch基准。
4. 测试窗口上限1小时，测试GPU全部退出后释放同一监督器，让原队列继续后续原配置。测试失败或条件不足也释放，不混入候选代码，不清锁、不恢复训练checkpoint。

该方法不改变当前组计算轨迹，但**下一组会晚启动一个有上限的测试窗口**；这段时间必须纳入净收益计算，不能承诺零墙钟成本。原队列接近结束前不存在独占GPU窗口；不抢占、热替换、强制中断当前训练，也不在两个进程间抢准入竞态。

当前完整CUDA测试入口和新测试RunSpec尚未冻结/发布，窗口尚未实际预留。控制核心CPU通过不等于正式hold、CUDA测试或重启已经执行。若错过可核验的自然边界，保持原实验，不能追赶式暂停新科学子进程。

## 6. 选择、剩余时间与重启条件

目前唯一已有完整前验收并正式运行的配置仍为Original+Batch36。候选02/03和新Batch无真实GPU排序，不能选择“最大Batch”为最佳。

以当前13455batch进度和新读到的最近5个完成epoch均值直接外推，旧方案剩余训练与既有epoch内验证/保存部分约 **167.1小时**。这不是保证ETA，且未计尚未测出的最终额外评价/初始化成本，未来close_mosaic与热状态也会变化。新方案全四组重训耗时、GPU增益、净节省均为null。

若等到CleanAT4完整结束才获得窗口，旧方案大致剩600个epoch，而新Batch按授权须重训800个epoch。在假设零额外成本、四组单位epoch成本相同且全部成本都随加速缩放的理想化下限里，要净省10%需速度比至少800/(600×0.9)=1.4815；**这是决策下限，不是测得48.15%提速**。现实还需加上验证、测试、科学审查、初始化与窗口延迟，因此不能只满足15%单项吞吐就自动重启。

只有逐项CUDA正确性/公平性、一次科学审查及主闭环、稳定实测吞吐≥15%、包含不确定性的总剩余时间节约≥10%、新源码/配置/RunSpec不可变冻结、合法安全退出与回滚都闭合后，才可采用新协议。现在没有新正式RunID/源码SHA、没有停止/重启回执。

## 7. 最终评价及未完成项

四组仍要求200epoch，clean VOC4952/all20、last.pt@200主评价和best次评价、全类AP/AP75/AP50:95、Person/NT19/All、WCD/TCD3，按同预算Clean/Poison计算P/U/S并处理零分母。不依据早期曲线选择有利checkpoint。没有新AT正式结果。

下一可执行条件：先完成有界CUDA入口及本地绑定检查；在同一监控器捕获的自然阶段边界按已验证监督器控制通路预留窗口；GPU真正独占且热状态可比时测试。控制/科学/性能任何一步不足则维持原实现。

证据：[STATE_AND_GPU_WINDOW.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/cuda_batch_optimization_20261030/STATE_AND_GPU_WINDOW.json)；[POST_CONTROL_FORMAL_HEALTH.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/cuda_batch_optimization_20261030/POST_CONTROL_FORMAL_HEALTH.json)；[RESOURCE_GATE_READONLY.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/cuda_batch_optimization_20261030/RESOURCE_GATE_READONLY.json)；[CPU_BATCH_POLICY_AUDIT.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/cuda_batch_optimization_20261030/CPU_BATCH_POLICY_AUDIT.json)；[MAIN_CONTROL_CLOSURE.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/cuda_batch_optimization_20261030/s4_at_queue_window_20261030_cpu04/MAIN_CONTROL_CLOSURE.json)；[RRCTL_IDENTITY_CAPTURE_FIX.patch](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/cuda_batch_optimization_20261030/RRCTL_IDENTITY_CAPTURE_FIX.patch)；[NATURAL_BOUNDARY_WINDOW_PLAN.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/cuda_batch_optimization_20261030/NATURAL_BOUNDARY_WINDOW_PLAN.json)；[PENDING_BATCH_CUDA_CONTRACT.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/cuda_batch_optimization_20261030/PENDING_BATCH_CUDA_CONTRACT.json)。

## 8. 本次安全准备补充（服务器UTC 2026-10-29T17:25:24.096437+00:00，香港10月30日01:25）

原正式CleanAT4仍健康：epoch32、31个完整epoch、14326个batch、7601次实际SGD、6次累计合法AMP跳步；原source/三层PID-start均精确一致。没有hold、信号、GPU探针或重启。

已实现独立的固定输入原生循环等价入口 execution/cuda_batch_optimization_20261030/cuda_equivalence.py，源码SHA 16fb360fb9abb47b0299d26cfcc3504e7f7096508b44bd5df524d95d3bea4ed8。其CPU新入口管线最终cpu_entry04通过：实际两个原生outer batch/20次PGD前向/两侧各一次SGD，零容差比较十步梯度和TAL、最终扰动、完整外层损失与梯度、BN/EMA/optimizer/scaler/scheduler及固定输入边界RNG。测试仅Candidate03/CleanAT4、batch2/img160/workers0/AMP关闭；CUDA未初始化，性能未测。

保留cpu_entry01/02管线失败：首轮只有全局RNG指纹不同；进一步记录发现Python RNG在native on_train_start、PGD尚未开始前已不同，NumPy/Torch相同。最终检查显式回放同一真实增强输入和其入口RNG，符合受控计算比较前提；所有计算后零容差检查保留。记录外部DataLoader RNG但不声称两个独立DataLoader具有逐位相同轨迹。cpu_entry03中间通过也保留。新入口不是完整CUDA/科学/性能验收，不重跑此前已通过的五例CPU候选测试。

入口含真实GPU准入、正确已提交源码/夹具hash、固定batch36/640/16/AMP和32个总outer batch上限检查。尚未发布、尚无当前候选CUDA/resume/ABBA/Batch性能证据，READY仍false。下一步完成其余独立有界入口/RunSpec并冻结发布，再等待既定自然阶段边界；现在不预约或暂停队列。原训练和四组协议完全保留。结论仍INCONCLUSIVE。

## 9. 有界CUDA入口与有限测速准备（2026-10-29T19:47:50.581564+00:00）

新增了GPU固定输入等价、原生有限测速/尾batch、当前候选生产resume绑定、短期只读资源采样与ABBA判定入口。未修改robustness_s4_at、ue_framework、configs或当前Release。测速入口实际跑完原生fixture warmup后再计时，不强设accumulate/scaler；所有PGD/full-GT/有限值/AMP守卫继续执行。独立CUDA事件分解样本与主测速分开，带追踪样本不能作为提速证据。

本地CPU原实现与Candidate03各完成9个原生outer batch/90次PGD；初始状态、进入稳定区状态、增强GT序列、模型/optimizer/scaler/EMA最终指纹与实际更新次数一致。仅batch2/img160/workers0/AMP关闭；CPU速度、CUDA峰值和GPU增益均为null。两次新入口工程失败完整保留：小fixture的原生accumulate末次插值22并非nominal32，以及每epoch仅一batch无法形成同epoch吞吐区间。只修复测试断言/CPU诊断表示，不修改原生数值或GPU计时门槛。

当前resume适配器按所选batch保留原生三完整fixture epoch与真实optimizer phase：36使用已验收684图/19batch的56→[56,58]边界；48/64/72仍用真实原生累积1和各自fixture长度，名义weight decay明确取已审计提案以保持实际decay。该部分目前仅AST/契约检查；没有实际CUDA/resume通过声明。

标准库契约检查通过143个唯一条件RunSpec的真实RRCTL schema验证，以及12个明确SIMULATION_ONLY的判定器测试（拒绝CPU冒充CUDA、错误顺序、GT/参数/AMP差异、Profiler污染、GPU竞争、温度/时钟/CPU漂移与缺监测）。它们不是143次实验，既未注册也未启动。新源码正在准备范围内发布；正式science READY仍false。

实际窗口仅一小时，条件清单不能全部盲跑；若剩余窗口、科学/性能证据或安全退役条件不足，释放同一监督器继续旧方案。当前不hold原队列，不运行任何并发GPU探针。全部真实GPU增益与新batch排序仍未测，结论INCONCLUSIVE。
